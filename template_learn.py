# -*- coding: utf-8 -*-
"""模板自学习：增量吸收存档帧样本 + 回放对比门禁

流程（防反馈回路退化，只升不降）：
1. 扫描 logs/frames 中新增的「接受帧」（文件名无 drop_ 标记，时间戳晚于上次学习）
2. 每帧分割数字段，与当前模板匹配——标签=文件名 t/a 值；
   质量门槛：匹配到标签数字且分数 ≥ 0.80 才入样本池（错标签/异常形态被挡）
3. 从样本池生成每数字多个变体（贪心多样性选择，覆盖形态差异）
4. 回放门禁：全部接受帧用新旧模板各识别一遍，
   新模板识别正确率 ≥ 旧模板才生效写入 templates/digits/，否则回滚保留旧模板
5. 学习状态记录到 templates/learn_state.json
"""
import json
import os
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from starrail_monitor import (OcrEngine, locate_bar, extract_column,  # noqa: E402
                              Extractor, mask_to_image)
from template_matcher import TemplateMatcher, TARGET_H  # noqa: E402

MIN_NEW = 20          # 新增帧少于该数不学习
MAX_PER_DIGIT = 30    # 每数字样本池上限
VARIANT_COUNT = 3     # 每数字模板变体数（质量优先，主流形态代表）
MIN_ACCEPT_SCORE = 0.78  # 样本入池质量门槛（标签分数须为该段最高且≥此值）
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "templates", "learn_state.json")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "templates", "samples")
DIGITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "templates", "digits")


def _ncc(a, b):
    """归一化互相关（两图须同尺寸）"""
    if a.shape != b.shape:
        return -1.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0][0])


def _frame_ts(name):
    m = re.match(r"frame_(\d{8})_(\d{6})_(\d{3})", name)
    if not m:
        return None
    return "%s%s%s" % (m.group(1), m.group(2), m.group(3))


def _load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_ts": "", "learn_count": 0,
                "acc_old": None, "acc_new": None, "applied": None}


def _save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def _load_pool():
    """从 templates/samples/{d}/ 加载样本池：{d: [数组]}"""
    pool = {d: [] for d in range(10)}
    for d in range(10):
        dd = os.path.join(SAMPLE_DIR, str(d))
        if not os.path.isdir(dd):
            continue
        for fn in sorted(os.listdir(dd)):
            if not fn.endswith(".png"):
                continue
            try:
                img = np.array(Image.open(os.path.join(dd, fn)).convert("L"))
            except Exception:
                continue
            ys, xs = np.where(img > 127)
            if len(ys) == 0:
                continue
            sub = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            pool[d].append(sub)
    return pool


def _save_sample(d, arr, idx):
    dd = os.path.join(SAMPLE_DIR, str(d))
    os.makedirs(dd, exist_ok=True)
    Image.fromarray(arr, "L").save(os.path.join(dd, "s%02d.png" % idx))


def _trim_pool(pool, max_per_digit=MAX_PER_DIGIT):
    """池超限时淘汰与池内平均最不接近的样本"""
    for d in range(10):
        lst = pool[d]
        if len(lst) <= max_per_digit:
            continue
        # 尺寸对齐到最大宽
        max_w = max(s.shape[1] for s in lst)
        max_h = max(s.shape[0] for s in lst)
        mats = []
        for s in lst:
            c = np.zeros((max_h, max_w), np.uint8)
            c[:s.shape[0], :s.shape[1]] = s
            mats.append(c)
        avg = (np.mean(np.stack(mats), axis=0) > 127).astype(np.uint8) * 255
        sims = [_ncc(m, avg) for m in mats]
        keep = sorted(range(len(lst)), key=lambda i: -sims[i])[:max_per_digit]
        pool[d] = [lst[i] for i in keep]


def _pick_variants(pool, k=VARIANT_COUNT):
    """质量优先选择：取与池内平均最接近的 k 个样本作变体。
    主流形态的鲁棒代表——异常/噪声样本与平均相似度低，自动排除。
    （不做多样性选择：覆盖异常形态会把识别带偏，如 9 变体像 7）"""
    variants = {}
    for d in range(10):
        lst = pool[d]
        if not lst:
            continue
        if len(lst) <= k:
            variants[d] = lst
            continue
        max_w = max(s.shape[1] for s in lst)
        max_h = max(s.shape[0] for s in lst)
        mats = []
        for s in lst:
            c = np.zeros((max_h, max_w), np.uint8)
            c[:s.shape[0], :s.shape[1]] = s
            mats.append(c)
        avg = (np.mean(np.stack(mats), axis=0) > 127).astype(np.uint8) * 255
        sims = [_ncc(m, avg) for m in mats]
        picked = sorted(range(len(lst)), key=lambda i: -sims[i])[:k]
        variants[d] = [lst[i] for i in picked]
    return variants


def _write_digits(variants):
    """把变体模板写入 templates/digits/（{d}_{i}.png）；无变体的数字保留旧模板"""
    # 先备份旧模板（单模板或变体）
    old = {}
    for fn in os.listdir(DIGITS_DIR):
        if fn.endswith(".png"):
            try:
                img = np.array(Image.open(os.path.join(DIGITS_DIR, fn)).convert("L"))
            except Exception:
                continue
            m = re.match(r"(\d+)(?:_\d+)?\.png", fn)
            if m:
                old.setdefault(int(m.group(1)), []).append((fn, img))
    for fn in os.listdir(DIGITS_DIR):
        if fn.endswith(".png"):
            os.remove(os.path.join(DIGITS_DIR, fn))
    for d in range(10):
        if d in variants:
            for i, arr in enumerate(variants[d]):
                Image.fromarray(arr, "L").save(
                    os.path.join(DIGITS_DIR, "%d_%d.png" % (d, i)))
        elif d in old:
            # 无新变体：保留旧模板（首个旧文件）
            fn, img = old[d][0]
            Image.fromarray(img, "L").save(os.path.join(DIGITS_DIR, fn))


def _collect_samples(frames, matcher, ocr):
    """对新增接受帧：分割数字，质量门槛入池。返回新增样本计数"""
    added = 0
    for f in frames:
        img = Image.open(f).convert("RGB")
        m = re.match(r"frame_.*_t(\d+)_a(\d+)\.png", os.path.basename(f))
        if not m:
            continue
        st, sa = int(m.group(1)), int(m.group(2))
        bar = locate_bar(img)
        if bar is None:
            continue
        crop = img.crop(bar)
        ex = Extractor(ocr, matcher=matcher)
        tg, ag, rg, info = ex.extract(crop)
        mm = re.search(r"行动值区域\((\d+),(\d+)\)-\((\d+),(\d+)\)", info)
        if not mm or not ag:
            continue
        x0, y0, x1, y1 = (int(v) for v in mm.groups())
        a = np.array(crop.convert("RGB")).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        white = (r > 190) & (g > 190) & (b > 190)
        # 与识别路径一致的放大域掩码图（原始尺度数字间隙会被膨胀核填平合并）
        img_m = mask_to_image(white, max(0, x0 - 3), max(0, y0 - 3),
                              min(crop.width - 1, x1 + 3),
                              min(crop.height - 1, y1 + 3))
        arr_m = np.array(img_m.convert("L"))
        mask = (255 - arr_m)        # 反转：数字=亮
        mask[mask > 127] = 255
        mask[mask <= 127] = 0
        segs = matcher._split_digits(mask)
        label_str = str(ag[0])
        if len(segs) != len(label_str):
            continue
        for seg, ch in zip(segs, label_str):
            lab = int(ch)
            # 宽松匹配：标签数字模板的最高分；要求标签分数是该段最高数字且 ≥ 门槛
            h = seg.shape[0]
            if h <= 0:
                continue
            w_new = max(1, int(round(seg.shape[1] * TARGET_H / h)))
            resized = cv2.resize(seg, (w_new, TARGET_H),
                                 interpolation=cv2.INTER_LANCZOS4)
            ys, xs = np.where(resized > 127)
            if len(ys) == 0:
                continue
            resized = resized[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            best_d, best_s = None, -1.0
            for dd, variants in matcher.templates.items():
                var_best = -1.0
                for tpl in variants:
                    th, tw = tpl.shape
                    rh, rw = resized.shape
                    t = (tpl if (tw == rw and th == rh)
                         else cv2.resize(tpl, (rw, rh),
                                         interpolation=cv2.INTER_LANCZOS4))
                    s = float(cv2.matchTemplate(resized, t,
                                                cv2.TM_CCOEFF_NORMED)[0][0])
                    if s > var_best:
                        var_best = s
                if var_best > best_s:
                    best_s, best_d = var_best, dd
            # 错标签（如真实7被标成1）时该段最高分是7≠1 → 拒；分数低 → 拒
            if best_d != lab or best_s < MIN_ACCEPT_SCORE:
                continue
            # 归一化数字段（高 TARGET_H，收缩内容）入池
            arr = resized[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            idx = len(pool_cache[lab])
            pool_cache[lab].append(arr)
            _save_sample(lab, arr, idx)
            added += 1
    return added


# 全局样本池缓存（进程内复用）
pool_cache = {d: [] for d in range(10)}


def _replay_accuracy(frames, matcher, ocr, sample=1):
    """回放测试：接受帧用指定模板识别，返回 (正确数, 总数)"""
    ok_n = total = 0
    for i, f in enumerate(frames):
        if i % sample != 0:
            continue
        m = re.match(r"frame_.*_t(\d+)_a(\d+)\.png", os.path.basename(f))
        if not m:
            continue
        st, sa = int(m.group(1)), int(m.group(2))
        try:
            img = Image.open(f).convert("RGB")
            t, a, _ = extract_column(ocr, img, matcher=matcher)
        except Exception:
            continue
        total += 1
        if t == st and a == sa:
            ok_n += 1
    return ok_n, total


def _digits_images():
    """加载磁盘模板每数字的变体 PIL 图（用于图形化对比）"""
    from PIL import Image as _I
    imgs = {}
    for d in range(10):
        lst = []
        i = 0
        while True:
            p = os.path.join(DIGITS_DIR, "%d_%d.png" % (d, i))
            if not os.path.isfile(p):
                break
            try:
                lst.append(_I.open(p).convert("L"))
            except Exception:
                pass
            i += 1
        if not lst:
            p = os.path.join(DIGITS_DIR, "%d.png" % d)
            if os.path.isfile(p):
                try:
                    lst.append(_I.open(p).convert("L"))
                except Exception:
                    pass
        imgs[d] = lst
    return imgs


def _arrs_images(arrs):
    """numpy 数组列表 → PIL 图列表"""
    from PIL import Image as _I
    return [_I.fromarray(a, "L") for a in arrs]


def learn_once(script_dir=None, force=False):
    """执行一轮模板学习。返回结构化结果 dict；无需学习时返回 None。
    force=True 时全量重学（清空样本池重新收集，用于手动执行）"""
    global pool_cache
    base = script_dir or os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(base, "logs", "frames")
    if not os.path.isdir(frames_dir):
        return None
    state = _load_state()
    # 新增接受帧（文件名时间戳 > last_ts）
    all_frames = sorted(f for f in os.listdir(frames_dir)
                        if f.startswith("frame_") and "_drop_" not in f)
    new_frames = []
    for fn in all_frames:
        ts = _frame_ts(fn)
        if ts and ts > state.get("last_ts", ""):
            new_frames.append(os.path.join(frames_dir, fn))
    if not force and len(new_frames) < MIN_NEW:
        return None
    if force:
        new_frames = [os.path.join(frames_dir, f) for f in all_frames]
        for d in range(10):
            dd = os.path.join(SAMPLE_DIR, str(d))
            if os.path.isdir(dd):
                shutil.rmtree(dd)

    ocr = OcrEngine()
    old_matcher = TemplateMatcher()
    pool_cache = _load_pool()
    # 旧模板缺失时（首次）用当前磁盘模板匹配；池从新帧全量吸收
    if not old_matcher.ok():
        return None

    added = _collect_samples(new_frames, old_matcher, ocr)
    if added == 0 and not force:
        return None
    _trim_pool(pool_cache)
    for d in range(10):
        dd = os.path.join(SAMPLE_DIR, str(d))
        if os.path.isdir(dd):
            for fn in os.listdir(dd):
                if fn.endswith(".png"):
                    os.remove(os.path.join(dd, fn))
        for i, arr in enumerate(pool_cache[d]):
            _save_sample(d, arr, i)

    variants = _pick_variants(pool_cache)
    if len(variants) < 10:
        return {"message": "样本不足：%d/10 个数字有样本" % len(variants),
                "applied": False, "acc_old": None, "acc_new": None,
                "old_total": 0, "new_total": 0,
                "old_images": _digits_images(),
                "new_images": {d: _arrs_images(variants[d])
                               for d in sorted(variants)},
                "samples": {d: len(pool_cache[d]) for d in range(10)}}

    # 回放门禁：新旧模板对比（接受帧全量）
    test_frames = [os.path.join(frames_dir, f) for f in all_frames]
    old_ok, old_total = _replay_accuracy(test_frames, old_matcher, ocr, sample=1)
    tmp_dir = tempfile.mkdtemp(prefix="tmpl_")
    try:
        # 新模板写入临时目录，加载对比
        for fn in os.listdir(DIGITS_DIR):
            if fn.endswith(".png"):
                shutil.copy(os.path.join(DIGITS_DIR, fn), tmp_dir)
        # 先写变体到临时目录：命名 {d}_{i}.png
        for d, lst in variants.items():
            for i, arr in enumerate(lst):
                Image.fromarray(arr, "L").save(
                    os.path.join(tmp_dir, "%d_%d.png" % (d, i)))
        new_matcher = TemplateMatcher(template_dir=tmp_dir)
        new_ok, new_total = _replay_accuracy(test_frames, new_matcher, ocr,
                                             sample=1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    old_acc = old_ok / old_total if old_total else 0
    new_acc = new_ok / new_total if new_total else 0
    applied = new_total > 0 and new_acc >= old_acc
    if applied:
        _write_digits(variants)
    # 更新状态
    state["last_ts"] = _frame_ts(os.path.basename(all_frames[-1])) if all_frames \
        else state.get("last_ts", "")
    state["learn_count"] = state.get("learn_count", 0) + 1
    state["acc_old"] = round(old_acc, 4)
    state["acc_new"] = round(new_acc, 4)
    state["applied"] = applied
    _save_state(state)
    message = ("模板学习生效：新模板识别率 %.1f%% (旧 %.1f%%)，样本+%d"
               % (new_acc * 100, old_acc * 100, added)
               if applied else
               "模板学习回滚：新模板识别率 %.1f%% < 旧 %.1f%%，保留旧模板（样本+%d）"
               % (new_acc * 100, old_acc * 100, added))
    return {"message": message, "applied": applied,
            "acc_old": round(old_acc, 4), "acc_new": round(new_acc, 4),
            "old_total": old_total, "new_total": new_total,
            "old_images": _digits_images(),
            "new_images": {d: _arrs_images(variants[d])
                           for d in sorted(variants)},
            "samples": {d: len(pool_cache[d]) for d in range(10)}}


def show_report(result):
    """图形化对比报告：新旧模板图片并排 + 识别率对比"""
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("模板学习报告")
    acc_old = result.get("acc_old")
    acc_new = result.get("acc_new")
    if acc_old is not None:
        head = ("%s\n旧模板识别率 %.1f%%（%d帧）  新模板识别率 %.1f%%（%d帧）"
                % (result["message"], acc_old * 100, result.get("old_total", 0),
                   acc_new * 100, result.get("new_total", 0)))
    else:
        head = result["message"]
    tk.Label(root, text=head, font=("Microsoft YaHei UI", 12),
             fg="#b00020" if not result.get("applied") else "#006400",
             justify="left").pack(pady=10, padx=12)

    def row_images(title, images_map, color):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=12, pady=4)
        tk.Label(frame, text=title, font=("Microsoft YaHei UI", 10, "bold"),
                 fg=color).pack(anchor="w")
        cells = tk.Frame(frame)
        cells.pack(anchor="w", pady=2)
        for d in range(10):
            cell = tk.Frame(cells, relief="groove", bd=1)
            cell.pack(side="left", padx=2)
            imgs = images_map.get(d, [])
            tk.Label(cell, text=str(d), font=("Consolas", 9)).pack()
            inner = tk.Frame(cell)
            inner.pack()
            if not imgs:
                tk.Label(inner, text="无", width=4, height=2).pack()
            for im in imgs:
                big = im.resize((im.width * 8, im.height * 8),
                                Image.NEAREST)
                tk.Label(inner, image=ImageTk.PhotoImage(big)).pack(
                    side="left", padx=1)
                # 保持引用防 GC
                cell._imgs = getattr(cell, "_imgs", []) + [ImageTk.PhotoImage(big)]
        samples = result.get("samples", {})
        tk.Label(frame, text="样本数: %s" % {d: samples.get(d, 0)
                                             for d in range(10)},
                 font=("Consolas", 8), fg="gray").pack(anchor="w")

    row_images("旧模板（当前生效）", result.get("old_images", {}), "#00008b")
    row_images("新模板（候选）", result.get("new_images", {}), "#006400")

    tk.Button(root, text="关闭", command=root.destroy,
              font=("Microsoft YaHei UI", 11), width=10).pack(pady=10)
    root.mainloop()


if __name__ == "__main__":
    print("开始模板学习（全量收集 + 回放对比，约 5-8 分钟）...")
    print("提示：期间请勿关闭窗口；日志帧文件夹在滚动时对比会自动重跑")
    result = learn_once(force=True)
    if result is None:
        print("无需学习（无新增接受帧）")
        sys.exit(0)
    print(result["message"])
    show_report(result)
