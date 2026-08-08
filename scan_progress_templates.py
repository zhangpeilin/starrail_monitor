# -*- coding: utf-8 -*-
"""进度数字模板自动采集工具

扫描 logs/frames/**/*_progress_*.png 全量存档：
1. 进度区（帧内 x115-155）+ 关卡区（帧内 x55-115）提取数字组件
2. 已有模板匹配（像素差 <45 + gap 比例）→ 高置信确认 → 新形态自动
   存入变体（差 >20 才存）
3. 匹配失败的段 → 单段 OCR 读标签 → 候选；同一标签候选段两两
   像素差 <20 聚簇（簇 ≥2 自证成立）→ 补入模板（修复 4/5 缺失）
4. 输出报告：新增模板/未确认候选

用法: venv\\Scripts\\python.exe scan_progress_templates.py
"""
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(SCRIPT_DIR, "logs", "frames")
DIGITS_DIR = os.path.join(SCRIPT_DIR, "templates", "battle", "digits")

TARGET_H = 16
DIFF_OK = 45        # 模板匹配确认阈值（跨帧渲染差异可达 38）
DIFF_NEW = 12       # 新形态变体采集阈值（差>12 才存新变体）
GAP_RATIO = 0.85    # 与次优数字的差距比例
CLUSTER_DIFF = 20   # 候选自证聚簇阈值

# 存档帧为条带裁剪（剑+关卡+进度+进度条）：进度数字区 / 关卡数字区
PROGRESS_REGION = (115, 148, 24, 56)   # (x0, x1, y0, y1) 右界避开 %（x150 起）
STAGE_REGION = (55, 112, 24, 56)


def load_templates():
    """加载现有模板 → {d: [变体归一化图]}"""
    tpls = {}
    for fn in os.listdir(DIGITS_DIR):
        m = re.match(r"(\d)(?:_(\d+))?\.png$", fn)
        if not m:
            continue
        d = int(m.group(1))
        arr = np.array(Image.open(os.path.join(DIGITS_DIR, fn)).convert("L"))
        # 与 TemplateMatcher._norm_one 一致：>127 bbox → 16 高归一化
        ys, xs = np.where(arr > 127)
        if len(ys) == 0:
            continue
        sub = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h = sub.shape[0]
        w_new = max(1, int(round(sub.shape[1] * TARGET_H / h)))
        r = np.array(Image.fromarray(sub).resize((w_new, TARGET_H), Image.LANCZOS))
        r = (r > 127).astype(np.uint8) * 255
        ys2, xs2 = np.where(r > 0)
        if len(ys2):
            r = r[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
            tpls.setdefault(d, []).append(r)
    return tpls


def save_template(d, seg):
    """存入模板（找下一个空序号）"""
    idx = 0
    while os.path.exists(os.path.join(DIGITS_DIR, "%d_%d.png" % (d, idx))):
        idx += 1
    w = seg.shape[1]
    canvas = np.zeros((TARGET_H, w), np.uint8)
    canvas[:seg.shape[0], :seg.shape[1]] = seg
    Image.fromarray(canvas, "L").save(os.path.join(DIGITS_DIR, "%d_%d.png" % (d, idx)))


def norm_seg(seg):
    h = seg.shape[0]
    if h <= 0:
        return None
    w_new = max(1, int(round(seg.shape[1] * TARGET_H / h)))
    r = np.array(Image.fromarray(seg).resize((w_new, TARGET_H), Image.LANCZOS))
    r = (r > 127).astype(np.uint8) * 255
    ys, xs = np.where(r > 0)
    if len(ys) == 0:
        return None
    return r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pixdiff(a, b):
    w = max(a.shape[1], b.shape[1])
    c1 = np.zeros((TARGET_H, w), np.uint8)
    c2 = np.zeros((TARGET_H, w), np.uint8)
    c1[:a.shape[0], :a.shape[1]] = a
    c2[:b.shape[0], :b.shape[1]] = b
    return float(np.abs(c1.astype(int) - c2.astype(int)).mean())


def match_digit(seg, tpls):
    """模板匹配：返回 (数字, 最佳差) 或 None"""
    if not tpls:
        return None
    order = []
    for d, variants in tpls.items():
        bd = min(pixdiff(seg, t) for t in variants)
        order.append((bd, d))
    order.sort()
    best_diff, best_d = order[0]
    second_diff = order[1][0] if len(order) > 1 else 1e9
    if best_diff < DIFF_OK and best_diff < second_diff * GAP_RATIO:
        return best_d, best_diff
    return None


def ocr_digit(seg):
    """单段 OCR（候选标签）"""
    try:
        import pytesseract
        if not pytesseract.pytesseract.tesseract_cmd or \
                not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Programs", "Tesseract-OCR", "tesseract.exe")
        h = seg.shape[0]
        w_new = max(1, int(round(seg.shape[1] * TARGET_H / h)))
        crop = Image.fromarray((255 - seg.astype(np.uint8) * 255)).resize(
            (w_new * 8, TARGET_H * 8), Image.LANCZOS)
        text = pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
        if len(text) == 1 and text.isdigit():
            return int(text)
    except Exception:
        pass
    return None


def extract_components(png, region):
    """区域掩码（>180）→ 数字组件（x 重叠聚类）"""
    im = Image.open(png).convert("RGB")
    a = np.array(im).astype(int)
    gray = (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114)
    x0, x1, y0, y1 = region
    x1 = min(x1, gray.shape[1] - 1)
    y1 = min(y1, gray.shape[0] - 1)
    if x1 <= x0 or y1 <= y0:
        return []
    mask = gray[y0:y1 + 1, x0:x1 + 1] > 180
    import starrail_monitor as _sm
    comps = [c for c in _sm.components(mask) if c[4] >= 15]
    if not comps:
        return []
    max_area = max(c[4] for c in comps)
    comps = [c for c in comps if c[4] >= max(20, max_area * 0.3)]
    comps.sort(key=lambda c: c[0])
    # 无膨胀列投影分割（4+4 相邻合并场景分离；纵向断裂由 x 重叠合并修复）
    if len(comps) >= 1 and max(c[2] - c[0] + 1 for c in comps) > 14:
        colsum = mask.sum(axis=0)
        segs = []
        cur = None
        for i, v in enumerate(colsum):
            if v > 0:
                if cur is None:
                    cur = [i, i]
                else:
                    cur[1] = i
            else:
                if cur is not None and i - cur[1] > 2:
                    segs.append(tuple(cur))
                    cur = None
        if cur is not None:
            segs.append(tuple(cur))
        comps = []
        for sx, ex in segs:
            ys, xs = np.where(mask[:, sx:ex + 1])
            if len(ys):
                comps.append((sx, ys.min(), ex, ys.max(), int((mask[:, sx:ex + 1]).sum())))
    # x 重叠合并（0 断裂修复）
    merged = []
    for c in comps:
        if merged and c[0] < merged[-1][2] - 2:
            mm = merged[-1]
            merged[-1] = (min(mm[0], c[0]), min(mm[1], c[1]),
                          max(mm[2], c[2]), max(mm[3], c[3]), mm[4] + c[4])
        else:
            merged.append(c)
    out = []
    for c in merged:
        seg = mask[c[1]:c[3] + 1, c[0]:c[2] + 1].astype(np.uint8) * 255
        n = norm_seg(seg)
        if n is not None:
            out.append((n, os.path.basename(png)))
    return out


def main():
    tpls = load_templates()
    print("现有模板: %s" % {d: len(v) for d, v in sorted(tpls.items())})
    if not os.path.isdir(FRAMES_DIR):
        print("无存档目录", FRAMES_DIR)
        return 1

    frames = sorted(os.path.join(root, f)
                    for root, _d, fs in os.walk(FRAMES_DIR)
                    for f in fs if "_progress_" in f)
    print("扫描进度帧: %d 张" % len(frames))

    added = 0
    candidates = {}          # label -> [seg]
    for f in frames:
        for region in (PROGRESS_REGION, STAGE_REGION):
            for seg, fname in extract_components(f, region):
                r = match_digit(seg, tpls)
                if r is not None:
                    d, best_diff = r
                    # 新形态（与现有变体差 >20）→ 存变体
                    if best_diff > DIFF_NEW and len(tpls[d]) < 8:
                        save_template(d, seg)
                        tpls[d].append(seg)
                        added += 1
                        print("  变体 %s ← %s (差%.1f)" % (d, fname, best_diff))
                else:
                    lab = ocr_digit(seg)
                    if lab is not None:
                        candidates.setdefault(lab, []).append(seg)

    # 候选聚类自证：同标签段两两差 <20 成簇（簇 ≥2 自证成立 → 入模板）
    confirmed = 0
    for lab in sorted(candidates):
        segs = candidates[lab]
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            cluster = [segs[i]]
            used[i] = True
            for j in range(i + 1, len(segs)):
                if not used[j] and pixdiff(segs[i], segs[j]) < CLUSTER_DIFF:
                    cluster.append(segs[j])
                    used[j] = True
            if len(cluster) >= 2 and lab not in tpls:
                # 取与簇平均最接近的代表
                avg = np.mean(np.stack(
                    [np.pad(s, ((0, TARGET_H - s.shape[0]), (0, 0)))
                     for s in cluster]), axis=0)
                best = min(cluster, key=lambda s: pixdiff(s, avg))
                save_template(lab, best)
                tpls[lab] = [best]
                confirmed += 1
                print("  自证新数字 %d ← %d 个候选段聚类" % (lab, len(cluster)))

    print("\n结果：新增变体 %d 个，自证新数字 %d 个" % (added, confirmed))
    print("模板现状: %s" % {d: len(v) for d, v in sorted(tpls.items())})
    if not confirmed and not added:
        print("提示：候选段不足（需同标签 ≥2 段且形态接近才自证）——"
              "全量存档后帧增多即可自动补全")
    return 0


if __name__ == "__main__":
    sys.exit(main())
