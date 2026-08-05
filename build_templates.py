# -*- coding: utf-8 -*-
"""从存档帧提取 0-9 数字模板（白色行动值数字）

用法: venv\\Scripts\\python.exe build_templates.py
输出: templates/digits/{0..9}.png  （黑字白底，统一归一化到高 16px）
"""
import sys
import os
import re
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import numpy as np

from starrail_monitor import OcrEngine, locate_bar, Extractor

FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "frames")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "digits")

TARGET_H = 16  # 模板统一高度（像素）


def digit_segments(white_mask, x0, y0, x1, y1):
    """行动值区域白色掩码 → 按列投影分割出单个数字的 bbox 列表"""
    sub = white_mask[y0:y1 + 1, x0:x1 + 1]
    colsum = sub.sum(axis=0)
    segs = []
    cur = None
    for i, v in enumerate(colsum):
        if v > 0:
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        else:
            if cur is not None and i - cur[1] > 2:  # 间隙>2列 → 分割
                segs.append(tuple(cur))
                cur = None
    if cur is not None:
        segs.append(tuple(cur))
    result = []
    for sx, ex in segs:
        ys, xs = np.where(sub[:, sx:ex + 1])
        if len(ys):
            result.append((x0 + sx, y0 + ys.min(), x0 + ex, y0 + ys.max()))
    return result


def norm_digit(white_mask, bbox):
    """按 bbox 裁剪数字，归一化到 TARGET_H 高（等比）"""
    x0, y0, x1, y1 = bbox
    sub = white_mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255
    h = sub.shape[0]
    if h <= 0:
        return None
    w_new = max(1, int(round(sub.shape[1] * TARGET_H / h)))
    img = Image.fromarray(sub, "L")
    img = img.resize((w_new, TARGET_H), Image.LANCZOS)
    a = np.array(img)
    # 再次收缩到实际内容 bbox（去掉 resize 产生的边缘）
    ys, xs = np.where(a > 127)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    if not os.path.isdir(FRAMES):
        print("未找到存档帧目录", FRAMES)
        return 1
    ocr = OcrEngine()
    files = sorted(f for f in os.listdir(FRAMES) if f.startswith("frame_"))
    samples = {str(d): [] for d in range(10)}
    n_ok = 0
    for f in files:
        img = Image.open(os.path.join(FRAMES, f)).convert("RGB")
        bar = locate_bar(img)
        if bar is None:
            continue
        crop = img.crop(bar)
        ex = Extractor(ocr)
        tg, ag, rg, info = ex.extract(crop)
        m = re.search(r"行动值区域\((\d+),(\d+)\)-\((\d+),(\d+)\)", info)
        if not m:
            continue
        x0, y0, x1, y1 = (int(v) for v in m.groups())
        a = np.array(crop.convert("RGB")).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        white = (r > 190) & (g > 190) & (b > 190)
        segs = digit_segments(white, x0, y0, x1, y1)
        if not segs:
            continue
        # 判定该帧行动值位数：2段=2位数，1段=1位数
        if len(segs) == 1:
            digits = [ag[0] if ag else None]
        elif len(segs) == 2:
            digits = None
            if ag and len(str(ag[0])) == 2:
                digits = [int(c) for c in str(ag[0])]
            elif rg:
                # 兜底：整区域识别结果
                all_d = "".join(rg)
                if len(all_d) >= len(segs):
                    digits = [int(c) for c in all_d[:len(segs)]]
        else:
            continue
        if digits is None:
            continue
        ok = True
        for d_val, seg in zip(digits, segs):
            try:
                d_val = int(d_val)
            except (TypeError, ValueError):
                ok = False
                break
            if not (0 <= d_val <= 9):
                ok = False
                break
            nd = norm_digit(white, seg)
            if nd is None:
                ok = False
                break
            samples[str(d_val)].append(nd)
        if ok:
            n_ok += 1
    print("成功处理帧数: %d" % n_ok)
    for d in range(10):
        print("数字%d: %d 个样本" % (d, len(samples[str(d)])))

    # 选代表样本：与同数字平均最接近的
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
    for d in range(10):
        lst = samples[str(d)]
        if not lst:
            print("警告: 缺少数字%d 的模板，需补充样本" % d)
            continue
        # 尺寸对齐到最大宽高（样本高度≤16，收缩过内容）
        max_w = max(s.shape[1] for s in lst)
        max_h = max(s.shape[0] for s in lst)
        mats = []
        for s in lst:
            canvas = np.zeros((max_h, max_w), np.uint8)
            canvas[:s.shape[0], :s.shape[1]] = s
            mats.append(canvas)
        avg = np.mean(np.stack(mats), axis=0)
        best = min(range(len(mats)), key=lambda i: np.mean((mats[i].astype(float) - avg) ** 2))
        Image.fromarray(mats[best], "L").save(os.path.join(OUT_DIR, "%d.png" % d))
        print("数字%d 模板已保存 (%dx%d)" % (d, mats[best].shape[1], mats[best].shape[0]))
    print("模板目录:", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
