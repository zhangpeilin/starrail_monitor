# -*- coding: utf-8 -*-
"""进度回退 debug 模式：重现 read_progress 全链路，输出彩色原图+编号图+逐段匹配明细。
用法: python debug_fallback.py <frame1.png> <frame2.png> ...
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import starrail_monitor as _sm
from battle_progress import BattleTracker
from template_matcher import TemplateMatcher

FRAME_W, FRAME_H = 1280, 720


def font(size):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def zoom(img, k, interp=Image.LANCZOS):
    return img.resize((img.width * k, img.height * k), interp)


def save_png(img, path):
    img.save(path)


def main():
    out_dir = "logs/debug_fallback_003516"
    os.makedirs(out_dir, exist_ok=True)
    frames = sys.argv[1:]
    bp = BattleTracker(_sm.OcrEngine())
    bp._progress_matcher = TemplateMatcher()
    bp._progress_matcher._load("templates/battle/digits")

    for fi, path in enumerate(frames):
        name = os.path.basename(path)[: 8 + 9]  # frame_20260809_003516
        rgb = np.array(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        H, W = gray.shape

        # ---------- 存档帧 = 条带裁剪图：帧内匹配剑图标 ----------
        sword_tpl = bp._scale(bp._sword, 1280, 720)  # 按全帧尺寸缩放
        res = cv2.matchTemplate(gray, sword_tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        th, tw = sword_tpl.shape
        sx1 = maxloc[0] + tw   # 剑右缘（帧内坐标）
        print(f"[{name}] 剑匹配分数={maxv:.3f} 位置=({maxloc[0]},{maxloc[1]}) "
              f"尺寸={tw}x{th} 右缘={sx1}")

        # ---------- read_progress 链路重现（帧内坐标） ----------
        rx0 = sx1 + 42
        rx1 = min(W, sx1 + 120)
        ry0 = int(H * 0.2)
        ry1 = int(H * 0.85)
        sub = gray[ry0:ry1, rx0:rx1]
        mask = sub > 180

        # 原图彩色搜索区
        sub_rgb = rgb[ry0:ry1, rx0:rx1]

        # 1) 整帧标注图
        img_full = Image.fromarray(rgb)
        d = ImageDraw.Draw(img_full)
        d.rectangle((maxloc[0], maxloc[1], maxloc[0] + tw, maxloc[1] + th),
                    outline=(0, 0, 255), width=2)
        d.rectangle((rx0, ry0, rx1, ry1), outline=(255, 0, 255), width=2)
        save_png(img_full, os.path.join(out_dir, f"{name}_0_full.png"))

        # 2) 搜索区放大彩色图
        k = 5
        sub_zoom = zoom(Image.fromarray(sub_rgb), k)
        save_png(sub_zoom, os.path.join(out_dir, f"{name}_1_search_zoom.png"))

        # ---------- 组件提取 ----------
        comps = [c for c in _sm.components(mask) if c[4] >= 15]
        max_area = max(c[4] for c in comps) if comps else 0
        max_h = max(c[3] - c[1] + 1 for c in comps) if comps else 0
        print(f"[{name}] mask像素={int(mask.sum())} 初始组件数={len(comps)} max_area={max_area} max_h={max_h}")
        comps = [c for c in comps
                 if c[4] >= max(20, max_area * 0.3)
                 and (c[3] - c[1] + 1) >= max_h * 0.6]
        comps.sort(key=lambda c: c[0])
        print(f"[{name}] 过滤后组件数={len(comps)}")
        for c in comps:
            print(f"    comp x{c[0]}-{c[2]} y{c[1]}-{c[3]} 面积{c[4]}")

        # 列投影分割（宽组件>14）
        if comps and max(c[2] - c[0] + 1 for c in comps) > 14:
            colsum = mask.sum(axis=0)
            col_segs = []
            cur = None
            for i, v in enumerate(colsum):
                if v > 0:
                    if cur is None:
                        cur = [i, i]
                    else:
                        cur[1] = i
                else:
                    if cur is not None and i - cur[1] > 2:
                        col_segs.append(tuple(cur))
                        cur = None
            if cur is not None:
                col_segs.append(tuple(cur))
            comps = []
            for sx, ex in col_segs:
                ys, xs = np.where(mask[:, sx:ex + 1])
                if len(ys):
                    comps.append((sx, ys.min(), ex, ys.max(),
                                  int(mask[:, sx:ex + 1].sum())))
            print(f"[{name}] 列投影分割后组件数={len(comps)}")
            for c in comps:
                print(f"    col x{c[0]}-{c[2]} y{c[1]}-{c[3]} 面积{c[4]}")

        # x 重叠合并
        merged = []
        for c in comps:
            if merged and c[0] < merged[-1][2] - 2:
                m = merged[-1]
                merged[-1] = (min(m[0], c[0]), min(m[1], c[1]),
                              max(m[2], c[2]), max(m[3], c[3]), m[4] + c[4])
            else:
                merged.append(c)
        print(f"[{name}] 合并后段数={len(merged)}")
        for c in merged:
            print(f"    seg x{c[0]}-{c[2]} y{c[1]}-{c[3]} 面积{c[4]}")

        # ---------- 3) 编号图（搜索区放大，每段画框+编号） ----------
        anno = Image.fromarray(sub_rgb)
        da = ImageDraw.Draw(anno)
        for i, c in enumerate(merged):
            da.rectangle((c[0], c[1], c[2], c[3]), outline=(255, 0, 0), width=1)
            da.text((c[0], max(0, c[1] - 10)), f"#{i}",
                    fill=(255, 0, 0), font=font(11))
        save_png(zoom(anno, k), os.path.join(out_dir, f"{name}_2_segments.png"))

        # ---------- 4) 逐段匹配明细 ----------
        digits = []
        result = None
        m = bp._progress_matcher
        for i, c in enumerate(merged):
            seg = mask[c[1]:c[3] + 1, c[0]:c[2] + 1].astype(np.uint8) * 255
            h = seg.shape[0]
            w_new = max(1, int(round(seg.shape[1] * 16 / h)))
            r = cv2.resize(seg, (w_new, 16), interpolation=cv2.INTER_LANCZOS4)
            ys, xs = np.where(r > 127)
            r = r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            d_min = {}
            for d, tpls in m.templates.items():
                best_for_d = 1e9
                for tpl in tpls:
                    w = max(r.shape[1], tpl.shape[1])
                    c1 = np.zeros((16, w), np.uint8)
                    c2 = np.zeros((16, w), np.uint8)
                    c1[:r.shape[0], :r.shape[1]] = r
                    c2[:tpl.shape[0], :tpl.shape[1]] = tpl
                    diff = float(np.abs(c1.astype(int) - c2.astype(int)).mean())
                    best_for_d = min(best_for_d, diff)
                d_min[d] = best_for_d
            order = sorted(d_min.items(), key=lambda kv: kv[1])
            best_d, best_diff = order[0]
            second_diff = order[1][1] if len(order) > 1 else 1e9
            hit = best_diff < 45 and best_diff < second_diff * 0.85
            top5 = " ".join(f"{d}:{diff:.1f}" for d, diff in order[:5])
            print(f"[{name}] 段#{i} x{c[0]}-{c[2]} 尺寸{r.shape[1]}x{r.shape[0]}")
            print(f"        匹配: {top5} | best={best_d}({best_diff:.1f}) "
                  f"second={second_diff:.1f} -> {'命中' if hit else '失败'}")
            # 段彩色裁剪放大 + 掩码
            seg_rgb = sub_rgb[c[1]:c[3] + 1, c[0]:c[2] + 1]
            seg_zoom = zoom(Image.fromarray(seg_rgb), k)
            save_png(seg_zoom, os.path.join(out_dir,
                                            f"{name}_3_seg{i}_x{c[0]}.png"))
            seg_mask = Image.fromarray(seg).resize(
                (seg.shape[1] * k, seg.shape[0] * k), Image.NEAREST)
            save_png(seg_mask, os.path.join(out_dir,
                                            f"{name}_3_seg{i}_mask.png"))
            if not hit:
                # OCR 兜底结果
                crop = Image.fromarray((255 - seg.astype(np.uint8) * 255)).resize(
                    (w_new * 8, 16 * 8), Image.LANCZOS)
                text = bp._ocr_digits(crop)
                print(f"        OCR兜底: '{text}'")
                if len(text) == 1 and text.isdigit():
                    digits.append(int(text))
                    continue
                print(f"        段{i} 失败 -> 连续前缀停止")
                break
            digits.append(best_d)
        if digits:
            v = 0
            for d in digits:
                v = v * 10 + d
            if v <= 100:
                result = v
        print(f"[{name}] 最终识别: {result if result is not None else 'None(拒绝)'} "
              f"(前缀={digits})")
        print("-" * 60)


if __name__ == "__main__":
    main()
