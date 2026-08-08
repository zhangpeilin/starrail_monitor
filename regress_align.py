# -*- coding: utf-8 -*-
"""全量回归：左对齐 vs 右对齐 模板匹配对全部存档进度帧的识别差异"""
import numpy as np, cv2, glob, os, sys
from PIL import Image
import starrail_monitor as _sm
from template_matcher import TemplateMatcher

m = TemplateMatcher()
m._load('templates/battle/digits')

# 1920x1080 几何：剑右缘=59(12+47)，数字区 rx0=102 rx1=179 ry0=21 ry1=53
RX0, RY0, RX1, RY1 = 102, 21, 179, 53


def recognize(gray, align):
    sub = gray[RY0:RY1, RX0:RX1]
    mask = sub > 180
    if int(mask.sum()) < 10:
        return None
    comps = [c for c in _sm.components(mask) if c[4] >= 15]
    if not comps:
        return None
    max_area = max(c[4] for c in comps)
    max_h = max(c[3] - c[1] + 1 for c in comps)
    comps = [c for c in comps
             if c[4] >= max(20, max_area * 0.3)
             and (c[3] - c[1] + 1) >= max_h * 0.6]
    comps.sort(key=lambda c: c[0])
    if comps and max(c[2] - c[0] + 1 for c in comps) > 14:
        colsum = mask.sum(axis=0)
        col_segs, cur = [], None
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
    merged = []
    for c in comps:
        if merged and c[0] < merged[-1][2] - 2:
            mm = merged[-1]
            merged[-1] = (min(mm[0], c[0]), min(mm[1], c[1]),
                          max(mm[2], c[2]), max(mm[3], c[3]), mm[4] + c[4])
        else:
            merged.append(c)
    digits = []
    for c in merged:
        seg = mask[c[1]:c[3] + 1, c[0]:c[2] + 1].astype(np.uint8) * 255
        h = seg.shape[0]
        w_new = max(1, int(round(seg.shape[1] * 16 / h)))
        r = cv2.resize(seg, (w_new, 16), interpolation=cv2.INTER_LANCZOS4)
        ys, xs = np.where(r > 127)
        if len(ys) == 0:
            break
        r = r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        d_min = {}
        for d, tpls in m.templates.items():
            best = 1e9
            for tpl in tpls:
                w = max(r.shape[1], tpl.shape[1])
                c1 = np.zeros((16, w), np.uint8)
                c2 = np.zeros((16, w), np.uint8)
                if align == 'cond':
                    if r.shape[1] > tpl.shape[1]:
                        c1[:r.shape[0], w - r.shape[1]:] = r
                        c2[:tpl.shape[0], w - tpl.shape[1]:] = tpl
                        best = min(best, float(np.abs(c1.astype(int) - c2.astype(int)).mean()))
                    else:
                        c1[:r.shape[0], :r.shape[1]] = r
                        c2[:tpl.shape[0], :tpl.shape[1]] = tpl
                        best = min(best, float(np.abs(c1.astype(int) - c2.astype(int)).mean()))
                elif align == 'resize':
                    t = cv2.resize(tpl, (r.shape[1], r.shape[0]),
                                   interpolation=cv2.INTER_LANCZOS4)
                    best = min(best, float(np.abs(r.astype(int) - t.astype(int)).mean()))
                elif align == 'both':
                    a1 = np.zeros((16, w), np.uint8); a2 = np.zeros((16, w), np.uint8)
                    a1[:r.shape[0], :r.shape[1]] = r
                    a2[:tpl.shape[0], :tpl.shape[1]] = tpl
                    dL = float(np.abs(a1.astype(int) - a2.astype(int)).mean())
                    b1 = np.zeros((16, w), np.uint8); b2 = np.zeros((16, w), np.uint8)
                    b1[:r.shape[0], w - r.shape[1]:] = r
                    b2[:tpl.shape[0], w - tpl.shape[1]:] = tpl
                    dR = float(np.abs(b1.astype(int) - b2.astype(int)).mean())
                    best = min(best, min(dL, dR))
                elif align == 'right':
                    c1[:r.shape[0], w - r.shape[1]:] = r
                    c2[:tpl.shape[0], w - tpl.shape[1]:] = tpl
                    best = min(best, float(np.abs(c1.astype(int) - c2.astype(int)).mean()))
                else:
                    c1[:r.shape[0], :r.shape[1]] = r
                    c2[:tpl.shape[0], :tpl.shape[1]] = tpl
                    best = min(best, float(np.abs(c1.astype(int) - c2.astype(int)).mean()))
            d_min[d] = best
        order = sorted(d_min.items(), key=lambda kv: kv[1])
        best_d, best_diff = order[0]
        second_diff = order[1][1] if len(order) > 1 else 1e9
        if best_diff < 45 and best_diff < second_diff * 0.85:
            digits.append(best_d)
        else:
            break
    if digits:
        v = 0
        for d in digits:
            v = v * 10 + d
        if v <= 100:
            return v
    return None


def main():
    dirs = sys.argv[1:] or ['logs/frames/2026-08-09']
    diffs = []
    total = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*_progress_*.png'))):
            total += 1
            gray = np.array(Image.open(f).convert('L'))
            rl = recognize(gray, 'left')
            rr = recognize(gray, 'cond')
            if rl != rr:
                diffs.append((os.path.basename(f), rl, rr))
    print('总帧数:', total)
    print('左/右对齐识别不一致帧数:', len(diffs))
    for name, rl, rr in diffs:
        print('  %s  左=%s 右=%s' % (name, rl, rr))
    # 汇总不一致模式
    from collections import Counter
    pat = Counter((rl, rr) for _, rl, rr in diffs)
    print('差异模式:', dict(pat))


if __name__ == '__main__':
    main()
