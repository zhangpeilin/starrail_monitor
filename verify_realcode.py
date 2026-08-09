# -*- coding: utf-8 -*-
"""真实代码验证：1920x1080 几何重现 + 调用 BattleTracker._match_progress_digit（条件对齐已生效）"""
import numpy as np, cv2, glob, os
from PIL import Image
import starrail_monitor as _sm
from battle_progress import BattleTracker

bp = BattleTracker(_sm.OcrEngine())
RX0, RY0, RX1, RY1 = 102, 21, 179, 53


def recognize(gray):
    sub = gray[RY0:RY1, RX0:RX1]
    mask = sub > 180
    if int(mask.sum()) < 50:
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
        d = bp._match_progress_digit(seg)
        if d is None:
            break
        digits.append(d)
    if digits:
        v = 0
        for d in digits:
            v = v * 10 + d
        if v <= 100:
            return v
    return None


def main():
    dirs = sys.argv[1:] or ['logs/frames/2026-08-09']
    total = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*_progress_*.png'))):
            total += 1
            gray = np.array(Image.open(f).convert('L'))
            v = recognize(gray)
            print('%s -> %s' % (os.path.basename(f), v))
    print('总帧数:', total)


if __name__ == '__main__':
    import sys
    main()
