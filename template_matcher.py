# -*- coding: utf-8 -*-
"""OpenCV 模板匹配数字识别器（固定字体，抗像素级缺陷）

输入为 mask_to_image 生成的数字区域图（黑字白底，任意比例缩放），
内部反转成「数字=亮」的掩码域，按列投影分割单个数字，
归一化到模板高度后与 0-9 模板做归一化互相关（NCC），取最高分。

优势：数字完整性由整体匹配保证——缺 1-2 像素只会略降分数，
不会像 Tesseract 那样把 77 判成 7（匹配的是整个字形，不是特征）。
"""
import os

import cv2
import numpy as np

TARGET_H = 16        # 模板统一高度
SCORE_MIN = 0.60     # 匹配分数阈值（低于=放弃，宁可不识别不误报）
SCORE_GAP = 0.15     # 最高分与次高分的最小差距（防噪声误判）
GAP_MIN = 3          # 列投影分割间隙阈值（像素）


class TemplateMatcher:
    def __init__(self, template_dir=None, height=TARGET_H, score_min=SCORE_MIN):
        if template_dir is None:
            template_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "templates", "digits")
        self.height = height
        self.score_min = score_min
        self.templates = {}   # d -> [变体图列表]（数字=255，背景=0，高=height）
        self._load(template_dir)

    def _norm_one(self, path):
        """加载单张模板图并归一化到统一高度"""
        from PIL import Image
        try:
            img = np.array(Image.open(path).convert("L"))
        except Exception:
            return None
        ys, xs = np.where(img > 127)
        if len(ys) == 0:
            return None
        sub = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h = sub.shape[0]
        w_new = max(1, int(round(sub.shape[1] * self.height / h)))
        sub = cv2.resize(sub, (w_new, self.height), interpolation=cv2.INTER_LANCZOS4)
        return (sub > 127).astype(np.uint8) * 255

    def _load(self, template_dir):
        # 多样本变体优先：{d}_N.png 全部加载（编号可跳跃——重命名/删除
        # 变体后不依赖连续编号，避免后续变体被漏加载）；
        # 变体存在时旧单模板 {d}.png 一并保留（覆盖多形态）
        try:
            files = os.listdir(template_dir)
        except OSError:
            files = []
        for d in range(10):
            variants = []
            idxs = []
            for fn in files:
                if fn.startswith("%d_" % d) and fn.endswith(".png"):
                    core = fn[len(str(d)) + 1:-4]
                    if core.isdigit():
                        idxs.append(int(core))
            for i in sorted(idxs):
                t = self._norm_one(os.path.join(template_dir, "%d_%d.png" % (d, i)))
                if t is not None:
                    variants.append(t)
            if variants:
                p = os.path.join(template_dir, "%d.png" % d)
                if os.path.isfile(p):
                    t = self._norm_one(p)
                    if t is not None:
                        variants.insert(0, t)   # 旧单模板放最前（默认形态优先）
            else:
                p = os.path.join(template_dir, "%d.png" % d)
                if os.path.isfile(p):
                    t = self._norm_one(p)
                    if t is not None:
                        variants.append(t)
            if variants:
                self.templates[d] = variants

    def ok(self):
        return len(self.templates) == 10

    # ------------------------------------------------------------------
    def _split_digits(self, mask):
        """数字掩码（数字=255）→ 按列投影分割，返回每个数字的掩码子图

        分割在「水平膨胀后的掩码」上进行（填平数字内部的小裂痕，如 1 的
        横笔与竖笔之间的空隙），再用原始掩码裁剪数字内容用于匹配。
        """
        # 水平膨胀 ±4px：数字内部裂痕填平，数字间空隙（>8px）保留
        kernel = np.ones((1, 9), np.uint8)
        dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel)
        colsum = dilated.sum(axis=0)
        segs = []
        cur = None
        for i, v in enumerate(colsum):
            if v > 0:
                if cur is None:
                    cur = [i, i]
                else:
                    cur[1] = i
            else:
                if cur is not None and i - cur[1] > GAP_MIN:
                    segs.append(tuple(cur))
                    cur = None
        if cur is not None:
            segs.append(tuple(cur))
        out = []
        for sx, ex in segs:
            ys, xs = np.where(mask[:, sx:ex + 1] > 0)
            if len(ys):
                out.append(mask[ys.min():ys.max() + 1, sx:ex + 1])
        return out

    def _match_one(self, digit_mask):
        """单个数字掩码 → 与 0-9 模板 NCC 匹配 → (数字, 分数) 或 None"""
        h = digit_mask.shape[0]
        if h <= 0:
            return None
        w_new = max(1, int(round(digit_mask.shape[1] * self.height / h)))
        resized = cv2.resize(digit_mask, (w_new, self.height),
                             interpolation=cv2.INTER_LANCZOS4)
        ys, xs = np.where(resized > 127)
        if len(ys) == 0:
            return None
        resized = resized[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        best_d, best_s = None, -1.0
        second_s = -1.0
        for d, variants in self.templates.items():
            # 每数字取所有变体的最高分（覆盖形态差异）
            var_best = -1.0
            for tpl in variants:
                th, tw = tpl.shape
                rh, rw = resized.shape
                if tw == rw and th == rh:
                    t = tpl
                else:
                    t = cv2.resize(tpl, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
                s = float(cv2.matchTemplate(resized, t, cv2.TM_CCOEFF_NORMED)[0][0])
                if s > var_best:
                    var_best = s
            if var_best > best_s:
                second_s = best_s
                best_s, best_d = var_best, d
            elif var_best > second_s:
                second_s = var_best
        if best_s < self.score_min or (best_s - second_s) < SCORE_GAP:
            return None
        return best_d, best_s

    # ------------------------------------------------------------------
    def read_mask(self, mask):
        """数字掩码图（数字=亮）→ 数字串列表，失败返回 []"""
        if not self.ok() or mask is None:
            return []
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        subs = self._split_digits(m)
        if not subs:
            return []
        # 噪声碎片过滤（修复：碎片段导致整体失败，如 133543 帧 13x5 碎片）
        max_area = max(int((s > 0).sum()) for s in subs)
        digits = []
        for sub in subs:
            area = int((sub > 0).sum())
            if area < max(20, max_area * 0.12):
                continue                       # 面积过小的噪声碎片 → 丢弃
            h = sub.shape[0]
            if h > 0 and round(sub.shape[1] * self.height / h) < 4:
                continue                       # 极窄竖笔碎片（断裂数字残段）→ 丢弃
            r = self._match_one(sub)
            if r is None:
                return []                      # 任一数字失败 → 整体放弃（宁缺毋滥）
            digits.append(str(r[0]))
        return ["".join(digits)] if digits else []

    def read(self, pil_img):
        """mask_to_image 生成的图（黑字白底 PIL Image）→ 数字串列表"""
        if pil_img is None:
            return []
        a = np.array(pil_img.convert("L"))
        mask = (255 - a)           # 反转：数字=亮
        mask[mask > 127] = 255
        mask[mask <= 127] = 0
        return self.read_mask(mask)
