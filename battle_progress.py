# -*- coding: utf-8 -*-
"""对局进度/状态监测模块

对局状态机（三态）：
  IDLE ──剑图标进度条带出现──▶ BATTLE（新对局开始：记录开始时间、重置基线）
  BATTLE ──检测到「挑战成功/挑战结束」标题──▶ SETTLEMENT（记录结果/最终进度/时长）
  SETTLEMENT ──剑图标条带再次出现──▶ BATTLE（新对局）

- 进度区域定位：顶部条带内剑图标（左锚）+ 心形血量图标（右锚）模板匹配框定，
  区域内 OCR 读百分比数字（"22%" → 22）
- 结束判定：固定区域模板匹配「挑战成功」/「挑战结束」（从结算截图标定）
- 记录：logs/battle_YYYY-MM-DD.csv（开始时间/结束时间/结果/最终进度/时长）
"""
import csv
import os
import re
import threading
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TPL_DIR = os.path.join(SCRIPT_DIR, "templates", "battle")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

# 模板提取时的标定窗口尺寸（1922x1239 游戏窗口）
CALIB_W, CALIB_H = 1922, 1239
# 顶部条带搜索区域（窗口相对比例：x 25%-65%，y 2%-10%）
BAND_X0, BAND_X1 = 0.25, 0.65
BAND_Y0, BAND_Y1 = 0.02, 0.10
# 标题检测区域（窗口相对比例，标定自结算截图 x697-838/y465-563）
TITLE_X0, TITLE_X1 = 0.36, 0.44
TITLE_Y0, TITLE_Y1 = 0.375, 0.455

SWORD_THRESHOLD = 0.45      # 剑图标匹配阈值（细线图标，NCC 偏低）
TITLE_THRESHOLD = 0.55      # 标题匹配阈值
PROGRESS_OCR_MIN = 0        # 进度下限


class BattleTracker:
    """对局状态机 + 进度/标题检测"""

    def __init__(self, ocr, log_fn=None):
        self.ocr = ocr
        self.log = log_fn or (lambda msg: None)
        self.state = "IDLE"          # IDLE / BATTLE / SETTLEMENT
        self.battle_start = None     # 对局开始时间（datetime）
        self.battle_start_ts = 0.0
        self.last_progress = None    # 最近一次进度（0-100）
        self._sword = self._load("sword.png")
        self._heart = self._load("heart.png")
        self._title_ok = self._load("title_success.png")
        self._title_fail = self._load("title_fail.png")
        self._lock = threading.Lock()
        self._csv_path = None
        self._csv_lock = threading.Lock()

    @staticmethod
    def _load(name):
        p = os.path.join(TPL_DIR, name)
        if not os.path.isfile(p):
            return None
        return np.array(Image.open(p).convert("L"), dtype=np.uint8)

    @property
    def templates_ok(self):
        return all(t is not None for t in
                   (self._sword, self._heart, self._title_ok, self._title_fail))

    # ---------------- 模板匹配 ----------------
    def _scale(self, tpl, w, h):
        """按当前窗口尺寸缩放标定模板"""
        tw = max(8, int(round(tpl.shape[1] * w / CALIB_W)))
        th = max(8, int(round(tpl.shape[0] * h / CALIB_H)))
        if (tw, th) == (tpl.shape[1], tpl.shape[0]):
            return tpl
        return cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)

    def _match(self, gray, tpl, threshold):
        if tpl is None or gray is None:
            return None
        H, W = gray.shape
        th, tw = tpl.shape
        if th > H or tw > W:
            return None
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv >= threshold:
            return (float(maxv), maxloc[0], maxloc[1], tw, th)
        return None

    # ---------------- 检测 ----------------
    def find_band(self, gray):
        """顶部条带定位：剑图标 + 心形图标 → (band_rect, sword_rect, heart_rect) 或 None"""
        if gray is None or self._sword is None:
            return None
        H, W = gray.shape
        x0, x1 = int(W * BAND_X0), int(W * BAND_X1)
        y0, y1 = int(H * BAND_Y0), int(H * BAND_Y1)
        if x1 - x0 < 40 or y1 - y0 < 20:
            return None
        band = gray[y0:y1, x0:x1]
        sword = self._match(band, self._scale(self._sword, W, H), SWORD_THRESHOLD)
        if sword is None:
            return None
        sw = (x0 + sword[1], y0 + sword[2], x0 + sword[1] + sword[3], y0 + sword[2] + sword[4])
        # 心形：剑图标右侧 100~600px 范围（同 y 带）
        hx0 = min(x1, sw[2] + int(W * 0.05))
        hx1 = min(x1, sw[2] + int(W * 0.35))
        heart = None
        if self._heart is not None and hx1 > hx0:
            heart = self._match(band[:, hx0 - x0:hx1 - x0],
                                self._scale(self._heart, W, H), 0.45)
        if heart is not None:
            hw = (hx0 + heart[1], y0 + heart[2],
                  hx0 + heart[1] + heart[3], y0 + heart[2] + heart[4])
        else:
            hw = None
        return ((x0, y0, x1, y1), sw, hw)

    def read_progress(self, gray, band_rect, sword_rect, heart_rect=None):
        """进度数字提取：剑图标右缘 +固定窗口 → 白名单 OCR → 取 ≤100 数字"""
        if gray is None:
            return None
        H, W = gray.shape
        if band_rect is None or sword_rect is None:
            return None
        bx0, by0, bx1, by1 = band_rect
        sx1 = sword_rect[2]
        rx0 = sx1 + max(20, int(W * 0.023))
        rx1 = min(bx1, sx1 + max(90, int(W * 0.06)))
        ry0, ry1 = by0, by1
        if rx1 <= rx0 or ry1 <= ry0:
            return None
        sub = gray[ry0:ry1, rx0:rx1]
        # 白字黑描边：亮色掩码（>150 灰度）→ 黑字白底放大
        mask = (sub > 150).astype(np.uint8) * 255
        if int((mask > 0).sum()) < 10:
            return None
        crop = Image.fromarray(255 - mask).resize(
            (mask.shape[1] * 6, mask.shape[0] * 6), Image.LANCZOS)
        text = self._ocr_digits(crop)
        if not text:
            return None
        # 规则：≤100 的匹配取最后一个；否则取最后两位
        for m in re.findall(r"(\d{1,3})", text):
            v = int(m)
            if v <= 100:
                return v
        if len(text) >= 2:
            v = int(text[-2:])
            if v <= 100:
                return v
        return None

    @staticmethod
    def _ocr_digits(crop):
        """数字白名单 OCR（tesseract 默认路径；失败返回 ''）"""
        try:
            import pytesseract
            if not pytesseract.pytesseract.tesseract_cmd or \
                    not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = os.path.join(
                    os.environ.get("LOCALAPPDATA", ""),
                    "Programs", "Tesseract-OCR", "tesseract.exe")
            return pytesseract.image_to_string(
                crop, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
        except Exception:
            return ""

    def detect_title(self, gray):
        """结算标题检测：「后两字像素差 + y 滑动对齐」→ success/fail/None

        NCC 对共享的「挑战」两字不敏感（交叉 0.93+），改用灰度像素差：
        标题区域与 success/fail 模板的后两字（成功/结束）最小差，阈值 45 判无标题。
        """
        if gray is None:
            return None
        H, W = gray.shape
        x0, x1 = int(W * TITLE_X0), int(W * TITLE_X1)
        y0, y1 = int(H * TITLE_Y0), int(H * TITLE_Y1)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        region = gray[y0:y1, x0:x1].astype(np.float32)
        t_ok = self._scale(self._title_ok, W, H)
        t_fail = self._scale(self._title_fail, W, H)
        cut_ok = int(t_ok.shape[1] * 0.52)
        cut_fail = int(t_fail.shape[1] * 0.52)
        d_ok = self._best_pixdiff(region, t_ok[:, cut_ok:], dx=cut_ok)
        d_fail = self._best_pixdiff(region, t_fail[:, cut_fail:], dx=cut_fail)
        if min(d_ok, d_fail) > 45:
            return None
        return "success" if d_ok < d_fail else "fail"

    @staticmethod
    def _best_pixdiff(region, tpl, dx=0, dy_range=range(-12, 13)):
        """模板在区域内 y 滑动对齐的最小灰度差（均值）"""
        h, w = tpl.shape
        if h > region.shape[0] or w > region.shape[1] - dx:
            return 1e9
        best = 1e9
        for dy in dy_range:
            y0 = max(0, dy)
            y1 = y0 + h
            if y1 > region.shape[0]:
                continue
            r = region[y0:y1, dx:dx + w]
            best = min(best, float(np.abs(r - tpl).mean()))
        return best

    # ---------------- 状态机 ----------------
    def update(self, gray):
        """低频调用（约 1s/次）。返回事件列表 [(类型, 说明), ...]"""
        events = []
        with self._lock:
            band = self.find_band(gray)
            in_battle_ui = band is not None
            title = self.detect_title(gray) if self.state != "IDLE" else None

            if self.state == "IDLE":
                if in_battle_ui:
                    self.state = "BATTLE"
                    self.battle_start = datetime.now()
                    self.battle_start_ts = __import__("time").time()
                    self.last_progress = self.read_progress(gray, band[0], band[1])
                    events.append(("battle_start", "对局开始"))
            elif self.state == "BATTLE":
                if title is not None:
                    # 结算：记录对局结果
                    self._record(title)
                    events.append(("battle_end", "对局%s（进度%s%%）" % (
                        "成功" if title == "success" else "结束未完成",
                        self.last_progress if self.last_progress is not None else "?")))
                    self.state = "SETTLEMENT"
                else:
                    p = self.read_progress(gray, band[0], band[1], band[2]) if band else None
                    if p is not None:
                        self.last_progress = p
                        events.append(("progress", p))
            elif self.state == "SETTLEMENT":
                if in_battle_ui and not title:
                    # 结算页消失、进度条带重现 → 新对局
                    self.state = "BATTLE"
                    self.battle_start = datetime.now()
                    self.battle_start_ts = __import__("time").time()
                    self.last_progress = self.read_progress(gray, band[0], band[1])
                    events.append(("battle_start", "对局开始"))
        return events

    def force_reset(self):
        """停止监测/手动停止时清状态"""
        with self._lock:
            if self.state == "BATTLE":
                self._record("interrupted")
            self.state = "IDLE"
            self.last_progress = None

    # ---------------- 记录 ----------------
    def _csv(self):
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(LOGS_DIR, "battle_%s.csv" % today)
        if path != self._csv_path:
            self._csv_path = path
            if not os.path.isfile(path):
                try:
                    with open(path, "w", encoding="utf-8-sig", newline="") as f:
                        csv.writer(f).writerow(
                            ["开始时间", "结束时间", "结果", "最终进度", "持续时间(秒)"])
                except Exception:
                    pass
        return path

    def _record(self, result):
        """写入对局记录（开始时间/结束时间/结果/最终进度/时长）"""
        if self.battle_start is None:
            return
        try:
            end = datetime.now()
            dur = int((end - self.battle_start).total_seconds())
            with self._csv_lock:
                with open(self._csv(), "a", encoding="utf-8-sig", newline="") as f:
                    csv.writer(f).writerow([
                        self.battle_start.strftime("%H:%M:%S"),
                        end.strftime("%H:%M:%S"),
                        result,
                        self.last_progress if self.last_progress is not None else "",
                        dur])
        except Exception:
            pass
