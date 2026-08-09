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
# Y0=0：新UI锚点(Y形图标)顶部贴近 y0；旧UI剑图标在带内不受影响
BAND_Y0, BAND_Y1 = 0.0, 0.10
# 标题检测区域（窗口相对比例，与模板裁剪坐标一致 x830-1090/y293-362）
TITLE_X0, TITLE_X1 = 0.4318, 0.5671
TITLE_Y0, TITLE_Y1 = 0.2365, 0.2922

SWORD_THRESHOLD = 0.75      # 剑图标匹配阈值（UI 虚化过渡帧分数 0.5-0.7，正常 0.94+）
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
        self._yicon = self._load("yicon.png")   # 新UI锚点（Y形图标，与剑形并列二选一）
        self._bossicon = self._load("bossicon.png")  # BOSS战锚点（翼状菱形，与旧存档帧同形）
        self._heart = self._load("heart.png")
        self._progress_matcher = None
        try:
            from template_matcher import TemplateMatcher
            self._progress_matcher = TemplateMatcher(
                template_dir=os.path.join(TPL_DIR, "digits"))
        except Exception:
            pass
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
        """顶部条带定位：锚点图标（剑形优先，Y形兜底）+ 心形图标
        → (band_rect, sword_rect, heart_rect) 或 None"""
        if gray is None or self._sword is None:
            return None
        H, W = gray.shape
        x0, x1 = int(W * BAND_X0), int(W * BAND_X1)
        y0, y1 = int(H * BAND_Y0), int(H * BAND_Y1)
        if x1 - x0 < 40 or y1 - y0 < 20:
            return None
        band = gray[y0:y1, x0:x1]
        sword = self._match(band, self._scale(self._sword, W, H), SWORD_THRESHOLD)
        if sword is None and self._yicon is not None:
            # 新UI锚点：Y形图标（与剑形并列，不同关卡二选一，位置一致）
            sword = self._match(band, self._scale(self._yicon, W, H), SWORD_THRESHOLD)
        if sword is None and self._bossicon is not None:
            # BOSS战锚点：翼状菱形（旧存档帧同形，sword宽版含大块黑边在
            # 渐变背景上匹配失败0.516，bossicon紧版1.000）
            sword = self._match(band, self._scale(self._bossicon, W, H), SWORD_THRESHOLD)
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
        # 精准条带：剑左缘 → 进度条右缘（检测橙色填充段；无则按固定间隙序列）
        band_x1 = sw[0] + int(W * 0.11)
        try:
            orange = (gray[sw[1]:sw[3], sw[0]:min(x1, sw[0] + int(W * 0.13))] > 0)
            # 橙色检测用原图颜色（此处灰度近似，改由调用方传 RGB；退化用固定宽度）
        except Exception:
            pass
        # y 收窄到剑图标 y 带（±2px 边距），避免把条带上下背景卷进区域
        by0 = max(0, sw[1] - 2)
        by1 = min(H - 1, sw[3] + 2)
        return ((sw[0], by0, band_x1, by1), sw, hw)

    def read_progress(self, gray, band_rect, sword_rect, heart_rect=None):
        """进度数字提取（内容驱动）：
        条带右侧 >180 掩码 → 数字组件（x 重叠聚类合并修复断裂）→
        进度字体模板匹配（多变体）→ 缺模板数字 OCR 兜底 → 拼接 ≤100"""
        if gray is None:
            return None
        H, W = gray.shape
        if band_rect is None or sword_rect is None:
            return None
        bx0, by0, bx1, by1 = band_rect
        sx1 = sword_rect[2]
        # 搜索区：剑右缘 +30 ~ +125（覆盖 1-3 位进度数字+%，避开左侧关卡）
        rx0 = sx1 + max(42, int(W * 0.022))    # 避开左侧 "1-3"（剑右缘+39 结束）
        rx1 = min(bx1, sx1 + max(120, int(W * 0.062)))
        # y 收窄到数字带（剑带中间 50%），避开进度条（剑带下部）干扰
        ry0 = by0 + int((by1 - by0) * 0.2)
        ry1 = by0 + int((by1 - by0) * 0.85)
        if rx1 <= rx0 or ry1 <= ry0:
            return None
        sub = gray[ry0:ry1, rx0:rx1]
        mask = sub > 180
        # 数字区内容极少（<50）= 数字切换过渡帧/残影（正常帧最小 mask 95，
        # 过渡帧仅 ~25 且只剩底部小段）；直接拒绝避免 OCR 兜底随机误读
        # （如 21 过渡帧被读成 7），保持上次值不污染基线
        if int(mask.sum()) < 50:
            return None
        # 数字组件 + 列投影分离（44 合并）+ x 重叠聚类合并（0 断裂）
        import starrail_monitor as _sm
        comps = [c for c in _sm.components(mask) if c[4] >= 15]
        if not comps:
            return None
        max_area = max(c[4] for c in comps)
        # 过滤：% 碎片/噪声（面积 < 最大 30% 或高度 < 最大 60%）
        max_h = max(c[3] - c[1] + 1 for c in comps)
        comps = [c for c in comps
                 if c[4] >= max(20, max_area * 0.3)
                 and (c[3] - c[1] + 1) >= max_h * 0.6]
        comps.sort(key=lambda c: c[0])
        # 相邻数字合并（如 44）→ 无膨胀列投影分离
        if max(c[2] - c[0] + 1 for c in comps) > 14:
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
        merged = []
        for c in comps:
            if merged and c[0] < merged[-1][2] - 2:
                m = merged[-1]
                merged[-1] = (min(m[0], c[0]), min(m[1], c[1]),
                              max(m[2], c[2]), max(m[3], c[3]), m[4] + c[4])
            else:
                merged.append(c)
        # 前导大间隔过滤：BOSS战关卡文本（2-7）比普通战宽~4px，rx0(+42)
        # 会切入其尾部组件（误读如 7），与进度数字间隔~25px；数字内部
        # 间距仅 4-7px → 间隔>12px 的前导组件判定为关卡残留，丢弃
        if len(merged) >= 2 and merged[1][0] - merged[0][2] > 12:
            merged = merged[1:]
        # 逐组件识别（连续前缀）：从左到右匹配，首个失败段（% 碎片/噪声）
        # 之后的组件忽略——% 符号恒在数字右侧
        digits = []
        for c in merged:
            seg = mask[c[1]:c[3] + 1, c[0]:c[2] + 1].astype(np.uint8) * 255
            d = self._match_progress_digit(seg)
            if d is None:
                break
            digits.append(d)
        if digits:
            v = 0
            for d in digits:
                v = v * 10 + d
            if v <= 100:
                return v
        # 断裂模式防御：组件 ≥2 且模板全失败 = 断裂数字（0/8 渲染断裂，
        # 段形态与模板差大），整窗 OCR 对断裂形态不稳定（同一帧读 1/0 随机，
        # 曾致 0→1 误读被接受后连环告警）→ 拒绝识别，保持上次值
        if len(merged) >= 2:
            return None
        # 兜底：整窗口 OCR（单数字完整形态场景）
        crop = Image.fromarray(255 - (sub > 150).astype(np.uint8) * 255).resize(
            (mask.shape[1] * 6, mask.shape[0] * 6), Image.LANCZOS)
        text = self._ocr_digits(crop)
        for m_ in re.findall(r"(\d{1,3})", text):
            v = int(m_)
            if v <= 100:
                return v
        if len(text) >= 2:
            v = int(text[-2:])
            if v <= 100:
                return v
        return None

    def _match_progress_digit(self, seg):
        """单数字掩码段 → 像素差模板匹配（进度字体 2/3/8 相似，
        NCC 的 SCORE_GAP 0.15 不可靠）→ OCR 兜底 → int 或 None"""
        m = self._progress_matcher
        h = seg.shape[0]
        if h <= 0:
            return None
        w_new = max(1, int(round(seg.shape[1] * 16 / h)))
        r = cv2.resize(seg, (w_new, 16), interpolation=cv2.INTER_LANCZOS4)
        ys, xs = np.where(r > 127)
        if len(ys) == 0:
            return None
        r = r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if m is not None:
            # 按数字聚合（每数字取所有变体最小差，重复变体不影响次优）
            d_min = {}
            for d, tpls in m.templates.items():
                best_for_d = 1e9
                for tpl in tpls:
                    w = max(r.shape[1], tpl.shape[1])
                    c1 = np.zeros((16, w), np.uint8)
                    c2 = np.zeros((16, w), np.uint8)
                    if r.shape[1] > tpl.shape[1]:
                        # 段比模板宽（左界多出 1px 抗锯齿残影，如 9 宽体变体）：
                        # 左对齐会把整列笔画推错位，MAE 从 0 飙到 67 超阈值；
                        # 右对齐让残影列越界、真笔画与模板重合
                        c1[:r.shape[0], w - r.shape[1]:] = r
                        c2[:tpl.shape[0], w - tpl.shape[1]:] = tpl
                    else:
                        c1[:r.shape[0], :r.shape[1]] = r
                        c2[:tpl.shape[0], :tpl.shape[1]] = tpl
                    diff = float(np.abs(c1.astype(int) - c2.astype(int)).mean())
                    best_for_d = min(best_for_d, diff)
                d_min[d] = best_for_d
            order = sorted(d_min.items(), key=lambda kv: kv[1])
            best_d, best_diff = order[0]
            second_diff = order[1][1] if len(order) > 1 else 1e9
            # 阈值：正确模板差 <45（渲染变体差异可达 38）且与次优差距显著
            if best_diff < 45 and best_diff < second_diff * 0.85:
                # 自动变体采集已关闭（误配段会污染模板库，如 8 段误配 3）；
                # 模板积累改由 scan_progress_templates.py 离线扫描完成
                return best_d
        # OCR 兜底（0/4/5/6/9 等缺模板数字）
        crop = Image.fromarray((255 - seg.astype(np.uint8) * 255)).resize(
            (w_new * 8, 16 * 8), Image.LANCZOS)
        text = self._ocr_digits(crop)
        if len(text) == 1 and text.isdigit():
            return int(text)
        return None

    def read_stage(self, gray, sword_rect):
        """关卡识别（"1-3" → 1层3关）：剑右缘+固定窗口 → 组件定位 →
        数字模板匹配（- 为矮横条跳过）→ (层, 关)"""
        if gray is None:
            return None
        H, W = gray.shape
        if sword_rect is None:
            return None
        sx1 = sword_rect[2]
        rx0 = sx1 + max(2, int(W * 0.002))
        # +48：新UI关卡文本（2-5）比旧UI（1-3）宽~2px，原+38会切掉第二数字右缘
        rx1 = sx1 + max(48, int(W * 0.025))
        ry0 = ry1 = None
        if sword_rect[1] is not None:
            ry0, ry1 = sword_rect[1], sword_rect[3]
        if rx1 <= rx0 or ry1 is None or ry0 is None or ry1 <= ry0:
            return None
        sub = gray[ry0:ry1, rx0:rx1]
        mask = sub > 180
        if int(mask.sum()) < 8:
            return None
        import starrail_monitor as _sm
        comps = [c for c in _sm.components(mask) if c[4] >= 15]
        comps.sort(key=lambda c: c[0])
        if not comps:
            return None
        max_h = max(c[3] - c[1] + 1 for c in comps)
        digits = []
        for c in comps:
            h = c[3] - c[1] + 1
            if h < max_h * 0.6:
                continue                       # "-" 矮横条跳过
            seg = mask[c[1]:c[3] + 1, c[0]:c[2] + 1].astype(np.uint8) * 255
            d = self._match_progress_digit(seg)
            if d is None:
                return None
            digits.append(d)
        if len(digits) == 2:
            return (digits[0], digits[1])
        if len(digits) == 1:
            return (digits[0], None)
        return None

    @staticmethod
    def _ocr_text(crop):
        """全字符 OCR（读 1-3 关卡格式）"""
        try:
            import pytesseract
            if not pytesseract.pytesseract.tesseract_cmd or                     not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = os.path.join(
                    os.environ.get("LOCALAPPDATA", ""),
                    "Programs", "Tesseract-OCR", "tesseract.exe")
            return pytesseract.image_to_string(
                crop, config="--psm 7").strip()
        except Exception:
            return ""

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
    def update(self, gray, rgb=None):
        """低频调用（约 1s/次）。返回事件列表 [(类型, 说明), ...]
        rgb：彩色帧（可选），进度变化时用于存档进度条带截图"""
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
                    self.last_progress = None   # 新对局进度归零，重置单调基线
                    self._progress_warn = None
                    self._new_battle_ts = self.battle_start_ts
                    self._read_band_values(gray, band, rgb)
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
                    self._read_band_values(gray, band, rgb)
                    if self._stage_changed:
                        events.append(("stage", "关卡 %d-%d" % self.last_stage))
                    if getattr(self, "_progress_warn", None):
                        events.append(("progress_warn", self._progress_warn))
                    elif self.last_progress is not None:
                        events.append(("progress", self.last_progress))
            elif self.state == "SETTLEMENT":
                if in_battle_ui and not title:
                    # 结算页消失、进度条带重现 → 新对局
                    self.state = "BATTLE"
                    self.battle_start = datetime.now()
                    self.battle_start_ts = __import__("time").time()
                    self.last_progress = None
                    self._progress_warn = None
                    self._new_battle_ts = self.battle_start_ts
                    self._read_band_values(gray, band, rgb)
                    events.append(("battle_start", "对局开始"))
        return events

    def _read_band_values(self, gray, band, rgb=None):
        """读取条带数值：关卡（1-3）+ 进度百分比；关卡变化置 _stage_changed"""
        if band is None:
            return
        stage = self.read_stage(gray, band[1])
        self._stage_changed = False
        if stage is not None:
            if stage != getattr(self, "last_stage", None):
                self._stage_changed = True
            self.last_stage = stage
        p = self.read_progress(gray, band[0], band[1], band[2])
        # 全量存档：每 tick 保存进度条带帧（正常+异常，供事后排查/模板采集）
        self._save_progress_frame(rgb, band, p)
        if p is not None:
            # 新对局 5 秒内高位值（>10）= 结算页过渡残留误读（新对局进度起步低）
            new_ts = getattr(self, "_new_battle_ts", 0.0)
            if p > 10 and __import__("time").time() - new_ts < 5:
                self._progress_warn = "新对局过渡帧高位 %d 已忽略" % p
                return
            # 单调性校验：对局内进度只增不减，回退=识别错误（如 50→5），
            # 保持上次值并记录告警（真实回退仅在新对局，已在 battle_start 重置）
            if self.last_progress is not None and p < self.last_progress:
                self._progress_warn = "进度回退 %d→%d 已忽略（识别错误）" % (
                    self.last_progress, p)
                return
            self._progress_warn = None
            self.last_progress = p

    def _maybe_add_variant(self, seg_norm, d):
        """自动变体采集：匹配成功的数字段与现有变体差异大时存入模板库
        （templates/battle/digits/{d}_{i}.png，上限 8 张/数字），
        覆盖不同帧的渲染形态差异（同一数字跨帧像素差可达 38）"""
        m = self._progress_matcher
        if m is None:
            return
        tpls = m.templates.get(d, [])
        for tpl in tpls:
            w = max(seg_norm.shape[1], tpl.shape[1])
            c1 = np.zeros((16, w), np.uint8)
            c2 = np.zeros((16, w), np.uint8)
            c1[:seg_norm.shape[0], :seg_norm.shape[1]] = seg_norm
            c2[:tpl.shape[0], :tpl.shape[1]] = tpl
            if float(np.abs(c1.astype(int) - c2.astype(int)).mean()) < 20:
                return                    # 已有相似变体
        if len(tpls) >= 8:
            return                        # 每数字上限
        try:
            w = seg_norm.shape[1]
            canvas = np.zeros((16, w), np.uint8)
            canvas[:seg_norm.shape[0], :seg_norm.shape[1]] = seg_norm
            Image.fromarray(canvas, "L").save(
                os.path.join(TPL_DIR, "digits", "%d_%d.png" % (d, len(tpls))))
            m.templates.setdefault(d, []).append(seg_norm.copy())
        except Exception:
            pass

    def _save_progress_frame(self, rgb, band, progress):
        """进度条带截图存档（logs/frames/日期/，文件名带 _progress_ 标记，
        prune 清理时豁免 = 调试存档无上限）；识别值可为 None（未识别帧也存）"""
        if rgb is None or band is None:
            return
        try:
            bx0, by0, bx1, by1 = band[0]
            pad = 12
            crop = rgb.crop((max(0, bx0 - pad), max(0, by0 - pad),
                             min(rgb.width - 1, bx1 + pad),
                             min(rgb.height - 1, by1 + pad)))
            now = datetime.now()
            sub = os.path.join(LOGS_DIR, "frames", now.strftime("%Y-%m-%d"))
            os.makedirs(sub, exist_ok=True)
            if progress is None:
                name = "frame_%s_progress_unk.png" % (
                    now.strftime("%Y%m%d_%H%M%S_%f")[:-3])
            else:
                name = "frame_%s_progress_a%d.png" % (
                    now.strftime("%Y%m%d_%H%M%S_%f")[:-3], progress)
            crop.save(os.path.join(sub, name))
        except Exception:
            pass

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
