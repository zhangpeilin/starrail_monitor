# -*- coding: utf-8 -*-
"""
星穹铁道 回合数/行动值 监控提醒程序 (v2)
=======================================
功能：
  1. 捕获游戏窗口画面（Windows Graphics Capture，OBS 同款 API，支持游戏在后台被遮挡时捕获）
  2. 在左侧行动序列列中动态定位倒计时条（位置随行动上下浮动）
  3. 识别「回合数」(沙漏右侧黄色数字) 与「行动值」(黑底白字条)
  4. 回合数==0 且 行动值<阈值 时弹窗提醒并响铃

用法：
  python starrail_monitor.py            正常启动
  python starrail_monitor.py --selftest <图片>   自检：定位红色矩形框区域并识别
"""
import ctypes
import ctypes.wintypes
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import queue
import winsound
import tkinter as tk
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter, ImageGrab, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------------
# DPI 感知
# --------------------------------------------------------------------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

TESS_DEFAULT_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe")


def find_tesseract():
    p = shutil.which("tesseract")
    if p:
        return p
    for cand in (TESS_DEFAULT_DIR,
                 r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.isfile(cand):
            return cand
    return None


# --------------------------------------------------------------------------
# OCR 引擎
# --------------------------------------------------------------------------
class OcrEngine:
    """Tesseract 优先，Windows OCR 兜底。read() 返回从左到右的数字串列表。"""

    def __init__(self):
        self.tess = None
        self.tess_path = find_tesseract()
        self.win = None
        if self.tess_path:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.tess_path
                self.tess = pytesseract
            except Exception:
                self.tess = None
        if self.tess is None:
            try:
                import winocr
                self.win = winocr
            except Exception:
                self.win = None

    def ok(self):
        return self.tess is not None or self.win is not None

    def engine_name(self):
        if self.tess:
            return "Tesseract"
        if self.win:
            return "WindowsOCR"
        return "无"

    def _ocr_psm(self, img, psm):
        try:
            text = self.tess.image_to_string(
                img, config="--psm %s -c tessedit_char_whitelist=0123456789" % psm)
            return re.findall(r"\d+", text)
        except Exception:
            return []

    def read(self, img, retry_big=False, vote=False):
        """识别一张图像，返回所有数字串（按读取顺序）。
        psm8 对数字掩码最可靠，命中即返回（避免多次启动 tesseract 进程拖慢识别）。
        vote=True 时 psm8+psm7 双跑，不一致用 psm10 裁决（用于行动值，防 2/4 类误读）。
        retry_big=True 时失败会用更大缩放重试一次（用于回合数字等小目标）。"""
        if img is None:
            return []
        if self.tess is not None:
            if vote:
                g8 = self._ocr_psm(img, "8")
                if g8:
                    g7 = self._ocr_psm(img, "7")
                    if g7 == g8:
                        return g8
                    n8 = len("".join(g8))
                    n7 = len("".join(g7))
                    if n8 != n7:
                        # 位数不同：取位数多的（更完整，防丢位）
                        return g8 if n8 > n7 else g7
                    g10 = self._ocr_psm(img, "10")
                    if g10 and (g10 == g8 or g10 == g7):
                        return g10
                    return g8
                for psm in ("7", "10"):
                    g = self._ocr_psm(img, psm)
                    if g:
                        return g
                return []
            for psm in ("8", "7", "10"):
                groups = self._ocr_psm(img, psm)
                if groups:
                    return groups
            if retry_big:
                try:
                    big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
                    groups = self._ocr_psm(big, "8")
                    if groups:
                        return groups
                except Exception:
                    pass
        if self.win is not None:
            try:
                res = self.win.recognize_pil_sync(img)
                return re.findall(r"\d+", res.get("text", ""))
            except Exception:
                pass
        return []


# --------------------------------------------------------------------------
# 图像处理基础
# --------------------------------------------------------------------------
def components(mask):
    """稀疏 BFS 连通域，返回 [(x0, y0, x1, y1, area)]"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    pts = set(zip(xs.tolist(), ys.tolist()))
    comps = []
    while pts:
        seed = pts.pop()
        stack = [seed]
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (x + dx, y + dy)
                    if nb in pts:
                        pts.discard(nb)
                        stack.append(nb)
        xs_c = [p[0] for p in comp]
        ys_c = [p[1] for p in comp]
        comps.append((min(xs_c), min(ys_c), max(xs_c), max(ys_c), len(comp)))
    return comps


def mask_to_image(mask, x0, y0, x1, y1, pad=6, scale=5):
    """把掩码区域转成黑字白底放大图，供 OCR 使用"""
    h, w = mask.shape
    px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
    px1, py1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    sub = mask[py0:py1 + 1, px0:px1 + 1]
    im = Image.fromarray(((~sub) * 255).astype(np.uint8))
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def dilate_mask(mask, k=2):
    """对二值掩码做方形膨胀，合并被细线/描边分割的碎片（如沙漏的黑描边）"""
    if k <= 0:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8))
    for _ in range(k):
        im = im.filter(ImageFilter.MaxFilter(3))
    return np.array(im) > 127


def merge_components(comps, gap=6):
    """按 x 方向近邻合并组件（x 间距 ≤ gap 且 y 有重叠），用于合并被描边分割的碎片"""
    comps = sorted(comps, key=lambda c: c[0])
    merged = []
    for c in comps:
        if (merged and c[0] - merged[-1][2] <= gap
                and c[1] < merged[-1][3] and c[3] > merged[-1][1]):
            m = merged[-1]
            merged[-1] = (min(m[0], c[0]), min(m[1], c[1]),
                          max(m[2], c[2]), max(m[3], c[3]), m[4] + c[4])
        else:
            merged.append(c)
    return merged


# --------------------------------------------------------------------------
# 倒计时条提取
# --------------------------------------------------------------------------
class Extractor:
    """从倒计时条区域图像中提取 回合数 / 行动值 数字串

    数字识别优先用 TemplateMatcher（OpenCV 模板匹配，固定字体抗像素级缺陷），
    失败时回退 Tesseract OCR。
    """

    def __init__(self, ocr, matcher=None):
        self.ocr = ocr
        if matcher is None:
            try:
                from template_matcher import TemplateMatcher
                matcher = TemplateMatcher()
            except Exception:
                matcher = None
        self.matcher = matcher

    def extract(self, img):
        """
        返回 (turn_groups, action_groups, raw_groups, info)
        groups 为数字串列表；失败为空列表；info 为调试文本
        支持两种敌人状态样式：金色六角图标(黄沙漏/黄回合数) / 红色六角图标(红沙漏/红回合数)，
        行动值均为白色数字。
        """
        info = []
        a = np.array(img.convert("RGB")).astype(int)
        H, W = a.shape[:2]
        if H < 5 or W < 5:
            return [], [], [], "区域过小"
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        min_area = max(15, int(W * H * 0.0003))

        yellow = (r > 170) & (g > 110) & (b < 110) & (r - b > 100)
        red = (r > 170) & (g < 110) & (b < 110) & (r - g > 60)
        colored = yellow | red
        white = (r > 190) & (g > 190) & (b > 190)

        yc_all = merge_components(
            [c for c in components(colored) if c[4] >= min_area], gap=3)
        wc = [c for c in components(white) if c[4] >= min_area]

        def contains_white(c):
            for wc_ in wc:
                if (wc_[0] >= c[0] and wc_[2] <= c[2]
                        and wc_[1] >= c[1] and wc_[3] <= c[3]):
                    return True
            return False

        # 锚点：内部含白色图案的有色组件（六角图标），金/红两种敌人状态；
        # 绝对尺寸过滤（六角图标 8-90px，排除大色块如注释框）
        anchors = [c for c in yc_all if contains_white(c)
                   and 8 <= c[2] - c[0] + 1 <= 90 and 8 <= c[3] - c[1] + 1 <= 90]
        if anchors:
            anchor = anchors[0]

            def mask_color(anchor):
                """判定锚点组件属于黄色还是红色掩码"""
                n_y = yellow[anchor[1]:anchor[3] + 1, anchor[0]:anchor[2] + 1].sum()
                n_r = red[anchor[1]:anchor[3] + 1, anchor[0]:anchor[2] + 1].sum()
                return "yellow" if n_y >= n_r else "red"

            color = mask_color(anchor)
            same_mask = yellow if color == "yellow" else red
            info.append("六角图标(%s)@(%d,%d)-(%d,%d)" % (
                color, anchor[0], anchor[1], anchor[2], anchor[3]))
            # 同色组件（沙漏 + 回合数字）
            yc = merge_components(
                [c for c in components(same_mask) if c[4] >= min_area], gap=3)
            # 沙漏 = 锚点右侧最左侧不含白图案的同色组件
            yc_by_x = [c for c in sorted(yc, key=lambda c: c[0]) if c[0] > anchor[2] - 4]
            hourglass = None
            for c in yc_by_x:
                if contains_white(c):
                    continue
                hourglass = c
                break
            if hourglass is None and yc_by_x:
                hourglass = yc_by_x[0]
        else:
            # 回退：无六角图标（部分界面/测试场景），用最左不含白的有色组件作沙漏
            same_mask = colored
            yc = yc_all
            yc_by_x = sorted(yc, key=lambda c: c[0])
            hourglass = None
            for c in yc_by_x:
                if contains_white(c):
                    continue
                hourglass = c
                break
            if hourglass is None and yc_by_x:
                hourglass = yc_by_x[0]
            info.append("无六角图标锚点，回退定位")

        turn_img = action_img = None
        if hourglass is not None:
            hy0, hy1 = hourglass[1], hourglass[3]
            info.append("沙漏区域(%d,%d)-(%d,%d)" % hourglass[:4])
            # 行动值：沙漏右侧、y 中心与沙漏对齐的白色组件（防把上下方
            # 其他白色元素如角色立绘/发光卷进区域）；
            # 按 x 间距聚类取最左簇（排除右侧无关元素如 2/6 的发光描边）
            hyc = (hy0 + hy1) / 2.0
            acts_all = [c for c in wc if c[0] > hourglass[2]
                        and abs((c[1] + c[3]) / 2.0 - hyc) <= 15]
            acts_all.sort(key=lambda c: c[0])
            clusters = []
            for c in acts_all:
                if clusters and c[0] - clusters[-1][-1][2] <= 12:
                    clusters[-1].append(c)
                else:
                    clusters.append([c])
            # 行动值 = 面积最大的簇（数字组件最大；排除噪声碎片与右侧 2/6 发光描边）
            acts = max(clusters, key=lambda cl: sum(c[4] for c in cl)) if clusters else []
            if acts:
                ax0 = min(c[0] for c in acts)
                ax1 = max(c[2] for c in acts)
                ay0 = min(c[1] for c in acts)
                ay1 = max(c[3] for c in acts)
                # 行动值区域 = 白色组件聚类结果。条框右边界已由
                # locate_bar 的黑底纹修正兜底（黑底不随数字淡出），
                # 过渡帧残影数字的组件在条框完整后自然存在，无需再扩展。
                action_img = mask_to_image(white, ax0, ay0, ax1, ay1)
                info.append("行动值区域(%d,%d)-(%d,%d)" % (ax0, ay0, ax1, ay1))
                # 回合数：沙漏右侧、行动值左侧、垂直与行动值有交叠的同色组件
                turn_cands = [c for c in yc if c[0] > hourglass[2] and c[2] < ax0
                              and c[3] > ay0 - 5 and c[1] < ay1 + 5]
                if turn_cands:
                    # 间隔断簇取最左簇（方案A）：沙漏→回合数字间隔 ~5px，
                    # 回合数字→右侧杂物（装饰/碎片）间隔 ≥9px——簇间隙 >8px 断开，
                    # 只取最左簇（回合数字本体），排除右侧碎片（如 133543 帧
                    # x94-96 黄色碎片曾被卷入导致 OCR 把 0 读成 1）
                    turn_cands.sort(key=lambda c: c[0])
                    tcl = []
                    for c in turn_cands:
                        if tcl and c[0] - tcl[-1][-1][2] <= 8:
                            tcl[-1].append(c)
                        else:
                            tcl.append([c])
                    turns = tcl[0]
                    tx0 = min(c[0] for c in turns)
                    tx1 = max(c[2] for c in turns)
                    ty0 = min(c[1] for c in turns)
                    ty1 = max(c[3] for c in turns)
                    turn_img = mask_to_image(same_mask, tx0, ty0, tx1, ty1)
                    info.append("回合数字区域(%d,%d)-(%d,%d)" % (tx0, ty0, tx1, ty1))

        turn_groups = action_groups = []
        if turn_img is not None:
            turn_groups = self.matcher.read(turn_img) if self.matcher else []
            if not turn_groups:
                turn_groups = self.ocr.read(turn_img, retry_big=True)
        if action_img is not None:
            action_groups = self.matcher.read(action_img) if self.matcher else []
            if not action_groups:
                action_groups = self.ocr.read(action_img, vote=True)

        # 兜底：整区域直接识别，取最后两个数字串
        raw_groups = []
        if not turn_groups or not action_groups:
            gray = ImageOps.autocontrast(img.convert("L"))
            gray = gray.resize((gray.width * 3, gray.height * 3), Image.LANCZOS)
            raw_groups = self.ocr.read(gray)
            info.append("整区域识别: %s" % raw_groups)

        return turn_groups, action_groups, raw_groups, "; ".join(info)


def parse_values(turn_groups, action_groups, raw_groups):
    """返回 (turn, action) 或 (None, None)"""
    turn = action = None
    if turn_groups:
        try:
            turn = int(turn_groups[0])
        except ValueError:
            turn = None
    if action_groups:
        try:
            action = int(action_groups[-1])
        except ValueError:
            action = None
    if (turn is None or action is None) and len(raw_groups) >= 2:
        try:
            t = int(raw_groups[-2])
            a = int(raw_groups[-1])
        except ValueError:
            t = a = None
        if turn is None and t is not None:
            turn = t
        if action is None and a is not None:
            action = a
    if turn is not None and not (0 <= turn <= 999):
        turn = None
    if action is not None and not (0 <= action <= 999):
        action = None
    return turn, action


# --------------------------------------------------------------------------
# 识别值合理性校验：范围 + 时序单调（丢弃离谱结果）
# --------------------------------------------------------------------------
class ValueFilter:
    """识别值合理性校验（依据货币战争机制）：
      1) 回合数、行动值必须在合理范围内，超出=识别错误，丢弃
      2) 同一回合内行动值只递减（突增=错误）
      3) 合法重置：行动值归零时回合数减 1、行动值重置回高位（回合数减小时接受）
      4) 同回合行动值突变：开关开启时，连续 3 帧突变且数值等值/递减、
         且发生在 行动值>0、回合数>0 时 → 允许（视为新回合重置），否则丢弃
      5) 新对局：回合数增大时（旧对局结束、新对局从高位回合开始），
         开关开启时同样三帧确认（连续 3 帧回合数等值/递减）→ 接受为新对局
      6) 同回合向下突变：一帧内行动值降幅超过 20（如 99 被误读成 9 的丢位）
         → 拒绝且不更新基线，避免丢位污染基线导致后续正确值被误判为突变
    丢弃帧不记录、不存档、不参与提醒判断。突变确认/新对局确认返回事件标记。
    """

    def __init__(self, max_turn=99, max_action=100, action_tolerance=5,
                 reset_after=15, allow_reset=True, reset_turn_min=1,
                 reset_action_min=1, action_drop_max=20,
                 turn_drop_action_min=30):
        self.max_turn = max_turn
        self.max_action = max_action
        self.action_tolerance = action_tolerance   # 行动值允许的微小回弹容差
        self.reset_after = reset_after             # 连续丢弃 N 帧后重置基线（识别长期失败兜底）
        self.allow_reset = allow_reset             # 行动值突变允许开关
        self.reset_turn_min = reset_turn_min       # 突变允许的回合数下限（回合数>0 才允许）
        self.reset_action_min = reset_action_min   # 突变允许的行动值下限（行动值>0 才允许）
        self.action_drop_max = action_drop_max     # 历史参数（保留兼容）：骤降判定已改为「多位数→个位数」检查
        self.turn_drop_action_min = turn_drop_action_min  # 回合减小帧行动值下限（重置必为高位）
        self.last = None
        self.drop_streak = 0
        self.reset_candidates = []                 # 突变候选帧（三帧确认）
        self.reset_ts = 0.0                        # 基线兜底重置时间（重置后提醒冷却用）

    def check(self, turn, action):
        """返回 (通过?, 事件标记或丢弃原因)
        通过时第二项为空；突变确认/新对局等特殊事件返回非空标记（用于日志高亮）。
        """
        # 闸门1：数值范围
        if not (0 <= turn <= self.max_turn):
            return False, "回合数超范围(%d)" % turn
        if not (0 <= action <= self.max_action):
            return False, "行动值超范围(%d)" % action
        if self.last is None:
            # 兜底重置后短时间内：首帧低位（<30）视为丢位拒绝。
            # 重置前已连续失败 ~7.5 秒，真实值必为高位（100 递减仍在 60+），
            # 低位首帧 = 丢位误读（如真实 94 被读成 4）；冷却过后低位即真实（战斗尾声）
            if self.reset_ts and time.time() - self.reset_ts < 10 \
                    and action < self.turn_drop_action_min:
                return False, "重置后首帧低位(%d)" % action
            return True, ""
        lt, la = self.last
        # 闸门3：回合数减小 = 合法重置（接受）；行动值必为高位（重置回100附近），
        # 低位 = 丢位误读（如 0回合99 被读成 9）→ 拒绝，防误报链
        if turn < lt:
            if action < self.turn_drop_action_min:
                return False, "回合重置行动值低位(%d)" % action
            return True, ""
        if turn > lt:
            # 回合数增大：疑似新对局开始。与行动值突变一样走三帧确认
            # （连续 3 帧回合数等值或递减才接受，防识别错误单帧误报新对局）
            if not self.allow_reset:
                return False, "回合数增大(%d→%d)" % (lt, turn)
            self.reset_candidates.append((turn, action))
            if len(self.reset_candidates) < 3:
                return False, "回合增大待确认(%d/3)" % len(self.reset_candidates)
            seq = self.reset_candidates
            if all(seq[i][0] >= seq[i + 1][0] for i in range(len(seq) - 1)):
                # 确认值取 3 帧最大行动值（防首帧恰好是丢位值：如 7/74/6 → 取 74）
                best = max(s[1] for s in seq)
                if best < self.turn_drop_action_min:
                    # 新对局确认值低位（如 99 被读成 9 的持续丢位）→ 拒绝
                    self.reset_candidates = []
                    return False, "新对局确认值低位(%d)" % best
                # 确认：新对局，基线 = 首个增大帧的回合 + 最大行动值
                self.last = (seq[0][0], best)
                self.reset_candidates = []
                return True, "新对局(回合增大确认)"
            self.reset_candidates = []
            return False, "回合增大无效(数值跳动)"
        # 闸门2：同一回合内行动值只递减（允许微小回弹容差）
        if action <= la + self.action_tolerance:
            if la >= 10 and action < 10 and action < la - 5:
                # 丢位骤降：从多位数掉到个位数（十位消失，如 99→9/86→8），
                # 拒绝且基线保持，下一帧正确值仍可正常接受；
                # 同位数骤降（86→54）是游戏真实机制/识别失败后的正常下降，
                # 且误读也能被后续容差自愈（54 相对误读值回弹在容差内）→ 不拦
                return False, "行动值骤降(%d→%d)" % (la, action)
            if la == 0 and action > 0:
                # 归零回弹：行动值归零=回合切换/战斗结束（机制上同回合不回弹），
                # 回弹值=新对局识别错值（如真实1回合100被读成(0,1)）→ 拒绝
                return False, "行动值归零回弹(%d→%d)" % (la, action)
            self.reset_candidates = []
            return True, ""
        # 同回合行动值突变
        if not self.allow_reset:
            return False, "行动值突变(开关关闭)"
        if turn < self.reset_turn_min or action <= self.reset_action_min:
            # 不满足突变允许条件（回合数>0、行动值>0）
            self.reset_candidates = []
            return False, "突变不允许(回合%d/行动值%d)" % (turn, action)
        # 三帧确认：连续 3 帧突变，数值等值/递减或每帧回弹 ≤2（识别抖动容差）
        # → 视为新回合重置；确认值取 3 帧最大行动值（抗丢位帧）
        self.reset_candidates.append((turn, action))
        if len(self.reset_candidates) < 3:
            return False, "突变待确认(%d/3)" % len(self.reset_candidates)
        seq = self.reset_candidates
        if all(seq[i][1] >= seq[i + 1][1] - 2 for i in range(len(seq) - 1)):
            best = max(s[1] for s in seq)
            if best < self.turn_drop_action_min:
                # 突变确认值低位（如 99 被读成 9 的持续丢位，3帧等值仍不可信）→ 拒绝
                self.reset_candidates = []
                return False, "突变确认值低位(%d)" % best
            # 确认：新回合重置，基线 = 首个突变帧的回合 + 最大行动值
            self.last = (seq[0][0], best)
            self.reset_candidates = []
            return True, "突变确认"
        self.reset_candidates = []
        return False, "突变序列无效(数值跳动)"

    def accept(self, turn, action):
        """接受一帧，更新基线"""
        self.last = (turn, action)
        self.reset_candidates = []
        self.drop_streak = 0

    def reject(self):
        """外部丢弃（未识别等）计数；连续丢弃过多时重置基线（识别长期失败兜底）"""
        self.drop_streak += 1
        if self.drop_streak >= self.reset_after:
            self.last = None
            self.reset_candidates = []
            self.drop_streak = 0
            self.reset_ts = time.time()

    def reset(self):
        self.last = None
        self.reset_candidates = []
        self.drop_streak = 0


# --------------------------------------------------------------------------
# 列内动态定位倒计时条
# --------------------------------------------------------------------------
def find_action_black_right(gray, hyc, x_start, W, H, dark_thr=130,
                            max_gap=3, min_w=6):
    """行动值黑底纹右缘检测（借鉴黑底白字 UI 特征：黑底不随数字淡出）

    行动值数字为黑底白字，黑底矩形在过渡帧（数字淡出）时依然完整，
    而白色组件会消失导致条框收缩。本函数在 y 带 hyc±12 内做列平均
    亮度剖面，从 x_start 向右找暗段（允许 ≤max_gap 像素的亮隙合并），
    返回最右侧暗段 (左缘, 右缘) 或 None。
    """
    y0, y1 = max(0, int(hyc - 12)), min(H - 1, int(hyc + 12))
    if y1 <= y0 or x_start >= W:
        return None
    band = gray[y0:y1 + 1, x_start:].mean(axis=0)
    dark = band < dark_thr
    segs = []
    cur = None
    gap = 0
    for i, v in enumerate(dark):
        if v:
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > max_gap:
                if cur[1] - cur[0] + 1 >= min_w:
                    segs.append(tuple(cur))
                cur = None
                gap = 0
    if cur is not None and cur[1] - cur[0] + 1 >= min_w:
        segs.append(tuple(cur))
    if not segs:
        return None
    s, e = segs[-1]
    return (x_start + s, x_start + e)


def locate_bar(img):
    """
    在整列区域中定位倒计时条。
    锚点 = 内部含白色图案的有色组件（六角图标，金/红两种敌人状态），
    其右侧需有同色组件（沙漏/回合数字）与白色数字（行动值）。
    找不到锚点/组合时返回 None（宁可未识别，不误读列表里的假目标）。
    返回 (x0, y0, x1, y1) 或 None。
    """
    a = np.array(img.convert("RGB")).astype(int)
    H, W = a.shape[:2]
    if H < 30 or W < 30:
        return None
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    min_area = max(15, int(W * H * 0.0002))
    # 有颜色组件 = 金色/黄色 ∪ 红色（六角图标有金/红两种敌人状态）
    yellow = (r > 170) & (g > 110) & (b < 110) & (r - b > 100)
    red = (r > 170) & (g < 110) & (b < 110) & (r - g > 60)
    colored = yellow | red
    white = (r > 190) & (g > 190) & (b > 190)

    yc = [c for c in components(colored) if c[4] >= min_area]
    wc = [c for c in components(white) if c[4] >= min_area]
    if not yc or not wc:
        return None

    def contains_white(c):
        """有色组件内部是否含白色图案（六角图标中心的图案）"""
        cx0, cy0, cx1, cy1 = c[0], c[1], c[2], c[3]
        inside = 0
        for w in wc:
            if w[0] >= cx0 and w[2] <= cx1 and w[1] >= cy0 and w[3] <= cy1:
                inside += w[4]
        return inside >= max(8, c[4] * 0.02)

    best = None
    for c in yc:
        if not contains_white(c):
            continue
        # 六角图标尺寸合理范围（过滤列中其他含白的大元素）
        cw = c[2] - c[0] + 1
        ch = c[3] - c[1] + 1
        if cw > W * 0.3 or ch > H * 0.15:
            continue
        # 锚点颜色决定同色掩码（金色→黄色组件，红色→红色组件）
        n_y = yellow[c[1]:c[3] + 1, c[0]:c[2] + 1].sum()
        n_r = red[c[1]:c[3] + 1, c[0]:c[2] + 1].sum()
        same = yellow if n_y >= n_r else red
        cy0, cy1 = c[1], c[3]
        # 锚点右侧的同色组件（沙漏/回合数字）
        digs = [d for d in components(same) if d[4] >= min_area
                and d[0] > c[2] and d[2] < c[2] + W * 0.6
                and d[3] > cy0 - 8 and d[1] < cy1 + 8]
        if not digs:
            continue
        dmax = max(d[2] for d in digs)
        # 更右侧的白色组件（行动值）
        acts = [w for w in wc if w[0] > dmax and w[0] < dmax + W * 0.7
                and w[3] > cy0 - 15 and w[1] < cy1 + 15]
        if not acts:
            continue
        score = c[4] + sum(d[4] for d in digs) + sum(w[4] for w in acts)
        if best is None or score > best[0]:
            # 行动值黑底纹扩展右边界：黑底不随数字淡出，
            # 过渡帧白色组件收缩时防止条框切掉残影数字（如 54 只剩 5）
            gray_a = (r * 0.299 + g * 0.587 + b * 0.114)
            br = find_action_black_right(gray_a, (cy0 + cy1) / 2.0,
                                         c[2] + 15, W, H)
            act_r = max(w[2] for w in acts)
            if br:
                act_r = max(act_r, br[1] + 4)
            x0 = min(c[0], min(d[0] for d in digs), min(w[0] for w in acts))
            x1 = max(c[2], dmax, act_r)
            y0 = min(c[1], min(d[1] for d in digs), min(w[1] for w in acts))
            y1 = max(c[3], max(d[3] for d in digs), max(w[3] for w in acts))
            best = (score, x0, y0, x1, y1)
    if best is None:
        return None
    _, x0, y0, x1, y1 = best
    pad = max(4, int(W * 0.02))
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(W - 1, x1 + pad), min(H - 1, y1 + pad))


def extract_column(ocr, column_img, matcher=None):
    """整列 → 定位倒计时条 → 提取数字。返回 (turn, action, info)。
    matcher 可注入指定模板的 TemplateMatcher（模板学习回放对比用）"""
    bar = locate_bar(column_img)
    if bar is None:
        return None, None, "未定位到倒计时条"
    crop = column_img.crop(bar)
    ex = Extractor(ocr, matcher=matcher)
    tg, ag, rg, info = ex.extract(crop)
    turn, action = parse_values(tg, ag, rg)
    return turn, action, "条@(%d,%d)-(%d,%d); %s" % (bar[0], bar[1], bar[2], bar[3], info)


# --------------------------------------------------------------------------
# 窗口查找
# --------------------------------------------------------------------------
BROWSER_MARKERS = ("Chrome", "Edge", "Firefox", "浏览器", "哔哩", "bilibili",
                   "B站", "视频", "网页", "页面", "搜索")


def find_game_hwnd(keyword):
    """按标题关键字找游戏窗口：优先精确匹配「崩坏：星穹铁道」，排除浏览器页面。
    返回 hwnd 或 None"""
    user32 = ctypes.windll.user32
    wins = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            if user32.GetWindowTextW(hwnd, buf, 256) > 0 and keyword in buf.value:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                wins.append((hwnd, buf.value, w * h, w, h))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    if not wins:
        return None

    def is_browser(title):
        return any(m in title for m in BROWSER_MARKERS)

    # 1) 精确匹配游戏标题（或以其开头）——即使窗口被隐藏/缩小也优先
    exact = [w for w in wins if w[1].startswith("崩坏：") or w[1].startswith(keyword + "：")]
    if exact:
        exact.sort(key=lambda t: -t[2])
        return exact[0][0]
    # 2) 含关键字、尺寸正常、不含浏览器标记
    no_browser = [w for w in wins if w[3] > 200 and w[4] > 200 and not is_browser(w[1])]
    if no_browser:
        no_browser.sort(key=lambda t: -t[2])
        return no_browser[0][0]
    # 3) 兜底：最大窗口
    wins.sort(key=lambda t: -t[2])
    return wins[0][0]


# --------------------------------------------------------------------------
# 捕获后端：PrintWindow 窗口直捕（游戏可见即可，无需前台）
# --------------------------------------------------------------------------
def printwindow_capture(hwnd):
    """PrintWindow + PW_RENDERFULLCONTENT 捕获窗口内容，返回 PIL Image 或 None"""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    hwnd_dc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    old = gdi32.SelectObject(mem_dc, bmp)
    user32.PrintWindow(hwnd, mem_dc, 2)  # PW_RENDERFULLCONTENT

    class BMIH(ctypes.Structure):
        _fields_ = [("biSize", ctypes.wintypes.DWORD), ("biWidth", ctypes.c_long),
                    ("biHeight", ctypes.c_long), ("biPlanes", ctypes.wintypes.WORD),
                    ("biBitCount", ctypes.wintypes.WORD), ("biCompression", ctypes.wintypes.DWORD),
                    ("biSizeImage", ctypes.wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                    ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", ctypes.wintypes.DWORD),
                    ("biClrImportant", ctypes.wintypes.DWORD)]
    class BMI(ctypes.Structure):
        _fields_ = [("bmiHeader", BMIH)]
    bmi = BMI()
    bmi.bmiHeader.biSize = ctypes.sizeof(BMIH)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    buf = ctypes.create_string_buffer(w * h * 4)
    got = gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    gdi32.SelectObject(mem_dc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)
    if got == 0:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(arr[..., 2::-1].copy(), "RGB")


def is_black(img, threshold=0.01):
    a = np.array(img)
    return (a.max(axis=2) > 30).mean() < threshold


class PrintWindowCapture:
    """PrintWindow 直捕游戏窗口：游戏可见（即使未在前台/被部分遮挡）即可工作"""

    def __init__(self, hwnd):
        self.hwnd = hwnd

    def grab(self):
        try:
            img = printwindow_capture(self.hwnd)
        except Exception:
            return None
        if img is None or is_black(img):
            return None
        return img

    def close(self):
        pass


class FlashCapture:
    """闪烁捕获：置顶→截屏→恢复Z序。游戏被完全遮挡且本进程与游戏同权限时可用"""

    def __init__(self, hwnd):
        self.hwnd = hwnd
        self._above = None

    def grab(self):
        user32 = ctypes.windll.user32
        SWP_NOACTIVATE = 0x0010
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return None
        self._above = user32.GetWindow(self.hwnd, 3)  # GW_HWNDPREV
        if not user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0,
                                   SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE):
            return None  # 权限不足
        try:
            time.sleep(0.35)
            shot = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom),
                                  all_screens=True)
            return shot
        finally:
            if self._above:
                user32.SetWindowPos(self.hwnd, self._above, 0, 0, 0, 0,
                                    SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE)

    def close(self):
        pass


class ScreenCapture:
    """屏幕区域捕获（游戏必须可见）"""

    def __init__(self, region):
        self.region = region

    def grab(self):
        try:
            return grab_region(self.region)
        except Exception:
            return None


def grab_region(region):
    """按虚拟屏幕坐标截取区域。region = [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = [int(v) for v in region]
    x1, y1 = min(x1, x2), min(y1, y2)
    x2, y2 = max(x1, x2), max(y1, y2)
    shot = ImageGrab.grab(all_screens=True)
    try:
        vx = ctypes.windll.user32.GetSystemMetrics(76)
        vy = ctypes.windll.user32.GetSystemMetrics(77)
    except Exception:
        vx = vy = 0
    return shot.crop((x1 - vx, y1 - vy, x2 - vx, y2 - vy))


def is_foreground(keyword):
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
        return keyword in buf.value
    except Exception:
        return True


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "capture_mode": "window",        # window=窗口直捕 / screen=屏幕区域
    "win_region": None,              # 窗口内列区域 [x1, y1, x2, y2]
    "region": None,                  # 屏幕坐标区域（screen 模式用）
    "turn_threshold": 0,
    "action_threshold": 30,
    "interval_ms": 300,
    "sound": True,
    "foreground_only": False,
    "window_keyword": "崩坏",
    "confirm_ticks": 2,
    "cooldown_s": 10,
    "gamma": 1.0,                    # 图像亮度校正（HDR 过曝时可调低）
    "save_frames": True,             # 数字变化/提醒时保存识别画面
    "max_frames": 10000,             # 画面存档上限（按日期分文件夹，超出自动删最旧）
    "log_history": True,             # 记录识别历史 CSV
    "max_turn": 99,                  # 回合数合理上限（超出=识别错误丢弃）
    "max_action": 100,               # 行动值合理上限（超出=识别错误丢弃）
    "allow_action_reset": True,      # 行动值突变允许开关（三帧确认）
    "reset_turn_min": 1,             # 突变允许的回合数下限（回合数>0 才允许）
    "reset_action_min": 1,           # 突变允许的行动值下限（行动值>0 才允许）
    "action_drop_max": 20,           # 同回合一帧内最大允许降幅（超过=丢位误读丢弃）
    "rapid_recheck": False,          # RapidOCR 复核默认关闭（本地推理与游戏抢CPU会卡，需手动开启）
    "rapid_baseline_s": 6,           # 周期基线校准间隔（秒，RapidOCR 复核）
    "sound_trigger": False,          # 声音触发（WASAPI 进程环回，需 templates/sounds/ 有音效样本）
    "sound_threshold": 0.13,         # 音效匹配触发阈值（越低越灵敏）
    "game_process": "StarRail.exe",  # 游戏进程名（声音捕获目标）
}


# --------------------------------------------------------------------------
# 识别记录：历史 CSV + 成功画面存档
# --------------------------------------------------------------------------
_GUI_LOG_LOCK = threading.Lock()


class LogGate:
    """日志门控（借鉴 ok-nte LogGate 精简版）：
      - allow(key, interval)：同 key 在 interval 秒内最多放行一次（节流高频日志）
      - allow_message(key, msg, interval, changed)：节流 + 消息内容变化时立即放行
    线程安全。
    """

    def __init__(self, time_func=time.time):
        self._time = time_func
        self._states = {}
        self._lock = threading.Lock()

    def allow(self, key, interval):
        if interval <= 0:
            return True
        now = self._time()
        with self._lock:
            st = self._states.setdefault(key, [0.0, None])
            if now - st[0] < interval:
                return False
            st[0] = now
            return True

    def allow_message(self, key, message, interval, changed=False):
        now = self._time()
        with self._lock:
            st = self._states.setdefault(key, [0.0, None])
            if changed and st[1] != message:
                st[0], st[1] = now, message
                return True
            if interval <= 0:
                return True
            if now - st[0] < interval:
                return False
            st[0], st[1] = now, message
            return True


def append_gui_log(ts, msg):
    """把 GUI 滚动日志同步追加到 logs/gui_YYYY-MM-DD.log（按天滚动）"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(SCRIPT_DIR, "logs", "gui_%s.log" % today)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _GUI_LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write("[%s] %s\n" % (ts, msg))
    except Exception:
        pass


class Recorder:
    """监控记录器：
      - logs/history_YYYY-MM-DD.csv  每次识别成功一行（时间,回合数,行动值,提醒,说明）
      - logs/frames/YYYY-MM-DD/      数字变化或触发提醒时的列区域画面（按日期分文件夹，总量保留最近 N 张）
      - logs/gui_YYYY-MM-DD.log      程序界面滚动日志（未识别/丢弃/事件等）完整落盘
    """

    def __init__(self, base_dir, max_frames=10000):
        self.logs_dir = os.path.join(base_dir, "logs")
        self.frames_dir = os.path.join(self.logs_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.max_frames = max_frames
        self._last = None          # 上次 (turn, action)，用于变化检测
        self._csv_path = None
        self._lock = threading.Lock()
        self._migrate_legacy_frames()

    def _migrate_legacy_frames(self):
        """旧版帧迁移：logs/frames/ 根目录的 frame_*.png → 按文件名日期子目录"""
        try:
            for f in os.listdir(self.frames_dir):
                p = os.path.join(self.frames_dir, f)
                if not (f.startswith("frame_") and os.path.isfile(p)):
                    continue
                ts = f[6:14]   # frame_YYYYMMDD_...
                if len(ts) == 8 and ts.isdigit():
                    d = "%s-%s-%s" % (ts[:4], ts[4:6], ts[6:8])
                    sub = os.path.join(self.frames_dir, d)
                    os.makedirs(sub, exist_ok=True)
                    try:
                        shutil.move(p, os.path.join(sub, f))
                    except Exception:
                        pass
        except Exception:
            pass

    def _date_dir(self):
        d = datetime.now().strftime("%Y-%m-%d")
        p = os.path.join(self.frames_dir, d)
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass
        return p

    def _csv(self):
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.logs_dir, "history_%s.csv" % today)
        if path != self._csv_path:
            self._csv_path = path
            if not os.path.isfile(path):
                with open(path, "w", encoding="utf-8-sig", newline="") as f:
                    csv.writer(f).writerow(["时间", "回合数", "行动值", "提醒", "说明"])
        return path

    def record(self, turn, action, alert, info, col_img=None,
               log_enabled=True, save_enabled=True):
        """记录一次识别结果。col_img 为列区域图像，仅在值变化或提醒时保存"""
        with self._lock:
            now = datetime.now()
            if log_enabled:
                try:
                    with open(self._csv(), "a", encoding="utf-8-sig", newline="") as f:
                        csv.writer(f).writerow([
                            now.strftime("%H:%M:%S"), turn, action,
                            "是" if alert else "", info or ""])
                except Exception:
                    pass
            changed = (self._last is None or self._last != (turn, action))
            self._last = (turn, action)
            if save_enabled and col_img is not None and (changed or alert):
                try:
                    name = "frame_%s_t%d_a%d.png" % (
                        now.strftime("%Y%m%d_%H%M%S_%f")[:-3], turn, action)
                    col_img.save(os.path.join(self._date_dir(), name))
                except Exception:
                    pass

    def save_drop(self, turn, action, reason, col_img):
        """保存被丢弃帧的画面（供事后排查丢弃是否正确），受 max_frames 上限约束"""
        if col_img is None:
            return
        with self._lock:
            try:
                now = datetime.now()
                tag = "".join(ch for ch in reason[:8] if ch.isalnum())
                name = "frame_%s_t%d_a%d_drop_%s.png" % (
                    now.strftime("%Y%m%d_%H%M%S_%f")[:-3], turn, action, tag)
                col_img.save(os.path.join(self._date_dir(), name))
            except Exception:
                pass
            # 丢弃帧不经过 record_count，自行定期清理超限帧
            self._drop_count = getattr(self, "_drop_count", 0) + 1
            if self._drop_count % 20 == 0:
                self.prune()

    def prune(self):
        """删除超出上限的最旧帧文件（递归所有日期子目录 + 根目录旧文件）"""
        try:
            files = []
            for root, _dirs, fs in os.walk(self.frames_dir):
                for f in fs:
                    if f.startswith("frame_"):
                        files.append(os.path.join(root, f))
            if len(files) > self.max_frames:
                files.sort(key=os.path.getmtime)
                for f in files[:len(files) - self.max_frames]:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        except Exception:
            pass


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 主界面
# --------------------------------------------------------------------------
class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.ocr = OcrEngine()
        self.capture = None
        self.recorder = Recorder(SCRIPT_DIR, self.cfg.get("max_frames", 500))
        self.recorder.prune()
        self.filter = ValueFilter(
            max_turn=self.cfg.get("max_turn", 99),
            max_action=self.cfg.get("max_action", 100),
            allow_reset=self.cfg.get("allow_action_reset", True),
            reset_turn_min=self.cfg.get("reset_turn_min", 1),
            reset_action_min=self.cfg.get("reset_action_min", 1),
            action_drop_max=self.cfg.get("action_drop_max", 20))
        # RapidOCR 低频复核器（突变/骤降帧二次确认 + 6秒基线校准）
        try:
            from rapid_recheck import RapidRechecker
            self.rapid = RapidRechecker(min_interval=3.0)
        except Exception:
            self.rapid = None
        self._init_sound()
        self.last_rapid_base = 0.0
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.msg_queue = queue.Queue()
        self.alert_win = None
        self.alert_beep = False
        self.last_dismiss = 0.0
        self.silent_round = False   # 本局静默（点"本局不再提醒"后置真，新对局自动恢复）
        self.consecutive = 0
        self.running = False
        # 日志门控（借鉴 ok-nte LogGate）：未识别节流 + 数值变化轨迹
        self._log_gate = LogGate()
        self._last_logged_val = None
        self._build_ui()
        self.root.after(100, self._poll_queue)
        if self.ocr.ok():
            self.log("OCR 引擎: %s (%s)" % (self.ocr.engine_name(),
                                             self.ocr.tess_path or "系统内置"))
        else:
            self.log("警告: 未找到 OCR 引擎，请运行 start.bat 安装依赖")
        # 窗口模式下强制关闭前台限制（PrintWindow 不需要游戏在前台）
        if self.cfg.get("capture_mode") == "window":
            self.cfg["foreground_only"] = False
            self.var_fg.set(False)
            # 旧版屏幕坐标区域 → 窗口内坐标迁移
            if not self.cfg.get("win_region") and self.cfg.get("region"):
                hwnd = find_game_hwnd(self.cfg["window_keyword"])
                if hwnd:
                    rect = ctypes.wintypes.RECT()
                    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    # 仅当窗口位置正常（在屏幕附近）时才迁移，避免被移出屏幕时产生垃圾坐标
                    if -2000 < rect.left < 5000 and -2000 < rect.top < 5000:
                        x0, y0, x1, y1 = self.cfg["region"]
                        wregion = [x0 - rect.left, y0 - rect.top, x1 - rect.left, y1 - rect.top]
                        self.cfg["win_region"] = wregion
                        save_config(self.cfg)
                        self.log("已把旧屏幕区域迁移为窗口内区域 %s" % (wregion,))
        if self.cfg.get("capture_mode") == "window" and self.cfg.get("win_region"):
            self.log("已加载窗口内监测区域 %s" % (self.cfg["win_region"],))
        elif self.cfg.get("region"):
            self.log("已加载屏幕监测区域 %s" % (self.cfg["region"],))
        else:
            self.log("尚未选择监测区域，请点击「选择监测区域」")

    # ---------------- UI ----------------
    def _build_ui(self):
        root = self.root
        root.title("星穹铁道 · 回合/行动值监控 v2")
        root.geometry("680x600")
        root.resizable(False, False)

        pad = {"padx": 10, "pady": 4}

        frm = tk.LabelFrame(root, text="状态", padx=8, pady=6)
        frm.pack(fill="x", **pad)
        self.lbl_status = tk.Label(frm, text="未开始", font=("Microsoft YaHei UI", 11, "bold"))
        self.lbl_status.pack(side="left")
        self.lbl_values = tk.Label(frm, text="回合数: --  行动值: --",
                                   font=("Microsoft YaHei UI", 11))
        self.lbl_values.pack(side="right")

        frm2 = tk.LabelFrame(root, text="提醒参数", padx=8, pady=6)
        frm2.pack(fill="x", **pad)
        row = tk.Frame(frm2)
        row.pack(fill="x")
        tk.Label(row, text="回合数等于:").pack(side="left")
        self.sp_turn = tk.Spinbox(row, from_=0, to=99, width=5,
                                  textvariable=tk.StringVar(value=str(self.cfg["turn_threshold"])))
        self.sp_turn.pack(side="left", padx=(4, 16))
        tk.Label(row, text="行动值小于:").pack(side="left")
        self.sp_act = tk.Spinbox(row, from_=0, to=999, width=6,
                                 textvariable=tk.StringVar(value=str(self.cfg["action_threshold"])))
        self.sp_act.pack(side="left", padx=(4, 16))
        tk.Label(row, text="间隔(毫秒):").pack(side="left")
        self.sp_int = tk.Spinbox(row, from_=200, to=5000, increment=100, width=6,
                                 textvariable=tk.StringVar(value=str(self.cfg["interval_ms"])))
        self.sp_int.pack(side="left", padx=(4, 16))
        tk.Label(row, text="gamma:").pack(side="left")
        self.sp_gamma = tk.Spinbox(row, from_=0.4, to=1.6, increment=0.1, width=5,
                                   textvariable=tk.StringVar(value=str(self.cfg["gamma"])))
        self.sp_gamma.pack(side="left", padx=(4, 0))
        self.var_sound = tk.BooleanVar(value=self.cfg["sound"])
        self.var_fg = tk.BooleanVar(value=self.cfg["foreground_only"])
        self.var_log = tk.BooleanVar(value=self.cfg.get("log_history", True))
        self.var_save = tk.BooleanVar(value=self.cfg.get("save_frames", True))
        self.var_reset = tk.BooleanVar(value=self.cfg.get("allow_action_reset", True))
        row2 = tk.Frame(frm2)
        row2.pack(fill="x", pady=(4, 0))
        self.cb_sound = tk.Checkbutton(row2, text="声音提醒", variable=self.var_sound)
        self.cb_sound.pack(side="left")
        self.cb_log = tk.Checkbutton(row2, text="记录历史", variable=self.var_log)
        self.cb_log.pack(side="left", padx=16)
        self.cb_save = tk.Checkbutton(row2, text="保存识别画面", variable=self.var_save)
        self.cb_save.pack(side="left", padx=16)
        self.cb_reset = tk.Checkbutton(row2, text="允许行动值突变(3帧确认)", variable=self.var_reset)
        self.cb_reset.pack(side="left", padx=16)
        row3 = tk.Frame(frm2)
        row3.pack(fill="x", pady=(2, 0))
        self.cb_fg = tk.Checkbutton(row3, text="仅游戏窗口在前台时监测(屏幕模式)",
                                    variable=self.var_fg)
        self.cb_fg.pack(side="left")
        self.var_sound_trig = tk.BooleanVar(value=self.cfg.get("sound_trigger", False))
        self.cb_sound_trig = tk.Checkbutton(row3, text="声音触发(需音效样本)",
                                            variable=self.var_sound_trig)
        self.cb_sound_trig.pack(side="left", padx=16)

        frm3 = tk.Frame(root)
        frm3.pack(fill="x", **pad)
        tk.Label(frm3, text="捕获方式:").pack(side="left")
        self.var_mode = tk.StringVar(value=self.cfg["capture_mode"])
        tk.Radiobutton(frm3, text="窗口捕获(推荐,后台可用)", variable=self.var_mode,
                       value="window").pack(side="left")
        tk.Radiobutton(frm3, text="屏幕区域(需可见)", variable=self.var_mode,
                       value="screen").pack(side="left")

        frm4 = tk.Frame(root)
        frm4.pack(fill="x", **pad)
        self.btn_region = tk.Button(frm4, text="选择监测区域", command=self.calibrate)
        self.btn_region.pack(side="left", padx=(0, 6))
        self.btn_test = tk.Button(frm4, text="测试识别", command=self.test_once)
        self.btn_test.pack(side="left", padx=6)
        self.btn_start = tk.Button(frm4, text="开始监测", command=self.start_monitor,
                                   bg="#d9f2d9")
        self.btn_start.pack(side="left", padx=6)
        self.btn_stop = tk.Button(frm4, text="停止", command=self.stop_monitor, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_logs = tk.Button(frm4, text="打开日志", command=self.open_logs)
        self.btn_logs.pack(side="left", padx=6)

        frm5 = tk.LabelFrame(root, text="日志", padx=8, pady=6)
        frm5.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(frm5, height=12, state="disabled", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True)
        # 日志高亮：丢弃=红，突变确认/新对局=橙，声音事件=蓝
        self.txt_log.tag_configure("drop", foreground="#cc0000")
        self.txt_log.tag_configure("event", foreground="#b06000")
        self.txt_log.tag_configure("sound", foreground="#0060cc")

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg, tag=None):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.configure(state="normal")
        if tag:
            self.txt_log.insert("end", "[%s] %s\n" % (ts, msg), tag)
        else:
            self.txt_log.insert("end", "[%s] %s\n" % (ts, msg))
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")
        append_gui_log(ts, msg)

    # ---------------- 捕获后端管理 ----------------
    def _make_capture(self):
        mode = self.var_mode.get()
        self.cfg["capture_mode"] = mode
        if mode == "window":
            hwnd = find_game_hwnd(self.cfg["window_keyword"])
            if hwnd is None:
                raise RuntimeError("未找到游戏窗口（标题含 '%s'），请先启动游戏" % self.cfg["window_keyword"])
            return self._select_window_backend(hwnd)
        else:
            if not self.cfg.get("region"):
                raise RuntimeError("屏幕模式需要先选择监测区域")
            return ScreenCapture(self.cfg["region"])

    def _select_window_backend(self, hwnd):
        """窗口模式：优先 PrintWindow 直捕；黑屏时尝试闪烁捕获（需管理员）"""
        # 检查窗口尺寸（可能被移出屏幕/隐藏）
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w < 300 or h < 200:
            self.log("警告: 游戏窗口过小(%dx%d)，可能被最小化或移出屏幕，请恢复游戏窗口" % (w, h))
        # 探测 PrintWindow（成功不记日志，避免识别失败重建捕获器时刷屏）
        try:
            img = printwindow_capture(hwnd)
            if img is not None and not is_black(img):
                return PrintWindowCapture(hwnd)
        except Exception:
            pass
        # 探测闪烁捕获
        fc = FlashCapture(hwnd)
        test = fc.grab()
        if test is not None:
            self.log("窗口被遮挡：启用闪烁捕获（游戏会周期性置顶）")
            return fc
        raise RuntimeError("无法捕获游戏窗口（被完全遮挡且未以管理员运行，或窗口不可见）")

    # ---------------- 区域校准 ----------------
    def calibrate(self):
        if self.running:
            self.stop_monitor()
        mode = self.var_mode.get()
        if mode == "window":
            try:
                cap = self._make_capture()
            except Exception as e:
                self.log("窗口捕获不可用: %s" % e)
                return
            shot = cap.grab()
            cap.close()
            self.win_cap_shot = shot
            self.log("已捕获游戏窗口画面（%dx%d），请框选倒计时所在整列（上下留足浮动空间）"
                     % shot.size)
        else:
            shot = ImageGrab.grab(all_screens=True)
            self.win_cap_shot = None
            self.log("已截取屏幕，请框选倒计时所在整列（游戏需可见）")
        self._calibrate_dialog(shot, mode)

    def _calibrate_dialog(self, shot, mode):
        W, H = shot.size
        scale = min(1.0, 1050.0 / W, 720.0 / H)
        disp = shot.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        photo = ImageTkImage(disp)

        win = tk.Toplevel(self.root)
        win.title("框选监测区域：倒计时条所在的整列（随行动上下浮动，请框满整列高度）")
        win.attributes("-topmost", True)
        cv = tk.Canvas(win, width=disp.width, height=disp.height, cursor="cross")
        cv.pack()
        cv.create_image(0, 0, anchor="nw", image=photo)
        cv.image = photo

        rect_id = [None]
        start = [None, None]

        def on_press(e):
            start[0], start[1] = e.x, e.y
            if rect_id[0] is not None:
                cv.delete(rect_id[0])
            rect_id[0] = cv.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

        def on_drag(e):
            if rect_id[0] is not None:
                cv.coords(rect_id[0], start[0], start[1], e.x, e.y)

        def on_release(e):
            x0, y0 = start[0] / scale, start[1] / scale
            x1, y1 = e.x / scale, e.y / scale
            if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
                return
            region = [int(round(min(x0, x1))), int(round(min(y0, y1))),
                      int(round(max(x0, x1))), int(round(max(y0, y1)))]
            if mode == "window":
                self.cfg["win_region"] = region
            else:
                try:
                    vx = ctypes.windll.user32.GetSystemMetrics(76)
                    vy = ctypes.windll.user32.GetSystemMetrics(77)
                except Exception:
                    vx = vy = 0
                self.cfg["region"] = [region[0] + vx, region[1] + vy,
                                      region[2] + vx, region[3] + vy]
            self.cfg["capture_mode"] = mode
            save_config(self.cfg)
            win.destroy()
            self.log("监测区域已保存: %s" % (region,))
            self.test_once()

        cv.bind("<ButtonPress-1>", on_press)
        cv.bind("<B1-Motion>", on_drag)
        cv.bind("<ButtonRelease-1>", on_release)

    # ---------------- 识别 ----------------
    def recognize(self, use_persistent=False):
        """返回 (turn, action, info, col_img)；失败时 col_img 为 None"""
        mode = self.cfg["capture_mode"]
        region = self.cfg.get("win_region" if mode == "window" else "region")
        if not region:
            return None, None, "未设置区域", None
        cap = self.capture if (use_persistent and self.capture is not None) else None
        close_after = cap is None
        try:
            if cap is None:
                cap = self._make_capture()
            try:
                shot = cap.grab()
            finally:
                if close_after:
                    cap.close()
        except Exception as e:
            return None, None, "捕获失败: %s" % e, None
        if shot is None:
            return None, None, "捕获失败(助手无响应)", None
        g = self.cfg.get("gamma", 1.0)
        if abs(g - 1.0) > 0.01:
            shot = shot.point(lambda v: int(((v / 255.0) ** g) * 255))
        col = shot.crop(region)
        turn, action, info = extract_column(self.ocr, col)
        return turn, action, info, col

    def test_once(self):
        turn, action, info, _ = self.recognize()
        if turn is None and action is None:
            self.log("测试结果: 未识别到数字（%s）" % info)
            self.lbl_values.configure(text="回合数: --  行动值: --")
            return
        self.lbl_values.configure(text="回合数: %s  行动值: %s" % (turn, action))
        self.log("测试结果: 回合数=%s 行动值=%s（%s）" % (turn, action, info))

    # ---------------- 监控循环 ----------------
    def start_monitor(self):
        if not self.ocr.ok():
            self.log("OCR 引擎不可用，无法监测")
            return
        self._sync_params()
        mode = self.var_mode.get()
        self.cfg["capture_mode"] = mode
        region = self.cfg.get("win_region" if mode == "window" else "region")
        if not region:
            self.log("请先选择监测区域")
            self.calibrate()
            return
        if self.running:
            return
        try:
            self.capture = self._make_capture()
        except Exception as e:
            self.log("启动失败: %s" % e)
            return
        self.running = True
        self.consecutive = 0
        self.filter.reset()
        self.stop_event.clear()
        # 声音触发：开关打开且模板已加载 → 启动监听（独立线程，崩溃自动重启）
        if self.cfg.get("sound_trigger") and self.sound is not None:
            try:
                self.sound.start()
            except Exception as e:
                self.log("声音触发启动失败: %s" % e)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_region.configure(state="disabled")
        self.log("开始监测（%s模式，间隔 %dms，回合数==%s 且 行动值<%s 时提醒）"
                 % (mode, self.cfg["interval_ms"], self.cfg["turn_threshold"],
                    self.cfg["action_threshold"]))

    def stop_monitor(self):
        self.running = False
        self.stop_event.set()
        if self.sound is not None:
            try:
                self.sound.stop()
            except Exception:
                pass
        if self.capture is not None:
            try:
                self.capture.close()
            except Exception:
                pass
            self.capture = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_region.configure(state="normal")
        self.lbl_status.configure(text="已停止")
        self.log("监测已停止")

    # ---------------- 声音触发（借鉴 ok-nte 声音驱动思路） ----------------
    def _init_sound(self):
        """加载 templates/sounds/*.wav 为音效模板，构造 SoundListener（不启动）"""
        self.sound = None
        try:
            sounds_dir = os.path.join(SCRIPT_DIR, "templates", "sounds")
            samples = {}
            if os.path.isdir(sounds_dir):
                for f in sorted(os.listdir(sounds_dir)):
                    if f.lower().endswith(".wav") and not f.lower().startswith("cand"):
                        samples[os.path.splitext(f)[0]] = os.path.join(sounds_dir, f)
            if not samples:
                return
            from sound_trigger import SoundListener
            self.sound = SoundListener(
                self.cfg.get("game_process", "StarRail.exe"),
                samples,
                threshold=float(self.cfg.get("sound_threshold", 0.13)))
            self.sound.on_triggered = self._on_sound_triggered
            self.log("声音触发就绪：%d 个音效模板（%s）"
                     % (len(samples), ", ".join(sorted(samples))))
        except Exception as e:
            self.log("声音触发不可用: %s" % e)
            self.sound = None

    def _on_sound_triggered(self, name, score):
        self.msg_queue.put({"status": "声音事件",
                            "info": "%s(分数%.2f)" % (name, score)})

    def _sync_params(self):
        def getv(spin):
            try:
                return int(spin.get())
            except ValueError:
                return None
        t = getv(self.sp_turn)
        a = getv(self.sp_act)
        i = getv(self.sp_int)
        if t is not None:
            self.cfg["turn_threshold"] = t
        if a is not None:
            self.cfg["action_threshold"] = a
        if i is not None:
            self.cfg["interval_ms"] = i
        try:
            self.cfg["gamma"] = float(self.sp_gamma.get())
        except ValueError:
            pass
        self.cfg["sound"] = self.var_sound.get()
        self.cfg["foreground_only"] = self.var_fg.get()
        self.cfg["log_history"] = self.var_log.get()
        self.cfg["save_frames"] = self.var_save.get()
        self.cfg["allow_action_reset"] = self.var_reset.get()
        self.cfg["sound_trigger"] = self.var_sound_trig.get()
        save_config(self.cfg)

    def open_logs(self):
        """打开日志目录（识别历史 CSV + 画面存档）"""
        try:
            os.startfile(self.recorder.logs_dir)
        except Exception:
            try:
                os.startfile(SCRIPT_DIR)
            except Exception:
                pass

    def _monitor_loop(self):
        interval = max(100, self.cfg["interval_ms"])
        fail_streak = 0
        record_count = 0
        while not self.stop_event.is_set():
            t0 = time.time()
            try:
                if self.alert_win is not None:
                    time.sleep(0.4)
                    continue
                if self.cfg["foreground_only"] and not is_foreground(self.cfg["window_keyword"]):
                    self.msg_queue.put({"status": "等待游戏窗口（前台检测）"})
                    time.sleep(0.4)
                    continue
                turn, action, info, col_img = self.recognize(use_persistent=True)
                if turn is None or action is None:
                    fail_streak += 1
                    # 未识别帧不计入兜底计数：基线未被污染，新版低位闸门
                    # 已能在读到真实值时直接恢复（回合重置高位接受）
                    if fail_streak >= 3 and self.capture is not None:
                        # 捕获器可能已失效（如游戏窗口重建），重建一次
                        try:
                            self.capture.close()
                        except Exception:
                            pass
                        self.capture = None
                        fail_streak = 0
                    self.msg_queue.put({"status": "未识别到数字", "info": info})
                    self.consecutive = 0
                    # 识别失败快速重试，提高有效采样率
                    time.sleep(max(0.05, interval / 1000.0 - (time.time() - t0)))
                    continue
                fail_streak = 0
                # 合理性闸门：范围 + 同回合递减 + 回合减小重置，离谱结果直接丢弃
                ok_val, drop_reason = self.filter.check(turn, action)
                accepted = ok_val
                if not ok_val:
                    self.consecutive = 0
                    # RapidOCR 低频复核：向上突变候选帧 / 向下骤降帧
                    # （复核一致 → 直接确认基线；骤降复核出高位 → 确认丢位拒绝）
                    if (drop_reason.startswith("突变待确认")
                            or drop_reason.startswith("行动值骤降")) \
                            and self.cfg.get("rapid_recheck", True) \
                            and self.rapid is not None:
                        ra = self.rapid.recheck_action(col_img)
                        if ra is not None and ra == action:
                            self.filter.accept(turn, action)
                            drop_reason = ("突变确认(Rapid复核)"
                                           if drop_reason.startswith("突变待确认")
                                           else "骤降确认(Rapid复核)")
                            accepted = True
                        elif ra is not None:
                            # 复核不一致（如骤降帧 RapidOCR 读出高位）→ 确认丢位
                            self.filter.reject()
                            self.recorder.save_drop(turn, action, drop_reason, col_img)
                            self.msg_queue.put({
                                "status": "丢弃(骤降Rapid复核=%d)" % ra,
                                "info": "t%d/a%d" % (turn, action)})
                            continue
                    if not accepted:
                        # 兜底计数仅限「基线污染信号」：突变不允许/突变序列无效
                        # 只在基线低位可疑时出现（基线正确时回合0不可能突变）；
                        # 骤降/超范围/低位拒绝等识别错误类是成功防御，不计数——
                        # 基线未污染，读到真实值即可恢复，计数反而引发无谓重置
                        if (drop_reason.startswith("突变不允许")
                                or drop_reason.startswith("突变序列无效")):
                            self.filter.reject()
                        # 被丢弃帧存档（文件名带原因标记，供事后排查丢弃是否正确）
                        self.recorder.save_drop(turn, action, drop_reason, col_img)
                        self.msg_queue.put({"status": "丢弃(%s)" % drop_reason,
                                            "info": "t%d/a%d" % (turn, action)})
                        continue
                self.filter.accept(turn, action)
                if drop_reason:
                    # 特殊事件（突变确认/新对局/骤降确认）：在说明与日志中标记
                    info = "%s %s" % (drop_reason, info) if info else drop_reason
                # 本局静默自动恢复：新对局开始（回合回到 1）或行动值重置回高位
                if self.silent_round and (turn >= 1 or action >= 50):
                    self.silent_round = False
                    self.msg_queue.put({"status": "提醒恢复",
                                        "info": "新对局开始，本局提醒已恢复"})
                hit = (turn == self.cfg["turn_threshold"]
                       and action < self.cfg["action_threshold"]
                       and time.time() - self.filter.reset_ts > 10
                       and not self.silent_round)
                alert_flag = False
                if hit:
                    self.consecutive += 1
                    if (self.consecutive >= self.cfg["confirm_ticks"]
                            and time.time() - self.last_dismiss >= self.cfg["cooldown_s"]):
                        self.msg_queue.put({"alert": True, "turn": turn, "action": action})
                        self.consecutive = 0
                        alert_flag = True
                else:
                    self.consecutive = 0
                # 记录历史与画面
                try:
                    self.recorder.record(
                        turn, action, alert_flag, info, col_img,
                        log_enabled=self.cfg.get("log_history", True),
                        save_enabled=self.cfg.get("save_frames", True))
                    record_count += 1
                    if record_count % 50 == 0:
                        self.recorder.prune()
                except Exception:
                    pass
                self.msg_queue.put({"status": "监测中", "turn": turn, "action": action, "info": info})
                # 周期基线校准：每 6 秒用 RapidOCR 复核一次行动值，
                # 与基线差异大（如首帧/回合切换帧吞入丢位值）→ 校准基线
                if self.cfg.get("rapid_recheck", True) and self.rapid is not None \
                        and col_img is not None \
                        and time.time() - self.last_rapid_base \
                        >= self.cfg.get("rapid_baseline_s", 6):
                    self.last_rapid_base = time.time()
                    ra = self.rapid.recheck_action(col_img)
                    if ra is not None and self.filter.last:
                        lt, la = self.filter.last
                        if ra != la and abs(ra - la) > 10:
                            self.filter.accept(lt, ra)
                            self.msg_queue.put({
                                "status": "基线校准",
                                "info": "a%d→a%d(Rapid复核)" % (la, ra)})
            except Exception as e:
                self.msg_queue.put({"status": "异常: %s" % e})
                self.consecutive = 0
            time.sleep(max(0.05, interval / 1000.0 - (time.time() - t0)))

    # ---------------- 提醒弹窗 ----------------
    def show_alert(self, turn, action):
        if self.alert_win is not None:
            return
        win = tk.Toplevel(self.root)
        win.title("⚠ 注意！")
        win.configure(bg="#b00020")
        win.attributes("-topmost", True)
        win.geometry("460x240+%d+%d" % (self.root.winfo_screenwidth() // 2 - 230,
                                        self.root.winfo_screenheight() // 2 - 160))
        tk.Label(win, text="⚠ 倒计时告急 ⚠", font=("Microsoft YaHei UI", 20, "bold"),
                 bg="#b00020", fg="white").pack(pady=(28, 8))
        tk.Label(win, text="回合数 %d   行动值 %d" % (turn, action),
                 font=("Microsoft YaHei UI", 26, "bold"), bg="#b00020", fg="#ffe600").pack(pady=6)
        tk.Label(win, text="（行动值低于 %d，请关注！）" % self.cfg["action_threshold"],
                 font=("Microsoft YaHei UI", 11), bg="#b00020", fg="#ffd0d0").pack()

        def dismiss():
            self.last_dismiss = time.time()
            self.alert_beep = False
            self.alert_win = None
            win.destroy()
            self.log("提醒已关闭，下次继续提醒")

        def dismiss_silent():
            self.last_dismiss = time.time()
            self.alert_beep = False
            self.alert_win = None
            self.silent_round = True
            win.destroy()
            self.log("本局不再提醒（新对局开始时自动恢复）")

        btn_row = tk.Frame(win, bg="#b00020")
        btn_row.pack(pady=(18, 0))
        tk.Button(btn_row, text="本局不再提醒", command=dismiss_silent,
                  font=("Microsoft YaHei UI", 12), width=14, height=1,
                  bg="white", fg="#b00020").pack(side="left", padx=10)
        tk.Button(btn_row, text="下次继续提醒", command=dismiss,
                  font=("Microsoft YaHei UI", 12), width=14, height=1,
                  bg="#ffe6e6", fg="#b00020").pack(side="left", padx=10)

        self.alert_win = win
        if self.cfg["sound"]:
            self.alert_beep = True
            threading.Thread(target=self._beep_loop, daemon=True).start()
        self.log("⚠ 触发提醒：回合数=%d 行动值=%d" % (turn, action))

    def _beep_loop(self):
        while self.alert_beep:
            try:
                winsound.Beep(1200, 250)
                time.sleep(0.25)
            except Exception:
                time.sleep(0.5)

    # ---------------- 事件循环 ----------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if "alert" in msg:
                    self.show_alert(msg["turn"], msg["action"])
                if "status" in msg:
                    self.lbl_status.configure(text=msg["status"])
                if "turn" in msg:
                    self.lbl_values.configure(text="回合数: %s  行动值: %s"
                                               % (msg["turn"], msg["action"]))
                    # 数值变化轨迹（changed 模式）：回合/行动值变化时记录一条，
                    # 不变时静默——事后可从日志还原完整数值变化时间线
                    cur = (msg["turn"], msg["action"])
                    if self._last_logged_val != cur:
                        self._last_logged_val = cur
                        self.log("数值: 回合%d 行动值%d" % (cur[0], cur[1]))
                if msg.get("status") == "声音事件" and msg.get("info"):
                    self.log("声音: %s" % msg["info"], tag="sound")
                if msg.get("status") == "模板学习" and msg.get("info"):
                    self.log("事件: %s" % msg["info"], tag="event")
                if msg.get("info") and msg.get("status") == "未识别到数字":
                    # 精简滚动日志：未定位到倒计时条是高频噪音（条浮动/遮挡常态），隐藏；
                    # 保留有条但提取失败/整区域识别异常的诊断信息
                    if not msg["info"].startswith("未定位到倒计时条") \
                            and self._log_gate.allow_message(
                                "unrecog", msg["info"], 5.0, changed=True):
                        self.log("未识别: %s" % msg["info"])
                elif msg.get("status", "").startswith("丢弃"):
                    drop_note = ""
                    if "行动值骤降" in msg["status"]:
                        drop_note = "（疑似数字切换过渡帧，像素级缺位，防御正确）"
                    self.log("%s (%s)%s" % (msg["status"], msg.get("info", ""), drop_note),
                             tag="drop")
                elif ("突变确认" in msg.get("info", "") or "新对局" in msg.get("info", "")
                      or "骤降确认" in msg.get("info", "")
                      or "基线校准" in msg.get("info", "")
                      or "提醒恢复" in msg.get("info", "")):
                    self.log("事件: %s" % msg["info"], tag="event")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def on_close(self):
        self.stop_monitor()
        self.alert_beep = False
        self.root.destroy()


def ImageTkImage(img):
    from PIL import ImageTk
    return ImageTk.PhotoImage(img)


# --------------------------------------------------------------------------
# 自检模式
# --------------------------------------------------------------------------
def selftest(image_path):
    img = Image.open(image_path).convert("RGB")
    a = np.array(img).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 150) & (g < 110) & (b < 110) & (r - g > 60)
    comps = components(red)
    cands = [c for c in comps if c[2] - c[0] >= 100 and 15 <= c[3] - c[1] <= 200]
    if not cands:
        print("未找到红色矩形框")
        return 1
    x0, y0, x1, y1, _ = max(cands, key=lambda c: c[4])
    pad = int(max(20, (x1 - x0) * 0.1))
    region = [max(0, x0 - pad), max(0, y0 - pad),
              min(img.width, x1 + pad), min(img.height, y1 + pad)]
    crop = img.crop(tuple(region))
    # 抹掉红色标注框（全部红色像素涂黑，否则会干扰颜色掩码/锚点检测）
    ca = np.array(crop).astype(int)
    redm = (ca[..., 0] > 150) & (ca[..., 1] < 110) & (ca[..., 2] < 110) & (ca[..., 0] - ca[..., 1] > 60)
    ca[redm] = (0, 0, 0)
    crop = Image.fromarray(ca.astype(np.uint8))
    ocr = OcrEngine()
    ex = Extractor(ocr)
    turn_groups, action_groups, raw_groups, info = ex.extract(crop)
    turn, action = parse_values(turn_groups, action_groups, raw_groups)
    print("OCR 引擎:", ocr.engine_name())
    print("区域(红框+边距):", region)
    print("信息:", info)
    print("回合数字串:", turn_groups, "行动值数字串:", action_groups, "整区域:", raw_groups)
    print("==> 回合数 = %s , 行动值 = %s" % (turn, action))
    return 0 if (turn is not None and action is not None) else 2


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    root = tk.Tk()
    MonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
