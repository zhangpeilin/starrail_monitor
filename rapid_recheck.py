# -*- coding: utf-8 -*-
"""RapidOCR 复核器（低频、按需）

用于对模板匹配/Tesseract 的「可疑帧」做二次确认：
- 向上突变帧（行动值突增）：RapidOCR 复核一致 → 直接确认基线（跳过三帧等待）
- 向下骤降帧（降幅 >20）：RapidOCR 复核判别是真实骤降还是丢位误读
- 周期基线校准（默认每 6 秒）：RapidOCR 读出与基线差异大 → 校准基线

RapidOCR 检测模式单帧约 1-1.7s（限 2 线程），只允许低频调用；
引擎懒加载（首次复核时才初始化，不阻塞程序启动）。
"""
import os
import re
import threading
import time


class RapidRechecker:
    def __init__(self, min_interval=3.0):
        self.engine = None
        self.failed = False
        self._lock = threading.Lock()
        self.last_call = 0.0
        self.min_interval = min_interval   # 两次复核的最小间隔（秒）

    # ------------------------------------------------------------------
    def ready(self):
        return self.engine is not None

    def ensure(self):
        """初始化 RapidOCR 引擎（懒加载，设置线程限制防吃满 CPU）"""
        if self.engine is not None:
            return True
        if self.failed:
            return False
        try:
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            return True
        except Exception:
            self.engine = None
            self.failed = True
            return False

    # ------------------------------------------------------------------
    def recheck_action(self, column_img):
        """对整列图复核行动值：RapidOCR 检测倒计时条原图，
        返回最右侧的数字文本（行动值）转 int，失败/节流返回 None。"""
        if column_img is None:
            return None
        with self._lock:
            now = time.time()
            if now - self.last_call < self.min_interval:
                return None
            self.last_call = now
        if not self.ensure():
            return None
        from starrail_monitor import locate_bar
        bar = locate_bar(column_img)
        if bar is None:
            return None
        crop = __import__("numpy").array(column_img.crop(bar).convert("RGB"))
        try:
            result, _ = self.engine(crop)
        except Exception:
            return None
        if not result:
            return None
        items = []
        for box, text, score in result:
            text = str(text).strip()
            if re.fullmatch(r"\d+", text):
                items.append((min(p[0] for p in box), int(text)))
        if not items:
            return None
        items.sort(key=lambda t: t[0])
        return items[-1][1]     # 最右侧数字 = 行动值
