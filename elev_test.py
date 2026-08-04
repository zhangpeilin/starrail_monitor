# -*- coding: utf-8 -*-
"""提权环境验证 + 捕获方案测试（请以管理员身份运行）
输出到：elev_result.txt 和两张对比图"""
import ctypes
import ctypes.wintypes as wt
import sys
import time
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageGrab

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elev_result.txt")
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

def find_game_window():
    """找到标题以「崩坏：」开头且可见的游戏窗口（排除隐藏辅助窗口）"""
    wins = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, lparam):
        buf = ctypes.create_unicode_buffer(256)
        if user32.GetWindowTextW(hwnd, buf, 256) > 0 and buf.value.startswith("崩坏："):
            if user32.IsWindowVisible(hwnd):
                rect = wt.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 200 and h > 200:
                    wins.append((hwnd, w * h))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    if not wins:
        return 0
    wins.sort(key=lambda t: -t[1])
    return wins[0][0]

log_lines = []

def log(msg):
    log_lines.append(msg)
    print(msg)

def main():
    log("=== 提权捕获测试 ===")
    # 管理员检查
    try:
        import ctypes.wintypes
        hToken = ctypes.wintypes.HANDLE()
        kernel32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x8, ctypes.byref(hToken))
        class TOKEN_ELEVATION(ctypes.Structure):
            _fields_ = [("TokenIsElevated", wt.DWORD)]
        te = TOKEN_ELEVATION()
        size = wt.DWORD()
        advapi32 = ctypes.windll.advapi32
        advapi32.GetTokenInformation(hToken, 20, ctypes.byref(te), ctypes.sizeof(te), ctypes.byref(size))
        log("本进程提权状态: %s" % ("已提权(管理员)" if te.TokenIsElevated else "未提权"))
    except Exception as e:
        log("提权检查失败: %s" % e)

    GAME = find_game_window()
    if GAME == 0:
        log("未找到可见的游戏窗口（标题以「崩坏：」开头）")
        save()
        return
    log("游戏窗口: %d" % GAME)
    rect = wt.RECT()
    user32.GetWindowRect(GAME, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    log("游戏尺寸: %dx%d @(%d,%d)" % (w, h, rect.left, rect.top))

    # ---- 测试1: PrintWindow (PW_RENDERFULLCONTENT) ----
    log("--- 测试1: PrintWindow ---")
    try:
        hwnd_dc = user32.GetWindowDC(GAME)
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        gdi32.SelectObject(mem_dc, bmp)
        ok = user32.PrintWindow(GAME, mem_dc, 2)
        class BMIH(ctypes.Structure):
            _fields_ = [("biSize", wt.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                        ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                        ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                        ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]
        class BMI(ctypes.Structure):
            _fields_ = [("bmiHeader", BMIH)]
        bmi = BMI()
        bmi.bmiHeader.biSize = ctypes.sizeof(BMIH)
        bmi.bmiHeader.biWidth = w; bmi.bmiHeader.biHeight = -h
        bmi.bmiHeader.biPlanes = 1; bmi.bmiHeader.biBitCount = 32
        buf = ctypes.create_string_buffer(w * h * 4)
        got = gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        gdi32.DeleteObject(bmp); gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(GAME, hwnd_dc)
        log("PrintWindow ok=%d got=%d" % (ok, got))
        if got:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            img = Image.fromarray(arr[..., 2::-1].copy(), "RGB")
            img.save(os.path.join(os.path.dirname(OUT), "elev_pw.png"))
            nonblack = (arr[..., :3].max(axis=2) > 30).mean()
            log("PrintWindow 非黑像素比=%.3f 均值=%s" % (nonblack, arr[..., :3].reshape(-1, 3).mean(axis=0).round(1)))
    except Exception as e:
        log("PrintWindow 异常: %s" % e)

    # ---- 测试2: 置顶截图（不抢焦点） ----
    log("--- 测试2: 置顶截图 ---")
    SWP_NOACTIVATE = 0x0010; SWP_NOMOVE = 0x0002; SWP_NOSIZE = 0x0001
    above = user32.GetWindow(GAME, 3)
    ret = user32.SetWindowPos(GAME, -1, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE)
    log("置顶 ret=%d err=%d" % (ret, kernel32.GetLastError()))
    time.sleep(0.5)
    cx, cy = (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
    top = user32.WindowFromPoint(wt.POINT(cx, cy))
    buf2 = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(top, buf2, 256)
    log("置顶后中心点顶层窗口=%d '%s'（游戏=%d）" % (top, buf2.value[:30], GAME))
    if top == GAME:
        shot = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True)
        shot.save(os.path.join(os.path.dirname(OUT), "elev_flash.png"))
        a = np.array(shot)
        log("置顶截图 均值=%s 过曝比=%.3f" % (a.reshape(-1, 3).mean(axis=0).round(1), (a.max(axis=2) > 250).mean()))
    else:
        log("置顶未生效，跳过截图")
    user32.SetWindowPos(GAME, above, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE)
    log("恢复Z序 ret=%d err=%d" % (user32.SetWindowPos(GAME, above, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE), kernel32.GetLastError()))
    log("=== 完成 ===")
    save()

def save():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print("结果已保存到:", OUT)
    except Exception as e:
        print("保存失败:", e)

if __name__ == "__main__":
    main()
