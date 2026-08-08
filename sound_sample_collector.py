# -*- coding: utf-8 -*-
"""音效样本采集工具：从游戏录音中自动切分候选音效片段

用法（双击 collect_sounds.bat 或命令行）：
  python sound_sample_collector.py <录音.wav> [-i]

流程：
  1. 加载录音（支持常见 wav），按短时能量自动切分「音效事件」候选片段
  2. 非交互模式：列出所有候选片段（编号/时间/长度/峰值），
     用户试听后把目标片段复制到 templates/sounds/<名称>.wav
  3. 交互模式（-i）：程序逐个播放候选（winsound），输入名称保存为模板
     或回车跳过；输入 q 结束

保存位置：templates/sounds/<名称>.wav（程序启动时自动加载为音效模板）
"""
import os
import sys

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(SCRIPT_DIR, "templates", "sounds")
CAND_DIR = os.path.join(SOUNDS_DIR, "candidates")
SR = 48000


def _load(path):
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        n = int(round(len(data) * SR / sr))
        data = np.interp(np.linspace(0, 1, len(data), endpoint=False),
                         np.linspace(0, 1, n, endpoint=False),
                         data) if False else np.interp(
                             np.linspace(0, 1, n, endpoint=False),
                             np.linspace(0, 1, len(data), endpoint=False), data)
    return data.astype(np.float32), sr


def split_candidates(data, energy_mult=6.0, min_dur=0.12, max_dur=3.0,
                     merge_gap=0.25, pad=0.05):
    """短时能量包络切分：返回 [(start_s, end_s, peak), ...]"""
    win = int(SR * 0.01)          # 10ms 能量窗
    hop = int(SR * 0.005)         # 5ms 步进
    n = len(data)
    if n < SR:
        return []
    energy = []
    for i in range(0, n - win, hop):
        e = float(np.sqrt(np.mean(data[i:i + win] ** 2)))
        energy.append(e)
    energy = np.array(energy)
    # 能量基准：取中位数×系数（抗个别大噪声），下限防全静音
    base = float(np.median(energy))
    thresh = max(base * energy_mult, 0.01)
    idx = np.where(energy > thresh)[0]
    segs = []
    if len(idx):
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i - prev > int(merge_gap / 0.005):
                segs.append((start, prev))
                start = i
            prev = i
        segs.append((start, prev))
    out = []
    for s, e in segs:
        s_s = max(0.0, s * 0.005 - pad)
        e_s = min(n / SR, e * 0.005 + 0.005 + pad)
        if e_s - s_s >= min_dur and e_s - s_s <= max_dur:
            peak = float(np.max(np.abs(data[int(s_s * SR):int(e_s * SR)])))
            out.append((s_s, e_s, peak))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    interactive = "-i" in sys.argv
    if not args:
        print("用法: python sound_sample_collector.py <录音.wav> [-i]")
        print("  -i  交互模式：逐个试听候选片段并命名保存")
        return 1
    rec = args[0]
    if not os.path.isfile(rec):
        print("文件不存在: %s" % rec)
        return 1
    os.makedirs(CAND_DIR, exist_ok=True)
    os.makedirs(SOUNDS_DIR, exist_ok=True)

    print("加载录音 %s …" % rec)
    data, _ = _load(rec)
    print("录音时长 %.1f 秒" % (len(data) / SR))

    cands = split_candidates(data)
    if not cands:
        print("未切分出候选音效（能量阈值内无事件？试试录音里让音效更突出）")
        return 2
    print("\n切分出 %d 个候选片段（保存到 templates/sounds/candidates/）：" % len(cands))
    saved = []
    for i, (s, e, p) in enumerate(cands):
        name = "cand_%02d_%06.2fs_%06.2fs.wav" % (i + 1, s, e)
        path = os.path.join(CAND_DIR, name)
        from scipy.io import wavfile
        pcm = (data[int(s * SR):int(e * SR)] * 32767.0).astype(np.int16)
        wavfile.write(path, SR, pcm)
        saved.append((i + 1, s, e, p, path))
        print("  [%02d] %.2f-%.2fs (%.2fs, 峰值%.3f) → %s"
              % (i + 1, s, e, e - s, p, os.path.basename(path)))

    if not interactive:
        print("\n试听 candidates/ 下的片段，把目标音效重命名复制到")
        print("templates/sounds/<名称>.wav（如 turn_end.wav）即可被程序加载。")
        return 0

    import winsound
    print("\n交互模式：自动播放每个候选，输入名称保存 / 回车跳过 / q 退出")
    for i, s, e, p, path in saved:
        print("\n[%02d] %.2f-%.2fs 播放中…" % (i, s, e), end="", flush=True)
        winsound.PlaySound(path, winsound.SND_FILENAME)
        name = input("\n保存名称（如 turn_end，回车跳过，q 退出）: ").strip()
        if name.lower() == "q":
            break
        if name:
            dst = os.path.join(SOUNDS_DIR, name if name.endswith(".wav") else name + ".wav")
            import shutil
            shutil.copyfile(path, dst)
            print("已保存 → %s" % dst)
    print("\n完成。templates/sounds/ 下的 wav 将在程序启动时加载为音效模板。")


if __name__ == "__main__":
    sys.exit(main())
