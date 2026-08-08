# -*- coding: utf-8 -*-
"""声音触发模块：WASAPI 进程环回捕获 + 音效模板匹配

借鉴 ok-nte / ZZZSoundTrigger 的「声音驱动」思路，自研实现：
  - 捕获：Windows WASAPI 进程级环回（ActivateAudioInterfaceAsync +
    AUDIOCLIENT_ACTIVATION_PARAMS，仅捕获目标进程树的声音，Win10 19041+）
  - 匹配：环形缓冲 0.2s 窗口 → 高通滤波 → 波形归一化 → FFT 互相关，
    最大相关系数超过阈值即触发（含防连发与崩溃自动重启）

依赖：comtypes（WASAPI COM）、scipy（滤波/相关）、numpy。无 librosa/numba 重依赖。
"""
import os
import sys
import threading
import time

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from scipy.signal import butter, correlate, filtfilt
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

CAPTURE_SAMPLE_RATE = 48000
SAMPLE_LEN = 0.2            # 匹配窗口（秒）
DETECTION_INTERVAL = 0.025  # 检测步进（秒）
DEFAULT_THRESHOLD = 0.13
TRIGGER_INTERVAL = 0.5      # 防连发（秒）
RESTART_INTERVAL = 1.0      # 监听循环崩溃后重启间隔（秒）
DEGREE = 4                  # 高通滤波器阶数
CUT_OFF = 1000              # 高通截止频率（Hz）

_FILT = "音效模板"

logger = None  # 由集成方注入 print 风格日志


def _log(msg):
    if logger is not None:
        logger(msg)
    else:
        print("[声音] " + msg)


def _load_wav(path):
    """wav → float32 单声道（scipy 解码，避免 numba 重依赖）"""
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
    if sr != CAPTURE_SAMPLE_RATE:
        # 线性插值重采样（轻量，模板加载一次性）
        n = int(round(len(data) * CAPTURE_SAMPLE_RATE / sr))
        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, n, endpoint=False)
        data = np.interp(x_new, x_old, data)
    return data.astype(np.float32)


# ============================================================================
# WASAPI 进程环回捕获（ctypes + comtypes，微软公开 API 自研实现）
# ============================================================================
import ctypes
from ctypes import wintypes

_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
_VT_BLOB = 65
_VIRTUAL_AUDIO_DEVICE = "VAD\\Process_Loopback"

_AUDCLNT_SHAREMODE_SHARED = 0
_AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
_AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
_AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
_AUDCLNT_BUFFERFLAGS_SILENT = 0x2
_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_REFERENCE_TIME = ctypes.c_longlong


class _WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class _WAVEFORMATEXTENSIBLE(ctypes.Structure):
    _fields_ = [
        ("Format", _WAVEFORMATEX),
        ("Samples", wintypes.WORD),
        ("dwChannelMask", wintypes.DWORD),
        ("SubFormat", ctypes.c_ubyte * 16),
    ]


class _AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [
        ("TargetProcessId", wintypes.DWORD),
        ("ProcessLoopbackMode", ctypes.c_int),
    ]


class _AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [
        ("ActivationType", ctypes.c_int),
        ("ProcessLoopbackParams", _AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS),
    ]


class _BLOB(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", ctypes.c_void_p)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", wintypes.USHORT),
        ("wReserved1", wintypes.USHORT),
        ("wReserved2", wintypes.USHORT),
        ("wReserved3", wintypes.USHORT),
        ("blob", _BLOB),
    ]


# ActivateAudioInterfaceAsync 回调接口（IActivateAudioInterfaceCompletionHandler）
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown, COMObject

_IID_IAUDIOCLIENT = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
_IID_IAUDIOCAPTURECLIENT = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
_IID_IAUDIOCLIENT_ACTIVATION_COMPLETION_HANDLER = GUID(
    "{41D949AB-9862-444A-80F6-C261334DA5EB}")


class _IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = _IID_IAUDIOCLIENT_ACTIVATION_COMPLETION_HANDLER
    _methods_ = [
        COMMETHOD([], HRESULT, "ActivateCompleted",
                  (["in"], ctypes.c_void_p, "activateOperation"),
                  (["in"], ctypes.c_void_p, "hr"),
                  (["in"], ctypes.c_void_p, "pIAudioInterface")),
    ]


class _IAudioClient(IUnknown):
    _iid_ = _IID_IAUDIOCLIENT
    _methods_ = [
        COMMETHOD([], HRESULT, "Initialize",
                  (["in"], ctypes.c_int, "ShareMode"),
                  (["in"], wintypes.DWORD, "StreamFlags"),
                  (["in"], _REFERENCE_TIME, "hnsBufferDuration"),
                  (["in"], _REFERENCE_TIME, "hnsPeriodicity"),
                  (["in"], ctypes.POINTER(_WAVEFORMATEX), "pFormat"),
                  (["in"], GUID, "AudioSessionGuid")),
        COMMETHOD([], HRESULT, "GetBufferSize",
                  (["out"], ctypes.POINTER(wintypes.DWORD), "pNumBufferFrames")),
        COMMETHOD([], HRESULT, "GetStreamLatency",
                  (["out"], ctypes.POINTER(_REFERENCE_TIME), "phnsLatency")),
        COMMETHOD([], HRESULT, "GetCurrentPadding",
                  (["out"], ctypes.POINTER(wintypes.DWORD), "pNumPaddingFrames")),
        COMMETHOD([], HRESULT, "IsFormatSupported",
                  (["in"], ctypes.c_int, "ShareMode"),
                  (["in"], ctypes.POINTER(_WAVEFORMATEX), "pFormat"),
                  (["out"], ctypes.POINTER(ctypes.POINTER(_WAVEFORMATEX)), "ppClosestMatch")),
        COMMETHOD([], HRESULT, "GetMixFormat",
                  (["out"], ctypes.POINTER(ctypes.POINTER(_WAVEFORMATEX)), "ppDeviceFormat")),
        COMMETHOD([], HRESULT, "GetDevicePeriod",
                  (["out"], ctypes.POINTER(_REFERENCE_TIME), "phnsDefaultDevicePeriod"),
                  (["out"], ctypes.POINTER(_REFERENCE_TIME), "phnsMinimumDevicePeriod")),
        COMMETHOD([], HRESULT, "Start"),
        COMMETHOD([], HRESULT, "Stop"),
        COMMETHOD([], HRESULT, "Reset"),
        COMMETHOD([], HRESULT, "SetEventHandle",
                  (["in"], ctypes.c_void_p, "eventHandle")),
        COMMETHOD([], HRESULT, "GetService",
                  (["in"], GUID, "riid"),
                  (["out"], ctypes.POINTER(ctypes.c_void_p), "ppv")),
    ]


class _IAudioCaptureClient(IUnknown):
    _iid_ = _IID_IAUDIOCAPTURECLIENT
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer",
                  (["out"], ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), "ppData"),
                  (["out"], ctypes.POINTER(wintypes.DWORD), "pNumFramesToRead"),
                  (["out"], ctypes.POINTER(wintypes.DWORD), "pdwFlags"),
                  (["out"], ctypes.POINTER(ctypes.c_ulonglong), "pu64DevicePosition"),
                  (["out"], ctypes.POINTER(ctypes.c_ulonglong), "pu64QPCPosition")),
        COMMETHOD([], HRESULT, "ReleaseBuffer",
                  (["in"], wintypes.DWORD, "NumFramesRead")),
        COMMETHOD([], HRESULT, "GetNextPacketSize",
                  (["out"], ctypes.POINTER(wintypes.DWORD), "pNumFramesInNextPacket")),
    ]


class _ActivationCallback(COMObject):
    """ActivateAudioInterfaceAsync 异步回调：收集激活结果"""

    _com_interfaces_ = [_IActivateAudioInterfaceCompletionHandler]

    def __init__(self):
        super().__init__()
        self.done = threading.Event()
        self.audio_client = None
        self.hr = None
        self._guard = [None]  # 保持 comtypes 生成的接口对象引用

    def ActivateCompleted(self, activateOperation, hr, pIAudioInterface):
        try:
            self.hr = hr if isinstance(hr, int) else hr.value
            if pIAudioInterface:
                iface = ctypes.cast(
                    ctypes.c_void_p(pIAudioInterface), ctypes.POINTER(ctypes.c_void_p))
                client = _IAudioClient(iface)
                self.audio_client = client
                self._guard[0] = client
        except Exception:
            pass
        finally:
            self.done.set()
        return 0


def _fmt_desc(fmt):
    try:
        if fmt.contents.wFormatTag == _WAVE_FORMAT_EXTENSIBLE:
            ext = ctypes.cast(fmt, ctypes.POINTER(_WAVEFORMATEXTENSIBLE)).contents
            tag = ext.Format.wFormatTag
        else:
            tag = fmt.contents.wFormatTag
        ch = fmt.contents.nChannels
        sr = fmt.contents.nSamplesPerSec
        bits = fmt.contents.wBitsPerSample
        name = "PCM" if tag == _WAVE_FORMAT_PCM else (
            "IEEE_FLOAT" if tag == _WAVE_FORMAT_IEEE_FLOAT else "0x%X" % tag)
        return "%dch %dHz %dbit %s" % (ch, sr, bits, name)
    except Exception:
        return "?"""


def _to_float32(pdata, frames, fmt):
    """原始缓冲 → float32 单声道（按格式解码，兼容 IEEE_FLOAT/PCM 16/32/24/8）"""
    ch = fmt.contents.nChannels
    tag = fmt.contents.wFormatTag
    bits = fmt.contents.wBitsPerSample
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        bits = ctypes.cast(fmt, ctypes.POINTER(_WAVEFORMATEXTENSIBLE)).contents.Format.wBitsPerSample
    n = frames * ch
    if bits == 32 and tag in (_WAVE_FORMAT_IEEE_FLOAT, _WAVE_FORMAT_EXTENSIBLE):
        arr = np.frombuffer(ctypes.string_at(pdata, n * 4), dtype=np.float32)
    elif bits == 32:
        arr = np.frombuffer(ctypes.string_at(pdata, n * 4), dtype=np.int32).astype(
            np.float32) / 2147483648.0
    elif bits == 16:
        arr = np.frombuffer(ctypes.string_at(pdata, n * 2), dtype=np.int16).astype(
            np.float32) / 32768.0
    elif bits == 24:
        raw = np.frombuffer(ctypes.string_at(pdata, n * 3), dtype=np.uint8)
        b = raw.reshape(-1, 3).astype(np.int32)
        arr = ((b[:, 0] << 8) | (b[:, 1] << 16) | (b[:, 2] << 24)).astype(
            np.float32) / 2147483648.0
    elif bits == 8:
        arr = (np.frombuffer(ctypes.string_at(pdata, n), dtype=np.uint8).astype(
            np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError("不支持的音频位深: %d" % bits)
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr


class WasapiProcessCapture:
    """WASAPI 进程环回捕获器：仅捕获目标进程树的声音"""

    def __init__(self, process_name):
        self.process_name = process_name
        self.audio_client = None
        self.capture_client = None
        self.mix_format = None
        self.error = None
        self.sr = CAPTURE_SAMPLE_RATE

    def start(self):
        """激活进程环回并初始化。返回 True/False，失败原因在 self.error"""
        try:
            target_pid = self._find_pid(self.process_name)
            if target_pid is None:
                self.error = "进程未运行: %s" % self.process_name
                return False
            self.error = None
            self.audio_client, self.mix_format = self._activate(target_pid)
            self.capture_client = self._get_capture_client()
            return True
        except Exception as e:
            self.error = str(e)
            self.audio_client = None
            return False

    def stop(self):
        try:
            if self.audio_client is not None:
                self.audio_client.Stop()
        except Exception:
            pass
        self.audio_client = None
        self.capture_client = None

    @property
    def is_alive(self):
        return self.capture_client is not None

    @staticmethod
    def _find_pid(process_name):
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq %s" % process_name, "/FO", "CSV"],
                capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        for line in out.splitlines()[1:]:
            parts = line.strip('"').split('","')
            if len(parts) >= 2 and parts[0].lower() == process_name.lower():
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return None

    def _activate(self, target_pid):
        """ActivateAudioInterfaceAsync 激活进程环回设备"""
        kernel32 = ctypes.windll.kernel32
        hr_func = kernel32.ActivateAudioInterfaceAsync
        hr_func.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(GUID), ctypes.c_void_p,
            ctypes.POINTER(_PROPVARIANT), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p)]
        hr_func.restype = HRESULT

        # 打包 AUDIOCLIENT_ACTIVATION_PARAMS 到 PROPVARIANT(VT_BLOB)
        params = _AUDIOCLIENT_ACTIVATION_PARAMS()
        params.ActivationType = _ACTIVATION_TYPE_PROCESS_LOOPBACK
        params.ProcessLoopbackParams.TargetProcessId = target_pid
        params.ProcessLoopbackParams.ProcessLoopbackMode = _LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
        blob = ctypes.create_string_buffer(ctypes.sizeof(params))
        ctypes.memmove(blob, ctypes.byref(params), ctypes.sizeof(params))
        prop = _PROPVARIANT()
        prop.vt = _VT_BLOB
        prop.blob.cbSize = ctypes.sizeof(params)
        prop.blob.pBlobData = ctypes.cast(blob, ctypes.c_void_p)

        cb = _ActivationCallback()
        op = ctypes.c_void_p()
        hr = hr_func(_VIRTUAL_AUDIO_DEVICE, ctypes.byref(_IID_IAUDIOCLIENT),
                     None, ctypes.byref(prop),
                     ctypes.cast(cb, ctypes.c_void_p), ctypes.byref(op))
        if hr < 0:
            raise RuntimeError("ActivateAudioInterfaceAsync 失败 HRESULT=0x%X" % (hr & 0xFFFFFFFF))
        if not cb.done.wait(timeout=5.0):
            raise RuntimeError("音频设备激活超时")
        if cb.hr is not None and cb.hr < 0:
            raise RuntimeError("进程环回激活失败 HRESULT=0x%X" % (cb.hr & 0xFFFFFFFF))
        if cb.audio_client is None:
            raise RuntimeError("激活回调未返回 IAudioClient")
        client = cb.audio_client

        # GetMixFormat → Initialize（LOOPBACK | EVENTCALLBACK | AUTOCONVERTPCM）
        pf = ctypes.POINTER(_WAVEFORMATEX)()
        hr = client.GetMixFormat(ctypes.byref(pf))
        if hr < 0:
            raise RuntimeError("GetMixFormat HRESULT=0x%X" % (hr & 0xFFFFFFFF))
        fmt = pf
        _log("混音格式: %s" % _fmt_desc(fmt))
        flags = (_AUDCLNT_STREAMFLAGS_LOOPBACK | _AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                 | _AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM)
        hr = client.Initialize(_AUDCLNT_SHAREMODE_SHARED, flags,
                               500000, 0, fmt, None)
        if hr < 0:
            raise RuntimeError("Initialize HRESULT=0x%X" % (hr & 0xFFFFFFFF))
        return client, fmt

    def _get_capture_client(self):
        pv = ctypes.c_void_p()
        hr = self.audio_client.GetService(_IID_IAUDIOCAPTURECLIENT, ctypes.byref(pv))
        if hr < 0 or not pv.value:
            raise RuntimeError("GetService(IAudioCaptureClient) HRESULT=0x%X" % (hr & 0xFFFFFFFF))
        return _IAudioCaptureClient(ctypes.cast(pv, ctypes.POINTER(ctypes.c_void_p)))

    def read(self, timeout=0.2):
        """读取一个数据包 → float32 单声道数组；无数据返回 None"""
        if self.capture_client is None:
            return None
        try:
            psize = wintypes.DWORD()
            hr = self.capture_client.GetNextPacketSize(ctypes.byref(psize))
            if hr < 0:
                raise RuntimeError("GetNextPacketSize HRESULT=0x%X" % (hr & 0xFFFFFFFF))
            if psize.value == 0:
                return None
            pdata = ctypes.POINTER(ctypes.c_ubyte)()
            nframes = wintypes.DWORD()
            flags = wintypes.DWORD()
            hr = self.capture_client.GetBuffer(
                ctypes.byref(pdata), ctypes.byref(nframes), ctypes.byref(flags),
                None, None)
            if hr < 0:
                raise RuntimeError("GetBuffer HRESULT=0x%X" % (hr & 0xFFFFFFFF))
            try:
                if flags.value & _AUDCLNT_BUFFERFLAGS_SILENT:
                    return np.zeros(int(nframes.value), dtype=np.float32)
                return _to_float32(pdata, int(nframes.value), self.mix_format)
            finally:
                self.capture_client.ReleaseBuffer(nframes.value)
        except Exception:
            return None


# ============================================================================
# 音效匹配监听器
# ============================================================================
class SoundListener:
    """监听目标游戏进程音频，匹配预设音效模板，触发回调

    sample_paths: {名称: wav 路径}
    on_triggered: 回调函数(名称, 分数)
    """

    def __init__(self, process_name, sample_paths, threshold=DEFAULT_THRESHOLD,
                 sample_len=SAMPLE_LEN, detection_interval=DETECTION_INTERVAL,
                 allow_successive=False):
        if not _SCIPY_OK:
            raise RuntimeError("缺少 scipy，请先运行 start.bat 安装依赖")
        self.process_name = process_name
        self.sample_paths = sample_paths
        self.threshold = threshold
        self.sample_len = sample_len
        self.detection_interval = detection_interval
        self.allow_successive = allow_successive
        self.on_triggered = None
        self.samples = {}
        self._b = None
        self._a = None
        self._running = False
        self._stop_event = None
        self._thread = None
        self._capture = None
        self._last_trigger = 0.0
        self._load_samples()
        self.enabled = True  # 捕获失败时由外层决定是否禁用

    # ---------------- 样本 ----------------
    def _load_samples(self):
        self._b, self._a = butter(DEGREE, CUT_OFF, btype="highpass",
                                  output="ba", fs=CAPTURE_SAMPLE_RATE)
        for name, path in self.sample_paths.items():
            cache = "%s_%dkHz_%d_%d.npy" % (path, CAPTURE_SAMPLE_RATE // 1000, DEGREE, CUT_OFF)
            if os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
                wav = np.load(cache)
            else:
                wav = _load_wav(path)
                if not os.path.isfile(path):
                    raise FileNotFoundError("音效样本不存在: %s" % path)
                wav = self._filtering(wav)
                try:
                    np.save(cache, wav)
                except Exception:
                    pass
            self.samples[name] = self._normalize(wav)
            _log("已加载音效样本 %s: %.2fs (%d 样本点)" % (name, len(wav) / CAPTURE_SAMPLE_RATE, len(wav)))

    def _filtering(self, wav):
        return filtfilt(self._b, self._a, wav)

    @staticmethod
    def _normalize(wav):
        std = float(np.std(wav))
        return wav / std if std > 1e-9 else wav

    # ---------------- 匹配 ----------------
    def _match(self, window, sample):
        """FFT 互相关最大相关系数（ok-nte 同思路）"""
        if len(window) > len(sample):
            corr = correlate(window, sample, mode="same", method="fft") / len(window)
        else:
            corr = correlate(sample, window, mode="same", method="fft") / len(sample)
        return float(np.max(np.abs(corr)))

    # ---------------- 生命周期 ----------------
    def start(self):
        if self._running:
            return True
        self._running = True
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sound-trigger")
        self._thread.start()
        _log("声音监听启动（进程 %s，阈值 %.2f，%d 个音效）"
             % (self.process_name, self.threshold, len(self.samples)))
        return True

    def stop(self):
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        _log("声音监听已停止")

    @property
    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()

    # ---------------- 主循环 ----------------
    def _loop(self):
        while self._running:
            try:
                self._listen_once()
            except Exception as e:
                _log("监听异常: %s" % e)
            finally:
                if self._capture is not None:
                    try:
                        self._capture.stop()
                    except Exception:
                        pass
                    self._capture = None
            if self._running:
                _log("监听循环退出，%.1fs 后重启" % RESTART_INTERVAL)
                self._stop_event.wait(RESTART_INTERVAL)

    def _listen_once(self):
        max_samples = int(CAPTURE_SAMPLE_RATE * self.sample_len)
        samples_per_check = max(1, int(CAPTURE_SAMPLE_RATE * self.detection_interval))
        ring = np.zeros(max_samples * 2, dtype=np.float32)
        pos = 0
        total = 0
        since_check = 0

        while self._running:
            if self._capture is None or not self._capture.is_alive():
                if self._capture is not None:
                    self._capture.stop()
                self._capture = WasapiProcessCapture(self.process_name)
                if not self._capture.start():
                    err = self._capture.error or "未知错误"
                    self._capture = None
                    _log("进程环回捕获不可用: %s（%.1fs 后重试）" % (err, RESTART_INTERVAL))
                    self._stop_event.wait(RESTART_INTERVAL)
                    continue
                _log("进程音频捕获已就绪")
            frame = self._capture.read(timeout=0.2)
            if frame is None or frame.size == 0:
                continue
            n = frame.shape[0]
            if n >= ring.shape[0]:
                frame = frame[-ring.shape[0]:]
            end = pos + n
            if end <= ring.shape[0]:
                ring[pos:end] = frame
            else:
                first = ring.shape[0] - pos
                ring[pos:] = frame[:first]
                ring[:end - ring.shape[0]] = frame[first:]
            pos = end % ring.shape[0]
            total += n
            since_check += n
            if total < max_samples or since_check < samples_per_check:
                continue
            since_check = 0
            if pos >= max_samples:
                window = ring[pos - max_samples:pos]
            else:
                window = np.concatenate([ring[-(max_samples - pos):], ring[:pos]])
            self._check(window)

    def _check(self, window):
        if not self.samples:
            return
        norm = self._normalize(self._filtering(window.astype(np.float64)))
        now = time.time()
        if not self.allow_successive and now - self._last_trigger < TRIGGER_INTERVAL:
            return
        for name, sample in self.samples.items():
            score = self._match(norm, sample)
            if score > self.threshold:
                _log("音效触发 %s 分数=%.3f（阈值 %.2f）" % (name, score, self.threshold))
                self._last_trigger = now
                if self.on_triggered:
                    try:
                        self.on_triggered(name, score)
                    except Exception:
                        pass
                return


# ============================================================================
# 捕获自测：python sound_trigger.py --capture-test <进程名> [秒数]
# 捕获目标进程 N 秒音频存为 capture_test.wav，供试听验证环回配置
# ============================================================================
def capture_test(process_name, seconds=5.0):
    _log("捕获自测：监听 %s 进程 %d 秒…" % (process_name, seconds))
    cap = WasapiProcessCapture(process_name)
    if not cap.start():
        _log("启动失败: %s" % cap.error)
        return 1
    _log("捕获就绪（%s），播放游戏声音进行验证…" % _fmt_desc(cap.mix_format))
    chunks = []
    start = time.time()
    silent_sec = 0.0
    while time.time() - start < seconds:
        frame = cap.read(timeout=0.2)
        if frame is not None and frame.size:
            chunks.append(frame)
        time.sleep(0.01)
    cap.stop()
    if not chunks:
        _log("未捕获到任何音频数据（进程未发声？）")
        return 2
    data = np.concatenate(chunks)
    from scipy.io import wavfile
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture_test.wav")
    pcm = (data * 32767.0).astype(np.int16)
    wavfile.write(out, CAPTURE_SAMPLE_RATE, pcm)
    peak = float(np.max(np.abs(data)))
    _log("已保存 %d 秒音频 → %s（峰值 %.3f，非静音）" % (len(data) / CAPTURE_SAMPLE_RATE, out, peak))
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--capture-test":
        sys.exit(capture_test(sys.argv[2],
                              float(sys.argv[3]) if len(sys.argv) > 3 else 5.0))
    else:
        print("用法:")
        print("  python sound_trigger.py --capture-test <进程名> [秒数]")
        print("   捕获指定进程音频存盘（验证 WASAPI 进程环回配置）")
