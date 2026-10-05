"""
Process resource probes used by the benchmark and the performance-budget tests.

Dependency-free: uses psutil when it is installed, otherwise the Win32 API via
ctypes (GetProcessMemoryInfo, Toolhelp32 thread snapshots, handle counts, I/O
counters) or /proc on Linux. Every probe degrades to ``None`` instead of raising.

Metrics
  * ``rss``      working set (what Task Manager shows as "Memory (active)").
  * ``private``  committed private bytes — the honest "memory this process
                 owns" number; trimming the working set does NOT lower it.
  * ``threads``  native OS threads (includes BLAS/OpenMP/runtime pools).
  * ``handles``  OS handles (files, events, sections …) — a leak detector.
"""

import ctypes
import os
import sys
import threading
import time

_IS_WINDOWS = sys.platform == "win32"

try:  # optional
    import psutil as _psutil  # type: ignore
except Exception:  # pragma: no cover - depends on the environment
    _psutil = None


if _IS_WINDOWS:
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _IOC(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
    _k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
    _k32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IOC)]
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]

    _PROCESS_QUERY = 0x0400 | 0x0010  # QUERY_INFORMATION | VM_READ

    def _handle(pid):
        if pid is None or pid == os.getpid():
            return _k32.GetCurrentProcess(), False
        h = _k32.OpenProcess(_PROCESS_QUERY, False, int(pid))
        return h, bool(h)


def memory(pid=None):
    """{'rss': bytes, 'private': bytes, 'peak_rss': bytes} or None."""
    if _IS_WINDOWS:
        h, close = _handle(pid)
        try:
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            if _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return {"rss": int(pmc.WorkingSetSize), "private": int(pmc.PrivateUsage),
                        "peak_rss": int(pmc.PeakWorkingSetSize)}
        finally:
            if close:
                _k32.CloseHandle(h)
        return None
    if _psutil is not None:
        try:
            info = _psutil.Process(pid or os.getpid()).memory_full_info()
            return {"rss": int(info.rss), "private": int(getattr(info, "uss", info.rss)),
                    "peak_rss": int(info.rss)}
        except Exception:
            return None
    try:
        values = {}
        with open(f"/proc/{pid or 'self'}/status", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM", "RssAnon"):
                    values[key] = int(rest.split()[0]) * 1024
        rss = values.get("VmRSS", 0)
        return {"rss": rss, "private": values.get("RssAnon", rss), "peak_rss": values.get("VmHWM", rss)}
    except Exception:
        return None


def thread_count(pid=None):
    """Native OS thread count of the process (not just Python threads)."""
    target = int(pid or os.getpid())
    if _IS_WINDOWS:
        snap = _k32.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
        if not snap or snap == wintypes.HANDLE(-1).value:
            return None
        try:
            entry = _THREADENTRY32()
            entry.dwSize = ctypes.sizeof(entry)
            count = 0
            ok = _k32.Thread32First(snap, ctypes.byref(entry))
            while ok:
                if entry.th32OwnerProcessID == target:
                    count += 1
                ok = _k32.Thread32Next(snap, ctypes.byref(entry))
            return count
        finally:
            _k32.CloseHandle(snap)
    if _psutil is not None:
        try:
            return _psutil.Process(target).num_threads()
        except Exception:
            return None
    try:
        return len(os.listdir(f"/proc/{target}/task"))
    except Exception:
        return None


def handle_count(pid=None):
    if _IS_WINDOWS:
        h, close = _handle(pid)
        try:
            count = wintypes.DWORD()
            if _k32.GetProcessHandleCount(h, ctypes.byref(count)):
                return int(count.value)
        finally:
            if close:
                _k32.CloseHandle(h)
        return None
    try:
        return len(os.listdir(f"/proc/{pid or 'self'}/fd"))
    except Exception:
        return None


def io_counters(pid=None):
    """{'read_bytes', 'write_bytes'} transferred by the process itself."""
    if _IS_WINDOWS:
        h, close = _handle(pid)
        try:
            ioc = _IOC()
            if _k32.GetProcessIoCounters(h, ctypes.byref(ioc)):
                return {"read_bytes": int(ioc.ReadTransferCount), "write_bytes": int(ioc.WriteTransferCount)}
        finally:
            if close:
                _k32.CloseHandle(h)
        return None
    try:
        values = {}
        with open(f"/proc/{pid or 'self'}/io", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                values[key] = int(rest)
        return {"read_bytes": values.get("read_bytes", 0), "write_bytes": values.get("write_bytes", 0)}
    except Exception:
        return None


# ── GPU (NVML via ctypes; no subprocess per sample) ─────────────────────────

class _Nvml:
    def __init__(self):
        self.lib = None
        self.device = None
        names = ["nvml.dll", r"C:\Windows\System32\nvml.dll",
                 r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"] if _IS_WINDOWS else ["libnvidia-ml.so.1"]
        for name in names:
            try:
                lib = ctypes.CDLL(name)
                if lib.nvmlInit_v2() != 0:
                    continue
                device = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device)) != 0:
                    continue
                self.lib, self.device = lib, device
                return
            except Exception:
                continue

    def used_bytes(self):
        if self.lib is None:
            return None

        class _Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        mem = _Mem()
        if self.lib.nvmlDeviceGetMemoryInfo(self.device, ctypes.byref(mem)) != 0:
            return None
        return int(mem.used)


_nvml = None
_nvml_lock = threading.Lock()


def gpu_used_bytes():
    """Device-wide used VRAM in bytes (None without an NVIDIA driver)."""
    global _nvml
    with _nvml_lock:
        if _nvml is None:
            _nvml = _Nvml()
    try:
        return _nvml.used_bytes()
    except Exception:
        return None


def snapshot(pid=None):
    mem = memory(pid) or {}
    return {
        "rss": mem.get("rss"),
        "private": mem.get("private"),
        "threads": thread_count(pid),
        "handles": handle_count(pid),
        "python_threads": threading.active_count() if pid in (None, os.getpid()) else None,
    }


class PeakSampler:
    """Background sampler of RSS / private bytes / VRAM peaks for a code block."""

    def __init__(self, interval=0.02, pid=None, gpu=True):
        self.interval = interval
        self.pid = pid
        self.gpu = gpu
        self.peak_rss = 0
        self.peak_private = 0
        self.peak_gpu = None
        self.gpu_baseline = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        mem = memory(self.pid)
        if mem:
            self.peak_rss = max(self.peak_rss, mem["rss"])
            self.peak_private = max(self.peak_private, mem["private"])
        if self.gpu:
            used = gpu_used_bytes()
            if used is not None:
                self.peak_gpu = used if self.peak_gpu is None else max(self.peak_gpu, used)
        self.samples += 1

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        if self.gpu:
            self.gpu_baseline = gpu_used_bytes()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="perf-peak-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()
        return False

    def result(self):
        out = {"peak_rss": self.peak_rss, "peak_private": self.peak_private, "samples": self.samples}
        if self.peak_gpu is not None and self.gpu_baseline is not None:
            out["peak_vram_delta"] = max(0, self.peak_gpu - self.gpu_baseline)
        return out


def mb(value):
    return None if value is None else round(value / (1024 * 1024), 1)


def wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
