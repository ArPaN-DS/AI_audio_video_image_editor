"""
Runtime tuning — thread pools, lazy heavy imports and inference-session options.

Import this module FIRST (before numpy/scipy/onnxruntime are loaded): it sets
thread-pool environment defaults that native libraries only read at load time.

Why it matters (measured on a 10-core / 16-thread laptop):
  * numpy and scipy each bundle OpenBLAS; with the default of one thread per
    logical CPU each library commits ~32 MB of private buffers per thread
    (~1 GB committed for the two) and parks 30 idle native threads. This app's
    numeric work is FFT / filtering / element-wise — not BLAS-bound — so a
    small BLAS pool costs nothing in speed and returns ~800 MB of commit.
  * Compute-heavy engines (inference sessions, transcription) are sized to
    physical cores minus one, so the web server and the OS stay responsive and
    hyper-threads do not oversubscribe the FPU.

Every value is only a *default*: an environment variable set by the user wins.
"""

import importlib
import os
import sys
import threading
import types

_DEFAULT_BLAS_THREADS = "2"


def _apply_env_defaults():
    blas = os.environ.get("MEDIA_BLAS_THREADS", _DEFAULT_BLAS_THREADS)
    # OpenBLAS reads OPENBLAS_NUM_THREADS before OMP_NUM_THREADS. OMP_NUM_THREADS is
    # deliberately left alone: neural engines size their own pools explicitly.
    for key in ("OPENBLAS_NUM_THREADS",):
        os.environ.setdefault(key, blas)


_apply_env_defaults()


# ── CPU topology ────────────────────────────────────────────────────────────

_physical_cores = None


def physical_cores():
    """Physical core count (never raises; falls back to logical count)."""
    global _physical_cores
    if _physical_cores is not None:
        return _physical_cores
    logical = os.cpu_count() or 1
    count = None
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32")
            length = wintypes.DWORD(0)
            # RelationProcessorCore == 0
            kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
            buffer = ctypes.create_string_buffer(length.value)
            if kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
                offset, cores = 0, 0
                while offset < length.value:
                    size = int.from_bytes(buffer.raw[offset + 4:offset + 8], "little")
                    if size <= 0:
                        break
                    cores += 1
                    offset += size
                count = cores or None
        elif os.path.exists("/proc/cpuinfo"):
            pairs = set()
            physical = core = None
            with open("/proc/cpuinfo", encoding="ascii", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("physical id"):
                        physical = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        pairs.add((physical, core))
            count = len(pairs) or None
    except Exception:
        count = None
    _physical_cores = max(1, min(logical, count or logical))
    return _physical_cores


def compute_threads(reserve=1):
    """Threads for one heavy compute job: physical cores minus a reserve for the server."""
    override = os.environ.get("MEDIA_COMPUTE_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return max(1, physical_cores() - max(0, int(reserve)))


def blas_threads():
    try:
        return max(1, int(os.environ.get("OPENBLAS_NUM_THREADS", _DEFAULT_BLAS_THREADS)))
    except ValueError:
        return int(_DEFAULT_BLAS_THREADS)


# ── Lazy modules ───────────────────────────────────────────────────────────

class LazyModule(types.ModuleType):
    """
    Module proxy that imports its target on first attribute access.

    ``librosa = lazy_module("librosa")`` keeps module-level call sites unchanged
    (``librosa.load(...)``) while the import cost is only paid by the first
    request that actually needs the library. ``unittest.mock.patch`` works on the
    proxy (patched attributes shadow the real module until restored).
    """

    def __init__(self, name, loader=None):
        super().__init__(name)
        self.__dict__["_lazy_target"] = name
        self.__dict__["_lazy_loader"] = loader

    def _lazy_load(self):
        module = sys.modules.get(self.__dict__["_lazy_target"])
        if module is None or module is self:
            loader = self.__dict__["_lazy_loader"]
            module = loader() if loader else importlib.import_module(self.__dict__["_lazy_target"])
        return module

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return getattr(self._lazy_load(), attr)

    def __dir__(self):
        return dir(self._lazy_load())

    def __repr__(self):
        return f"<lazy module {self.__dict__['_lazy_target']!r}>"


def lazy_module(name, loader=None):
    return LazyModule(name, loader)


_noisereduce_lock = threading.Lock()


def import_noisereduce():
    """
    Import the spectral-gating denoiser WITHOUT dragging in its optional deep
    learning backend (only used for its GPU variant, which this app never
    requests). Saves ~1.5 s and ~150 MB of resident memory per process.
    """
    module = sys.modules.get("noisereduce")
    if module is not None:
        return module
    with _noisereduce_lock:
        module = sys.modules.get("noisereduce")
        if module is not None:
            return module
        # Its numeric dependencies probe sys.modules for the backend at import
        # time, so load them first, unblocked.
        importlib.import_module("scipy.signal")
        blocked = "torch" not in sys.modules
        try:
            import _imp
            _imp.acquire_lock()  # no other thread may observe the temporary block
        except Exception:
            _imp = None
        try:
            if blocked:
                sys.modules["torch"] = None  # makes `import torch` raise ImportError
            return importlib.import_module("noisereduce")
        finally:
            if blocked and "torch" in sys.modules and sys.modules["torch"] is None:
                del sys.modules["torch"]
            if _imp is not None:
                _imp.release_lock()


# ── Inference session options ──────────────────────────────────────────────

def onnx_session_options(low_memory=True):
    """
    Session options for CPU inference sessions.

    * intra-op threads = physical cores - 1 (no hyper-thread oversubscription),
      one inter-op thread and sequential execution (graphs here are chains);
    * the CPU memory arena is disabled in low-memory mode: the arena grows to
      the high-water mark and never returns it, which keeps hundreds of MB
      committed after the session is released;
    * full graph optimisation.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = compute_threads()
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if low_memory:
        options.enable_cpu_mem_arena = False
    return options


def ffmpeg_thread_args(active_jobs=1):
    """`-threads` for one encode so concurrent encodes share the cores instead of
    each spawning a thread per logical CPU."""
    override = os.environ.get("MEDIA_FFMPEG_THREADS", "").strip()
    if override.isdigit():
        return [] if override == "0" else ["-threads", override]
    jobs = max(1, int(active_jobs or 1))
    if jobs <= 1:
        return []  # a lone encode keeps the encoder's own (fastest) thread layout
    return ["-threads", str(max(2, (os.cpu_count() or 2) // jobs))]
