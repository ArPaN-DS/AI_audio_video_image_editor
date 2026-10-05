"""
Model Manager — Dynamic AI Model Lifecycle & Memory Offloader.

Architectural Purpose:
  1. Zero-Idle Memory Footprint: Models are loaded ONLY on-demand when requested.
  2. Single Active Model Mutex: Prevents multiple heavy AI models (STT, Super-Res,
     Object Detection, etc.) from co-existing in RAM/VRAM simultaneously.
  3. Immediate Post-Task Sleep: Automatically unloads models and releases VRAM/RAM
     immediately after a task completes (or after an idle timeout).
  4. Future-Proof Registry: Seamlessly registers future AI models (Speech-to-Text,
     Image Upscaling, Background Removal, Video AI, etc.).
  5. Adaptive Quality Governor: Probes free RAM / CPU / GPU memory and selects the
     highest-fidelity variant of each capability that the machine can safely run,
     falling back to the baseline (lightweight) variant under memory pressure or
     after resource failures, with cooldown-based demotion so a failing heavy
     variant is not retried on every request.
"""

import os
import gc
import re
import sys
import time
import shutil
import logging
import threading
import subprocess
from contextlib import contextmanager

_log = logging.getLogger("model_manager")
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[ModelManager] %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


class ModelLifecycleManager:
    """
    Central Manager enforcing dynamic model loading, mutual exclusion,
    and aggressive RAM/VRAM memory reclamation.
    """

    def __init__(self, idle_timeout_sec: float = 0.0):
        """
        :param idle_timeout_sec: Seconds after which an idle model auto-unloads.
               0.0 = Immediate offload after task completion (Zero-Idle Memory).
        """
        self._lock = threading.RLock()
        self._active_model_id = None
        self._active_model_instance = None
        self._active_model_metadata = {}
        self._unload_hook = None
        self._last_used_time = 0.0
        self._idle_timeout_sec = idle_timeout_sec
        self._timer = None
        self._session_depth = 0
        self._idle_generation = 0

    def _get_memory_status(self):
        """Get telemetry on current RAM and VRAM availability."""
        status = []
        ram_free, ram_total = _probe_ram_gb()
        if ram_total > 0:
            status.append(f"RAM: {ram_free:.2f} GB free / {ram_total:.2f} GB total")
        try:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                status.append(f"VRAM: {free_bytes / (1024**3):.2f} GB free / {total_bytes / (1024**3):.2f} GB total")
        except Exception:
            pass

        return " | ".join(status) if status else "Memory telemetry active"

    def _ensure_headroom(self, min_free_ram_gb):
        """Reclaim memory before a heavy load; refuse to load into exhausted RAM."""
        if not min_free_ram_gb:
            return
        ram_free, ram_total = _probe_ram_gb()
        if ram_total <= 0 or ram_free >= min_free_ram_gb:
            return
        self._flush_system_memory()
        ram_free, ram_total = _probe_ram_gb()
        if ram_total > 0 and ram_free < min_free_ram_gb:
            raise ResourceExhaustedError(
                f"Not enough memory for this capability ({ram_free:.1f} GB free, {min_free_ram_gb:.1f} GB needed)."
            )

    def _flush_system_memory(self):
        """Reclaim RAM and VRAM completely back to OS."""
        gc.collect()
        try:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            import ctypes
            if sys.platform == "win32":
                # Reduce process working set size on Windows to return RAM to OS
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass

        _log.info(f"[Telemetry] Post-flush state -> {self._get_memory_status()}")

    def _cancel_idle_timer(self):
        self._idle_generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


    def unload_active_model(self):
        """Explicitly unload whichever model is currently resident in memory."""
        with self._lock:
            if self._session_depth:
                raise RuntimeError("The active media capability is still in use.")
            self._cancel_idle_timer()

            if self._active_model_id is not None:
                _log.info("Offloading the active media intelligence capability.")

                # Run specific cleanup hook if registered
                if self._unload_hook:
                    try:
                        self._unload_hook(self._active_model_instance)
                    except Exception as error:
                        _log.warning("Capability cleanup failed (%s).", type(error).__name__)

                self._active_model_instance = None
                self._active_model_id = None
                self._active_model_metadata = {}
                self._unload_hook = None

                self._flush_system_memory()
                _log.info("[OK] Idle media capability released; memory reclamation requested.")

    def release_active_instance(self):
        """
        Drop the manager's reference to the resident instance while a session
        is still open (e.g. after an out-of-memory failure) so its memory can be
        reclaimed before a lighter replacement is loaded.
        """
        with self._lock:
            self._active_model_instance = None
            self._active_model_metadata = {}
            self._flush_system_memory()

    def adopt_active_instance(self, instance, metadata=None):
        """Register a replacement instance for the currently active capability."""
        with self._lock:
            if self._active_model_id is None:
                raise RuntimeError("No active media capability to replace.")
            self._active_model_instance = instance
            self._active_model_metadata = metadata or {}
            self._last_used_time = time.time()

    def load_model(self, model_id: str, load_fn, unload_fn=None, min_free_ram_gb: float = 0.0):
        """
        Request a model by ID.
        If another model is currently in memory, it is automatically offloaded first.
        """
        with self._lock:
            if self._session_depth and self._active_model_id != model_id:
                raise RuntimeError("Finish the active media task before switching capabilities.")
            self._cancel_idle_timer()

            # If the exact same model is already loaded and active, reuse it
            if self._active_model_id == model_id and self._active_model_instance is not None:
                self._last_used_time = time.time()
                _log.info("Reusing the active media intelligence capability.")
                return self._active_model_instance, self._active_model_metadata

            # Otherwise, unload previous active model first (Single Active Model Policy)
            if self._active_model_id is not None:
                _log.info("Switching media intelligence capabilities.")
                self.unload_active_model()

            self._ensure_headroom(min_free_ram_gb)
            _log.info("Loading the requested media intelligence capability.")
            start_t = time.time()

            try:
                instance, metadata = load_fn()
            except Exception:
                self._flush_system_memory()
                raise

            self._active_model_id = model_id
            self._active_model_instance = instance
            self._active_model_metadata = metadata or {}
            self._unload_hook = unload_fn
            self._last_used_time = time.time()

            elapsed = time.time() - start_t
            _log.info("[OK] Media intelligence capability ready in %.2fs", elapsed)
            return instance, metadata

    @contextmanager
    def session(self, model_id: str, load_fn, unload_fn=None, auto_offload: bool = True,
                min_free_ram_gb: float = 0.0):
        """
        Hold exclusive capability ownership through execution and cleanup.

        Nested sessions may reuse the same capability; switching capabilities
        is permitted only after the outer session has finished.

        Usage:
            with manager.session("stt_whisper", load_stt, unload_stt) as (model, meta):
                results = model.transcribe(...)
            # Model is automatically offloaded here upon exit!
        """
        with self._lock:
            # Hand the instance out without keeping a frame-local reference, so
            # release_active_instance() can actually free it mid-session.
            loaded = list(self.load_model(model_id, load_fn, unload_fn, min_free_ram_gb))
            self._session_depth += 1
            try:
                yield loaded.pop(0), loaded.pop(0)
            finally:
                self._session_depth -= 1
                if self._session_depth == 0:
                    if auto_offload or self._idle_timeout_sec == 0.0:
                        self.unload_active_model()
                    else:
                        self._schedule_idle_unload()

    def _unload_if_idle(self, generation):
        with self._lock:
            if generation != self._idle_generation or self._session_depth:
                return
            self.unload_active_model()

    def _schedule_idle_unload(self):
        """Schedule automatic unload after idle timeout."""
        with self._lock:
            self._cancel_idle_timer()

            if self._idle_timeout_sec > 0.0:
                self._timer = threading.Timer(
                    self._idle_timeout_sec, self._unload_if_idle,
                    args=(self._idle_generation,))
                self._timer.daemon = True
                self._timer.start()


# Global Singleton Manager instance for the entire application
global_model_manager = ModelLifecycleManager(idle_timeout_sec=0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTIVE QUALITY GOVERNOR
# ═══════════════════════════════════════════════════════════════════════════

TIER_LITE = "lite"
TIER_BALANCED = "balanced"
TIER_MAX = "max"
TIER_ORDER = (TIER_LITE, TIER_BALANCED, TIER_MAX)
TIER_LABELS = {
    TIER_LITE: "Efficiency Mode",
    TIER_BALANCED: "Balanced Quality",
    TIER_MAX: "Maximum Fidelity",
}

_RESOURCE_ERROR_PATTERN = re.compile(
    r"out of memory|outofmemory|bad_alloc|cannot allocate|failed to allocate|"
    r"allocation failed|insufficient memory|not enough memory|memory error|"
    r"cuda error|cublas|cudnn|resource exhausted|resource_exhausted",
    re.IGNORECASE,
)


class ResourceExhaustedError(RuntimeError):
    """Raised when a capability variant cannot run within current resources."""


def is_resource_error(error) -> bool:
    if isinstance(error, (MemoryError, ResourceExhaustedError)):
        return True
    return bool(_RESOURCE_ERROR_PATTERN.search(f"{type(error).__name__}: {error}"))


def _probe_ram_gb():
    """Return (available_gb, total_gb); never raises."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3), mem.total / (1024 ** 3)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatus()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024 ** 3), stat.ullTotalPhys / (1024 ** 3)
        elif os.path.exists("/proc/meminfo"):
            values = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    values[key] = int(rest.strip().split()[0]) / (1024 ** 2)
            return values.get("MemAvailable", values.get("MemFree", 0.0)), values.get("MemTotal", 0.0)
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = pages * page_size / (1024 ** 3)
            available = os.sysconf("SC_AVPHYS_PAGES") * page_size / (1024 ** 3)
            return available, total
    except Exception:
        pass
    return 0.0, 0.0


def _probe_gpu_memory_gb():
    """Return (free_gb, total_gb) for the primary accelerator, or (0, 0)."""
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            return free_bytes / (1024 ** 3), total_bytes / (1024 ** 3)
    except Exception:
        pass
    try:
        smi = shutil.which("nvidia-smi")
        if smi:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            output = subprocess.run(
                [smi, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, creationflags=creationflags,
            ).stdout.strip().splitlines()
            if output:
                free_mb, total_mb = (float(part) for part in output[0].split(",")[:2])
                return free_mb / 1024, total_mb / 1024
    except Exception:
        pass
    return 0.0, 0.0


class ResourceSnapshot:
    __slots__ = ("ram_free_gb", "ram_total_gb", "cpu_cores", "gpu_free_gb", "gpu_total_gb", "taken_at")

    def __init__(self, ram_free_gb=0.0, ram_total_gb=0.0, cpu_cores=1,
                 gpu_free_gb=0.0, gpu_total_gb=0.0, taken_at=None):
        self.ram_free_gb = max(0.0, float(ram_free_gb or 0.0))
        self.ram_total_gb = max(0.0, float(ram_total_gb or 0.0))
        self.cpu_cores = max(1, int(cpu_cores or 1))
        self.gpu_free_gb = max(0.0, float(gpu_free_gb or 0.0))
        self.gpu_total_gb = max(0.0, float(gpu_total_gb or 0.0))
        self.taken_at = time.monotonic() if taken_at is None else taken_at

    def as_public_dict(self):
        return {
            "ram_free_gb": round(self.ram_free_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "cpu_cores": self.cpu_cores,
            "gpu_acceleration": self.gpu_total_gb > 0,
            "gpu_free_gb": round(self.gpu_free_gb, 2),
        }


class CapabilityVariant:
    """
    One selectable implementation of a capability, e.g. a heavier cutout network.

    Requirements are expressed in free resources at selection time. The last
    variant of a ladder is the baseline and is always eligible.
    """

    def __init__(self, variant_id, tier=TIER_LITE, min_ram_gb=0.0, min_cpu_cores=1,
                 min_gpu_gb=0.0, is_available=None, public_label=None, options=None):
        self.variant_id = variant_id
        self.tier = tier if tier in TIER_ORDER else TIER_LITE
        self.min_ram_gb = float(min_ram_gb)
        self.min_cpu_cores = int(min_cpu_cores)
        self.min_gpu_gb = float(min_gpu_gb)
        self.is_available = is_available
        self.public_label = public_label or variant_id
        self.options = dict(options or {})

    def available(self) -> bool:
        if self.is_available is None:
            return True
        try:
            return bool(self.is_available())
        except Exception:
            return False


class AdaptiveQualityGovernor:
    """
    Chooses capability variants from live hardware telemetry.

    Robustness guarantees:
      * Probes never raise; unknown hardware is treated as the lite tier.
      * Snapshots are cached briefly so bursts of requests do not re-probe.
      * Tier changes use hysteresis so borderline machines do not flap.
      * MEDIA_QUALITY_TIER=lite|balanced|max pins a ceiling (auto by default).
      * Variants that fail for resource reasons are demoted with an exponential
        cooldown; the baseline variant is always attempted last.
    """

    SNAPSHOT_TTL_SEC = 5.0
    HYSTERESIS_GB = 0.75
    BASE_COOLDOWN_SEC = 600.0
    MAX_COOLDOWN_SEC = 7200.0
    TIER_RULES = {
        # tier: (min free RAM GB, min CPU cores, min free GPU GB as alternative)
        TIER_MAX: (8.0, 8, 4.0),
        TIER_BALANCED: (3.0, 4, 2.0),
    }

    def __init__(self, probe=None, clock=time.monotonic, env=None):
        self._lock = threading.RLock()
        self._probe = probe or self._default_probe
        self._clock = clock
        self._env = os.environ if env is None else env
        self._snapshot = None
        self._tier = None
        self._strikes = {}
        self._banned_until = {}
        self._last_choice = {}

    @staticmethod
    def _default_probe():
        ram_free, ram_total = _probe_ram_gb()
        gpu_free, gpu_total = _probe_gpu_memory_gb()
        return ResourceSnapshot(ram_free, ram_total, os.cpu_count() or 1, gpu_free, gpu_total)

    def snapshot(self, force=False) -> ResourceSnapshot:
        with self._lock:
            now = self._clock()
            if (not force and self._snapshot is not None
                    and now - self._snapshot.taken_at < self.SNAPSHOT_TTL_SEC):
                return self._snapshot
        try:
            fresh = self._probe()
            if not isinstance(fresh, ResourceSnapshot):
                raise TypeError("invalid snapshot")
        except Exception as error:
            _log.warning("Resource probe failed (%s); assuming efficiency mode.", type(error).__name__)
            fresh = ResourceSnapshot()
        fresh.taken_at = self._clock()
        with self._lock:
            self._snapshot = fresh
        return fresh

    def tier_override(self):
        value = str(self._env.get("MEDIA_QUALITY_TIER", "auto") or "auto").strip().lower()
        aliases = {"low": TIER_LITE, "standard": TIER_BALANCED, "medium": TIER_BALANCED, "high": TIER_MAX}
        value = aliases.get(value, value)
        return value if value in TIER_ORDER else None

    def _qualifies(self, snap, tier, margin=0.0):
        min_ram, min_cores, min_gpu = self.TIER_RULES[tier]
        cpu_path = snap.ram_free_gb >= min_ram + margin and snap.cpu_cores >= min_cores
        gpu_path = snap.gpu_free_gb >= min_gpu + margin and snap.ram_free_gb >= min(min_ram, 2.0) + margin
        return cpu_path or gpu_path

    def _classify(self, snap, previous):
        detected = TIER_LITE
        for tier in (TIER_MAX, TIER_BALANCED):
            if self._qualifies(snap, tier):
                detected = tier
                break
        if previous is None or previous == detected:
            return detected
        # Upgrades require clearing the threshold by a margin; downgrades only
        # happen when the current tier's floor is no longer met at all.
        if TIER_ORDER.index(detected) > TIER_ORDER.index(previous):
            return detected if self._qualifies(snap, detected, self.HYSTERESIS_GB) else previous
        if previous != TIER_LITE and self._qualifies(snap, previous, -self.HYSTERESIS_GB):
            return previous
        return detected

    def current_tier(self, force=False) -> str:
        snap = self.snapshot(force=force)
        with self._lock:
            detected = self._classify(snap, self._tier)
            if detected != self._tier:
                if self._tier is not None:
                    _log.info("Adaptive quality changed to %s.", TIER_LABELS[detected])
                self._tier = detected
            override = self.tier_override()
            if override is not None:
                return override
            return detected

    def _is_banned(self, key):
        until = self._banned_until.get(key)
        if until is None:
            return False
        if self._clock() >= until:
            self._banned_until.pop(key, None)
            return False
        return True

    def _fits(self, variant, snap):
        if snap.cpu_cores < variant.min_cpu_cores:
            return False
        if variant.min_gpu_gb > 0:
            return snap.gpu_free_gb >= variant.min_gpu_gb and snap.ram_free_gb >= min(variant.min_ram_gb, 2.0)
        return snap.ram_free_gb >= variant.min_ram_gb

    def plan(self, capability, ladder):
        """
        Return the ordered variants to attempt, best first, always ending with
        the baseline (last ladder entry).
        """
        if not ladder:
            raise ValueError("A capability ladder needs at least one variant.")
        baseline = ladder[-1]
        tier = self.current_tier()
        snap = self.snapshot()
        ceiling = TIER_ORDER.index(tier)
        ordered = []
        with self._lock:
            for variant in ladder[:-1]:
                if TIER_ORDER.index(variant.tier) > ceiling:
                    continue
                if self._is_banned((capability, variant.variant_id)):
                    continue
                if not variant.available() or not self._fits(variant, snap):
                    continue
                ordered.append(variant)
        ordered.append(baseline)
        return ordered

    def is_paused(self, capability, variant_id):
        with self._lock:
            return self._is_banned((capability, variant_id))

    def select(self, capability, ladder):
        return self.plan(capability, ladder)[0]

    def report_failure(self, capability, variant_id, error):
        key = (capability, variant_id)
        with self._lock:
            strikes = self._strikes.get(key, 0) + 1
            self._strikes[key] = strikes
            resource_failure = is_resource_error(error)
            if resource_failure or strikes >= 2:
                cooldown = min(self.BASE_COOLDOWN_SEC * (2 ** (strikes - 1)), self.MAX_COOLDOWN_SEC)
                self._banned_until[key] = self._clock() + cooldown
                _log.warning(
                    "High-fidelity %s variant paused for %d min after %s.",
                    capability, int(cooldown // 60),
                    "a memory limit" if resource_failure else "repeated failures",
                )
            # Force a fresh probe so the next selection sees reclaimed memory.
            self._snapshot = None

    def report_success(self, capability, variant_id):
        key = (capability, variant_id)
        with self._lock:
            self._strikes.pop(key, None)
            self._banned_until.pop(key, None)
            self._last_choice[capability] = variant_id

    def run(self, capability, ladder, operation):
        """
        Execute `operation(variant)` with the best eligible variant, falling back
        down the ladder on failure. Returns (result, variant). The baseline's
        error propagates if every variant fails.
        """
        attempts = self.plan(capability, ladder)
        last_error = None
        for index, variant in enumerate(attempts):
            is_last = index == len(attempts) - 1
            try:
                result = operation(variant)
            except Exception as error:
                last_error = error
                if variant is not ladder[-1]:
                    self.report_failure(capability, variant.variant_id, error)
                if is_last:
                    raise
                _log.warning(
                    "%s variant fell back to a lighter configuration (%s).",
                    capability, type(error).__name__,
                )
                global_model_manager._flush_system_memory()
                continue
            self.report_success(capability, variant.variant_id)
            return result, variant
        raise last_error or RuntimeError("No capability variant could run.")

    def public_status(self):
        tier = self.current_tier()
        snap = self.snapshot()
        with self._lock:
            paused = sorted({capability for (capability, _), until in self._banned_until.items()
                             if self._clock() < until})
            return {
                "quality_tier": tier,
                "quality_mode": TIER_LABELS[tier],
                "pinned": self.tier_override() is not None,
                "hardware": snap.as_public_dict(),
                "paused_capabilities": paused,
            }

    def reset(self):
        with self._lock:
            self._snapshot = None
            self._tier = None
            self._strikes.clear()
            self._banned_until.clear()
            self._last_choice.clear()


global_quality_governor = AdaptiveQualityGovernor()
