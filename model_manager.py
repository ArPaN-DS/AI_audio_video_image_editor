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
"""

import os
import gc
import sys
import time
import logging
import threading
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

    def _get_memory_status(self):
        """Get telemetry on current RAM and VRAM availability."""
        status = []
        try:
            import psutil
            mem = psutil.virtual_memory()
            status.append(f"RAM: {mem.available / (1024**3):.2f} GB free / {mem.total / (1024**3):.2f} GB total")
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                status.append(f"VRAM: {free_bytes / (1024**3):.2f} GB free / {total_bytes / (1024**3):.2f} GB total")
        except Exception:
            pass

        return " | ".join(status) if status else "Memory telemetry active"

    def _flush_system_memory(self):
        """Reclaim RAM and VRAM completely back to OS."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        try:
            import ctypes
            if sys.platform == "win32":
                # Reduce process working set size on Windows to return RAM to OS
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass

        gc.collect()
        _log.info(f"[Telemetry] Post-flush state → {self._get_memory_status()}")


    def unload_active_model(self):
        """Explicitly unload whichever model is currently resident in memory."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

            if self._active_model_id is not None:
                _log.info(f"Offloading active model '{self._active_model_id}' from RAM/VRAM → Sleep")

                # Run specific cleanup hook if registered
                if self._unload_hook:
                    try:
                        self._unload_hook(self._active_model_instance)
                    except Exception as e:
                        _log.warning(f"Error during model unload hook: {e}")

                self._active_model_instance = None
                self._active_model_id = None
                self._active_model_metadata = {}
                self._unload_hook = None

                self._flush_system_memory()
                _log.info("✓ System memory reclaimed (RAM/VRAM back to base baseline)")

    def load_model(self, model_id: str, load_fn, unload_fn=None):
        """
        Request a model by ID.
        If another model is currently in memory, it is automatically offloaded first.
        """
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

            # If the exact same model is already loaded and active, reuse it
            if self._active_model_id == model_id and self._active_model_instance is not None:
                self._last_used_time = time.time()
                _log.info(f"Reusing currently loaded model '{model_id}'")
                return self._active_model_instance, self._active_model_metadata

            # Otherwise, unload previous active model first (Single Active Model Policy)
            if self._active_model_id is not None:
                _log.info(f"Switching AI task: Offloading '{self._active_model_id}' before loading '{model_id}'")
                self.unload_active_model()

            _log.info(f"Loading AI model '{model_id}' into RAM/VRAM...")
            start_t = time.time()

            instance, metadata = load_fn()

            self._active_model_id = model_id
            self._active_model_instance = instance
            self._active_model_metadata = metadata or {}
            self._unload_hook = unload_fn
            self._last_used_time = time.time()

            elapsed = time.time() - start_t
            _log.info(f"✓ Model '{model_id}' loaded in {elapsed:.2f}s")
            return instance, metadata

    @contextmanager
    def session(self, model_id: str, load_fn, unload_fn=None, auto_offload: bool = True):
        """
        Context manager for zero-idle memory execution.

        Usage:
            with manager.session("stt_whisper", load_stt, unload_stt) as (model, meta):
                results = model.transcribe(...)
            # Model is automatically offloaded here upon exit!
        """
        with self._lock:
            model_instance, metadata = self.load_model(model_id, load_fn, unload_fn)
        
        try:
            yield model_instance, metadata
        finally:
            if auto_offload or self._idle_timeout_sec == 0.0:
                self.unload_active_model()
            else:
                self._schedule_idle_unload()

    def _schedule_idle_unload(self):
        """Schedule automatic unload after idle timeout."""
        with self._lock:
            if self._timer:
                self._timer.cancel()

            if self._idle_timeout_sec > 0.0:
                self._timer = threading.Timer(self._idle_timeout_sec, self.unload_active_model)
                self._timer.daemon = True
                self._timer.start()


# Global Singleton Manager instance for the entire application
global_model_manager = ModelLifecycleManager(idle_timeout_sec=0.0)
