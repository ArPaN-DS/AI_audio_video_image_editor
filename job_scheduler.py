"""
Job Scheduler — admission control for heavy media work.

Why: the dev server is threaded, so N concurrent requests used to start N
heavy jobs at once (each sizing itself for the WHOLE machine) and jointly run
it out of memory. Every heavy entry point now passes through ``job()``:

  * Resource classes with slot limits
        model   1   one resident inference model at a time (pairs with the
                    Single-Active-Model lifecycle manager; FIFO instead of a
                    lock free-for-all)
        gpu     1   exclusive accelerator slot
        cpu     N   CPU-heavy DSP / imaging (sized from physical cores and RAM)
        encode  M   media-engine encodes (each gets cores // M threads)
  * Memory reservations: a job declares the RAM/VRAM it expects to need. It is
    admitted only if (free memory - reservations of running jobs) covers it, or
    if nothing else is running (a lone job is never starved; it adapts or
    falls back). The quality governor subtracts OTHER jobs' reservations from
    its hardware snapshot, so two concurrent jobs cannot both pick "max".
  * FIFO fairness per resource class, a queue-wait timeout, cooperative
    cancellation (``checkpoint()``) plus cancel hooks that kill child processes,
    a run-time limit watchdog, and queue positions for a progress UI.

Lock ordering (deadlock freedom — see test_performance_budgets.py):

    scheduler admission (outermost, at most once per thread)
      -> ModelLifecycleManager._lock
        -> AdaptiveQualityGovernor._lock (leaf; never held across calls)

  * Admission is re-entrant per thread: a nested ``job()`` inside an admitted
    job does not take new slots (no hold-and-wait between slots).
  * A thread that already holds the model lifecycle lock is never made to wait
    for a slot (its ``job()`` runs pass-through), so the lock is never waited
    on while a slot holder waits for that thread.
  * All classes a job needs are granted atomically (all-or-nothing), under one
    condition variable that is never held while user code runs.
"""

import itertools
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager

_log = logging.getLogger("model_manager")

CLASSES = ("model", "gpu", "cpu", "encode")
BUSY_MESSAGE = "The studio is busy with other jobs. Please try again in a moment."
CANCELLED_MESSAGE = "The job was cancelled."


class JobQueueTimeout(RuntimeError):
    """The job waited longer than its queue timeout for resources."""


class JobCancelled(RuntimeError):
    """The job was cancelled (by request or by its run-time limit)."""


def _env_int(env, key, default):
    try:
        value = int(str(env.get(key, "")).strip())
        return value if value > 0 else default
    except ValueError:
        return default


def _env_float(env, key, default):
    try:
        value = float(str(env.get(key, "")).strip())
        return value if value > 0 else default
    except ValueError:
        return default


def default_slots(env=None, cores=None, ram_total_gb=None):
    env = os.environ if env is None else env
    if cores is None:
        try:
            import runtime_tuning
            cores = runtime_tuning.physical_cores()
        except Exception:
            cores = os.cpu_count() or 2
    if ram_total_gb is None:
        try:
            from model_manager import _probe_ram_gb
            ram_total_gb = _probe_ram_gb()[1]
        except Exception:
            ram_total_gb = 8.0
    cpu = max(1, min(cores // 3, int(max(ram_total_gb, 4.0) // 4), 4))
    encode = max(1, min(2, cores // 4))
    return {
        "model": 1,
        "gpu": 1,
        "cpu": _env_int(env, "MEDIA_CPU_JOB_SLOTS", cpu),
        "encode": _env_int(env, "MEDIA_ENCODE_SLOTS", encode),
    }


class Job:
    """A unit of admitted (or waiting) work."""

    __slots__ = ("id", "name", "classes", "ram_gb", "vram_gb", "state", "enqueued_at", "started_at",
                 "thread_id", "_cancel", "_hooks", "_hooks_lock", "reason", "_scheduler", "__weakref__")

    def __init__(self, scheduler, job_id, name, classes, ram_gb, vram_gb, now):
        self._scheduler = scheduler
        self.id = job_id
        self.name = name
        self.classes = tuple(classes)
        self.ram_gb = max(0.0, float(ram_gb or 0.0))
        self.vram_gb = max(0.0, float(vram_gb or 0.0))
        self.state = "queued"
        self.enqueued_at = now
        self.started_at = None
        self.thread_id = threading.get_ident()
        self._cancel = threading.Event()
        self._hooks = []
        self._hooks_lock = threading.Lock()
        self.reason = None

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def check_cancelled(self):
        if self._cancel.is_set():
            raise JobCancelled(self.reason or CANCELLED_MESSAGE)

    def add_cancel_hook(self, hook):
        """Register ``hook()`` (e.g. kill a child process) to run on cancellation.
        Returns a callable that unregisters it."""
        with self._hooks_lock:
            if self._cancel.is_set():
                run_now = True
            else:
                self._hooks.append(hook)
                run_now = False
        if run_now:
            _safe_call(hook)
            return lambda: None

        def remove():
            with self._hooks_lock:
                if hook in self._hooks:
                    self._hooks.remove(hook)
        return remove

    def cancel(self, reason=None):
        self.reason = reason or CANCELLED_MESSAGE
        with self._hooks_lock:
            self._cancel.set()
            hooks, self._hooks = list(self._hooks), []
        for hook in hooks:
            _safe_call(hook)
        self._scheduler._wake()

    def public(self, position=None, now=None):
        now = time.monotonic() if now is None else now
        info = {"id": self.id, "task": self.name, "state": self.state}
        if self.state == "queued":
            info["queue_position"] = position
            info["waited_sec"] = round(now - self.enqueued_at, 1)
        elif self.started_at is not None:
            info["running_sec"] = round(now - self.started_at, 1)
        return info


def _safe_call(hook):
    try:
        hook()
    except Exception as error:  # a broken hook must not break cancellation
        _log.warning("A cancellation hook failed (%s).", type(error).__name__)


class _PassThroughJob:
    """Returned for nested/bypassed admissions: shares the outer job's controls."""

    def __init__(self, outer=None):
        self._outer = outer

    id = None
    name = None
    state = "running"

    @property
    def cancelled(self):
        return bool(self._outer and self._outer.cancelled)

    def check_cancelled(self):
        if self._outer is not None:
            self._outer.check_cancelled()

    def add_cancel_hook(self, hook):
        return self._outer.add_cancel_hook(hook) if self._outer is not None else (lambda: None)


class JobScheduler:

    def __init__(self, slots=None, ram_probe=None, clock=time.monotonic, env=None, model_lock_owned=None):
        self._env = os.environ if env is None else env
        self._slots = dict(slots or default_slots(self._env))
        for name in CLASSES:
            self._slots.setdefault(name, 1)
        self._ram_probe = ram_probe or _default_ram_probe
        self._clock = clock
        self._model_lock_owned = model_lock_owned or _default_model_lock_owned
        self._cond = threading.Condition(threading.Lock())
        self._queue = deque()          # waiting jobs, FIFO
        self._running = {}             # id -> Job
        self._in_use = {name: 0 for name in self._slots}
        self._extra = {}               # anonymous reservations: token -> (ram, vram, thread)
        self._ids = itertools.count(1)
        self._local = threading.local()
        self._stats = {"admitted": 0, "timeouts": 0, "cancelled": 0, "waited_sec": 0.0}

    # ── configuration ─────────────────────────────────────────────────
    @property
    def slots(self):
        return dict(self._slots)

    def queue_timeout(self):
        return _env_float(self._env, "MEDIA_JOB_QUEUE_TIMEOUT_SEC", 600.0)

    # ── admission ─────────────────────────────────────────────────────
    def current_job(self):
        stack = getattr(self._local, "stack", None)
        return stack[-1] if stack else None

    @contextmanager
    def job(self, name, classes=("cpu",), ram_gb=0.0, vram_gb=0.0, timeout=None, max_runtime=None):
        """
        Admit a heavy job. Yields a ``Job`` with ``check_cancelled()`` and
        ``add_cancel_hook()``. Raises ``JobQueueTimeout`` if resources do not
        free up within ``timeout`` seconds and ``JobCancelled`` if cancelled
        while queued.
        """
        outer = self.current_job()
        if outer is not None or self._model_lock_owned():
            # Re-entrant (nested call inside an admitted job) or the caller
            # already holds the model lifecycle lock: never wait for slots.
            self._mark_work()
            yield _PassThroughJob(outer)
            return
        classes = tuple(c for c in (classes or ()) if c in self._slots) or ("cpu",)
        job = self._enqueue(name, classes, ram_gb, vram_gb)
        timer = None
        try:
            self._wait_for_admission(job, self.queue_timeout() if timeout is None else timeout)
            if max_runtime:
                timer = threading.Timer(max_runtime, job.cancel,
                                        kwargs={"reason": "The job took longer than allowed and was stopped."})
                timer.daemon = True
                timer.start()
            stack = getattr(self._local, "stack", None)
            if stack is None:
                stack = self._local.stack = []
            stack.append(job)
            try:
                yield job
            finally:
                stack.pop()
        finally:
            if timer is not None:
                timer.cancel()
            self._release(job)

    def _enqueue(self, name, classes, ram_gb, vram_gb):
        with self._cond:
            job = Job(self, next(self._ids), str(name), classes, ram_gb, vram_gb, self._clock())
            self._queue.append(job)
            return job

    def _admissible(self, job, free_ram_gb):
        # FIFO per class: an earlier waiting job that wants any of our classes goes first.
        for earlier in self._queue:
            if earlier is job:
                break
            if earlier.state == "queued" and set(earlier.classes) & set(job.classes):
                return False
        for name in job.classes:
            if self._in_use[name] >= self._slots[name]:
                return False
        if not self._running:
            return True  # a lone job always runs (it adapts / falls back itself)
        if job.ram_gb > 0 and free_ram_gb is not None:
            reserved = sum(r.ram_gb for r in self._running.values()) + sum(v[0] for v in self._extra.values())
            if free_ram_gb - reserved < job.ram_gb:
                return False
        return True

    def _wait_for_admission(self, job, timeout):
        deadline = None if timeout is None else self._clock() + timeout
        free = self._probe_free()
        with self._cond:
            while True:
                if job.cancelled:
                    self._drop(job)
                    self._stats["cancelled"] += 1
                    raise JobCancelled(job.reason or CANCELLED_MESSAGE)
                if self._admissible(job, free):
                    self._queue.remove(job)
                    job.state = "running"
                    job.started_at = self._clock()
                    self._running[job.id] = job
                    for name in job.classes:
                        self._in_use[name] += 1
                    self._stats["admitted"] += 1
                    self._stats["waited_sec"] += job.started_at - job.enqueued_at
                    self._mark_work()
                    return
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    self._drop(job)
                    self._stats["timeouts"] += 1
                    raise JobQueueTimeout(BUSY_MESSAGE)
                # Re-probe memory periodically: running jobs free memory as they go.
                self._cond.wait(timeout=0.5 if remaining is None else min(0.5, remaining))
                self._cond.release()
                try:
                    free = self._probe_free()
                finally:
                    self._cond.acquire()

    def _drop(self, job):
        try:
            self._queue.remove(job)
        except ValueError:
            pass
        job.state = "cancelled" if job.cancelled else "timed_out"
        self._cond.notify_all()

    def _release(self, job):
        with self._cond:
            if self._running.pop(job.id, None) is not None:
                for name in job.classes:
                    self._in_use[name] = max(0, self._in_use[name] - 1)
                job.state = "cancelled" if job.cancelled else "done"
            self._cond.notify_all()

    def _wake(self):
        with self._cond:
            self._cond.notify_all()

    def _probe_free(self):
        try:
            free, total = self._ram_probe()
            return free if total > 0 else None
        except Exception:
            return None

    # ── cooperation helpers ───────────────────────────────────────────
    def checkpoint(self):
        """Raise JobCancelled if the calling thread's job was cancelled."""
        job = self.current_job()
        if job is not None:
            job.check_cancelled()

    def cancel(self, job_id, reason=None):
        with self._cond:
            job = self._running.get(job_id) or next((j for j in self._queue if j.id == job_id), None)
        if job is None:
            return False
        job.cancel(reason)
        return True

    def add_cancel_hook(self, hook):
        job = self.current_job()
        return job.add_cancel_hook(hook) if job is not None else (lambda: None)

    def active_count(self, name):
        with self._cond:
            return self._in_use.get(name, 0)

    # ── reservations (read by the quality governor) ───────────────────
    def reserved_by_others(self):
        """(ram_gb, vram_gb) reserved by running jobs other than the caller's."""
        me = self.current_job()
        ident = threading.get_ident()
        with self._cond:
            ram = sum(j.ram_gb for j in self._running.values() if j is not me)
            vram = sum(j.vram_gb for j in self._running.values() if j is not me)
            for r, v, owner in self._extra.values():
                if owner != ident:
                    ram += r
                    vram += v
        return ram, vram

    @contextmanager
    def reservation(self, ram_gb=0.0, vram_gb=0.0):
        """
        Hold an additional reservation for the duration of the block — used by
        the quality governor for the variant it picked. Inside an admitted job
        the job's own reservation is raised instead.
        """
        ram_gb, vram_gb = max(0.0, float(ram_gb or 0.0)), max(0.0, float(vram_gb or 0.0))
        job = self.current_job()
        if job is not None:
            with self._cond:
                before = (job.ram_gb, job.vram_gb)
                job.ram_gb, job.vram_gb = max(job.ram_gb, ram_gb), max(job.vram_gb, vram_gb)
            try:
                yield
            finally:
                with self._cond:
                    job.ram_gb, job.vram_gb = before
                    self._cond.notify_all()
            return
        token = object()
        with self._cond:
            self._extra[token] = (ram_gb, vram_gb, threading.get_ident())
        try:
            yield
        finally:
            with self._cond:
                self._extra.pop(token, None)
                self._cond.notify_all()

    # ── reclamation hint for the web layer ────────────────────────────
    def _mark_work(self):
        self._local.did_work = True

    def consume_work_flag(self):
        """True once if heavy work ran on this thread since the last call."""
        flag = getattr(self._local, "did_work", False)
        self._local.did_work = False
        return flag

    # ── status for a progress UI ──────────────────────────────────────
    def status(self):
        now = self._clock()
        with self._cond:
            running = [j.public(now=now) for j in self._running.values()]
            queued = [j.public(position=i + 1, now=now) for i, j in enumerate(self._queue)]
            return {
                "slots": dict(self._slots),
                "in_use": dict(self._in_use),
                "running": running,
                "queued": queued,
                "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self._stats.items()},
            }

    def position(self, job_id):
        with self._cond:
            if job_id in self._running:
                return 0
            for index, job in enumerate(self._queue):
                if job.id == job_id:
                    return index + 1
        return None


def _default_ram_probe():
    from model_manager import _probe_ram_gb
    return _probe_ram_gb()


def _default_model_lock_owned():
    try:
        from model_manager import global_model_manager
        return global_model_manager.owned_by_current_thread()
    except Exception:
        return False


global_scheduler = JobScheduler()


def job(name, classes=("cpu",), ram_gb=0.0, vram_gb=0.0, timeout=None, max_runtime=None):
    """Module-level shortcut for ``global_scheduler.job(...)``."""
    return global_scheduler.job(name, classes=classes, ram_gb=ram_gb, vram_gb=vram_gb, timeout=timeout,
                                max_runtime=max_runtime)


def scheduled(name, classes=("cpu",), ram_gb=0.0, vram_gb=0.0, estimate=None):
    """
    Decorator: run the function as an admitted job. ``estimate(*args, **kwargs)``
    may return a dict overriding ``ram_gb`` / ``vram_gb`` / ``classes`` per call
    (e.g. from the input's size); it must be cheap and never raise.
    """
    import functools

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            spec = {"classes": classes, "ram_gb": ram_gb, "vram_gb": vram_gb}
            if estimate is not None:
                try:
                    spec.update(estimate(*args, **kwargs) or {})
                except Exception:
                    pass
            with global_scheduler.job(name, **spec):
                return fn(*args, **kwargs)
        wrapper.__wrapped_unscheduled__ = fn
        return wrapper
    return decorate


def checkpoint():
    global_scheduler.checkpoint()


def file_gb(path, factor=1.0, floor=0.0):
    """Cheap reservation estimate: ``factor`` x the file size, in GB."""
    try:
        return max(floor, os.path.getsize(path) * factor / (1024 ** 3))
    except (OSError, TypeError):
        return floor


def audio_gb(path, bytes_per_sample=4.0, copies=1.0, floor=0.05):
    """Estimate RAM for ``copies`` decoded copies of an audio file (header only)."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return max(floor, info.frames * info.channels * bytes_per_sample * copies / (1024 ** 3))
    except Exception:
        # Compressed formats: assume ~10x expansion of the file size to float32.
        return file_gb(path, factor=10.0 * copies * bytes_per_sample / 4.0, floor=floor)
