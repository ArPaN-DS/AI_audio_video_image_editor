"""
Unit and integration tests for JobScheduler admission control, slot limits,
memory reservations, cooperative cancellation, deadlock freedom, and governor integration.
"""

import os
import sys
import threading
import time
import unittest
from unittest import mock

import job_scheduler
from job_scheduler import (
    JobScheduler,
    JobQueueTimeout,
    JobCancelled,
    CLASSES,
    default_slots,
    Job,
)
import model_manager
from model_manager import (
    ResourceSnapshot,
    AdaptiveQualityGovernor,
    TIER_LITE,
    TIER_BALANCED,
    TIER_MAX,
)


class TestJobSchedulerSlots(unittest.TestCase):
    """Test slot limits for model, gpu, cpu, and encode classes."""

    def setUp(self):
        self.scheduler = JobScheduler(
            slots={"model": 1, "gpu": 1, "cpu": 2, "encode": 1},
            ram_probe=lambda: (16.0, 16.0),  # Plenty of RAM
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

    def test_default_slots_initialization(self):
        slots = default_slots(cores=8, ram_total_gb=16.0)
        self.assertEqual(slots["model"], 1)
        self.assertEqual(slots["gpu"], 1)
        self.assertGreaterEqual(slots["cpu"], 1)
        self.assertGreaterEqual(slots["encode"], 1)

    def test_exclusive_gpu_slot(self):
        """Only 1 GPU job can run at a time; second waits until first finishes."""
        events = []

        def worker1():
            with self.scheduler.job("gpu_task_1", classes=("gpu",)):
                events.append("w1_start")
                time.sleep(0.08)
                events.append("w1_end")

        def worker2():
            # Wait a moment so worker1 enters first
            time.sleep(0.02)
            with self.scheduler.job("gpu_task_2", classes=("gpu",), timeout=1.0):
                events.append("w2_start")
                events.append("w2_end")

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Worker 2 must not start until worker 1 has finished
        self.assertEqual(events, ["w1_start", "w1_end", "w2_start", "w2_end"])

    def test_exclusive_model_slot(self):
        """Only 1 model inference job can run at a time."""
        events = []

        def worker1():
            with self.scheduler.job("model_task_1", classes=("model",)):
                events.append("m1_start")
                time.sleep(0.08)
                events.append("m1_end")

        def worker2():
            time.sleep(0.02)
            with self.scheduler.job("model_task_2", classes=("model",), timeout=1.0):
                events.append("m2_start")
                events.append("m2_end")

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(events, ["m1_start", "m1_end", "m2_start", "m2_end"])

    def test_cpu_concurrency_limit(self):
        """CPU slot limit (2) allows 2 concurrent jobs, 3rd waits."""
        active_counts = []
        lock = threading.Lock()

        def worker():
            with self.scheduler.job("cpu_task", classes=("cpu",), timeout=1.0):
                with lock:
                    cnt = self.scheduler.active_count("cpu")
                    active_counts.append(cnt)
                time.sleep(0.05)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The active count must never exceed 2
        for count in active_counts:
            self.assertLessEqual(count, 2)


class TestJobSchedulerMemoryReservations(unittest.TestCase):
    """Test memory reservations so concurrent jobs don't exceed available RAM."""

    def test_memory_reservation_queues_oversized_job(self):
        # 10 GB free RAM. Job 1 reserves 7 GB, Job 2 requests 5 GB.
        # Job 2 cannot start until Job 1 finishes.
        scheduler = JobScheduler(
            slots={"model": 2, "gpu": 2, "cpu": 4, "encode": 2},
            ram_probe=lambda: (10.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        events = []

        def worker1():
            with scheduler.job("heavy_1", classes=("cpu",), ram_gb=7.0):
                events.append("w1_start")
                time.sleep(0.08)
                events.append("w1_end")

        def worker2():
            time.sleep(0.02)
            with scheduler.job("heavy_2", classes=("cpu",), ram_gb=5.0, timeout=1.0):
                events.append("w2_start")
                events.append("w2_end")

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(events, ["w1_start", "w1_end", "w2_start", "w2_end"])

    def test_lone_job_never_starved(self):
        """Even if requested RAM exceeds free RAM, a lone job is admitted to adapt/fall back."""
        scheduler = JobScheduler(
            slots={"cpu": 2},
            ram_probe=lambda: (2.0, 16.0),  # Only 2 GB free
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        ran = False
        with scheduler.job("lone_heavy", classes=("cpu",), ram_gb=8.0, timeout=0.5) as j:
            ran = True
            self.assertEqual(j.state, "running")
        self.assertTrue(ran)

    def test_reserved_by_others_visible_to_governor(self):
        """The governor's effective_snapshot subtracts memory reserved by other running jobs."""
        scheduler = JobScheduler(
            slots={"cpu": 4},
            ram_probe=lambda: (12.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        governor = AdaptiveQualityGovernor(
            probe=lambda: ResourceSnapshot(12.0, 16.0, 8, 4.0, 8.0),
            reservations=scheduler.reserved_by_others,
            env={"MEDIA_QUALITY_TIER": "auto"},
        )

        # Baseline snapshot has 12.0 GB free RAM
        snap1 = governor.effective_snapshot()
        self.assertEqual(snap1.ram_free_gb, 12.0)

        # When worker 1 is running with 5.0 GB RAM reservation
        done_flag = threading.Event()
        started_flag = threading.Event()

        def background_worker():
            with scheduler.job("worker_job", classes=("cpu",), ram_gb=5.0, vram_gb=1.5):
                started_flag.set()
                done_flag.wait(timeout=2.0)

        t = threading.Thread(target=background_worker)
        t.start()
        started_flag.wait(timeout=1.0)

        try:
            # From another thread, governor effective_snapshot must reflect 12 - 5 = 7 GB RAM
            snap2 = governor.effective_snapshot()
            self.assertAlmostEqual(snap2.ram_free_gb, 7.0, places=2)
            self.assertAlmostEqual(snap2.gpu_free_gb, 2.5, places=2)
        finally:
            done_flag.set()
            t.join()


class TestJobSchedulerCancellationAndTimeouts(unittest.TestCase):
    """Test timeout, cooperative cancellation, and cancel hooks."""

    def test_queue_timeout_raises_job_queue_timeout(self):
        scheduler = JobScheduler(
            slots={"gpu": 1},
            ram_probe=lambda: (16.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        blocker_started = threading.Event()
        blocker_release = threading.Event()

        def blocker():
            with scheduler.job("blocker", classes=("gpu",)):
                blocker_started.set()
                blocker_release.wait(timeout=2.0)

        t = threading.Thread(target=blocker)
        t.start()
        blocker_started.wait(timeout=1.0)

        try:
            with self.assertRaises(JobQueueTimeout):
                with scheduler.job("blocked", classes=("gpu",), timeout=0.05):
                    pass
        finally:
            blocker_release.set()
            t.join()

    def test_cooperative_cancellation_and_hooks(self):
        scheduler = JobScheduler(
            slots={"cpu": 2},
            ram_probe=lambda: (16.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        hook_called = False

        def cancel_hook():
            nonlocal hook_called
            hook_called = True

        with scheduler.job("cancellable_job", classes=("cpu",)) as j:
            j.add_cancel_hook(cancel_hook)
            self.assertFalse(j.cancelled)
            j.cancel("User cancelled")
            self.assertTrue(j.cancelled)
            self.assertTrue(hook_called)

            with self.assertRaises(JobCancelled):
                j.check_cancelled()

    def test_max_runtime_watchdog(self):
        scheduler = JobScheduler(
            slots={"cpu": 2},
            ram_probe=lambda: (16.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        with scheduler.job("time_limited", classes=("cpu",), max_runtime=0.05) as j:
            time.sleep(0.1)
            self.assertTrue(j.cancelled)
            with self.assertRaises(JobCancelled):
                j.check_cancelled()


class TestJobSchedulerDeadlockFreedom(unittest.TestCase):
    """Verify re-entrancy and model-lock pass-through prevents deadlocks."""

    def test_reentrant_nested_job_does_not_deadlock(self):
        scheduler = JobScheduler(
            slots={"cpu": 1},  # Only 1 slot
            ram_probe=lambda: (16.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: False,
        )

        # Outer job takes the single CPU slot; nested job must run pass-through
        with scheduler.job("outer", classes=("cpu",)) as outer_j:
            self.assertEqual(outer_j.state, "running")
            with scheduler.job("inner", classes=("cpu",), timeout=0.1) as inner_j:
                self.assertEqual(inner_j.state, "running")

    def test_model_lock_holder_runs_passthrough(self):
        """A thread that already holds the model lifecycle lock is never blocked."""
        is_locked = True
        scheduler = JobScheduler(
            slots={"model": 0},  # 0 slots available
            ram_probe=lambda: (16.0, 16.0),
            clock=time.monotonic,
            model_lock_owned=lambda: is_locked,
        )

        # Even with 0 slots, if model lock is owned, job runs pass-through immediately
        with scheduler.job("lifecycle_held", classes=("model",), timeout=0.05) as j:
            self.assertEqual(j.state, "running")


class TestGovernorTimeBudget(unittest.TestCase):
    """Verify governor time budget calculation and MEDIA_TIME_BUDGET_SEC env handling."""

    def test_explicit_time_budget(self):
        gov = AdaptiveQualityGovernor(env={})
        self.assertEqual(gov.time_budget("enhance_video", work_units=30, override=120.0), 120.0)

    def test_env_time_budget(self):
        gov = AdaptiveQualityGovernor(env={"MEDIA_TIME_BUDGET_SEC": "45.0"})
        self.assertEqual(gov.time_budget("enhance_video", work_units=30), 45.0)

    def test_default_time_budget_calculation(self):
        gov = AdaptiveQualityGovernor(env={})
        budget = gov.time_budget("speech.transcribe", work_units=30)
        # DEFAULT_TIME_BUDGETS["speech.transcribe"] is (60.0, 0.5) -> 60.0 + 0.5 * 30 = 75.0
        self.assertAlmostEqual(budget, 75.0, places=1)


if __name__ == "__main__":
    unittest.main()
