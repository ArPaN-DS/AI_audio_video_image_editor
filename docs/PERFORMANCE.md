# Media Studio — Performance & Resource Architecture

This document details the resource management, admission control, and performance architecture that enables Media Studio to run on local desktop and laptop hardware without cloud dependencies.

---

## 1. Resource Governance Architecture

Media Studio enforces strict hardware limits to prevent Out-Of-Memory (OOM) crashes, CPU starvation, and UI freezes during concurrent usage:

### A. Single-Active-Model Lifecycle (`model_manager.py`)
- **Exclusive Model Slot:** Only one heavy deep learning model is resident in memory at any given time.
- **Zero Idle Memory:** Models are loaded lazily upon request and released immediately into garbage collection once execution completes.
- **Hardware-Adaptive Cascades:** Models fall back automatically from high-precision GPU layers to quantized CPU models if VRAM or RAM headroom drops below safety thresholds.

### B. Heavy Job Scheduler (`job_scheduler.py`)
Because the local server process handles multi-threaded requests, concurrent operations pass through admission control via `JobScheduler`:

| Resource Class | Default Slots | Description |
|---|---|---|
| `model` | 1 | Exclusive slot for neural inference (transcription, stem separation, vision models). |
| `gpu` | 1 | Exclusive accelerator access to protect VRAM from concurrent allocations. |
| `cpu` | $N$ (cores // 3, max 4) | Concurrency limit for CPU-intensive signal processing and image manipulation. |
| `encode` | $M$ (cores // 4, max 2) | Media conversion and timeline export pipeline. |

- **Memory Reservations:** Jobs declare estimated RAM/VRAM requirements. A new job is admitted only if $(\text{free RAM} - \sum \text{running reservations}) \ge \text{required RAM}$. A lone job is never starved; it is always admitted to adapt or fall back gracefully.
- **FIFO Fairness:** Waiting jobs are queued and admitted in first-in, first-out order per class.
- **Cooperative Cancellation:** Jobs register hooks (e.g. terminating subprocesses) and test `checkpoint()` to stop cleanly on user cancellation or timeout.
- **Deadlock Freedom:** Strict hierarchical lock ordering:
  $$\text{Scheduler Admission} \longrightarrow \text{ModelLifecycleManager.\_lock} \longrightarrow \text{AdaptiveQualityGovernor.\_lock}$$
  Nested admissions within the same thread run pass-through without holding additional slots.

### C. Adaptive Quality Governor & Time Budgeting
- Hardware snapshots evaluate free RAM, VRAM, and CPU cores to select among `lite`, `balanced`, or `max` quality variants.
- The governor uses an effective snapshot that subtracts reservations of other concurrent jobs, preventing two jobs from both claiming the maximum tier simultaneously.
- **Time Budget Control (`MEDIA_TIME_BUDGET_SEC`):**
  - Configurable execution time ceiling.
  - When unset, the system gives highest priority to output fidelity and quality, allowing complex operations (such as high-quality stem separation, studio loudness normalization, and large model transcription) to run to completion without arbitrary truncation.

### D. Streaming Audio Processing
- Long-file loudness normalization (BS.1770 / EBU R128) operates in rolling chunks with in-place streaming limiter passes.
- Peak memory for a 60-minute stereo audio file remains bounded under 300 MB (reduced from a 22.4 GB baseline).

---

## 2. Benchmark Results: Before vs. After

Measurements recorded using `perf/bench.py` on local workstation hardware (Intel/AMD multi-core, NVIDIA RTX GPU, 16 GB system RAM).

### Startup & Idle Memory

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| `import app` cold time | 6.44 s | 0.48 s | **92.5% faster** |
| Idle Working Set (RSS) | 265.3 MB | 48.5 MB | **81.7% reduction** |
| Idle Private Bytes | 1,204.4 MB | 96.2 MB | **92.0% reduction** |
| Top-level heavy imports | `torch`, `scipy`, `cv2`, `librosa`, `soundfile` | **0** (all deferred) | **Zero heavy libraries at idle** |
| Dev server idle footprint | 2,424.0 MB | 194.0 MB | **92.0% reduction** |

### Capability Benchmarks

| Capability | Input | Baseline Peak Private | Optimized Peak Private | Baseline Latency | Optimized Latency |
|---|---|---|---|---|---|
| `audio.normalize` | 10 min | 4,775 MB | **293 MB** | 55.7 s | **50.4 s** |
| `audio.normalize` | 60 min | 22,391 MB (22.4 GB) | **294 MB** | 798.2 s | **413.9 s** |
| `audio.lufs` | 60 min | 4,191 MB | **259 MB** | 52.4 s | **40.0 s** |
| `audio.eq` | 10 min | 3,060 MB | **312 MB** | 9.9 s | **8.4 s** |
| `audio.trim_gaps` | 10 min | 3,054 MB | **285 MB** | 18.0 s | **16.1 s** |
| `image.cutout` | 12 MP | 10,909 MB | **1,850 MB** | 17.3 s | **14.2 s** |
| `stt_transcribe` | 10 min | 4,759 MB | **1,650 MB** | 117.2 s | **110.5 s** |

---

## 3. Running Performance Tests and Benchmarks

### Automated Test Suite
```bash
# Verify admission control, slot limits, memory reservations, and deadlock freedom:
venv\Scripts\python.exe test_performance_budgets.py

# Verify hardware governor and idle import checks:
venv\Scripts\python.exe test_resource_governor.py
```

### Running the Benchmark Suite
```bash
# Capture a benchmark run:
python perf/bench.py --label my_run

# Compare two benchmark outputs:
python perf/bench.py --compare perf/baseline.json perf/after.json
```
