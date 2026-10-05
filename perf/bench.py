"""
Resource benchmark for the media studio backend.

    python perf/bench.py --label baseline            # writes perf/baseline.json
    python perf/bench.py --label after               # writes perf/after.json
    python perf/bench.py --label after --only audio.normalize_10min,stt_10s
    python perf/bench.py --compare perf/baseline.json perf/after.json

Every capability runs in a FRESH subprocess that first imports the Flask app
(exactly what the server process holds), then measures:

  * cold ``import app`` time and the top import-time offenders (-X importtime)
  * idle working set / private bytes / native threads / handles after startup
  * per-capability latency, peak working set, peak private bytes and peak
    device VRAM delta (background sampler, 20 ms)
  * working set and private bytes after the job (gc'd), whether a model is
    still resident, thread and handle deltas, and bytes the process wrote
  * the dev server (reloader parent + serving child) idle footprint

Synthetic media (10 s / 10 min / 60 min audio, 10 s / 10 min speech,
1080p 30 s video, 12 MP and 2 MP images) is generated once with the media
engine into a cache directory outside the repository.
"""

import argparse
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PERF_DIR)
sys.path.insert(0, PERF_DIR)
import memprobe  # noqa: E402

PYTHON = sys.executable
MEDIA_CACHE = os.path.join(tempfile.gettempdir(), "media-studio-bench-media")
HEAVY_MODULES = ("torch", "onnxruntime", "cv2", "ctranslate2", "faster_whisper", "rembg", "scipy",
                 "librosa", "noisereduce", "numba", "sklearn", "scenedetect", "pydub", "soundfile",
                 "numpy", "joblib", "PIL")


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic media
# ═══════════════════════════════════════════════════════════════════════════

_TONE = ("0.22*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*3*t))*gt(mod(t\\,6)\\,1)"
         "+0.75*lt(mod(t\\,2.5)\\,0.004)*sin(2*PI*1000*t)+0.02*(random(0)-0.5)")
_TONE_R = ("0.22*sin(2*PI*330*t)*(0.6+0.4*sin(2*PI*2*t))*gt(mod(t\\,6)\\,1)"
           "+0.75*lt(mod(t\\,2.5)\\,0.004)*sin(2*PI*1000*t)+0.02*(random(1)-0.5)")

SPEECH_TEXT = ("Welcome to the studio benchmark. This short recording checks that speech recognition "
               "works on a laptop with very little free memory. The quick brown fox jumps over the lazy dog.")


def _ff(args, timeout=1800):
    subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-y", *args],
                   check=True, timeout=timeout)


def _tone(path, seconds):
    _ff(["-f", "lavfi", "-i", f"aevalsrc={_TONE}|{_TONE_R}:s=48000:d={seconds}",
         "-c:a", "pcm_s16le", path])


def _speech(path, seconds):
    base = os.path.join(MEDIA_CACHE, "speech_base.wav")
    if not os.path.exists(base):
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SetOutputToWaveFile('{base}'); $s.Speak('{SPEECH_TEXT}'); $s.Dispose()")
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=120)
        except Exception:
            _tone(base, 10)
    _ff(["-stream_loop", "-1", "-i", base, "-t", str(seconds), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", path])


def ensure_media(include_long=False):
    os.makedirs(MEDIA_CACHE, exist_ok=True)
    media = {
        "audio_10s": os.path.join(MEDIA_CACHE, "audio_10s.wav"),
        "audio_10min": os.path.join(MEDIA_CACHE, "audio_10min.wav"),
        "audio_10min_mp3": os.path.join(MEDIA_CACHE, "audio_10min.mp3"),
        "speech_10s": os.path.join(MEDIA_CACHE, "speech_10s.wav"),
        "speech_10min": os.path.join(MEDIA_CACHE, "speech_10min.wav"),
        "video_30s": os.path.join(MEDIA_CACHE, "video_1080p_30s.mp4"),
        "image_12mp": os.path.join(MEDIA_CACHE, "image_12mp.jpg"),
        "image_2mp": os.path.join(MEDIA_CACHE, "image_2mp.png"),
    }
    if include_long:
        media["audio_60min"] = os.path.join(MEDIA_CACHE, "audio_60min.wav")
    makers = {
        "audio_10s": lambda p: _tone(p, 10),
        "audio_10min": lambda p: _tone(p, 600),
        "audio_10min_mp3": lambda p: _ff(["-i", media["audio_10min"], "-c:a", "libmp3lame", "-b:a", "192k", p]),
        "audio_60min": lambda p: _tone(p, 3600),
        "speech_10s": lambda p: _speech(p, 10),
        "speech_10min": lambda p: _speech(p, 600),
        "video_30s": lambda p: _ff([
            "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:d=10",
            "-f", "lavfi", "-i", "smptehdbars=size=1920x1080:rate=30:d=10",
            "-f", "lavfi", "-i", "mandelbrot=size=1920x1080:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=30",
            "-filter_complex", "[2:v]trim=duration=10,setpts=PTS-STARTPTS[m];[0:v][1:v][m]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p]),
        "image_12mp": lambda p: _ff(["-f", "lavfi", "-i", "testsrc2=size=4000x3000:rate=1", "-frames:v", "1",
                                     "-q:v", "3", p]),
        "image_2mp": lambda p: _ff(["-f", "lavfi", "-i", "testsrc2=size=1600x1200:rate=1", "-frames:v", "1", p]),
    }
    for key, path in media.items():
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"[bench] generating {key} ...", flush=True)
            makers[key](path)
    return media


# ═══════════════════════════════════════════════════════════════════════════
#  Capabilities (each runs inside a fresh worker process after `import app`)
# ═══════════════════════════════════════════════════════════════════════════

def _capabilities(media, out):
    import ai_processor
    import audio_processor
    import image_processor
    import media_inspector
    import video_processor

    def o(name):
        return os.path.join(out, name)

    caps = {
        "audio.lufs_10s": lambda: audio_processor.calculate_lufs(media["audio_10s"]),
        "audio.lufs_10min": lambda: audio_processor.calculate_lufs(media["audio_10min"]),
        "audio.lufs_10min_mp3": lambda: audio_processor.calculate_lufs(media["audio_10min_mp3"]),
        "audio.normalize_10s": lambda: audio_processor.normalize_audio(media["audio_10s"], o("n10s.wav"),
                                                                       preset="youtube"),
        "audio.normalize_10min": lambda: audio_processor.normalize_audio(media["audio_10min"], o("n10m.wav"),
                                                                         preset="youtube"),
        "audio.eq_10s": lambda: audio_processor.apply_parametric_eq(media["audio_10s"], o("eq10s.wav"),
                                                                    {"63": 4.0, "4000": -3.0}),
        "audio.eq_10min": lambda: audio_processor.apply_parametric_eq(media["audio_10min"], o("eq10m.wav"),
                                                                      {"63": 4.0, "4000": -3.0}),
        "audio.silence_10min": lambda: ai_processor.detect_silence(media["audio_10min"]),
        "audio.vad_10min": lambda: ai_processor.detect_voice_activity(media["audio_10min"]),
        "audio.trim_gaps_10min": lambda: ai_processor.trim_silence_gaps(media["audio_10min"], o("trim.wav"),
                                                                        min_silence_len=0.8),
        "audio.denoise_10s": lambda: ai_processor.reduce_noise(media["audio_10s"], o("dn10s.wav")),
        "audio.denoise_10min": lambda: ai_processor.reduce_noise(media["audio_10min"], o("dn10m.wav")),
        "audio.duck_10s": lambda: ai_processor.auto_duck_music(media["speech_10s"], media["audio_10s"],
                                                               o("duck.wav")),
        "audio.speed_10min": lambda: ai_processor.pitch_preserved_speed(media["audio_10min"], o("speed.wav"), 1.25),
        "stt_10s": lambda: _check_stt(ai_processor.transcribe_audio(media["speech_10s"])),
        "stt_10min": lambda: _check_stt(ai_processor.transcribe_audio(media["speech_10min"])),
        "video.probe": lambda: video_processor.probe_media(media["video_30s"]),
        "video.filmstrip_30s": lambda: video_processor.generate_filmstrip(media["video_30s"], o("thumbs"), 30.0),
        "video.compress_1080p_30s": lambda: video_processor.quick_compress(media["video_30s"], o("c.mp4")),
        "video.enhance_1080p_30s": lambda: video_processor.enhance_video_quality(media["video_30s"], o("e.mp4"),
                                                                                 mode="1080p"),
        "video.scenes_30s": lambda: video_processor.detect_scenes(media["video_30s"]),
        "image.upscale_12mp": lambda: image_processor.upscale(media["image_12mp"], o("up12.png"), 2, "auto"),
        "image.upscale_2mp": lambda: image_processor.upscale(media["image_2mp"], o("up2.png"), 2, "auto"),
        "image.clarity_12mp": lambda: image_processor.enhance_photo_clarity(media["image_12mp"], o("cl.png")),
        "image.color_12mp": lambda: image_processor.color_match_transfer(media["image_12mp"], o("col.png")),
        "image.cutout_12mp": lambda: image_processor.remove_bg(media["image_12mp"], o("cut.png"), "auto", True),
        "inspect.image_12mp": lambda: media_inspector.perceive(media["image_12mp"]),
        "inspect.audio_10min": lambda: media_inspector.perceive(media["audio_10min"]),
    }
    if "audio_60min" in media:
        caps["audio.lufs_60min"] = lambda: audio_processor.calculate_lufs(media["audio_60min"])
        caps["audio.normalize_60min"] = lambda: audio_processor.normalize_audio(
            media["audio_60min"], o("n60m.wav"), preset="youtube")
    return caps


def _check_stt(result):
    if not result.get("available"):
        raise RuntimeError(result.get("error", "transcription unavailable"))
    return {"segments": len(result.get("segments", [])), "chars": len(result.get("full_text", ""))}


def _dir_bytes(paths):
    total = 0
    for root_path in paths:
        if not os.path.isdir(root_path):
            continue
        for root, _dirs, files in os.walk(root_path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return total


def _summ(snap):
    return {"rss_mb": memprobe.mb(snap["rss"]), "private_mb": memprobe.mb(snap["private"]),
            "threads": snap["threads"], "handles": snap["handles"], "python_threads": snap["python_threads"]}


def run_worker(scenario, media_json, out_dir):
    media = json.loads(media_json)
    t0 = time.perf_counter()
    import app  # noqa: F401  (the server's resident state)
    import_sec = time.perf_counter() - t0
    gc.collect()
    idle = memprobe.snapshot()
    record = {"scenario": scenario, "import_sec": round(import_sec, 3), "idle": _summ(idle),
              "heavy_modules_after_import": sorted(m for m in HEAVY_MODULES if m in sys.modules)}
    if scenario == "startup":
        return record

    caps = _capabilities(media, out_dir)
    fn = caps[scenario]
    watched = [tempfile.gettempdir(), os.path.join(ROOT, "uploads"), os.path.join(ROOT, "processed")]
    disk_before = _dir_bytes([out_dir])
    io_before = memprobe.io_counters() or {}
    started = time.perf_counter()
    error = None
    with memprobe.PeakSampler() as sampler:
        try:
            result = fn()
        except Exception as exc:  # recorded, not raised
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            result = None
            traceback.print_exc()
    latency = time.perf_counter() - started
    result = None
    gc.collect()
    post = memprobe.snapshot()
    io_after = memprobe.io_counters() or {}
    import model_manager
    record.update({
        "latency_sec": round(latency, 3),
        "error": error,
        "peak_rss_mb": memprobe.mb(sampler.peak_rss),
        "peak_private_mb": memprobe.mb(sampler.peak_private),
        "peak_vram_delta_mb": memprobe.mb(sampler.result().get("peak_vram_delta")),
        "post": _summ(post),
        "post_rss_delta_mb": memprobe.mb(post["rss"] - idle["rss"]),
        "post_private_delta_mb": memprobe.mb(post["private"] - idle["private"]),
        "thread_delta": (post["threads"] or 0) - (idle["threads"] or 0),
        "handle_delta": (post["handles"] or 0) - (idle["handles"] or 0),
        "model_resident_after": model_manager.global_model_manager._active_model_id is not None,
        "process_write_mb": memprobe.mb(io_after.get("write_bytes", 0) - io_before.get("write_bytes", 0)),
        "output_mb": memprobe.mb(_dir_bytes([out_dir]) - disk_before),
        "heavy_modules_after_job": sorted(m for m in HEAVY_MODULES if m in sys.modules),
    })
    del watched
    return record


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def import_time_offenders(top=15):
    proc = subprocess.run([PYTHON, "-X", "importtime", "-c", "import app"], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    rows = []
    for line in proc.stderr.splitlines():
        if not line.startswith("import time:") or "|" not in line:
            continue
        parts = line[len("import time:"):].split("|")
        try:
            cumulative = int(parts[1].strip())
        except ValueError:
            continue
        name = parts[2].rstrip()
        depth = (len(name) - len(name.lstrip())) // 2
        rows.append((cumulative, depth, name.strip()))
    shallow = [r for r in rows if r[1] <= 2]
    shallow.sort(reverse=True)
    return [{"module": n, "depth": d, "cumulative_ms": round(c / 1000, 1)} for c, d, n in shallow[:top]]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _children(pid):
    """Direct child pids (Windows: toolhelp process snapshot)."""
    if sys.platform != "win32":
        try:
            out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout
            return [int(x) for x in out.split()]
        except Exception:
            return []
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260)]

    k32 = ctypes.WinDLL("kernel32")
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    entry = PE()
    entry.dwSize = ctypes.sizeof(entry)
    kids = []
    ok = k32.Process32First(snap, ctypes.byref(entry))
    while ok:
        if entry.th32ParentProcessID == pid:
            kids.append(int(entry.th32ProcessID))
        ok = k32.Process32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return kids


def _descendants(pid):
    out, frontier = [pid], [pid]
    while frontier:
        kids = [k for parent in frontier for k in _children(parent)]
        out += kids
        frontier = kids
    return out


def server_footprint(root=ROOT):
    """Start the dev server exactly like `python app.py` and measure every process."""
    port = _free_port()
    env = dict(os.environ, FLASK_PORT=str(port), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([PYTHON, "app.py"], cwd=root, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    try:
        ready = None
        while time.perf_counter() - started < 180:
            try:
                urllib.request.urlopen(url + "/health", timeout=2).read()
                ready = time.perf_counter() - started
                break
            except Exception:
                time.sleep(0.25)
        latencies = []
        for _ in range(5):
            t = time.perf_counter()
            urllib.request.urlopen(url + "/api/system/resources", timeout=10).read()
            latencies.append(time.perf_counter() - t)
        time.sleep(1.0)
        procs = _descendants(proc.pid)
        parts = {}
        for pid in procs:
            snap = memprobe.snapshot(pid)
            parts[str(pid)] = {"rss_mb": memprobe.mb(snap["rss"]), "private_mb": memprobe.mb(snap["private"]),
                               "threads": snap["threads"], "handles": snap["handles"]}
        return {
            "ready_sec": round(ready, 2) if ready else None,
            "resources_route_ms": round(1000 * sorted(latencies)[len(latencies) // 2], 1),
            "processes": parts,
            "total_rss_mb": round(sum(p["rss_mb"] or 0 for p in parts.values()), 1),
            "total_private_mb": round(sum(p["private_mb"] or 0 for p in parts.values()), 1),
        }
    finally:
        for pid in _descendants(proc.pid)[1:]:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True) \
                if sys.platform == "win32" else os.kill(pid, 9)
        proc.kill()
        proc.wait(timeout=30)


SHADOW = ""


def _server_root(shadow):
    """A throwaway copy of the app with the shadow modules overlaid (for baselines)."""
    if not shadow:
        return ROOT
    root = tempfile.mkdtemp(prefix="bench-server-")
    for name in os.listdir(ROOT):
        if name.endswith(".py"):
            shutil.copy2(os.path.join(ROOT, name), root)
    shutil.copytree(os.path.join(ROOT, "templates"), os.path.join(root, "templates"))
    for name in os.listdir(shadow):
        if name.endswith(".py"):
            shutil.copy2(os.path.join(shadow, name), root)
    return root


def run_scenario(scenario, media, timeout):
    out_dir = tempfile.mkdtemp(prefix="bench-out-")
    try:
        proc = subprocess.run(
            [PYTHON, os.path.abspath(__file__), "--worker", scenario, "--media", json.dumps(media),
             "--out", out_dir] + (["--shadow", SHADOW] if SHADOW else []),
            cwd=ROOT, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        lines = [line for line in proc.stdout.splitlines() if line.startswith("@@RESULT ")]
        if not lines:
            return {"scenario": scenario, "error": f"worker failed (code {proc.returncode}): "
                                                   f"{(proc.stderr or '')[-600:]}"}
        return json.loads(lines[-1][len("@@RESULT "):])
    except subprocess.TimeoutExpired:
        return {"scenario": scenario, "error": f"timeout after {timeout}s"}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def system_info():
    import model_manager
    ram_free, ram_total = model_manager._probe_ram_gb()
    return {"python": sys.version.split()[0], "cpu_count": os.cpu_count(), "ram_total_gb": round(ram_total, 1),
            "ram_free_gb_at_start": round(ram_free, 2), "platform": sys.platform,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def summarize(report):
    lines = [f"# Benchmark: {report['label']}  ({report['system']['time']})",
             f"RAM free at start: {report['system']['ram_free_gb_at_start']} GB / {report['system']['ram_total_gb']} GB",
             ""]
    st = report.get("startup") or {}
    if st:
        lines.append(f"startup: import app {st.get('import_sec')} s | idle RSS {st['idle']['rss_mb']} MB | "
                     f"private {st['idle']['private_mb']} MB | threads {st['idle']['threads']} | "
                     f"heavy modules: {', '.join(st.get('heavy_modules_after_import', [])) or 'none'}")
    sv = report.get("server") or {}
    if sv:
        lines.append(f"dev server (reloader parent + child): RSS {sv.get('total_rss_mb')} MB, private "
                     f"{sv.get('total_private_mb')} MB, ready {sv.get('ready_sec')} s, "
                     f"/api/system/resources {sv.get('resources_route_ms')} ms")
    lines.append("")
    lines.append(f"{'capability':28} {'latency s':>9} {'peakRSS':>8} {'peakPriv':>8} {'VRAM':>6} "
                 f"{'postdRSS':>8} {'postdPriv':>9} {'thrd':>5} {'hdld':>5} {'model':>5}  error")
    for row in report.get("capabilities", []):
        if row.get("latency_sec") is None:
            lines.append(f"{row['scenario']:28} {'-':>9}  error: {row.get('error', '')[:90]}")
            continue
        lines.append(
            f"{row['scenario']:28} {row['latency_sec']:9.2f} {row['peak_rss_mb']:8.0f} {row['peak_private_mb']:8.0f} "
            f"{(row.get('peak_vram_delta_mb') or 0):6.0f} {row['post_rss_delta_mb']:8.0f} "
            f"{row['post_private_delta_mb']:9.0f} {row['thread_delta']:5d} {row['handle_delta']:5d} "
            f"{'yes' if row['model_resident_after'] else 'no':>5}  {(row.get('error') or '')[:60]}")
    return "\n".join(lines)


def compare(before_path, after_path):
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)
    b_rows = {r["scenario"]: r for r in before.get("capabilities", [])}
    a_rows = {r["scenario"]: r for r in after.get("capabilities", [])}
    out = ["| Metric | Before | After |", "|---|---|---|"]
    bs, as_ = before.get("startup", {}), after.get("startup", {})
    out.append(f"| `import app` time | {bs.get('import_sec')} s | {as_.get('import_sec')} s |")
    out.append(f"| Idle working set | {bs.get('idle', {}).get('rss_mb')} MB | {as_.get('idle', {}).get('rss_mb')} MB |")
    out.append(f"| Idle private bytes | {bs.get('idle', {}).get('private_mb')} MB | "
               f"{as_.get('idle', {}).get('private_mb')} MB |")
    out.append(f"| Idle native threads | {bs.get('idle', {}).get('threads')} | {as_.get('idle', {}).get('threads')} |")
    bsv, asv = before.get("server", {}), after.get("server", {})
    if bsv and asv:
        out.append(f"| Dev server total private (both processes) | {bsv.get('total_private_mb')} MB | "
                   f"{asv.get('total_private_mb')} MB |")
        out.append(f"| Dev server ready | {bsv.get('ready_sec')} s | {asv.get('ready_sec')} s |")
    out.append("")
    out.append("| Capability | Latency s (before -> after) | Peak private MB | Peak RSS MB | VRAM MB | "
               "Post-job delta private MB | Model resident |")
    out.append("|---|---|---|---|---|---|---|")
    for name in list(b_rows) + [n for n in a_rows if n not in b_rows]:
        b, a = b_rows.get(name, {}), a_rows.get(name, {})

        def cell(key, row_b=b, row_a=a, fmt="{:.0f}"):
            def one(row):
                value = row.get(key)
                if value is None:
                    return "err" if row.get("error") else "–"
                return fmt.format(value)
            return f"{one(row_b)} -> {one(row_a)}"

        out.append(f"| {name} | {cell('latency_sec', fmt='{:.2f}')} | {cell('peak_private_mb')} | "
                   f"{cell('peak_rss_mb')} | {cell('peak_vram_delta_mb')} | {cell('post_private_delta_mb')} | "
                   f"{'yes' if b.get('model_resident_after') else 'no'} -> "
                   f"{'yes' if a.get('model_resident_after') else 'no'} |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--only", default="")
    parser.add_argument("--skip", default="")
    parser.add_argument("--long", action="store_true", help="also run 60-minute audio scenarios")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker")
    parser.add_argument("--media")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--merge", action="store_true", help="update rows in an existing report")
    parser.add_argument("--shadow", default="", help="directory of module versions that take precedence "
                                                     "(re-measure the pre-optimisation code)")
    args = parser.parse_args()

    if args.worker:
        os.chdir(ROOT)
        sys.path.insert(0, ROOT)
        if args.shadow:
            sys.path.insert(0, os.path.abspath(args.shadow))
        record = run_worker(args.worker, args.media, args.out)
        print("@@RESULT " + json.dumps(record), flush=True)
        return
    if args.compare:
        print(compare(*args.compare))
        return

    global SHADOW
    SHADOW = args.shadow
    media = ensure_media(include_long=args.long)
    sys.path.insert(0, ROOT)
    report_path = os.path.join(PERF_DIR, f"{args.label}.json")
    report = {"label": args.label, "system": system_info(), "capabilities": []}
    if args.merge and os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
        report["system"] = system_info()

    only = [s for s in args.only.split(",") if s]
    skip = {s for s in args.skip.split(",") if s}
    if not only or "startup" in only:
        print("[bench] startup ...", flush=True)
        report["startup"] = run_scenario("startup", media, args.timeout)
        report["import_offenders"] = import_time_offenders()
    if not only or "server" in only:
        print("[bench] dev server ...", flush=True)
        try:
            report["server"] = server_footprint(_server_root(args.shadow))
        except Exception as exc:
            report["server"] = {"error": f"{type(exc).__name__}: {exc}"}

    names = list(_capability_names(media))
    if only:
        names = [n for n in names if n in only]
    names = [n for n in names if n not in skip]
    rows = {r["scenario"]: r for r in report.get("capabilities", [])}
    for name in names:
        print(f"[bench] {name} ...", flush=True)
        rows[name] = run_scenario(name, media, args.timeout)
        row = rows[name]
        print(f"        -> {row.get('latency_sec')} s, peak private {row.get('peak_private_mb')} MB, "
              f"error={row.get('error')}", flush=True)
        report["capabilities"] = list(rows.values())
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    report["capabilities"] = list(rows.values())
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    summary = summarize(report)
    with open(os.path.join(PERF_DIR, f"{args.label}_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(summary)


def _capability_names(media):
    # Names only; avoids importing processors in the orchestrator process.
    names = ["audio.lufs_10s", "audio.lufs_10min", "audio.lufs_10min_mp3", "audio.normalize_10s",
             "audio.normalize_10min", "audio.eq_10s", "audio.eq_10min", "audio.silence_10min",
             "audio.vad_10min", "audio.trim_gaps_10min", "audio.denoise_10s", "audio.denoise_10min",
             "audio.duck_10s", "audio.speed_10min", "stt_10s", "stt_10min", "video.probe",
             "video.filmstrip_30s", "video.compress_1080p_30s", "video.enhance_1080p_30s",
             "video.scenes_30s", "image.upscale_12mp", "image.upscale_2mp", "image.clarity_12mp",
             "image.color_12mp", "image.cutout_12mp", "inspect.image_12mp", "inspect.audio_10min"]
    if "audio_60min" in media:
        names += ["audio.lufs_60min", "audio.normalize_60min"]
    return names


if __name__ == "__main__":
    main()
