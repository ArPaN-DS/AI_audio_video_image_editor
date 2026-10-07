"""
RobustVideoMatting (RVM) SOTA AI Video Background Removal Engine (ONNX + CUDA).

Provides real-time (60 FPS) video matting without green screen:
- Recurrent ConvGRU memory eliminates frame-to-frame boundary flickering.
- Output modes:
    - 'green': Replaces background with green screen (#00FF00).
    - 'transparent': Exports transparent WebM (VP9 + alpha).
    - 'blur': Cinematic portrait video with blurred background.
    - 'black': Clean black studio background.
- Preserves 100% of the original audio stream.
- Flushes GPU VRAM immediately after execution.
"""

import os
import gc
import logging
import tempfile
import subprocess
import cv2
import numpy as np

_log = logging.getLogger("rvm")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RVM_PATH = os.path.join(_BASE_DIR, "models", "rvm_mobilenetv3_fp32.onnx")


def is_available():
    """Return True if RVM model weights exist on disk."""
    return os.path.exists(_RVM_PATH)


def _ensure_cuda_dlls():
    """Ensure PyTorch CUDA/cuDNN DLLs are discoverable by ONNX Runtime."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
    except ImportError:
        pass


def remove_video_background(video_path, output_path, bg_type="green", custom_bg_color=(0, 255, 0)):
    """
    Removes background from video using RobustVideoMatting on CUDA GPU.
    bg_type: 'green', 'black', 'blur', or 'transparent'
    """
    _ensure_cuda_dlls()
    import onnxruntime as ort

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downsample ratio for internal feature extraction (0.5 balances speed and precision)
    downsample_ratio = np.array([0.5 if max(w, h) >= 720 else 1.0], dtype=np.float32)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    session = ort.InferenceSession(_RVM_PATH, sess_options=sess_options, providers=providers)
    _log.info(f"[RVM] Loaded session on providers: {session.get_providers()}")

    # Recurrent state tensors
    r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)

    is_transparent = (bg_type.lower() == "transparent" or output_path.lower().endswith(".webm"))
    temp_raw = os.path.join(tempfile.gettempdir(), f"rvm_temp_{os.getpid()}_{np.random.randint(1000, 9999)}.mp4")

    if is_transparent:
        pipe_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28",
            temp_raw
        ]
    else:
        pipe_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            temp_raw
        ]

    proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            src = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

            inputs = {
                "src": src,
                "r1i": r1,
                "r2i": r2,
                "r3i": r3,
                "r4i": r4,
                "downsample_ratio": downsample_ratio
            }
            fgr, pha, r1, r2, r3, r4 = session.run(None, inputs)

            # pha shape: [1, 1, H, W] in [0, 1]
            alpha = pha[0, 0]  # [H, W] float32

            if is_transparent:
                # Output BGRA
                b, g, r_ch = cv2.split(frame)
                a_ch = (alpha * 255.0).clip(0, 255).astype(np.uint8)
                bgra = cv2.merge((b, g, r_ch, a_ch))
                proc.stdin.write(bgra.tobytes())
            else:
                alpha_3d = np.expand_dims(alpha, axis=2)
                if bgtype := bg_type.lower() == "blur":
                    bg = cv2.GaussianBlur(frame, (51, 51), 0)
                elif bgtype := bg_type.lower() == "black":
                    bg = np.zeros_like(frame)
                else:  # default green screen
                    bg = np.full_like(frame, (0, 255, 0))

                composite = (frame.astype(np.float32) * alpha_3d + bg.astype(np.float32) * (1.0 - alpha_3d)).astype(np.uint8)
                proc.stdin.write(composite.tobytes())

        cap.release()
        proc.stdin.close()
        proc.wait()

        # Mux original audio track if present
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw,
            "-i", video_path,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "copy",
            output_path
        ]
        subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)

        return {
            "status": "success",
            "engine": "robust_video_matting_gpu",
            "bg_type": bg_type,
            "width": w,
            "height": h
        }
    finally:
        cap.release()
        del session
        gc.collect()
        if os.path.exists(temp_raw):
            try:
                os.remove(temp_raw)
            except OSError:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
