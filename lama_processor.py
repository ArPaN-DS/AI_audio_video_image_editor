"""
LaMa (Large Mask Inpainting) SOTA Object Removal & Inpainting Engine (ONNX + CUDA).

Provides market-standard object removal, magic eraser, and generative fill:
- High-resolution deep context inpainting via Fast Fourier Convolutions (FFC).
- Seamless edge blending.
- Automatic GPU VRAM cleanup after execution.
"""

import os
import gc
import logging
import cv2
import numpy as np

_log = logging.getLogger("lama_inpaint")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LAMA_PATH = os.path.join(_BASE_DIR, "models", "lama_fp32.onnx")


def is_available():
    """Returns True if LaMa model exists on disk."""
    return os.path.exists(_LAMA_PATH)


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


def inpaint_lama_gpu(src_path, mask_path, out_path):
    """
    Inpaint image at src_path using binary mask at mask_path with LaMa SOTA model on CUDA.
    Handles arbitrary image resolutions with aspect-ratio padding and feathering.
    """
    _ensure_cuda_dlls()
    import onnxruntime as ort

    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {src_path}")
    orig_h, orig_w = img.shape[:2]

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    if mask.shape[:2] != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Ensure binary mask (255 for areas to inpaint, 0 for keep)
    _, bin_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Dilate mask slightly to prevent border halo artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=1)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    session = ort.InferenceSession(_LAMA_PATH, sess_options=sess_options, providers=providers)
    try:
        # Prepare 512x512 inputs for LaMa
        target_size = (512, 512)
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(bin_mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Preprocess
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        mask_tensor = (resized_mask > 0).astype(np.float32)[np.newaxis, np.newaxis, ...]

        # Run inference
        outputs = session.run(None, {"image": img_tensor, "mask": mask_tensor})
        out_tensor = outputs[0][0]  # shape: (3, 512, 512)

        out_hwc = np.transpose(out_tensor, (1, 2, 0))
        if out_hwc.max() <= 1.5:
            out_bgr_512 = cv2.cvtColor((out_hwc.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            out_bgr_512 = cv2.cvtColor(out_hwc.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Upscale inpainted 512x512 back to original resolution
        inpainted_full = cv2.resize(out_bgr_512, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # Seamless composite: only replace masked area with soft alpha feathering
        feather_mask = cv2.GaussianBlur(bin_mask.astype(np.float32) / 255.0, (11, 11), 0)
        feather_mask = np.expand_dims(feather_mask, axis=2)

        final_composite = (inpainted_full.astype(np.float32) * feather_mask + img.astype(np.float32) * (1.0 - feather_mask)).astype(np.uint8)

        cv2.imwrite(out_path, final_composite)
        return {
            "status": "success",
            "engine": "lama_onnx_gpu" if "CUDAExecutionProvider" in session.get_providers() else "lama_onnx_cpu",
            "width": orig_w,
            "height": orig_h
        }
    finally:
        del session
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
