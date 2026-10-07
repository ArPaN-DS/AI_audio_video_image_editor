"""
Remini-Grade Generative Face & Portrait Restoration Engine (CodeFormer + GFPGAN v1.4).

Provides market-standard photorealistic facial reconstruction:
- Generates lifelike eye iris details, reflections, eyelashes, skin pores, and defined lips.
- Combines YuNet DNN face detection with CodeFormer (w=0.15 generative prior) and GFPGAN v1.4.
- Uses seamless feathered-edge blending to eliminate seam boundaries.
- Flushes GPU VRAM immediately after execution.
"""

import os
import gc
import logging
import cv2
import numpy as np

_log = logging.getLogger("codeformer")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CODEFORMER_PATH = os.path.join(_BASE_DIR, "models", "codeformer.onnx")
_GFPGAN_PATH = os.path.join(_BASE_DIR, "models", "GFPGANv1.4.onnx")
_YUNET_PATH = os.path.join(_BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")


def is_available():
    """Returns True if CodeFormer or GFPGAN weights exist on disk."""
    return os.path.exists(_CODEFORMER_PATH) or os.path.exists(_GFPGAN_PATH)


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


class FaceEnhanceInference:
    def __init__(self, use_gfpgan=False):
        _ensure_cuda_dlls()
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        
        self.use_gfpgan = use_gfpgan and os.path.exists(_GFPGAN_PATH)
        model_path = _GFPGAN_PATH if self.use_gfpgan else _CODEFORMER_PATH
        
        _log.info(f"[FaceEnhance] Loading model: {os.path.basename(model_path)} on providers: {providers}")
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.resolution = (512, 512)

    def enhance_crop(self, bgr_crop, fidelity=0.70):
        """
        Enhance cropped face ROI with identity-preserving balance.
        fidelity=0.70 preserves genuine facial anatomy, expressions, and identity
        while removing blur and compression artifacts. Lower values (e.g. 0.15)
        force aggressive hallucination which distorts non-frontal or expressive faces.
        """
        h, w_in = bgr_crop.shape[:2]
        resized = cv2.resize(bgr_crop, self.resolution, interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - 0.5) / 0.5
        x = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        if self.use_gfpgan:
            outputs = self.session.run(None, {"input": x})
        else:
            weight = np.array([fidelity], dtype=np.double)
            outputs = self.session.run(None, {"x": x, "w": weight})

        out_tensor = outputs[0][0]

        out_hwc = (np.transpose(out_tensor, (1, 2, 0)).clip(-1.0, 1.0) + 1.0) * 0.5
        out_rgb = (out_hwc * 255.0).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        return cv2.resize(out_bgr, (w_in, h), interpolation=cv2.INTER_LANCZOS4)


def restore_faces_gpu(src_path, out_path, fidelity=0.70, use_gfpgan=False):
    """
    Restore and reconstruct faces in the image.
    fidelity=0.70: Identity-preserving natural restoration without facial distortion.
    """
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {src_path}")
    h, w = img.shape[:2]

    # Detect faces with YuNet
    faces = None
    if os.path.exists(_YUNET_PATH):
        try:
            detector = cv2.FaceDetectorYN.create(
                model=_YUNET_PATH,
                config="",
                input_size=(w, h),
                score_threshold=0.45,
                nms_threshold=0.3,
                top_k=5000
            )
            _, faces = detector.detect(img)
        except Exception as e:
            _log.warning(f"YuNet face detection failed: {e}")

    # Fallback to Haar if YuNet finds nothing
    if faces is None or len(faces) == 0:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(detected) > 0:
                faces = [[x, y, fw, fh] for (x, y, fw, fh) in detected]
        except Exception:
            pass

    engine = FaceEnhanceInference(use_gfpgan=use_gfpgan)
    result = img.copy()

    try:
        if faces is not None and len(faces) > 0:
            for face in faces:
                fx, fy, fw, fh = map(int, face[:4])
                pad_w = int(fw * 0.35)
                pad_h = int(fh * 0.35)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(w, fx + fw + pad_w)
                y2 = min(h, fy + fh + pad_h)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                enhanced = engine.enhance_crop(crop, fidelity=fidelity)

                # Soft Gaussian feather blend for invisible seams
                crop_h, crop_w = crop.shape[:2]
                mask = np.ones((crop_h, crop_w), dtype=np.float32)
                feather = max(6, int(min(crop_w, crop_h) * 0.12))
                cv2.rectangle(mask, (0, 0), (crop_w, crop_h), 0.0, feather)
                mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather / 2)
                mask = np.expand_dims(mask, axis=2)

                blended = (enhanced.astype(np.float32) * mask + crop.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                result[y1:y2, x1:x2] = blended
        else:
            # If no face is detected, apply portrait-wide enhancement if compact
            if max(w, h) <= 1024:
                result = engine.enhance_crop(img, fidelity=fidelity)

        cv2.imwrite(out_path, result)
        return {
            "status": "success",
            "engine": "remini_gfpgan" if use_gfpgan else "remini_codeformer",
            "faces_detected": len(faces) if faces is not None else 0
        }
    finally:
        del engine
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
