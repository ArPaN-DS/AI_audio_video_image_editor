"""
Image Processor — real, local super-resolution ("Increase Quality").

Uses OpenCV's dnn_superres with pretrained learned models (FSRCNN / EDSR)
to genuinely reconstruct pixels — the same class of upscaling that online
platforms sell behind a subscription — but running 100% on your own CPU.

Falls back to high-quality Lanczos + unsharp mask if a model is missing.
"""

import os
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# model key -> (opencv model name, {scale: filename})
MODELS = {
    "fast": ("fsrcnn", {2: "FSRCNN_x2.pb", 4: "FSRCNN_x4.pb"}),
    "best": ("edsr", {2: "EDSR_x2.pb", 4: "EDSR_x4.pb"}),
}

# EDSR is a deep net and slow on CPU (~12s for a tiny 320x240 image), so cap
# the input size: above this we transparently downgrade to the fast model.
EDSR_MAX_INPUT_PIXELS = 250_000         # ~500x500
MAX_OUTPUT_PIXELS = 40_000_000          # safety ceiling on the result


def _model_path(model_key, scale):
    entry = MODELS.get(model_key)
    if not entry:
        return None, None
    cv_name, files = entry
    fname = files.get(scale)
    if not fname:
        return None, None
    path = os.path.join(MODELS_DIR, fname)
    return (cv_name, path) if os.path.exists(path) else (cv_name, None)


def _lanczos_upscale(src, out_path, scale):
    """Fallback: real resolution increase via Lanczos + unsharp mask."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=110, threshold=2))
    img.save(out_path, quality=95)
    return {"engine": "lanczos", "scale": scale, "size": img.size}


def upscale(src, out_path, scale=2, model_key="fast"):
    """
    Increase image resolution by `scale` (2 or 4) using a learned SR model.

    Returns a dict describing what actually ran (engine may differ from the
    request if we had to downgrade for performance or missing models).
    """
    if scale not in (2, 4):
        scale = 2

    try:
        import cv2
    except ImportError:
        return _lanczos_upscale(src, out_path, scale)

    if not hasattr(cv2, "dnn_superres"):
        return _lanczos_upscale(src, out_path, scale)

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return _lanczos_upscale(src, out_path, scale)

    h, w = img.shape[:2]
    in_pixels = w * h

    # Downgrade EDSR -> FSRCNN when the input is too big for a CPU render.
    if model_key == "best" and in_pixels > EDSR_MAX_INPUT_PIXELS:
        model_key = "fast"
        downgraded = True
    else:
        downgraded = False

    # Guard against an absurdly large output.
    if in_pixels * scale * scale > MAX_OUTPUT_PIXELS:
        if scale == 4:
            scale = 2
        if in_pixels * scale * scale > MAX_OUTPUT_PIXELS:
            return _lanczos_upscale(src, out_path, scale)

    cv_name, model_path = _model_path(model_key, scale)
    if not model_path:
        # Requested model file missing — try the other one before Lanczos.
        alt = "fast" if model_key == "best" else "best"
        cv_name, model_path = _model_path(alt, scale)
        model_key = alt
        if not model_path:
            return _lanczos_upscale(src, out_path, scale)

    try:
        from model_manager import global_model_manager

        def _load_sr():
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel(cv_name, scale)
            return sr, {"engine": cv_name, "scale": scale}

        def _unload_sr(sr_instance):
            del sr_instance

        model_id = f"image_superres_{cv_name}_x{scale}"
        with global_model_manager.session(model_id, _load_sr, _unload_sr) as (sr, meta):
            result = sr.upsample(img)
            ext = os.path.splitext(out_path)[1].lower()
            params = ([cv2.IMWRITE_JPEG_QUALITY, 95]
                      if ext in (".jpg", ".jpeg") else [])
            cv2.imwrite(out_path, result, params)
            rh, rw = result.shape[:2]
            return {
                "engine": cv_name,
                "scale": scale,
                "size": (rw, rh),
                "downgraded": downgraded,
            }
    except Exception:
        return _lanczos_upscale(src, out_path, scale)



def available_engines():
    """Report which upscalers are usable right now (for the UI/health check)."""
    engines = {"lanczos": True}
    try:
        import cv2
        engines["opencv_superres"] = hasattr(cv2, "dnn_superres")
    except ImportError:
        engines["opencv_superres"] = False
    engines["fast_model"] = os.path.exists(os.path.join(MODELS_DIR, "FSRCNN_x4.pb"))
    engines["best_model"] = os.path.exists(os.path.join(MODELS_DIR, "EDSR_x4.pb"))
    return engines
