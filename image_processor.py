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

EDSR_MAX_INPUT_PIXELS = 1_500_000       # ~1224x1224
MAX_OUTPUT_PIXELS = 64_000_000          # safety ceiling on the result


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


def enhance_photo_clarity(src, out_path, denoise_strength=5, sharpen_strength=1.2):
    """
    Applies real-time AI-style photo clarity enhancement:
    - Removes JPEG compression noise & grain (Bilateral Filtering)
    - Enhances micro-contrast & edge sharpness (Unsharp Masking)
    - Auto-corrects dynamic range and histogram contrast (CLAHE)
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            img_pil = Image.open(src).convert("RGB")
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 1. Bilateral Denoise (preserves sharp edges while smoothing flat noise/grain)
        if denoise_strength > 0:
            denoised = cv2.bilateralFilter(img, d=5, sigmaColor=denoise_strength * 10, sigmaSpace=5)
        else:
            denoised = img

        # 2. Detail Sharpening & Micro-contrast
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.0 + sharpen_strength, gaussian, -sharpen_strength, 0)

        # 3. Dynamic Range & Contrast Polish (CLAHE on L-channel of LAB color space)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_opt = clahe.apply(l)
        enhanced_lab = cv2.merge((l_opt, a, b))
        final_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        cv2.imwrite(out_path, final_img)
        return {"status": "success", "engine": "clarity_enhancer"}
    except Exception as e:
        # Fallback using PIL
        pil_img = Image.open(src).convert("RGB")
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
        pil_img.save(out_path, quality=95)
        return {"status": "success", "engine": "pil_fallback"}


def remove_bg(src_path, out_path, model_name="isnet-general-use", alpha_matting=True):
    """
    Strips background from image using rembg (ONNX model).
    Wrapped in global_model_manager for zero-idle memory lifecycle.
    model_name: 'isnet-general-use' (Ultra/HD Detail), 'u2net' (Studio), 'u2net_human_seg' (Portrait), 'u2netp' (Fast)
    """
    try:
        from rembg import remove, new_session
        from PIL import Image, ImageFilter
        from model_manager import global_model_manager

        valid_models = ("isnet-general-use", "u2net", "u2net_human_seg", "u2netp")
        if model_name not in valid_models:
            model_name = "isnet-general-use"

        def _load_rembg():
            session = new_session(model_name)
            return session, {"engine": f"rembg_{model_name}"}

        def _unload_rembg(session):
            del session

        model_id = f"rembg_session_{model_name}"
        with global_model_manager.session(model_id, _load_rembg, _unload_rembg) as (session, meta):
            input_img = Image.open(src_path)
            
            if alpha_matting:
                try:
                    output_img = remove(
                        input_img,
                        session=session,
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10,
                        post_process_mask=True
                    )
                except Exception:
                    output_img = remove(input_img, session=session, post_process_mask=True)
            else:
                output_img = remove(input_img, session=session, post_process_mask=True)

            # ── ALPHA EDGE ANTI-ALIASING PASS ──
            if output_img.mode == "RGBA":
                r, g, b, a = output_img.split()
                a_blurred = a.filter(ImageFilter.GaussianBlur(radius=0.6))
                import numpy as np
                a_arr = np.array(a, dtype=np.uint8)
                ab_arr = np.array(a_blurred, dtype=np.uint8)
                # Anti-alias boundary pixels to eliminate jagged stair-stepping
                edge_mask = (a_arr > 0) & (a_arr < 255)
                a_arr[edge_mask] = ab_arr[edge_mask]
                smooth_a = Image.fromarray(a_arr)
                output_img = Image.merge("RGBA", (r, g, b, smooth_a))

            output_img.save(out_path, format="PNG")
            return {"engine": f"rembg_{model_name}", "status": "success"}
    except Exception as e:
        raise RuntimeError(f"AI Background Removal failed: {str(e)}")


def inpaint_object(src_path, mask_path, out_path, radius=3, method="telea"):
    """
    AI Magic Eraser / Object Inpainting: Replaces brushed mask regions with background textures.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError("Could not load image or mask file.")

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, bin_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        result = cv2.inpaint(img, bin_mask, inpaintRadius=radius, flags=inpaint_flag)

        cv2.imwrite(out_path, result)
        return {"status": "success", "engine": f"opencv_inpaint_{method}"}
    except Exception as e:
        raise RuntimeError(f"Magic Eraser Inpainting failed: {str(e)}")


def restore_faces(src_path, out_path):
    """
    AI Portrait & Face Detail Restorer: Detects faces and applies high-frequency detail restoration,
    skin bilateral smoothing, and eye contrast sharpening.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        result = img.copy()

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img.shape[1], x + w + pad_w)
                y2 = min(img.shape[0], y + h + pad_h)

                face_roi = result[y1:y2, x1:x2]
                smooth_face = cv2.bilateralFilter(face_roi, d=7, sigmaColor=35, sigmaSpace=7)
                blur = cv2.GaussianBlur(smooth_face, (0, 0), 2.5)
                sharp_face = cv2.addWeighted(smooth_face, 1.4, blur, -0.4, 0)
                
                lab = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
                l_opt = clahe.apply(l)
                enhanced_face = cv2.cvtColor(cv2.merge((l_opt, a, b)), cv2.COLOR_LAB2BGR)

                result[y1:y2, x1:x2] = enhanced_face
        else:
            smooth = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=5)
            blur = cv2.GaussianBlur(smooth, (0, 0), 2)
            result = cv2.addWeighted(smooth, 1.3, blur, -0.3, 0)

        cv2.imwrite(out_path, result)
        return {"status": "success", "faces_found": len(faces), "engine": "opencv_face_restorer"}
    except Exception as e:
        raise RuntimeError(f"Face Restoration failed: {str(e)}")
