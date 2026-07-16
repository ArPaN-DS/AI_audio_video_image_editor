"""
Download the super-resolution models used by the Image Editor's
"Increase Quality" (local AI upscale) feature.

Models are saved into ./models/. They're git-ignored because the EDSR
files are large (~38 MB each). If they're missing, the app still works —
it falls back to high-quality Lanczos upscaling automatically.

Run:  python download_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    # Fast, tiny — near-instant on CPU
    "FSRCNN_x2.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    "FSRCNN_x4.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
    # Best quality (sharpest text/edges) — larger + slower on CPU
    "EDSR_x2.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
    "EDSR_x4.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        dst = os.path.join(MODELS_DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  ✓ {name} already present ({os.path.getsize(dst) // 1024} KB)")
            continue
        print(f"  ↓ downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  ✓ saved {name} ({os.path.getsize(dst) // 1024} KB)")
        except Exception as e:
            print(f"  ✗ FAILED {name}: {e}")
    print("\nDone. If any failed, the app will fall back to Lanczos upscaling.")


if __name__ == "__main__":
    main()
