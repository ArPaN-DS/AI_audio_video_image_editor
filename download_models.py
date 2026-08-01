"""
Download all AI models (Super-Resolution & Whisper Speech-to-Text)
used by the Audio & Image editors for 100% offline usage.

Models are saved into ./models/.

Run:  python download_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

IMAGE_MODELS = {
    # Fast, tiny — near-instant on CPU
    "FSRCNN_x2.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    "FSRCNN_x4.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
    # Best quality (sharpest text/edges) — larger + slower on CPU
    "EDSR_x2.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
    "EDSR_x4.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
}

WHISPER_MODELS = ["large-v3-turbo", "medium", "small"]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("==========================================================")
    print(" [Image] Pre-downloading Image Super-Resolution AI Models")
    print("==========================================================")
    for name, url in IMAGE_MODELS.items():
        dst = os.path.join(MODELS_DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  [OK] {name} already present ({os.path.getsize(dst) // 1024} KB)")
            continue
        print(f"  [>] downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  [OK] saved {name} ({os.path.getsize(dst) // 1024} KB)")
        except Exception as e:
            print(f"  [X] FAILED {name}: {e}")

    print("\n==========================================================")
    print(" [STT] Pre-downloading Speech-to-Text Whisper AI Models")
    print("==========================================================")
    try:
        from faster_whisper import WhisperModel
        for model_name in WHISPER_MODELS:
            print(f"  [>] Pre-fetching Whisper '{model_name}' for offline use...")
            try:
                # Pre-download weights into ./models/
                WhisperModel(model_name, device="cpu", compute_type="int8", download_root=MODELS_DIR)
                print(f"  [OK] {model_name} ready!")
            except Exception as e:
                print(f"  [X] FAILED {model_name}: {e}")
    except ImportError:
        print("  [X] faster_whisper not installed in current environment.")

    print("\nDone! All AI models are pre-downloaded for 100% offline use.")


if __name__ == "__main__":
    main()
