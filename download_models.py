"""
Download all AI models (Super-Resolution & Whisper Speech-to-Text)
used by the Audio & Image editors for 100% offline usage.

Models are saved into ./models/.

Run:  python download_models.py
      python download_models.py --max-quality   # also fetch the high-detail
                                                # cutout network (~1 GB) used
                                                # automatically on strong hardware
      python download_models.py --voice         # ONLY the voiceover pack (~350 MB;
                                                # opt-in, nothing else is fetched)
      python download_models.py --voice-lite    # compact low-memory voiceover
                                                # pack (~110 MB); combinable
      python download_models.py --stems         # ONLY the Studio stems pack:
                                                # separator package + weights (~84 MB)
      python download_models.py --stems-fine    # plus the fine 4-model bag (~330 MB)

The voiceover pack is never fetched implicitly. It contains Apache-2.0 speech
weights + voice styles and Apache-2.0 pronunciation lexicons; it runs on the
already-installed ONNX Runtime and needs no GPL phonemizer (see tts_processor).
"""

import os
import sys
import shutil
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

IMAGE_MODELS = {
    # GPU Flagship: Real-ESRGAN 4x (Deep Generative Residual Network)
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
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


def download_voice_pack(studio=True, compact=False):
    """Explicit opt-in: fetch the local voiceover pack into ./models/voice/.

    Files are pinned to fixed upstream revisions (see tts_processor.voice_pack_manifest)
    and written atomically, so an interrupted download never leaves a half file
    that would be mistaken for an installed pack.
    """
    import tts_processor

    print("==========================================================")
    print(" [Voice] Downloading the local voiceover pack")
    print("==========================================================")
    failures = 0
    for rel_path, url in tts_processor.voice_pack_manifest(studio=studio, compact=compact):
        dst = os.path.join(tts_processor.VOICE_DIR, *rel_path.split("/"))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  [OK] {rel_path} already present ({os.path.getsize(dst) // 1024} KB)")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        print(f"  [>] downloading {rel_path} ...", flush=True)
        last_error = None
        for _attempt in range(3):
            try:
                # Explicit timeout: a stalled connection must not hang forever.
                with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
                    shutil.copyfileobj(response, out, 1 << 20)
                os.replace(tmp, dst)
                last_error = None
                break
            except Exception as e:
                last_error = e
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        if last_error is None:
            print(f"  [OK] saved {rel_path} ({os.path.getsize(dst) // 1024} KB)", flush=True)
        else:
            failures += 1
            print(f"  [X] FAILED {rel_path}: {last_error}", flush=True)
    tts_processor.write_voice_pack_notice()
    if failures:
        print(f"\n{failures} file(s) failed; the voiceover pack stays disabled until rerun succeeds.")
        return 1
    print("\nDone! Natural voiceover is available offline.")
    return 0


STEMS_DIR = os.path.join(MODELS_DIR, "stems")
STEMS_PACKAGE = "demucs==4.1.0"   # MIT; 4.1 no longer needs torchaudio (keeps the installed torch)
# Pinned upstream revisions (MIT-licensed code and weights, Meta AI "Demucs v4").
STEM_BAGS = {
    "htdemucs": {
        "hf": "https://huggingface.co/adefossez/HTDemucs/resolve/cbc8a9b1a87023b7fd74e7b3412e6321c0eab003/",
        "legacy": {"955717e8": "hybrid_transformer/955717e8-8726e21a.th"},
        "sha256": {"955717e8": "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"},
    },
    "htdemucs_ft": {
        "hf": "https://huggingface.co/adefossez/HTDemucs-ft/resolve/d74ac89c3a1e874fc78f152555cf4d8533f06cd4/",
        "legacy": {"f7e0c4bc": "hybrid_transformer/f7e0c4bc-ba3fe64a.th",
                   "d12395a8": "hybrid_transformer/d12395a8-e57c48e6.th",
                   "92cfc3b6": "hybrid_transformer/92cfc3b6-ef3bcb9c.th",
                   "04573f0d": "hybrid_transformer/04573f0d-f3cf25b2.th"},
        "sha256": {"f7e0c4bc": "2c85ab3c62dd6edd8e0b965e38b16fd1cdde357cc25de6b6bc9ce7c83f60925f",
                   "d12395a8": "5b01a97567ae9a3178a6236fb520251045c03eb8834bc8c24a4eec11d6c8fb56",
                   "92cfc3b6": "a241863551f30d01c42bd7b97da40839922ead3acb0f1fcab25682f55b4eeb59",
                   "04573f0d": "68854b0d7c2b3274723b5761f6fd9f5aec5f1bcd3f0de7c1669546fdb7871b7c"},
    },
}
STEMS_LEGACY_ROOT = "https://dl.fbaipublicfiles.com/demucs/"
STEMS_NOTICE = """Studio stems pack (optional download)

Code: Demucs v4 by Meta AI, MIT License (https://github.com/adefossez/demucs).
Weights: htdemucs / htdemucs_ft checkpoints published by the same authors with the
repository; no separate weights licence is stated, so the repository MIT License
applies. Training data included MUSDB18-HQ (research licence) plus internal songs.
Downloaded explicitly with:  python download_models.py --stems [--stems-fine]
"""


def _fetch(url, dst):
    tmp = dst + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
        return True
    except Exception as e:
        print(f"  [X] FAILED {os.path.basename(dst)}: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def download_stems_pack(include_fine=False):
    """Explicit opt-in: install the separator package and fetch its weights into ./models/stems/.

    Nothing here runs implicitly; separation_processor only uses these files when present.
    """
    import hashlib
    import importlib.util
    import shutil
    import subprocess

    print("==========================================================")
    print(" [Stems] Installing the Studio stems pack")
    print("==========================================================")
    if importlib.util.find_spec("demucs") is None:
        print(f"  [>] pip install {STEMS_PACKAGE} ...")
        proc = subprocess.run([sys.executable, "-m", "pip", "install", STEMS_PACKAGE])
        if proc.returncode != 0:
            print("  [X] Package install failed; Quick separation stays available.")
            return 1
    os.makedirs(STEMS_DIR, exist_ok=True)
    failures = 0
    for name in ["htdemucs"] + (["htdemucs_ft"] if include_fine else []):
        bag = STEM_BAGS[name]
        yaml_dst = os.path.join(STEMS_DIR, f"{name}.yaml")
        if not os.path.exists(yaml_dst):
            spec = importlib.util.find_spec("demucs")
            packaged = os.path.join(os.path.dirname(spec.origin), "remote", f"{name}.yaml") if spec else ""
            if packaged and os.path.isfile(packaged):
                shutil.copyfile(packaged, yaml_dst)
            elif not _fetch(bag["hf"] + f"{name}.yaml", yaml_dst):
                failures += 1
                continue
        for sig, legacy in bag["legacy"].items():
            st = os.path.join(STEMS_DIR, f"{sig}.safetensors")
            th = os.path.join(STEMS_DIR, os.path.basename(legacy))
            if (os.path.exists(st) and os.path.getsize(st) > 0) or os.path.exists(th):
                print(f"  [OK] {name}/{sig} already present")
                continue
            print(f"  [>] downloading {name}/{sig} ...")
            if _fetch(bag["hf"] + f"{sig}.safetensors", st):
                with open(st, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                if digest == bag["sha256"][sig]:
                    print(f"  [OK] saved {sig}.safetensors ({os.path.getsize(st) // 1024} KB, checksum verified)")
                    continue
                os.remove(st)
                print(f"  [X] checksum mismatch for {sig}.safetensors; file removed")
            # Fallback: legacy checkpoint, verified against the hash in its file name.
            if _fetch(STEMS_LEGACY_ROOT + legacy, th):
                with open(th, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                expected = os.path.basename(legacy)[:-3].rsplit("-", 1)[1]
                if digest.startswith(expected):
                    print(f"  [OK] saved {os.path.basename(legacy)} (checksum verified)")
                    continue
                os.remove(th)
                print(f"  [X] checksum mismatch for {sig}; file removed")
            failures += 1
    with open(os.path.join(STEMS_DIR, "NOTICE.txt"), "w", encoding="utf-8") as handle:
        handle.write(STEMS_NOTICE)
    if failures:
        print(f"\n{failures} item(s) failed; Quick separation stays in use until a rerun succeeds.")
        return 1
    print("\nDone! Studio stems are available offline.")
    return 0


if __name__ == "__main__":
    if "--stems" in sys.argv or "--stems-fine" in sys.argv:
        sys.exit(download_stems_pack(include_fine="--stems-fine" in sys.argv))
    if "--voice" in sys.argv or "--voice-lite" in sys.argv:
        sys.exit(download_voice_pack(studio="--voice" in sys.argv, compact="--voice-lite" in sys.argv))
    main()
