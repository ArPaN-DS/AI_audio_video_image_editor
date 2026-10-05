# Third-Party Notices

This project includes the following third-party components and their licenses.

## Frontend Assets (Vendored)

### Font Awesome Free 6.x
- **Location**: `static/vendor/fontawesome/`
- **License**: Multiple licenses depending on file type
  - **Icons (SVG/JS)**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
  - **Fonts (Web/Desktop)**: SIL OFL 1.1 (SIL Open Font License)
  - **Code**: MIT License
- **Copyright**: Copyright (c) 2024 Fonticons, Inc. (https://fontawesome.com)
- **URL**: https://fontawesome.com/license/free

### JSZip
- **Location**: `static/vendor/jszip/`
- **License**: Dual MIT / GPLv3 (we use it under the MIT license)
- **Copyright**: Copyright (c) 2009-2016 Stuart Knightley, David Duponchel, Franz Buchinger, António Afonso
- **URL**: https://github.com/Stuk/jszip

### wavesurfer.js
- **Location**: `static/vendor/wavesurfer/`
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2012-2023, katspaugh and contributors
- **URL**: https://wavesurfer.xyz/

## Optional Model Packs (Downloaded via `download_models.py`)

### Studio Stems Pack (Optional)
**Installation**: `python download_models.py --stems [--stems-fine]`

- **Code & Weights**: Demucs v4 by Meta AI
  - **License**: MIT License
  - **URL**: https://github.com/adefossez/demucs
  - **Details**: HTDemucs and HTDemucs-ft checkpoints. Training data included MUSDB18-HQ (research license) plus internal songs.

### Natural Voiceover Pack (Optional)
**Installation**: `python download_models.py --voice [--voice-lite]`

- **Speech Synthesis Weights & Lexicons**: Apache License 2.0
  - **Runtime**: ONNX Runtime (MIT License)
  - **Details**: Weights and phonetic lexicons for text-to-speech synthesis, fetched only explicitly via the download command. Uses Apache-2.0 phoneme lexicons (no GPL dependency).

## Python Dependencies

For detailed version and license information, see `requirements.txt` and run:

```bash
pip-licenses
```

Key dependencies include:
- **Flask**: BSD 3-Clause License
- **Pillow (PIL)**: HPND License
- **OpenCV** (opencv-contrib-python): Apache 2.0 License
- **librosa**: ISC License
- **pydub**: MIT License
- **NumPy**: BSD License
- **scipy**: BSD License
- **faster-whisper**: MIT License
- **ONNX Runtime**: MIT License
- **rembg**: MIT License
- **scenedetect**: BSD License
- **PyAV (`av`)**: PyAV wheels bundle FFmpeg binaries licensed under LGPL v2.1+ or GPL v2+/v3 (depending on whether GPL filters such as `x264` or `postproc` are compiled in). This is relevant if packaging a standalone binary installer: ensure LGPL compliance or provide full source/license terms if redistributing a GPL build.
- **FFmpeg**: Executable / library used for media decoding and muxing. Under LGPL v2.1+ by default, or GPL v2+/v3 when compiled with GPL flags (`--enable-gpl`). Standalone distribution must respect corresponding redistribution terms.

## AI Models (Auto-Downloaded on First Use)

### Image Super-Resolution Models (Optional)
**Auto-fetched by**: Image editor's "Increase Quality" feature

- **FSRCNN**: https://github.com/Saafke/FSRCNN_Tensorflow (TensorFlow Research Model)
- **EDSR**: https://github.com/Saafke/EDSR_Tensorflow (TensorFlow Research Model)

### Speech-to-Text Model (Auto-Downloaded)
**Auto-fetched by**: `download_models.py` or first-use of transcription

- **Whisper**: OpenAI Whisper (Model: large-v3-turbo, medium, small)
  - **License**: MIT License
  - **URL**: https://github.com/openai/whisper

---

**Last Updated**: October 2026
