<p align="center">
  <img src="static/brand-mark.svg" alt="App Logo" width="80">
</p>

<h1 align="center">🎵 Audio, Video & Image Studio</h1>

<p align="center">
  <strong>A local, private, all-in-one multimedia editor with built-in AI assistant — no subscriptions, no cloud uploads, no accounts required</strong><br>
  Edit audio • Edit video • Edit images • AI-assisted workflows • Transcribe • Denoise • Separate stems • All offline, on your own machine
</p>

<p align="center">
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/stargazers"><img src="https://img.shields.io/github/stars/ArPaN-DS/Audio_Cutter?style=for-the-badge&logo=github&color=FFD700&labelColor=1a1a2e" alt="GitHub Stars"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/network/members"><img src="https://img.shields.io/github/forks/ArPaN-DS/Audio_Cutter?style=for-the-badge&logo=github&color=4CAF50&labelColor=1a1a2e" alt="GitHub Forks"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/issues"><img src="https://img.shields.io/github/issues/ArPaN-DS/Audio_Cutter?style=for-the-badge&color=FF6B6B&labelColor=1a1a2e" alt="Open Issues"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ArPaN-DS/Audio_Cutter?style=for-the-badge&color=blue&labelColor=1a1a2e" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python 3.11+">
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/actions"><img src="https://img.shields.io/github/actions/workflow/status/ArPaN-DS/Audio_Cutter/ci.yml?style=for-the-badge&label=CI&labelColor=1a1a2e" alt="CI Status"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/commits/main"><img src="https://img.shields.io/github/last-commit/ArPaN-DS/Audio_Cutter?style=for-the-badge&labelColor=1a1a2e" alt="Last Commit"></a>
</p>

---

## ✨ Why This App?

> **100% Private. Zero Cloud. No Subscriptions.**
> Your files never leave your machine — everything is processed locally on your own server.

| 🔒 Privacy-First | 🤖 AI-Assisted | ⚡ Feature-Rich | 🌐 Browser-Based |
|:---:|:---:|:---:|:---:|
| All files stay on your server — never sent to cloud APIs | Built-in AI assistant for workflow help; local models for transcription, denoising, stem separation | Four dedicated editors (audio/video/image) + quick tools + manual controls | Works on any device with a browser — no app install |

---

## 📋 What's Inside

### 🎙️ Audio Editor (`/audio`)
Edit, cut, and enhance audio with waveform precision:
- Multi-region cutting with drag-and-drop
- Audio effects (fade, normalize, reverse)
- Real-time audio analysis (silence detection, BPM detection)
- Speech-to-text transcription (on-device)
- Noise reduction and voice isolation
- Export to MP3 (320kbps) or WAV
- Full undo/redo history
- Keyboard shortcuts and responsive design

### 🎬 Video Editor (`/video`)
Build professional videos from a timeline:
- Multi-clip timeline with drag-to-reorder
- Per-clip speed control (0.25×–4×)
- Per-clip audio control (volume, fade, mute)
- Text overlays with custom styling and 9-point positioning
- Filters and transforms (brightness, saturation, B&W, sepia, rotate)
- Fade-through-black transitions
- Canvas presets (original, 16:9, 9:16, 1:1, 720p)
- Quick tools (extract audio, make GIF, compress, convert, grab frame, mute)
- Export to MP4 or audio as MP3/WAV

### 🖼️ Image Editor (`/image`)
Full-featured image editing, all in your browser (zero upload):
- Crop & transform (ratio presets, rotate, flip)
- Adjustments (brightness, contrast, saturation, warmth, blur)
- Filter presets (B&W, sepia, vintage, cool, warm, vivid, invert)
- Text & meme text (multiple draggable layers, outline support)
- Drawing tools (brush, shapes, arrows, any color/size)
- Paste from clipboard
- Real super-resolution upscale (2×/4× via learned models)
- Undo/redo history
- Export as PNG, JPG, or WebP

### 🤖 AI Assistant
An interactive AI assistant available throughout the app to help with:
- Workflow guidance and best practices
- Feature explanations and shortcuts
- Custom editing suggestions
- Memory of conversation context (local only)

---

## ⚡ Quick Start (Windows)

### Prerequisites
- **Python 3.11+** → [Download](https://www.python.org/downloads/)
- **FFmpeg** → [Download](https://ffmpeg.org/download.html) *(optional, but recommended for video/audio features)*

### Installation & Run

```bash
# Clone the repository
git clone https://github.com/ArPaN-DS/Audio_Cutter.git
cd Audio_Cutter

# Option A: Automatic (Windows)
# Simply double-click run.bat
# It will set up the venv, install dependencies, and start the server

# Option B: Manual
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and set your FLASK_SECRET_KEY

# 4. (Optional) Download AI models for offline use
python download_models.py

# 5. Run the app
python app.py
```

### 🎉 Open → [http://localhost:5000](http://localhost:5000)

> **📖 Need more help?** See the [full setup guide](docs/SETUP.md)

---

## 📦 Optional Model Packs

### Speech-to-Text & Image Super-Resolution
These are **auto-downloaded** on first use, but you can pre-fetch them:

```bash
python download_models.py
```

### Natural Voiceover (Text-to-Speech)
Optional TTS pack (~350 MB) — install explicitly:

```bash
python download_models.py --voice       # Full voiceover pack
python download_models.py --voice-lite  # Compact variant
```

### Studio Stems (Vocal/Instrument Separation)
Optional neural separator (~84 MB + weights) — install explicitly:

```bash
python download_models.py --stems       # Core separator
python download_models.py --stems-fine  # Plus 4-model ensemble
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_SECRET_KEY` | *(required)* | Session secret — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_PORT` | `5000` | Server port |
| `APP_PRODUCT_NAME` | `"Studio"` | Display name for the app (configured at runtime) |
| `APP_ASSISTANT_NAME` | `"Assistant"` | Display name for the AI helper (configured at runtime) |
| `DB_PASSWORD` | — | PostgreSQL password (optional logging backend) |

---

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Python e2e tests
python e2e_full_automated_test.py

# Individual component tests
python test_media_dsp_correctness.py
python test_platform_hardening.py
python test_reasoning_models.py
python test_identity_guard.py

# Agent evaluation (if offline reasoning is enabled)
python agent_orchestration_eval.py
python agent_skills_eval.py
python slm_eval.py

# JavaScript tests (frontend)
node agent_chat_robustness_test.js
node agent_session_test.js
node agent_skills_ui_test.js
node image_studio_test.js
node local_voice_input_test.js
node studio_agent_test.js
node studio_playback_test.js
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, Flask 3.x |
| **Media processing** | FFmpeg (installed separately) plus local audio, video and image processing |
| **AI features** | Local, on-device models only (optional packs via `download_models.py`) |
| **Frontend** | HTML5, CSS3, vanilla JavaScript |
| **Logging** | PostgreSQL via psycopg2 (optional) |

---

## 📁 Project Structure

```
Audio_Cutter/
├── app.py                      # Flask backend — routes & request handling
├── ai_processor.py             # Local AI processing (silence, beats, denoise, transcription)
├── model_manager.py            # AI model lifecycle & resource governance
├── agent_*.py                  # AI assistant modules (orchestration, memory, planner)
├── slm_*.py                    # Local reasoning model (if enabled)
├── {audio,video,image}_processor.py  # Specialized editors
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── download_models.py          # Optional model pack installer
├── THIRD_PARTY_NOTICES.md      # Third-party licenses
│
├── static/
│   ├── style.css, *.css        # Design system (glassmorphism)
│   ├── script.js, *.js         # Frontend logic
│   ├── vendor/                 # Vendored libraries (Font Awesome, JSZip, wavesurfer)
│   └── brand-mark.svg          # App logo
│
├── templates/
│   ├── index.html              # Landing/dashboard
│   ├── audio.html, video.html, image.html  # Editor templates
│   └── studio.html             # Unified editor interface
│
├── docs/
│   ├── SETUP.md                # Detailed setup guide
│   ├── STORAGE.md              # Data storage & privacy policy
│   └── ASSISTANT.md              # AI assistant guide (if applicable)
│
├── perf/
│   └── bench.py, memprobe.py   # Performance benchmarking
│
├── uploads/, processed/, logs/ # Runtime data (auto-managed)
└── models/                     # Downloaded AI models (local cache)
```

---

## 🔧 Environment & Dependencies

### Minimal Install
```bash
pip install -r requirements.txt
```

All core features work without optional packages. Advanced features gracefully degrade to fallbacks:
- No neural voice-cleanup pack? → Uses spectral noise reduction
- No stems pack? → Uses fast signal-processing separation
- No FFmpeg? → Audio editor still works; video features are limited
- No voice pack? → Voiceover uses the operating system's built-in voices

### Advanced Options
For offline reasoning, transcription, or stem separation, see [SETUP.md](docs/SETUP.md).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Third-party licenses are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 🤝 Contributing

Contributions are welcome! Whether it's a bug fix, feature, or documentation improvement:

1. Fork the repo
2. Create your branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add awesome feature"`
4. Push & open a Pull Request

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for full guidelines.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ArPaN-DS"><strong>ArPaN-DS</strong></a>
  <br><br>
  If this project helped you, please consider giving it a ⭐
</p>
