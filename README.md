<p align="center">
  <img src="static/logo.png" alt="Audio Cutter Pro Logo" width="80">
</p>

<h1 align="center">🎵 Audio Cutter Pro <span>+ 🎬 Video + 🖼️ Image Editor</span></h1>

<p align="center">
  <strong>A professional-grade, browser-based audio, video &amp; image studio built with Flask, WaveSurfer.js &amp; FFmpeg</strong><br>
  Cut audio • Edit video • Retouch images • Add text &amp; music • Extract audio • Export — all from your browser. No signup. No cloud.
</p>

<p align="center">
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/stargazers"><img src="https://img.shields.io/github/stars/ArPaN-DS/Audio_Cutter?style=for-the-badge&logo=github&color=FFD700&labelColor=1a1a2e" alt="GitHub Stars"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/network/members"><img src="https://img.shields.io/github/forks/ArPaN-DS/Audio_Cutter?style=for-the-badge&logo=github&color=4CAF50&labelColor=1a1a2e" alt="GitHub Forks"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/issues"><img src="https://img.shields.io/github/issues/ArPaN-DS/Audio_Cutter?style=for-the-badge&color=FF6B6B&labelColor=1a1a2e" alt="Open Issues"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ArPaN-DS/Audio_Cutter?style=for-the-badge&color=blue&labelColor=1a1a2e" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python 3.9+">
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/actions"><img src="https://img.shields.io/github/actions/workflow/status/ArPaN-DS/Audio_Cutter/ci.yml?style=for-the-badge&label=CI&labelColor=1a1a2e" alt="CI Status"></a>
  <a href="https://github.com/ArPaN-DS/Audio_Cutter/commits/main"><img src="https://img.shields.io/github/last-commit/ArPaN-DS/Audio_Cutter?style=for-the-badge&labelColor=1a1a2e" alt="Last Commit"></a>
</p>

---

## ✨ Why Audio Cutter Pro?

> **No subscriptions. No uploads to third-party servers. No account required.**
> Your audio files never leave your machine — everything is processed locally on your own server.

| 🔒 Privacy-First | 🤖 Local AI Tools | ⚡ Feature-Rich | 🌐 Browser-Based |
|:---:|:---:|:---:|:---:|
| Files stay on your server — never sent to cloud APIs | Silence detection, Transcript generator, auto-trim, BPM, and noise reduction (Runs 100% on CPU, even on resource-constrained devices) | Multi-region cutting, effects, undo/redo, mic recording | Works on any device with a browser — no app install |

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🎯 **Multi-Region Cutting** | Create unlimited cut regions on the waveform — each independently adjustable |
| 🔊 **Audio Effects** | Fade In, Fade Out, Normalize (loudness leveling), Reverse |
| 📤 **Flexible Export** | Export as **MP3** (320kbps) or **WAV** — merged or as separate ZIP files |
| 🎙️ **Microphone Recording** | Record audio directly from your browser microphone |
| 🖱️ **Drag & Drop Upload** | Drop your audio or video file directly onto the page |
| ↩️ **Undo/Redo** | Full undo system (`Ctrl+Z`) for all region operations |
| ⌨️ **Keyboard Shortcuts** | `Space`, `Delete`, `Ctrl+Z`, `← / →`, `M`, `?` |
| 📱 **Fully Responsive** | Works on mobile, tablet, and desktop |
| 🎨 **Premium Design** | Glassmorphism UI — clean, modern, professional |
| 🤖 **AI-Powered Analysis** | Local silence detection, auto-trim, beat tracking (BPM), voice activity detection (VAD), **speech-to-text transcription** (requires `openai-whisper`), and local noise reduction — runs 100% on CPU |

---

## 🎬 Video Editor

A full **audio + video combo editor** lives at **`/video`** — powered entirely by your local FFmpeg install. Built to be a *one-person studio* that's still simple enough for a complete beginner.

**Multi-clip timeline (for full control):**

| Feature | Description |
|---------|-------------|
| 🎞️ **Multi-Clip Timeline** | Import multiple videos/audio, drag to reorder, trim clip edges, and split at the playhead |
| ⏩ **Speed Control** | 0.25× – 4× per clip (video *and* audio stay in sync) |
| 🔊 **Per-Clip Audio** | Volume, mute, fade in/out — mix video sound with a background music track |
| 🎨 **Filters & Transforms** | Brightness, contrast, saturation, B&W, sepia, rotate 90° |
| 🔤 **Text Overlays** | Add captions with custom size, color, 9-point positioning, and timing |
| 🔀 **Transitions** | Fade-through-black between clips |
| 📐 **Canvas Presets** | Original, 16:9, 9:16 (Shorts/Reels), 1:1 (square), 720p |
| 📤 **Flexible Export** | Render to **MP4**, or export the timeline's sound as **MP3/WAV** (audio-only) |

**One-click Quick Tools (no timeline needed — perfect for laymen):**

| Tool | What it does |
|------|--------------|
| 🎵 **Extract Audio** | Pull the MP3/WAV soundtrack out of any video |
| 🖼️ **Make a GIF** | Turn a clip into an optimized animated GIF |
| 🗜️ **Shrink File Size** | Compress video for easy sharing (light / balanced / strong) |
| 🔄 **Convert / Resize** | Change format (MP4 / WebM / MKV) and resolution |
| 📸 **Grab a Frame** | Save any moment as a snapshot image |
| 🔇 **Mute Video** | Strip the audio track in one click |

> Everything runs locally through FFmpeg — your files never leave your machine.

---

## 🖼️ Image Editor

A full image editor lives at **`/image`** — and it's **100% client-side** (HTML Canvas), so your images are *never even uploaded*. Nothing touches the server.

| Feature | Description |
|---------|-------------|
| ✂️ **Crop & Transform** | Free crop + ratio presets (1:1, 16:9, 9:16, 4:3, 2:3), rotate, flip H/V |
| 🎚️ **Adjustments** | Brightness, contrast, saturation, warmth, blur — live sliders |
| 🎨 **Filter Presets** | B&W, Sepia, Vintage, Cool, Warm, Vivid, Invert — one click |
| 🔤 **Text & Memes** | Multiple draggable text layers; Impact + outline for classic meme text |
| ✏️ **Draw & Annotate** | Freehand brush, rectangle, ellipse, line, arrow — any color/size |
| ↩️ **Undo / Redo** | Full non-destructive history (`Ctrl+Z` / `Ctrl+Y`) |
| 📋 **Paste Support** | Paste an image straight from your clipboard (`Ctrl+V`) |
| 🔍 **Increase Quality (Local AI)** | **Real super-resolution** — genuinely upscales 2×/4× and reconstructs pixels, text & edges instead of stretching. The thing other platforms charge a subscription for, running free on your own CPU |
| 📤 **Export** | PNG, JPG or WEBP with a quality slider |

> Editing runs entirely in your browser (zero upload, complete privacy). The optional **Increase Quality** upscale sends the image only to *your own local server* — never the cloud.

### 🔍 About "Increase Quality" (super-resolution)

Unlike free tools that just bilinear-stretch an image (making text and edges blocky), this uses **learned super-resolution models** via OpenCV's `dnn_superres`:

- **Fast (FSRCNN)** — near-instant, great for most images
- **Best (EDSR)** — sharpest text & edges, slower on CPU (best for smaller images)

The models download automatically the first time, or you can pre-fetch them:

```bash
python download_models.py
```

If the models or OpenCV aren't available, the app **automatically falls back** to high-quality Lanczos + sharpening — so the feature always works.

---

## ⚡ Quick Start (5 minutes)

### Prerequisites

- **Python 3.9+** → [Download](https://www.python.org/downloads/)
- **FFmpeg** → [Download](https://ffmpeg.org/download.html) *(required for audio processing)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ArPaN-DS/Audio_Cutter.git
cd Audio_Cutter

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your FLASK_SECRET_KEY

# 5. Run the app
python app.py
```

### 🎉 Open → [http://localhost:5000](http://localhost:5000)

> 📖 **Need more help?** See the [full setup guide →](docs/SETUP.md)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.9+, Flask 3.x |
| **Audio Engine** | Pydub + FFmpeg |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Waveform** | WaveSurfer.js v7 (Regions + Timeline plugins) |
| **Fonts** | Google Fonts — Inter |
| **Icons** | Font Awesome 6 |
| **Logging** | PostgreSQL via psycopg2 (optional) |

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Play / Pause |
| `Double-Click` | Add new region on waveform |
| `Click Region` | Select a region |
| `Delete` | Remove selected region |
| `Ctrl + Z` | Undo last action |
| `← / →` | Skip back / forward 5 seconds |
| `M` | Mute / Unmute |
| `?` | Show shortcuts panel |

---

## 📁 Project Structure

```
Audio_Cutter/
├── app.py                  # Flask backend — routes & audio processing
├── ai_processor.py         # Local AI processing engine (silence, beats, denoise, speech detection)
├── logger.py               # PostgreSQL upload logger (optional)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment configuration template
├── CHANGELOG.md            # Version history
│
├── static/
│   ├── style.css           # Design system (1900+ lines)
│   ├── script.js           # Frontend JS — WaveSurfer, regions, UX (700+ lines)
│   └── logo.png            # App logo
│
├── templates/
│   └── index.html          # Main HTML template (Jinja2)
│
├── .github/
│   ├── workflows/ci.yml    # GitHub Actions CI — linting
│   ├── ISSUE_TEMPLATE/     # Bug report & feature request forms
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docs/
│   ├── SETUP.md            # Detailed setup guide
│   ├── PRODUCTION.md       # Production deployment guide
│   ├── USER_MANUAL.md      # End-user documentation
│   └── CONTRIBUTING.md     # Contribution guidelines
│
├── uploads/                # Temporary uploaded files (auto-cleaned)
├── processed/              # Temporary processed output files
└── logs/                   # Application logs
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [📋 Setup Guide](docs/SETUP.md) | Step-by-step setup for beginners |
| [🏭 Production Guide](docs/PRODUCTION.md) | Deploy to a real server |
| [📖 User Manual](docs/USER_MANUAL.md) | How to use every feature |
| [🤝 Contributing](docs/CONTRIBUTING.md) | How to contribute to this project |
| [📋 Changelog](CHANGELOG.md) | Version history |
| [🔒 Security](SECURITY.md) | Vulnerability reporting policy |

---

## 🔧 Configuration

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_SECRET_KEY` | *(required)* | Session secret key — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_PORT` | `5000` | Server port |
| `DB_PASSWORD` | — | PostgreSQL password (optional logging) |

---

## 🗺️ Roadmap

Planned features for upcoming releases:

- [x] **v1.1** — Waveform zoom controls + speed adjustment (0.5×–2×)
- [x] **v1.2** — Local AI integrations (Silence detection, auto-trim, BPM, denoise, Whisper transcription)
- [ ] **v1.3** — Batch file processing (multiple files in one session)
- [ ] **v1.4** — Audio merge from multiple source files
- [ ] **v1.5** — Dark / Light mode toggle
- [ ] **v2.0** — Docker one-command deployment

> 💡 Have an idea? [Open a feature request →](https://github.com/ArPaN-DS/Audio_Cutter/issues/new?template=feature_request.yml)

---

## 🤝 Contributing

Contributions are very welcome! Whether it's a bug fix, a new feature, or improved documentation:

1. Fork the repo
2. Create your branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add awesome feature"`
4. Push & open a Pull Request

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for full guidelines.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
Free to use, modify, and distribute.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ArPaN-DS"><strong>ArPaN-DS</strong></a>
  <br><br>
  If this project helped you, please consider giving it a ⭐ — it means a lot!
</p>
