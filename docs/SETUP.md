# 📋 Audio Cutter Pro — Setup Guide

> **This guide is written for beginners.** Follow each step carefully, even if you've never worked with Python before.

---

## 📌 What You'll Need

Before starting, make sure you have these installed on your computer:

| Software | Why You Need It | Download Link |
|----------|----------------|---------------|
| **Python 3.9 or newer** | Runs the backend server | [python.org/downloads](https://www.python.org/downloads/) |
| **FFmpeg** | Processes audio files (cut, convert, etc.) | [ffmpeg.org/download](https://ffmpeg.org/download.html) |
| **Git** | Clone the project from GitHub | [git-scm.com](https://git-scm.com/) |
| **A web browser** | Use the app (Chrome, Firefox, Edge, etc.) | Already installed! |
| **PostgreSQL** *(optional)* | Log upload statistics | [postgresql.org](https://www.postgresql.org/download/) |

---

## Step 1 — Install Python

### Windows
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the latest Python 3 installer
3. **IMPORTANT:** During installation, check the box that says **"Add Python to PATH"**
4. Click "Install Now"
5. Verify by opening Command Prompt and typing:
   ```
   python --version
   ```
   You should see something like `Python 3.11.5`

### macOS
```bash
brew install python3
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

---

## Step 2 — Install FFmpeg

FFmpeg is required by the audio processing library (Pydub).

### Windows
1. Go to [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
2. Download **"ffmpeg-release-essentials.zip"**
3. Extract the ZIP file to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your system PATH:
   - Search for "Environment Variables" in Windows
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System Variables", find `Path` and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click OK on all windows
5. Verify by opening a **new** Command Prompt:
   ```
   ffmpeg -version
   ```

### macOS
```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian)
```bash
sudo apt install ffmpeg
```

---

## Step 3 — Clone the Project

Open a terminal (Command Prompt, PowerShell, or Terminal) and run:

```bash
git clone https://github.com/ArPaN-DS/Audio_Cutter.git
cd Audio_Cutter
```

---

## Step 4 — Set Up Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects.

```bash
# Create the virtual environment
python -m venv venv

# Activate it:
# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# macOS / Linux:
source venv/bin/activate
```

> After activation, you should see `(venv)` at the beginning of your terminal prompt.

---

## Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the core dependencies, including:
- **Flask** — The web framework
- **Pydub** — Audio processing library
- **librosa** & **soundfile** — Local audio analysis & parsing
- **noisereduce** — Background noise reduction
- **psycopg2-binary** — PostgreSQL database logger

---

## Step 6 — Configure Environment (Optional)

If you want to change the default settings:

```bash
# Copy the example config
# Windows:
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```

Then edit `.env` with your preferred text editor to set the PostgreSQL password.

> **Note:** The app works perfectly fine without PostgreSQL. The database is only used for logging upload statistics. If PostgreSQL is not configured, the app will print a warning but continue working normally.

### Optional — Local reasoning models for the AI Copilot

The Copilot works without a language model (it falls back to built-in intent routing), but plans
complex multi-step requests much better with one. Models must run **on this machine**: only
`localhost` / `127.0.0.1` endpoints are accepted, so cloud tunnels (ngrok, Colab, etc.) are rejected.

1. Run any local server with an OpenAI-compatible API — on Windows use llama.cpp's `llama-server`,
   Ollama, or LM Studio (vLLM also works on Linux/WSL with a supported GPU).
2. Load a quantized instruction-tuned model (GGUF Q4_K_M or similar) and note its model id
   (the server's `GET /v1/models` lists it).
3. In `.env` set `LOCAL_REASONING_URL` (e.g. `http://127.0.0.1:8000/v1`) and `LOCAL_REASONING_MODEL`.
4. Optional larger tiers: set `LOCAL_REASONING_MODEL_BALANCED` / `LOCAL_REASONING_MODEL_MAX`
   (and `LOCAL_REASONING_URL_BALANCED` / `LOCAL_REASONING_URL_MAX` if they run on separate ports).
   The Copilot only uses tiers the server actually serves and your hardware can sustain. Short
   requests start on the smallest tier and escalate to a larger one only if its answer fails
   validation; long multi-step requests go to the largest tier first. Slow or failing tiers are
   paused automatically. `LOCAL_REASONING_TIER=lite|balanced|max` pins the ceiling.

**Small-model tuning (all optional, `slm_runtime.py` / `slm_context.py` / `slm_profiles.py`).**
Each request gets a budgeted prompt (~1–1.5k tokens on a ~2B model instead of ~2.5k+): a static
rules block first (so the server can reuse its prompt cache), then only the most relevant tools and
1–3 similar examples, memory and the request. Output is constrained to the plan's JSON schema when
the server supports it (llama.cpp, Ollama ≥ 0.5, LM Studio, vLLM); a bad answer gets one repair
retry, then the built-in planner takes over. The context window is read from the server
(llama.cpp `/props`, Ollama `/api/show`/`/api/ps`, LM Studio, vLLM `max_model_len`).

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_REASONING_ROUTING` | `auto` | `auto`: skip the model when the built-in planner is confident (fastest); `always`: always ask the model; `local`: never |
| `LOCAL_REASONING_CONTEXT_TOKENS` | from server, else 4096 | Force the context window used for budgeting |
| `LOCAL_REASONING_PROMPT_TOKENS` | 1536 / 3072 / 6144 per tier | Prompt size target (always capped to fit the window) |
| `LOCAL_REASONING_STRUCTURED_OUTPUT` | `auto` (`json_schema`) | `json_schema`, `json_object` or `off`; auto-downgrades if the server rejects it |
| `LOCAL_REASONING_SCHEMA` | `strict` | `strict` types every tool's args; `loose` only pins tool names |
| `LOCAL_REASONING_SERVER` | auto-detected | `llamacpp`, `ollama`, `lmstudio`, `vllm` or `generic` |
| `LOCAL_REASONING_PROBE` | `on` | `off` disables the server metadata probes |

Hybrid "thinking" models (e.g. Qwen3) have thinking switched off for planning automatically —
otherwise it consumes the whole output budget. With Ollama, raise the window if you want more memory
in the prompt: `set OLLAMA_CONTEXT_LENGTH=8192` before `ollama serve`. With llama.cpp, note that
`-c` is split across `--parallel` slots (e.g. `-c 8192 -np 4` gives each request 2048 tokens).
Evaluate a served model on the golden prompts with
`./venv/Scripts/python.exe slm_eval.py live --models <model-id>` (`slm_eval.py offline` runs a mock).

Design sources: fewer, retrieved tools improve small-model tool choice
([Less-is-More, arXiv:2411.15399](https://arxiv.org/abs/2411.15399);
[RAG-MCP, arXiv:2505.03275](https://arxiv.org/abs/2505.03275)); constrained decoding removes
structural JSON failures but not semantic ones, hence validation + repair + fallback
([arXiv:2609.23742](https://arxiv.org/abs/2609.23742)); reasoning field first in the schema
([Let Me Speak Freely?, EMNLP 2024](https://aclanthology.org/2024.emnlp-industry.91/),
[dottxt response](https://blog.dottxt.ai/say-what-you-mean.html)); server APIs:
[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
[LM Studio](https://lmstudio.ai/docs/developer/openai-compat/structured-output),
[vLLM](https://docs.vllm.ai/en/latest/features/structured_outputs/); model cards:
[Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B) (128K context, native system role),
[Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B-it) (32K, no system role).

Rough memory needs for 4-bit quantized models: ~2B ≈ 2–3 GB, ~4B ≈ 3–5 GB, ~12B ≈ 7–9 GB
(GPU VRAM, or system RAM when running on CPU, which is much slower).
`GET /api/system/resources` shows which reasoning tier is active.

### Optional — Natural voiceover pack

Voiceover (Audio editor, and `POST /ai/tts`) always works with your operating system's built-in
voices. For natural-sounding voices, download the local voice pack once:

```bash
python download_models.py --voice        # ~350 MB, best quality and fastest on CPU
python download_models.py --voice-lite   # optional ~110 MB low-memory variant
```

Files go to `models/voice/` (with a `NOTICE.txt`). Licences: speech weights and voice styles are
Apache-2.0 (Kokoro-82M), pronunciation lexicons are Apache-2.0 (misaki data), the runtime is
ONNX Runtime (MIT). The GPL phonemizer used by the upstream pipeline (espeak-ng) is deliberately
not used. Nothing is downloaded unless you run the command. Voice cloning is not supported.

---

## Step 7 — Run the Application

```bash
python app.py
```

You should see output like:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

## Step 8 — Open the App

Open your web browser and go to:

### 👉 [http://localhost:5000](http://localhost:5000)

You should see the Audio Cutter Pro interface!

---

## 🔧 Troubleshooting

### "Python is not recognized"
- **Solution:** Reinstall Python and make sure to check "Add Python to PATH"

### "FFmpeg not found" or audio processing fails
- **Solution:** Make sure FFmpeg is installed and added to your system PATH. Restart your terminal after adding it.

### "ModuleNotFoundError: No module named 'flask'"
- **Solution:** Make sure your virtual environment is activated (you should see `(venv)` in your prompt), then run `pip install -r requirements.txt`

### "DATABASE ERROR: password authentication failed"
- **Solution:** This is a non-critical warning. Either:
  - Set the correct PostgreSQL password in `logger.py` (line 9)
  - Or ignore it — the app works fine without the database

### Port 5000 is already in use
- **Solution:** Change the port in `app.py` (last line):
  ```python
  app.run(debug=True, port=5001)  # Change to any available port
  ```

---

## ✅ You're Done!

The app is now running. To learn how to use it, check the [User Manual](USER_MANUAL.md).

To stop the server, press `Ctrl + C` in your terminal.
