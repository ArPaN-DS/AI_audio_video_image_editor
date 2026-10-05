# How to Run

## Requirements
- Python 3.11+
- FFmpeg on PATH (`ffmpeg -version` must work)
- Optional: a local reasoning model server (llama.cpp / Ollama / LM Studio) — see `docs/SETUP.md`

## 1. Local computer (Windows)

```bat
git clone <your-repo-url>
cd "auido-video-image editor"
python -m venv venv
venv\Scripts\pip install -r requirements.txt
copy .env.example .env
venv\Scripts\python download_models.py            &:: base models (optional)
venv\Scripts\python download_models.py --voice    &:: voiceover pack (optional)
venv\Scripts\python download_models.py --stems    &:: song separation pack (optional)
run.bat
```
Open http://localhost:5000

## 1b. Local computer (macOS / Linux)

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
venv/bin/python download_models.py      # optional packs: --voice --stems
venv/bin/python app.py
```

## 2. Server (Linux, production)

`python app.py` is the development server (hot reload). On a server use a WSGI server:

```bash
sudo apt install ffmpeg python3-venv
python3 -m venv venv
venv/bin/pip install -r requirements.txt waitress
cp .env.example .env        # set FLASK_SECRET_KEY to a long random value
venv/bin/python download_models.py
venv/bin/waitress-serve --host=127.0.0.1 --port=5000 --threads=8 app:app
```

Put a reverse proxy (nginx / Caddy) in front for HTTPS and set an upload size limit, e.g. nginx:
```
client_max_body_size 2G;
location / { proxy_pass http://127.0.0.1:5000; proxy_read_timeout 1800; }
```

Keep it running with systemd (`/etc/systemd/system/media-studio.service`):
```ini
[Service]
WorkingDirectory=/opt/media-studio
ExecStart=/opt/media-studio/venv/bin/waitress-serve --host=127.0.0.1 --port=5000 --threads=8 app:app
Restart=always
[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now media-studio`

Notes:
- The reasoning model must also run on the same server (`localhost`); remote endpoints are rejected by design.
- Heavy AI features need RAM/GPU; the app picks lighter models automatically on small servers (`MEDIA_QUALITY_TIER` to pin).
- Writable folders: `uploads/`, `processed/`, `projects/`, `data/`, `logs/`, `models/`.

## 3. Check it works
```bash
venv/bin/python e2e_full_automated_test.py     # expect: Ran 24 tests ... OK
curl http://localhost:5000/health
```
