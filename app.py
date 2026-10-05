import runtime_tuning  # noqa: F401  (must be first: thread-pool defaults before numeric libraries load)

import os
import sys

if __name__ == '__main__':
    sys.modules['app'] = sys.modules['__main__']

import time
import uuid
import json
import zipfile
import logging
import math
import warnings
from PIL import Image

import re
import shutil
import tempfile
import threading
from urllib.parse import unquote, urlsplit
from flask import Flask, render_template, request, send_file, send_from_directory, jsonify, abort
from werkzeug.exceptions import HTTPException

# Load environment variables from .env BEFORE importing processors: several of
# them (e.g. the copilot reasoning endpoint) read configuration at import time.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables

import ai_processor
import video_processor
import image_processor
import audio_processor
import separation_processor
import agent_processor
import agent_memory
import agent_skills
from model_manager import global_quality_governor
import identity_guard
import branding
from pydub import AudioSegment
from logger import log_upload_details
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

app = Flask(__name__)


@app.context_processor
def inject_brand():
    """Product/assistant display names (configurable; see branding.py)."""
    return {"brand": branding.brand()}


secret = os.environ.get("FLASK_SECRET_KEY")
if not secret or secret == "change-me-in-production":
    import secrets
    app.secret_key = secrets.token_hex(32)
else:
    app.secret_key = secret

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
PROJECTS_FOLDER = os.path.join(BASE_DIR, 'projects')
os.makedirs(PROJECTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Config: 500MB Limit
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['PROJECTS_FOLDER'] = PROJECTS_FOLDER

PRIVATE_RESPONSE_KEYS = {
    'backend',
    'device',
    'engine',
    'implementation',
    'model',
    'model_accuracy',
    'model_name',
    'model_used',
    'note'
}
PRIVATE_TERM_REPLACEMENTS = (
    (re.compile(r'\b(?:google/)?gemma[\w.-]*', re.IGNORECASE), 'local reasoning engine'),
    (re.compile(r'\bvllm\w*', re.IGNORECASE), 'local reasoning service'),
    (re.compile(r'\b(?:faster[-_ ]?)?whisper[\w.-]*', re.IGNORECASE), 'speech recognition engine'),
    (re.compile(r'\b(?:isnet|u2net|birefnet|rembg)[\w.-]*', re.IGNORECASE), 'subject cutout engine'),
    (re.compile(r'\b(?:ffmpeg|ffprobe)\w*', re.IGNORECASE), 'media processing engine'),
    (re.compile(r'\b(?:opencv[\w.-]*|cv2)\b', re.IGNORECASE), 'image processing engine'),
    (re.compile(r'\b(?:real[-_ ]?)?esrgan[\w.-]*|\b(?:edsr|fsrcnn|gfpgan|codeformer)[\w.-]*|\blama\b',
                re.IGNORECASE), 'image enhancement engine'),
    (re.compile(r'\b(?:silero|deepfilternet|noisereduce|demucs|(?:py)?scenedetect)[\w.-]*', re.IGNORECASE),
     'media analysis engine'),
    (re.compile(r'\b(?:librosa|pydub|pytorch|torch|ctranslate2|onnx)[\w.-]*', re.IGNORECASE), 'local processing engine'),
    (re.compile(r'\b(?:kokoro|misaki|piper|espeak(?:-ng)?|sapi\d*|pyttsx\d*)\b[\w.-]*|\bSystem\.Speech\b',
                re.IGNORECASE), 'speech synthesis engine')
)


def _sanitize_public_text(value):
    sanitized = value
    for pattern, replacement in PRIVATE_TERM_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


class BrandPrivacyLogFilter(logging.Filter):
    _product_loggers = {'app', 'agent_processor', 'model_manager'}

    def filter(self, record):
        if record.name not in self._product_loggers:
            record.name = 'media_engine'
        record.msg = _sanitize_public_text(record.getMessage())
        record.args = ()
        if record.name == 'media_engine':
            record.exc_info = None
            record.exc_text = None
        return True


def _install_privacy_filters():
    """Attach the privacy filter to every handler, including processors' own stdout handlers."""
    loggers = [logging.getLogger()] + [
        candidate for candidate in logging.root.manager.loggerDict.values()
        if isinstance(candidate, logging.Logger)
    ]
    for active_logger in loggers:
        for active_handler in active_logger.handlers:
            if not any(isinstance(existing, BrandPrivacyLogFilter) for existing in active_handler.filters):
                active_handler.addFilter(BrandPrivacyLogFilter())


_install_privacy_filters()


def _sanitize_public_payload(value):
    if isinstance(value, dict):
        return {
            key: _sanitize_public_payload(item)
            for key, item in value.items()
            if key.lower() not in PRIVATE_RESPONSE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_public_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_public_payload(item) for item in value)
    if isinstance(value, str):
        return _sanitize_public_text(value)
    return value

ALLOWED_EXTENSIONS = {
    'audio': {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma', 'opus', 'aiff', 'webm'},
    'video': {'mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv', 'm4v'},
    'image': {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'gif'}
}
ALL_ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS['audio'] | ALLOWED_EXTENSIONS['video'] | ALLOWED_EXTENSIONS['image']

def allowed_file(filename, media_types=None):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if media_types:
        allowed = set()
        for mt in media_types:
            allowed |= ALLOWED_EXTENSIONS.get(mt, set())
        return ext in allowed
    return ext in ALL_ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════
#  PLATFORM MEMORY & STORAGE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

import gc
import sys

_last_cleanup_time = 0
_storage_lock = threading.RLock()


def _project_storage_references():
    protected_files = set()
    protected_directories = set()
    projects_folder = app.config['PROJECTS_FOLDER']
    if not os.path.exists(projects_folder):
        return protected_files, protected_directories
    reference_keys = {'id', 'mediaId', 'source', 'url', 'thumbs', 'activeMediaId'}
    upload_root = os.path.realpath(app.config['UPLOAD_FOLDER'])
    processed_root = os.path.realpath(app.config['PROCESSED_FOLDER'])

    def protect_path(folder, relative_path, directory=False):
        target = os.path.realpath(os.path.join(folder, relative_path))
        try:
            if target == folder or os.path.commonpath([folder, target]) != folder:
                return
        except ValueError:
            return
        if directory:
            protected_directories.add(target)
        else:
            protected_files.add(target)

    try:
        with os.scandir(projects_folder) as entries:
            project_entries = list(entries)
    except OSError:
        logger.warning('Storage cleanup deferred because saved projects could not be inspected.')
        return None
    for entry in project_entries:
        if not entry.name.endswith('.aviproject') or not entry.is_file(follow_symlinks=False):
            continue
        try:
            with open(entry.path, encoding='utf-8') as project_file:
                project = json.load(project_file)
        except (OSError, ValueError):
            logger.warning('Storage cleanup deferred because a saved project could not be read.')
            return None
        pending = [('', project)]
        while pending:
            key, value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.items())
            elif isinstance(value, list):
                pending.extend((key, item) for item in value)
            elif key in reference_keys and isinstance(value, str):
                try:
                    parsed = urlsplit(value)
                except ValueError:
                    continue
                if parsed.scheme or parsed.netloc:
                    continue
                path = unquote(parsed.path)
                if path.startswith('/media/'):
                    protect_path(upload_root, path[len('/media/'):])
                elif path.startswith('/processed/'):
                    protect_path(processed_root, path[len('/processed/'):])
                elif path.startswith('/video/thumb/'):
                    thumb_id = path.split('/')[3]
                    if re.fullmatch(r'[A-Za-z0-9-]+', thumb_id):
                        protect_path(processed_root, os.path.join('thumbs', thumb_id), True)
                elif re.fullmatch(r'[A-Za-z0-9_.-]+', path):
                    protect_path(upload_root, path)
                    protect_path(processed_root, path)
                if path.startswith('/media/'):
                    path = path[len('/media/'):]
                if re.fullmatch(r'[A-Za-z0-9-]+\.[A-Za-z0-9]+', path):
                    protect_path(processed_root, os.path.join('thumbs', os.path.splitext(path)[0]), True)
    return protected_files, protected_directories

def _cleanup_old_temp_files(max_age_seconds=3600):
    """
    Background purger: Removes uploaded and processed temporary files
    older than max_age_seconds to prevent SSD disk bloat.
    """
    with _storage_lock:
        references = _project_storage_references()
        if references is None:
            return
        protected_files, protected_directories = references
        now = time.time()
        _forensic_dirs = {'cases', 'forensic', 'forensics', 'evidence'}
        _forensic_pattern = re.compile(r'^(case[s_-]|forensic[s_-]|evidence[s_-]|\.case|\.evidence)', re.IGNORECASE)
        _forensic_exts = ('.case', '.evidence', '.audit', '.exhibit')

        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            folder = os.path.realpath(folder)
            if not os.path.exists(folder):
                continue
            for root, directories, files in os.walk(folder, topdown=False, followlinks=False):
                resolved_root = os.path.realpath(root)
                if os.path.commonpath([folder, resolved_root]) != folder:
                    continue
                rel_root = os.path.relpath(resolved_root, folder).replace('\\', '/')
                path_parts = [p.lower() for p in rel_root.split('/') if p and p != '.']
                if any(p in _forensic_dirs or _forensic_pattern.match(p) for p in path_parts):
                    continue
                if any(resolved_root == protected or resolved_root.startswith(protected + os.sep)
                       for protected in protected_directories):
                    continue
                try:
                    directory_age = now - os.path.getmtime(root)
                except OSError:
                    continue
                for filename in files:
                    filepath = os.path.join(root, filename)
                    try:
                        if os.path.islink(filepath) or os.path.realpath(filepath) in protected_files:
                            continue
                        if _forensic_pattern.match(filename) or filename.lower().endswith(_forensic_exts):
                            continue
                        if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > max_age_seconds:
                            os.remove(filepath)
                    except OSError:
                        pass
                if root != folder and directory_age > max_age_seconds:
                    try:
                        os.rmdir(root)
                    except OSError:
                        pass

def _maybe_cleanup_temp_files(max_age_seconds=3600, interval_seconds=300):
    """Debounced cleanup: only scans disk once every interval_seconds."""
    global _last_cleanup_time
    if not _storage_lock.acquire(blocking=False):
        return
    try:
        now = time.monotonic()
        if now - _last_cleanup_time < interval_seconds:
            return
        _cleanup_old_temp_files(max_age_seconds)
        _last_cleanup_time = time.monotonic()
    finally:
        _storage_lock.release()

HEAVY_ROUTES = ('/process', '/transcribe', '/separate', '/enhance', '/export', '/ai/', '/cut', '/filter', '/upscale', '/api/agent/chat')

@app.after_request
def end_to_end_memory_reclaim(response):
    """
    Selective memory reclaim: runs garbage collection and working set reclaim
    after heavy media processing routes to keep memory low without burdening lightweight requests.
    """
    try:
        path = request.path
        if any(h in path for h in HEAVY_ROUTES):
            gc.collect()
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass
    return response


COPILOT_PATH_PREFIX = '/api/agent/'


@app.before_request
def intercept_identity_probes():
    """Answer questions about internals / prompt-extraction attempts without reaching the model."""
    if request.method != 'POST' or request.path != '/api/agent/chat':
        return None
    data = request.get_json(silent=True)
    message = data.get('message') if isinstance(data, dict) else None
    if not isinstance(message, str):
        return None
    reply = identity_guard.probe_reply(message)
    if reply is None:
        return None
    logger.info("Copilot answered a question about the Studio itself.")
    return jsonify(reply), 200


@app.after_request
def guard_copilot_identity(response):
    """Scrub any identity disclosure from Copilot responses, whatever produced them."""
    if not request.path.startswith(COPILOT_PATH_PREFIX) or not response.is_json:
        return response
    payload = response.get_json(silent=True)
    if payload is None:
        return response
    response.set_data(app.json.dumps(identity_guard.guard_chat_response(payload)))
    return response


@app.after_request
def apply_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    return response


@app.after_request
def enforce_public_response_contract(response):
    if not response.is_json:
        return response

    payload = response.get_json(silent=True)
    if payload is None:
        return response

    if response.status_code >= 400:
        if isinstance(payload, dict):
            message = payload.get('error') or payload.get('message') or 'Request failed.'
        else:
            message = 'Request failed.'
        if response.status_code >= 500:
            message = 'The request could not be completed. Please try again.'
        payload = {
            'status': 'error',
            'error': _sanitize_public_text(str(message)),
            'code': response.status_code
        }
    else:
        payload = _sanitize_public_payload(payload)

    response.set_data(app.json.dumps(payload))
    response.mimetype = 'application/json'
    return response


@app.errorhandler(HTTPException)
def handle_http_error(error):
    return jsonify({
        'status': 'error',
        'error': error.description or 'Request failed.',
        'code': error.code
    }), error.code


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.error('Unhandled request failure (%s)', type(error).__name__)
    return jsonify({
        'status': 'error',
        'error': 'The request could not be completed. Please try again.',
        'code': 500
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for platform liveness and readiness."""
    return jsonify({
        "status": "healthy",
        "storage": {
            "uploads_exists": os.path.exists(UPLOAD_FOLDER),
            "processed_exists": os.path.exists(PROCESSED_FOLDER),
            "projects_exists": os.path.exists(PROJECTS_FOLDER)
        },
        "adaptive_quality": _adaptive_quality_status()
    }), 200


def _adaptive_quality_status():
    try:
        return global_quality_governor.public_status()
    except Exception:
        return {"quality_tier": "lite", "quality_mode": "Efficiency Mode", "pinned": False}


@app.route('/api/system/resources', methods=['GET'])
def system_resources():
    """Live hardware headroom, the adaptive quality tier, and Copilot reasoning readiness."""
    status = dict(_adaptive_quality_status())
    try:
        import reasoning_models
        status["copilot"] = reasoning_models.public_status()
    except Exception:
        status["copilot"] = {"reasoning_available": False, "reasoning_mode": "Built-in Intent Routing"}
    return jsonify(status), 200

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'logo.png', mimetype='image/png')

@app.route('/')
def studio_landing():
    _maybe_cleanup_temp_files()
    return render_template('landing.html')




@app.route('/studio')
def master_studio():
    _maybe_cleanup_temp_files()
    return render_template('studio.html')


@app.route('/studio/project/save', methods=['POST'])
def studio_project_save():
    """Save project JSON state (.aviproject)."""
    try:
        data = request.get_json(silent=True)
        if not data or 'name' not in data:
            return jsonify({"error": "Invalid project data"}), 400
        
        project_id = data.get('id') or str(uuid.uuid4())
        if not isinstance(project_id, str) or not re.fullmatch(r'[A-Za-z0-9-]+', project_id):
            return jsonify({"error": "Invalid project ID"}), 400
        data['id'] = project_id
        data['updated_at'] = time.time()
        
        filepath = os.path.join(app.config['PROJECTS_FOLDER'], f"{project_id}.aviproject")
        with _storage_lock:
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', dir=app.config['PROJECTS_FOLDER'],
                    prefix='project-save-', suffix='.tmp', delete=False
                ) as project_file:
                    temporary_path = project_file.name
                    json.dump(data, project_file, indent=2)
                os.replace(temporary_path, filepath)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)
            
        return jsonify({"status": "success", "id": project_id, "file": f"{project_id}.aviproject"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/studio/project/load/<project_id>', methods=['GET'])
def studio_project_load(project_id):
    """Load project JSON state (.aviproject)."""
    if not re.match(r'^[A-Za-z0-9-]+$', project_id):
        return jsonify({"error": "Invalid project ID"}), 400
    filepath = os.path.join(app.config['PROJECTS_FOLDER'], f"{project_id}.aviproject")
    if not os.path.exists(filepath):
        return jsonify({"error": "Project not found"}), 404
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  AI AGENT CHATBOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/processed/<path:filename>')
def serve_processed_file(filename):
    """Serve generated audio, video, or image files with range request support."""
    base_dir = os.path.abspath(app.config['PROCESSED_FOLDER'])
    target_path = os.path.abspath(os.path.join(base_dir, filename))
    if not (target_path == base_dir or target_path.startswith(base_dir + os.sep)):
        return jsonify({"error": "Access denied"}), 403
    # Forward slashes: on Windows a backslash in the path is rejected as unsafe,
    # which made files in sub-folders (e.g. stems_*/vocals.wav) unreachable.
    rel_path = os.path.relpath(target_path, base_dir).replace(os.sep, '/')
    if os.path.isfile(target_path):
        _touch_in_use(target_path)
    return send_from_directory(base_dir, rel_path, conditional=True)




def _process_and_register_uploaded_media(saved_path, original_filename):
    """Common media verification, probing, and registration for direct and chunked uploads."""
    file_size = os.path.getsize(saved_path) if os.path.exists(saved_path) else 0
    if not file_size:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return jsonify({'error': 'The uploaded media file is empty.'}), 400

    orig_ext = os.path.splitext(original_filename)[1].lower() or '.bin'
    saved_filename = os.path.basename(saved_path)

    try:
        if orig_ext[1:] in ALLOWED_EXTENSIONS['image']:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(saved_path) as image:
                    image.verify()
                with Image.open(saved_path) as image:
                    image.load()
                    metadata = {'width': image.width, 'height': image.height, 'duration': 0.0,
                                'has_video': False, 'has_audio': False}
            file_type = 'image'
        else:
            metadata = video_processor.probe_media(saved_path)
            if not math.isfinite(metadata['duration']) or metadata['duration'] <= 0:
                raise ValueError('Media duration must be positive.')
            if orig_ext[1:] in ALLOWED_EXTENSIONS['audio'] and orig_ext[1:] not in ALLOWED_EXTENSIONS['video']:
                if not metadata['has_audio']:
                    raise ValueError('The file has no audio stream.')
                file_type = 'audio'
            elif metadata['has_video']:
                file_type = 'video'
            elif metadata['has_audio']:
                file_type = 'audio'
            else:
                raise ValueError('The file contains no usable media streams.')
    except Exception:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return jsonify({'error': 'The file could not be read as supported media. Choose a valid audio, video, or image file.'}), 400

    return jsonify({
        "status": "success",
        "id": saved_filename,
        "filename": saved_filename,
        "original_name": original_filename,
        "type": file_type,
        "size": file_size,
        "url": f"/media/{saved_filename}",
        **metadata
    })












# ═══════════════════════════════════════════════════════════════════════════
#  COPILOT SKILLS (catalog, details, personal skills)
# ═══════════════════════════════════════════════════════════════════════════









_test_memory_lock = threading.Lock()


def _copilot_memory_ready():
    """Test runs get an isolated, throwaway Copilot Memory so they never touch the user's memory."""
    if app.config.get('TESTING') and not agent_memory.is_configured():
        with _test_memory_lock:
            if not agent_memory.is_configured():
                agent_memory.configure(os.path.join(tempfile.mkdtemp(prefix='copilot-memory-'), 'memory.db'),
                                       dense='off')


def _memory_repeat_plan(session_id, message, media_context):
    """Replay of this conversation's last successful edit, validated like any other plan; None if not applicable."""
    raw = agent_memory.safe_repeat_plan(session_id, message)
    if raw is None:
        return None
    try:
        return agent_processor.agent_planner.finalize_plan(
            raw, message, media_context, agent_processor._perception_for(media_context), source='memory')
    except Exception as error:
        logger.warning('Copilot Memory replay skipped (%s).', type(error).__name__)
        return None




def _agent_public_url(path):
    """Public URL for a Copilot input/output file (uploads are served from /media, results from /processed)."""
    rel_name = os.path.basename(path)
    if os.path.realpath(path).startswith(os.path.realpath(app.config['UPLOAD_FOLDER']) + os.sep):
        return f"/media/{rel_name}"
    return f"/processed/{rel_name}"


# ════════════════════════════════════════════════════════════════════════════
#  BACKGROUND JOBS  (progress · cancel · ETA)
# ════════════════════════════════════════════════════════════════════════════

_bg_jobs: dict = {}           # job_id -> {status, progress, message, result, error, cancel_event}
_bg_jobs_lock = threading.Lock()


def _bg_job_worker(job_id, file_path, tools_to_run, media_context, processed_folder):
    """Run execute_agent_plan in a background thread; update _bg_jobs as it progresses."""
    cancel_event = _bg_jobs[job_id]['cancel_event']
    try:
        steps = agent_processor.agent_planner.execution_steps(tools_to_run)
        total = max(len(steps), 1)
        patched_results = []

        def _step_done(i):
            pct = round((i + 1) / total * 100)
            with _bg_jobs_lock:
                if _bg_jobs.get(job_id):
                    _bg_jobs[job_id]['progress'] = pct
                    _bg_jobs[job_id]['message'] = f'Step {i + 1} of {total} complete.'

        results = agent_processor.execute_agent_plan(
            file_path, tools_to_run, processed_folder,
            context=media_context,
        )
        if cancel_event.is_set():
            with _bg_jobs_lock:
                _bg_jobs[job_id]['status'] = 'cancelled'
            return
        final_file = agent_processor.final_output_of(results)
        final_url = _agent_public_url(final_file) if final_file else None
        public_results = [{**{k: v for k, v in step.items() if k != 'extra_outputs'},
                           'output_file': os.path.basename(step['output_file']) if step.get('output_file') else None}
                          for step in results]
        failed = any(s.get('status') == 'error' for s in results)
        job_status = 'partial' if failed and final_file else 'failed' if failed else 'success'
        with _bg_jobs_lock:
            _bg_jobs[job_id].update({
                'status': job_status, 'progress': 100, 'message': 'Done.',
                'result': {'execution_results': public_results, 'output_file': os.path.basename(final_file) if final_file else None,
                           'output_url': final_url, 'status': job_status},
            })
    except Exception as err:
        with _bg_jobs_lock:
            if _bg_jobs.get(job_id):
                _bg_jobs[job_id].update({'status': 'error', 'message': str(err)[:300]})
















def _loudness_normalize_segment(segment, preset='youtube'):
    """Loudness-normalize an in-memory segment (-14 LUFS, true-peak protected)."""
    if segment.dBFS == float('-inf'):
        return segment
    fd_in, tmp_in = tempfile.mkstemp(suffix='.wav', dir=app.config['PROCESSED_FOLDER'])
    fd_out, tmp_out = tempfile.mkstemp(suffix='.wav', dir=app.config['PROCESSED_FOLDER'])
    os.close(fd_in)
    os.close(fd_out)
    try:
        segment.export(tmp_in, format='wav')
        audio_processor.normalize_audio(tmp_in, tmp_out, preset=preset, mode='loudness')
        return AudioSegment.from_file(tmp_out, format='wav')
    except Exception as error:
        logger.warning("Loudness normalization fell back to peak-safe gain (%s)", type(error).__name__)
        headroom = -1.0 - segment.max_dBFS
        return segment.apply_gain(min(-14.0 - segment.dBFS, headroom))
    finally:
        for path in (tmp_in, tmp_out):
            try:
                os.remove(path)
            except OSError:
                pass



# Helper to validate and save upload file temporarily
def get_uploaded_file(req, allowed_types=('audio', 'video')):
    if 'file' not in req.files:
        return None, (jsonify({"error": "No file uploaded"}), 400)
    f = req.files['file']
    if not f or f.filename == '':
        return None, (jsonify({"error": "No file selected"}), 400)
    if not allowed_file(f.filename, allowed_types):
        return None, (jsonify({"error": f"Invalid file type: {f.filename}"}), 400)
    return f, None

def save_temp_upload(file):
    unique_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename)[1] or ".webm"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{unique_id}{original_ext}")
    file.save(temp_path)
    return temp_path














# ── Source separation (vocals / karaoke / 4 stems / voice) and lyrics ──────

def _separation_base_name(filename):
    base = os.path.splitext(os.path.basename(filename or 'audio'))[0]
    base = re.sub(r'[^\w.-]+', '_', base, flags=re.UNICODE).strip('._')
    return (base or 'audio')[:80]


def _processed_url(path):
    rel = os.path.relpath(path, app.config['PROCESSED_FOLDER']).replace(os.sep, '/')
    return f"/processed/{rel}"








# ═══════════════════════════════════════════
#  VIDEO EDITOR (audio + video combo editor)
# ═══════════════════════════════════════════

THUMBS_FOLDER = os.path.join(PROCESSED_FOLDER, 'thumbs')
os.makedirs(THUMBS_FOLDER, exist_ok=True)

MEDIA_ID_RE = re.compile(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9]+$')


def _touch_in_use(path):
    """Refresh mtime so the age-based storage purge never removes media still in use."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def _media_path(media_id):
    """Resolve an uploaded media id to a safe absolute path (or None)."""
    if not media_id or not MEDIA_ID_RE.match(media_id):
        return None
    if media_id.rsplit('.', 1)[1].lower() not in ALL_ALLOWED_EXTENSIONS:
        return None
    path = os.path.join(app.config['UPLOAD_FOLDER'], media_id)
    if not os.path.isfile(path):
        return None
    _touch_in_use(path)
    return path




























# ── Voiceover (local text-to-speech). Public capability labels only. ──




















EXPORT_FORMATS = {
    'mp4': 'video/mp4',
    'webm': 'video/webm',
    'gif': 'image/gif',
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
}
GIF_EXPORT_MAX_SECONDS = 30.0




from routes import register_blueprints
register_blueprints(app)


if __name__ == '__main__':
    from werkzeug.serving import WSGIRequestHandler

    # Do not advertise the server software or runtime version in responses.
    WSGIRequestHandler.server_version = "MediaStudio"
    WSGIRequestHandler.sys_version = ""
    # Hot reloading stays on; the interactive debugger (which exposes internals)
    # is opt-in via FLASK_DEBUG=1 for local development only.
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    debug_enabled = os.environ.get('FLASK_DEBUG', '').strip().lower() in ('1', 'true', 'yes')
    app.run(debug=debug_enabled, port=int(os.environ.get('FLASK_PORT', 5000)), use_reloader=True)