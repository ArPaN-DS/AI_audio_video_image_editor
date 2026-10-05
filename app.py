import runtime_tuning  # noqa: F401  (must be first: thread-pool defaults before numeric libraries load)

import os
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
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            folder = os.path.realpath(folder)
            if not os.path.exists(folder):
                continue
            for root, directories, files in os.walk(folder, topdown=False, followlinks=False):
                resolved_root = os.path.realpath(root)
                if os.path.commonpath([folder, resolved_root]) != folder:
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


@app.route('/agent')
@app.route('/chat')
def agent_full_window():
    _maybe_cleanup_temp_files()
    return render_template('agent.html')


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


@app.route('/api/agent/tools', methods=['GET'])
def agent_get_tools():
    """Returns registered AI agent tool schemas and active Sub-Agents."""
    return jsonify({
        "status": "success",
        "subagents": ["VisionSubAgent", "VideoSubAgent", "AudioSubAgent", "InspectorSubAgent", "MasterOrchestrator"],
        "tools": agent_processor.TOOL_DEFINITIONS
    })


@app.route('/api/agent/upload', methods=['POST'])
def agent_upload_media():
    """Direct upload endpoint for AI Agent Full Window workspace."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Choose a supported audio, video, or image file.'}), 400

    unique_id = str(uuid.uuid4())
    orig_ext = os.path.splitext(file.filename)[1].lower() or '.bin'
    saved_filename = f"agent_{unique_id}{orig_ext}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
    file.save(saved_path)

    file_size = os.path.getsize(saved_path)
    if not file_size:
        os.remove(saved_path)
        return jsonify({'error': 'The uploaded media file is empty.'}), 400
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
        "original_name": file.filename,
        "type": file_type,
        "size": file_size,
        "url": f"/media/{saved_filename}"
        , **metadata
    })


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat_endpoint():
    """
    AI Multi-Agent Copilot Chatbot Endpoint.
    Receives user natural language input + media context + conversation history,
    orchestrates Vision, Video, Audio, and Inspector Sub-Agents.
    """
    try:
        _copilot_memory_ready()
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'error': 'Send an object containing your message.'}), 400
        if (not isinstance(data.get('message', ''), str)
                or not isinstance(data.get('filename', ''), (str, type(None)))
                or not isinstance(data.get('context', {}), (dict, type(None)))
                or not isinstance(data.get('history', []), (list, type(None)))
                or not isinstance(data.get('session_id', ''), (str, type(None)))):
            return jsonify({'error': 'Invalid message, media context, or conversation history.'}), 400
        # Optional conversation id for Copilot Memory; malformed ids simply disable session memory.
        memory_session = agent_memory.valid_session_id(data.get('session_id'))
        user_message = (data.get("message") or "").strip()
        filename = (data.get("filename") or "").strip()
        media_context = data.get("context") or {}
        conversation_history = data.get("history") or []
        if any(not isinstance(item, dict) or not isinstance(item.get('content', ''), str)
               or not isinstance(item.get('role', ''), str) for item in conversation_history):
            return jsonify({'error': 'Invalid conversation history.'}), 400

        if not user_message:
            return jsonify({"error": "Message parameter is required."}), 400
        # Personal skills: "save this as @name" / "delete skill @name" are handled without planning.
        skill_command = agent_skills.handle_command(user_message, memory_session or 'default')
        if skill_command is not None:
            return jsonify({'status': 'success', 'reply': skill_command['reply'], 'thought': '',
                            'delegated_subagent': 'Orchestrator', 'clarification_needed': False,
                            'clarification_options': [], 'tools_planned': [], 'execution_results': [],
                            'output_file': None, 'output_url': None, 'artifacts': [], 'plan_notes': [],
                            'skills_used': [], 'skill': skill_command.get('skill'),
                            'suggested_actions': skill_command.get('suggested_actions', [])})
        agent_memory.remember_preferences_async(memory_session, user_message)

        # Locate file on disk
        target_filepath = None
        if filename:
            safe_name = os.path.basename(filename)
            possible_paths = [
                os.path.join(app.config['UPLOAD_FOLDER'], safe_name),
                os.path.join(app.config['PROCESSED_FOLDER'], safe_name)
            ]
            for p in possible_paths:
                if safe_name and os.path.isfile(p):
                    target_filepath = p
                    _touch_in_use(p)
                    break

        # Underscore keys are server-internal (e.g. the resolved media path); never accept them from clients.
        media_context = {key: value for key, value in media_context.items() if not str(key).startswith('_')}
        if target_filepath:
            media_context = agent_processor._media_context_for_file(target_filepath, media_context)

        # Query Master Orchestrator Agent. Copilot Memory: an explicit "do the same as before" replays the
        # last successful edit of this conversation, re-validated by the planner for the current media.
        with agent_memory.request_scope(target_filepath):
            agent_plan = _memory_repeat_plan(memory_session, user_message, media_context) if target_filepath else None
            if agent_plan is None:
                if memory_session:
                    agent_plan = agent_processor.query_agent_orchestrator(
                        user_message, media_context, conversation_history, session_id=memory_session)
                else:
                    agent_plan = agent_processor.query_agent_orchestrator(user_message, media_context, conversation_history)

        tools_to_run = agent_plan.get("tools", [])
        delegated_subagent = agent_plan.get("delegated_subagent", "Orchestrator")
        clarification_needed = agent_plan.get("clarification_needed", False)
        clarification_options = agent_plan.get("clarification_options", [])
        suggested_actions = agent_plan.get("suggested_actions", [])
        if clarification_needed:
            tools_to_run = []
        if tools_to_run and not target_filepath:
            return jsonify({'error': 'Upload or select media before requesting an edit.'}), 400

        execution_results = []
        final_output_file = None
        final_output_url = None

        execution_started = time.perf_counter()
        if tools_to_run and target_filepath and os.path.exists(target_filepath):
            execution_results = agent_processor.execute_agent_plan(
                target_filepath,
                tools_to_run,
                app.config['PROCESSED_FOLDER'],
                context=media_context
            )

            # Main-chain result (side outputs such as thumbnails are listed as artifacts instead)
            final_output_file = agent_processor.final_output_of(execution_results)
            if final_output_file:
                final_output_url = _agent_public_url(final_output_file)

            # Inspector Sub-Agent checks outputs and generates proactive follow-up suggestions
            inspection = agent_processor.InspectorSubAgent.inspect(execution_results, final_output_file)
            if inspection.get("suggested_actions"):
                suggested_actions = inspection["suggested_actions"]

        failed = any(step.get('status') == 'error' for step in execution_results)
        status = 'partial' if failed and final_output_file else 'failed' if failed else 'success'
        reply = agent_processor.execution_reply(agent_plan, execution_results, final_output_file)
        public_results = [{**{k: v for k, v in step.items() if k != 'extra_outputs'},
                           'output_file': os.path.basename(step['output_file'])
                           if step.get('output_file') else None} for step in execution_results]
        artifacts = []
        for step in agent_processor.side_outputs_of(execution_results):
            artifacts.append({'tool': step.get('tool'), 'output_file': os.path.basename(step['output_file']),
                              'output_url': _agent_public_url(step['output_file'])})
            for extra in step.get('extra_outputs') or []:
                artifacts.append({'tool': step.get('tool'), 'label': extra.get('label'),
                                  'output_file': os.path.basename(extra['output_file']),
                                  'output_url': _agent_public_url(extra['output_file'])})
        if status == 'success' and execution_results:
            agent_skills.remember_plan(memory_session or 'default', tools_to_run, media_context.get('type'))
        # Copilot Memory records on a background writer; it can never delay or break this reply.
        agent_memory.observe_chat_async(
            session_id=memory_session, message=user_message, reply=reply,
            plan={'tools': tools_to_run}, results=execution_results, media_path=target_filepath,
            media_context=media_context,
            elapsed_ms=(time.perf_counter() - execution_started) * 1000.0 if execution_results else None)
        return jsonify({
            "status": status,
            "thought": agent_plan.get("thought", ""),
            "delegated_subagent": delegated_subagent,
            "clarification_needed": clarification_needed,
            "clarification_options": clarification_options,
            "reply": reply,
            "tools_planned": tools_to_run,
            "execution_results": public_results,
            "output_file": os.path.basename(final_output_file) if final_output_file else None,
            "output_url": final_output_url,
            "artifacts": artifacts,
            "plan_notes": agent_plan.get("plan_notes", []) if isinstance(agent_plan.get("plan_notes"), list) else [],
            "skills_used": agent_plan.get("skills_used", []) if isinstance(agent_plan.get("skills_used"), list) else [],
            "auto_skill": agent_plan.get("auto_skill") if isinstance(agent_plan.get("auto_skill"), str) else None,
            "suggested_actions": suggested_actions
        })

    except Exception as err:
        return jsonify({"error": str(err)}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  COPILOT SKILLS (catalog, details, personal skills)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/agent/skills', methods=['GET'])
def agent_skills_catalog():
    """Skill catalog for "@" autocomplete and the Skills library (capability language only)."""
    media_type = request.args.get('media_type')
    return jsonify({'status': 'success', **agent_skills.catalog(media_type)})


@app.route('/api/agent/skills/import', methods=['POST'])
def agent_skills_import():
    """Import a SKILL.md as a personal skill (validated like built-in skills; no code is ever run)."""
    content = None
    if 'file' in request.files:
        upload = request.files['file']
        raw = upload.read(agent_skills.MAX_SKILL_BYTES + 1)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'error': 'The skill file must be UTF-8 text.'}), 400
    else:
        data = request.get_json(silent=True) or {}
        content = data.get('content') if isinstance(data, dict) else None
    if not isinstance(content, str) or not content.strip():
        return jsonify({'error': 'Send a SKILL.md file or its text content.'}), 400
    try:
        skill = agent_skills.import_user_skill(content)
    except agent_skills.SkillError as err:
        return jsonify({'error': str(err)}), 400
    return jsonify({'status': 'success', 'skill': skill.public()})


@app.route('/api/agent/skills/<skill_id>', methods=['GET', 'DELETE'])
def agent_skill_detail(skill_id):
    """GET: one skill with its full procedure (read on demand). DELETE: remove a personal skill."""
    if request.method == 'DELETE':
        try:
            agent_skills.delete_user_skill(skill_id)
        except agent_skills.SkillError as err:
            return jsonify({'error': str(err)}), 400
        return jsonify({'status': 'success'})
    skill = agent_skills.registry().get(skill_id)
    if not skill:
        return jsonify({'error': 'Skill not found.'}), 404
    return jsonify({'status': 'success', 'skill': {**skill.public(), 'procedure': agent_skills.skill_body(skill.id) or ''}})


@app.route('/api/agent/skills/<skill_id>/rename', methods=['POST'])
def agent_skill_rename(skill_id):
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or not isinstance(data.get('new_id', ''), str) \
            or not isinstance(data.get('title', ''), (str, type(None))):
        return jsonify({'error': 'Send the new skill name.'}), 400
    try:
        skill = agent_skills.rename_user_skill(skill_id, data.get('new_id', ''), data.get('title'))
    except (agent_skills.SkillError, OSError) as err:
        return jsonify({'error': str(err) if isinstance(err, agent_skills.SkillError) else 'The skill could not be renamed.'}), 400
    return jsonify({'status': 'success', 'skill': skill.public()})


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


@app.route('/api/agent/memory', methods=['GET', 'DELETE'])
def agent_memory_endpoint():
    """Copilot Memory status (GET) and user-initiated forgetting (DELETE ?session_id=... or ?scope=all)."""
    _copilot_memory_ready()
    if request.method == 'GET':
        return jsonify(agent_memory.public_status())
    session_id = request.args.get('session_id')
    everything = request.args.get('scope') == 'all'
    if not everything and not agent_memory.valid_session_id(session_id):
        return jsonify({'error': 'Choose a conversation to forget, or scope=all to clear Copilot Memory.'}), 400
    try:
        removed = agent_memory.forget(session_id, everything=everything)
    except Exception as error:
        logger.warning('Copilot Memory could not be cleared (%s).', type(error).__name__)
        return jsonify({'error': 'Copilot Memory could not be cleared. Please try again.'}), 500
    return jsonify({'status': 'success', 'forgotten': removed})


def _agent_public_url(path):
    """Public URL for a Copilot input/output file (uploads are served from /media, results from /processed)."""
    rel_name = os.path.basename(path)
    if os.path.realpath(path).startswith(os.path.realpath(app.config['UPLOAD_FOLDER']) + os.sep):
        return f"/media/{rel_name}"
    return f"/processed/{rel_name}"


@app.route('/audio/lufs', methods=['POST'])
def audio_calculate_lufs():
    """Calculate integrated LUFS & True Peak loudness metrics."""
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400
    
    temp_path = save_temp_upload(file)
    try:
        metrics = audio_processor.calculate_lufs(temp_path)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/audio/eq', methods=['POST'])
def audio_apply_eq():
    """Apply 10-Band Parametric EQ."""
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400
    
    eq_json = request.form.get('eq_bands', '{}')
    try:
        eq_bands = json.loads(eq_json)
    except Exception:
        eq_bands = {}
        
    temp_path = save_temp_upload(file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"eq_{uuid.uuid4()}.wav")
    try:
        audio_processor.apply_parametric_eq(temp_path, out_path, eq_bands=eq_bands)
        return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name="eq_processed.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/audio/multitrack-mix', methods=['POST'])
def audio_multitrack_mix():
    """Mix multiple audio clips according to multitrack specs."""
    spec = request.get_json(silent=True)
    if not spec or 'tracks' not in spec:
        return jsonify({"error": "Invalid mix spec"}), 400
        
    out_format = spec.get('format', 'wav')
    out_name = f"mix_{uuid.uuid4()}.{out_format}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
    
    # Resolve relative paths inside tracks
    tracks = spec.get('tracks', [])
    resolved_tracks = []
    for t in tracks:
        media_id = t.get('media_id')
        p = _media_path(media_id)
        if p:
            t_copy = dict(t)
            t_copy['file_path'] = p
            resolved_tracks.append(t_copy)
            
    if not resolved_tracks:
        return jsonify({"error": "No valid media found in tracks spec"}), 400
        
    try:
        master_vol = float(spec.get('master_volume', 1.0))
        audio_processor.mix_audio_tracks(resolved_tracks, out_path, master_volume=master_vol, format=out_format)
        return send_file(out_path, mimetype=f"audio/{out_format}", as_attachment=True, download_name=f"multitrack_master.{out_format}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/audio')
def audio_editor():
    return render_template('index.html')

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


@app.route('/cut', methods=['POST'])
def cut_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename, ['audio', 'video']):
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        # 1. SETUP & SAVE INPUT
        unique_id = str(uuid.uuid4())
        original_ext = os.path.splitext(file.filename)[1] or ".webm"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}{original_ext}")
        file.save(input_path)
        
        # Calculate file size
        file_size = os.path.getsize(input_path)
        output_format = request.form.get('format', 'mp3')
        
        # Call the robust logger
        log_upload_details(
            request=request, 
            filename=file.filename, 
            file_size_bytes=file_size, 
            target_format=output_format
        )
        
        # 2. PARSE REGIONS (Multi-Region Support)
        regions_json = request.form.get('regions', '[]')
        regions = json.loads(regions_json)
        
        if not regions or len(regions) == 0:
            return jsonify({"error": "No regions provided"}), 400
        
        # 3. GET EXPORT MODE & EFFECTS
        export_mode = request.form.get('export_mode', 'merged')
        fade_in = request.form.get('fade_in') == 'true'
        fade_out = request.form.get('fade_out') == 'true'
        do_normalize = request.form.get('normalize') == 'true'
        do_reverse = request.form.get('reverse') == 'true'

        # 4. LOAD AUDIO
        audio = AudioSegment.from_file(input_path)
        
        # 5. PROCESS REGIONS
        processed_segments = []
        for region in regions:
            start_ms = float(region['start']) * 1000
            end_ms = float(region['end']) * 1000
            
            # Validation
            if end_ms > len(audio): 
                end_ms = len(audio)
            if start_ms >= end_ms:
                continue
            
            # Cut
            segment = audio[start_ms:end_ms]
            
            # Apply effects
            fade_duration = 2000 
            if len(segment) < 4000:
                fade_duration = min(2000, len(segment) // 2)

            if fade_in:
                segment = segment.fade_in(fade_duration)
            if fade_out:
                segment = segment.fade_out(fade_duration)
            if do_normalize:
                segment = _loudness_normalize_segment(segment)
            if do_reverse:
                segment = segment.reverse()
            
            processed_segments.append({
                'name': region.get('name', 'Region'),
                'audio': segment
            })
        
        if not processed_segments:
            return jsonify({"error": "No valid regions to process"}), 400
        
        # 6. EXPORT
        export_args = {}
        if output_format == 'mp3':
            export_args = {'format': 'mp3', 'bitrate': '320k'}
        else:
            export_args = {'format': 'wav'}
        
        if export_mode == 'merged' or len(processed_segments) == 1:
            # MERGE ALL SEGMENTS
            merged = processed_segments[0]['audio']
            for seg in processed_segments[1:]:
                merged += seg['audio']  # Concatenate
            
            output_filename = f"merged_{unique_id}.{output_format}"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            merged.export(output_path, **export_args)
            
            return send_file(
                output_path, 
                as_attachment=True, 
                download_name=f'merged_audio.{output_format}'
            )
        
        else:
            # EXPORT SEPARATE FILES (ZIP)
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, seg_data in enumerate(processed_segments, 1):
                    temp_buffer = BytesIO()
                    seg_data['audio'].export(temp_buffer, **export_args)
                    temp_buffer.seek(0)
                    
                    safe_name = seg_data['name'].replace(' ', '_').replace('/', '_')
                    filename = f"{i:02d}_{safe_name}.{output_format}"
                    zip_file.writestr(filename, temp_buffer.read())
            
            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='audio_cuts.zip'
            )

    except Exception as e:
        logger.error(f"Error cutting audio: {e}", exc_info=True)
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

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

@app.route('/ai/detect-silence', methods=['POST'])
def ai_detect_silence():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    
    min_silence_len = float(request.form.get('min_silence_len', 0.5))
    silence_thresh = float(request.form.get('silence_thresh', 40))
    
    temp_path = save_temp_upload(file)
    try:
        results = ai_processor.detect_silence(temp_path, min_silence_len, silence_thresh)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ai/auto-trim', methods=['POST'])
def ai_auto_trim():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    
    threshold = float(request.form.get('threshold', 40))
    
    temp_path = save_temp_upload(file)
    try:
        results = ai_processor.auto_trim_silence(temp_path, threshold)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ai/detect-beats', methods=['POST'])
def ai_detect_beats():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    
    temp_path = save_temp_upload(file)
    try:
        results = ai_processor.detect_beats(temp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ai/detect-vad', methods=['POST'])
def ai_detect_vad():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    
    threshold_db = float(request.form.get('threshold_db', -35.0))
    
    temp_path = save_temp_upload(file)
    try:
        results = ai_processor.detect_voice_activity(temp_path, threshold_db)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ai/transcribe', methods=['POST'])
def ai_transcribe():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    
    temp_path = save_temp_upload(file)
    try:
        results = ai_processor.transcribe_audio(temp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ai/noise-reduce', methods=['POST'])
def ai_noise_reduce():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    
    temp_path = save_temp_upload(file)
    try:
        unique_id = str(uuid.uuid4())
        output_filename = f"denoised_{unique_id}.wav"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        ai_processor.reduce_noise(temp_path, output_path)
        
        base_name = os.path.splitext(file.filename)[0]
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"denoised_{base_name}.wav"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/ai/filler-words', methods=['POST'])
def ai_filler_words():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err

    temp_path = save_temp_upload(file)
    try:
        results = ai_processor.detect_filler_words(temp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/ai/enhance-speech', methods=['POST'])
def ai_enhance_speech():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err

    temp_path = save_temp_upload(file)
    try:
        unique_id = str(uuid.uuid4())
        output_filename = f"enhanced_speech_{unique_id}.wav"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        res = ai_processor.enhance_speech_studio(temp_path, output_path)

        base_name = os.path.splitext(file.filename)[0]
        resp = send_file(
            output_path,
            as_attachment=True,
            download_name=f"enhanced_{base_name}.wav"
        )
        resp.headers['X-Enhance-Engine'] = res.get('engine', 'unknown')
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/ai/separate-stems', methods=['POST'])
def ai_separate_stems():
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err

    temp_path = save_temp_upload(file)
    out_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"stems_{uuid.uuid4()}")
    try:
        res = ai_processor.separate_stems(temp_path, out_dir)
        return jsonify(res)
    except ValueError as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        logger.error('Stem separation failed (%s)', type(e).__name__)
        return jsonify({"error": "Separation could not be completed. Please try again."}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Source separation (vocals / karaoke / 4 stems / voice) and lyrics ──────

def _separation_base_name(filename):
    base = os.path.splitext(os.path.basename(filename or 'audio'))[0]
    base = re.sub(r'[^\w.-]+', '_', base, flags=re.UNICODE).strip('._')
    return (base or 'audio')[:80]


def _processed_url(path):
    rel = os.path.relpath(path, app.config['PROCESSED_FOLDER']).replace(os.sep, '/')
    return f"/processed/{rel}"


@app.route('/ai/separate/capabilities', methods=['GET'])
def ai_separate_capabilities():
    return jsonify(separation_processor.capability_status())


@app.route('/ai/separate', methods=['POST'])
def ai_separate():
    """mode: vocals|karaoke|4stem|voice, format: wav|mp3, quality: auto|fast|best,
    delivery: file (default — ZIP for several stems, the file itself for one) | json."""
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    try:
        mode, fmt, quality = separation_processor.validate_request(
            request.form.get('mode'), request.form.get('format'), request.form.get('quality'))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    delivery = (request.form.get('delivery') or 'file').strip().lower()

    temp_path = save_temp_upload(file)
    out_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"stems_{uuid.uuid4().hex}")
    try:
        report = separation_processor.separate(temp_path, out_dir, mode=mode, quality=quality, fmt=fmt)
    except ValueError as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        logger.error('Separation failed (%s)', type(e).__name__)
        return jsonify({"error": "Separation could not be completed. Please try again."}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    base = _separation_base_name(file.filename)
    stems = report['stems']
    zip_path = None
    if len(stems) > 1:
        zip_path = os.path.join(out_dir, f"{base}_stems.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED if fmt == 'mp3' else zipfile.ZIP_DEFLATED) as zf:
            for name, path in stems.items():
                zf.write(path, f"{base}_{name}.{fmt}")
    headers = {
        'X-Separation-Quality': report['quality']['label'],
        'X-Separation-Mode': report['mode_label'],
    }

    if delivery == 'json':
        payload = {
            "status": "success",
            "mode": report['mode'],
            "mode_label": report['mode_label'],
            "format": fmt,
            "quality": report['quality'],
            "quality_note": report['quality_note'],
            "duration": report['duration'],
            "sample_rate": report['sample_rate'],
            "channels": report['channels'],
            "checks": report['checks'],
            "warnings": report['warnings'],
            "elapsed_sec": report['elapsed_sec'],
            "realtime_factor": report['realtime_factor'],
            "stems": [{
                "name": name,
                "label": report['stem_labels'][name],
                "url": _processed_url(path),
                "download_name": f"{base}_{name}.{fmt}",
                "loudness": report['loudness'].get(name, {}),
            } for name, path in stems.items()],
            "zip_url": _processed_url(zip_path) if zip_path else None,
        }
        resp = jsonify(payload)
    elif zip_path:
        resp = send_file(zip_path, mimetype='application/zip', as_attachment=True,
                         download_name=f"{base}_stems.zip")
    else:
        name, path = next(iter(stems.items()))
        resp = send_file(path, mimetype='audio/mpeg' if fmt == 'mp3' else 'audio/wav', as_attachment=True,
                         download_name=f"{base}_{name}.{fmt}")
    for key, value in headers.items():
        resp.headers[key] = value
    return resp


@app.route('/ai/lyrics', methods=['POST'])
def ai_lyrics():
    """Isolate the vocals, transcribe them; returns text, timed lines and SRT/VTT links."""
    file, err = get_uploaded_file(request, ('audio', 'video'))
    if err: return err
    try:
        _, _, quality = separation_processor.validate_request('vocals', 'wav', request.form.get('quality'))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    temp_path = save_temp_upload(file)
    try:
        result = separation_processor.lyrics(temp_path, quality=quality)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except separation_processor.SeparationUnavailableError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error('Lyrics extraction failed (%s)', type(e).__name__)
        return jsonify({"error": "Lyrics could not be extracted. Please try again."}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    base = _separation_base_name(file.filename)
    out_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"lyrics_{uuid.uuid4().hex}")
    os.makedirs(out_dir, exist_ok=True)
    links = {}
    for ext in ('srt', 'vtt', 'txt'):
        path = os.path.join(out_dir, f"{base}_lyrics.{ext}")
        body = result['text'] + "\n" if ext == 'txt' else result[ext]
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(body)
        links[f"{ext}_url"] = _processed_url(path)
    payload = {key: value for key, value in result.items() if key not in ('srt', 'vtt')}
    payload.update(links)
    resp = jsonify(payload)
    resp.headers['X-Separation-Quality'] = result['quality']['label']
    return resp


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


@app.route('/video')
def video_editor():
    return render_template('video.html')


@app.route('/video/detect-scenes', methods=['POST'])
def video_detect_scenes():
    if 'file' not in request.files: return jsonify({"error": "No video file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400

    temp_path = save_temp_upload(file)
    try:
        threshold = float(request.form.get('threshold', 27.0))
        scenes = video_processor.detect_scenes(temp_path, threshold=threshold)
        return jsonify({"status": "success", "scenes": scenes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/image')
def image_editor():
    # Editing is client-side (HTML Canvas). Only optional AI calls reach backend.
    return render_template('image.html')


@app.route('/image/remove-bg', methods=['POST'])
def image_remove_bg():
    if 'file' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image"}), 400

    model_name = request.form.get('model', 'auto')
    alpha_matting = request.form.get('alpha_matting', 'true').lower() == 'true'

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"bg_in_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"bg_out_{uid}.png")
    file.save(in_path)

    try:
        image_processor.remove_bg(in_path, out_path, model_name=model_name, alpha_matting=alpha_matting)
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


@app.route('/image/clarity', methods=['POST'])
def image_clarity():
    """Real-time AI Photo Clarity, Denoise & Dynamic Range Polish."""
    if 'file' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image"}), 400

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"clarity_in_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"clarity_out_{uid}.png")
    file.save(in_path)

    try:
        image_processor.enhance_photo_clarity(in_path, out_path)
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


@app.route('/image/inpaint', methods=['POST'])
def image_inpaint():
    if 'file' not in request.files or 'mask' not in request.files:
        return jsonify({"error": "Missing image or mask file"}), 400
    file = request.files['file']
    mask = request.files['mask']

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"inp_in_{uid}.png")
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"inp_mask_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"inp_out_{uid}.png")
    file.save(in_path)
    mask.save(mask_path)

    try:
        method = request.form.get('method', 'telea')
        image_processor.inpaint_object(in_path, mask_path, out_path, method=method)
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (in_path, mask_path):
            if os.path.exists(p): os.remove(p)


@app.route('/image/restore-faces', methods=['POST'])
def image_restore_faces():
    if 'file' not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files['file']

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"face_in_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"face_out_{uid}.png")
    file.save(in_path)

    try:
        image_processor.restore_faces(in_path, out_path)
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(in_path): os.remove(in_path)


@app.route('/video/burn-subtitles', methods=['POST'])
def video_burn_subtitles():
    if 'file' not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files['file']

    temp_path = save_temp_upload(file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"subbed_{uuid.uuid4()}.mp4")
    try:
        style = request.form.get('style', 'yellow_box')
        video_processor.burn_subtitles(temp_path, out_path, style=style)
        return send_file(out_path, mimetype='video/mp4', as_attachment=True, download_name="subtitled_video.mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.route('/video/enhance-quality', methods=['POST'])
def video_enhance_quality():
    if 'file' not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files['file']

    temp_path = save_temp_upload(file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"enhanced_{uuid.uuid4()}.mp4")
    try:
        mode = request.form.get('mode', '1080p')
        video_processor.enhance_video_quality(temp_path, out_path, mode=mode)
        return send_file(out_path, mimetype='video/mp4', as_attachment=True, download_name="enhanced_quality_video.mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.route('/video/interpolate', methods=['POST'])
def video_interpolate():
    if 'file' not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files['file']

    temp_path = save_temp_upload(file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"smooth60_{uuid.uuid4()}.mp4")
    try:
        target_fps = int(request.form.get('fps', 60))
        video_processor.interpolate_video_fps(temp_path, out_path, target_fps=target_fps)
        return send_file(out_path, mimetype='video/mp4', as_attachment=True, download_name="smooth_60fps_video.mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.route('/image/color-match', methods=['POST'])
def image_color_match():
    if 'file' not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files['file']

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"col_in_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"col_out_{uid}.png")
    file.save(in_path)

    try:
        palette = request.form.get('palette', 'teal_orange')
        image_processor.color_match_transfer(in_path, out_path, palette_mode=palette)
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(in_path): os.remove(in_path)


@app.route('/ai/auto-duck', methods=['POST'])
def ai_auto_duck():
    if 'speech' not in request.files or 'music' not in request.files:
        return jsonify({"error": "Missing speech or music audio file"}), 400
    speech_file = request.files['speech']
    music_file = request.files['music']

    speech_path = save_temp_upload(speech_file)
    music_path = save_temp_upload(music_file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"ducked_{uuid.uuid4()}.wav")

    try:
        ai_processor.auto_duck_music(speech_path, music_path, out_path)
        return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name="auto_ducked_music.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (speech_path, music_path):
            if os.path.exists(p): os.remove(p)


@app.route('/ai/pitch-speed', methods=['POST'])
def ai_pitch_speed():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files['file']

    temp_path = save_temp_upload(file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"speed_{uuid.uuid4()}.wav")

    try:
        speed = float(request.form.get('speed', 1.25))
        ai_processor.pitch_preserved_speed(temp_path, out_path, speed=speed)
        return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name="pitch_preserved_speed.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


# ── Voiceover (local text-to-speech). Public capability labels only. ──
@app.route('/ai/tts/voices', methods=['GET'])
def ai_tts_voices():
    import tts_processor
    voices = tts_processor.list_voices(refresh=request.args.get('refresh') == '1')
    return jsonify({
        "status": "success",
        "available": bool(voices),
        "voices": voices,
        "default_voice": next((v["id"] for v in voices if v.get("default")), None),
        "max_chars": tts_processor.MAX_TEXT_CHARS,
        "speed_range": [tts_processor.SPEED_MIN, tts_processor.SPEED_MAX],
        "formats": sorted(tts_processor.OUTPUT_FORMATS),
    })


@app.route('/ai/tts', methods=['POST'])
def ai_tts():
    import tts_processor
    data = request.get_json(silent=True) if request.is_json else None
    if not isinstance(data, dict):
        data = request.form
    fmt = str(data.get('format') or 'wav').strip().lower()
    if fmt not in tts_processor.OUTPUT_FORMATS:
        return jsonify({"status": "error", "error": "Choose WAV or MP3 for the voiceover file."}), 400
    filename = f"voiceover_{uuid.uuid4().hex[:12]}.{fmt}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    try:
        report = tts_processor.synthesize(
            data.get('text', ''), out_path, voice=data.get('voice') or None,
            speed=data.get('speed', 1.0), pitch=data.get('pitch', 0), fmt=fmt,
            quality=data.get('quality') or 'auto')
    except tts_processor.VoiceoverUnavailableError as e:
        return jsonify({"status": "error", "error": str(e)}), 422
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error('Voiceover failed (%s)', type(e).__name__)
        return jsonify({"status": "error", "error": "The voiceover could not be created. Please try again."}), 500
    resp = send_file(out_path, mimetype=tts_processor.OUTPUT_FORMATS[fmt], as_attachment=True,
                     download_name=f"voiceover.{fmt}")
    resp.headers['X-Voice-Quality'] = report['tier']
    resp.headers['X-Voice-Label'] = report['voice']
    resp.headers['X-Audio-Duration'] = f"{report['duration']:.2f}"
    resp.headers['X-Output-File'] = filename
    if report.get('notice'):
        resp.headers['X-Voice-Notice'] = report['notice']
    resp.headers['Access-Control-Expose-Headers'] = (
        'X-Voice-Quality, X-Voice-Label, X-Audio-Duration, X-Output-File, X-Voice-Notice')
    return resp


@app.route('/ai/youtube-chapters', methods=['POST'])
def ai_youtube_chapters():
    if 'file' not in request.files:
        return jsonify({"error": "No media file"}), 400
    file = request.files['file']

    temp_path = save_temp_upload(file)
    try:
        res = ai_processor.generate_youtube_chapters(temp_path)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.route('/ai/trim-silence', methods=['POST'])
def ai_trim_silence():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files['file']

    temp_path = save_temp_upload(file)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"trimmed_{uuid.uuid4()}.wav")
    try:
        min_silence_len = float(request.form.get('min_silence_len', 1.0))
        ai_processor.trim_silence_gaps(temp_path, out_path, min_silence_len=min_silence_len)
        return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name="trimmed_audio.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.route('/image/enhance', methods=['POST'])
def image_enhance():
    """Real local super-resolution ('Increase Quality') via OpenCV dnn_superres."""
    if 'file' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image"}), 400

    try:
        scale = int(request.form.get('scale', 2))
    except (TypeError, ValueError):
        scale = 2
    if scale not in (2, 4):
        scale = 2
    model_key = request.form.get('model', 'auto')
    if model_key not in ('auto', 'fast', 'best'):
        model_key = 'auto'

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enh_in_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"enh_out_{uid}.png")
    file.save(in_path)

    try:
        info = image_processor.upscale(in_path, out_path, scale=scale, model_key=model_key)
        resp = send_file(out_path, mimetype='image/png')
        public_tiers = {'edsr': 'best', 'fsrcnn': 'fast'}
        resp.headers['X-Enhance-Engine'] = public_tiers.get(str(info.get('engine', '')).lower(), 'standard')
        resp.headers['X-Enhance-Scale'] = str(info.get('scale', scale))
        resp.headers['X-Enhance-Downgraded'] = '1' if info.get('downgraded') else '0'
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


@app.route('/video/upload', methods=['POST'])
def video_upload():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400

    if not allowed_file(file.filename, ('audio', 'video')):
        return jsonify({"error": "Unsupported file type. Upload a video or audio file."}), 400

    unique_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    media_id = f"{unique_id}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], media_id)
    file.save(path)

    try:
        log_upload_details(
            request=request,
            filename=file.filename,
            file_size_bytes=os.path.getsize(path),
            target_format='video-editor-upload'
        )
    except Exception:
        pass  # logging must never block an upload

    try:
        info = video_processor.probe_media(path)
    except Exception as e:
        logger.warning("Uploaded media could not be read (%s)", type(e).__name__)
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({"error": "Could not read this media file. It may be corrupt or in an unsupported format."}), 400

    thumbs = []
    if info['has_video']:
        thumb_dir = os.path.join(THUMBS_FOLDER, unique_id)
        made = video_processor.generate_filmstrip(path, thumb_dir, info['duration'])
        thumbs = [f"/video/thumb/{unique_id}/{i}" for i in range(len(made))]

    return jsonify({
        "id": media_id,
        "name": file.filename,
        "url": f"/media/{media_id}",
        "size": os.path.getsize(path),
        "thumbs": thumbs,
        **info,
    })


@app.route('/media/<media_id>')
def serve_media(media_id):
    path = _media_path(media_id)
    if not path: abort(404)
    return send_file(path, conditional=True)  # range requests for <video> seek


@app.route('/video/thumb/<uid>/<int:n>')
def serve_thumb(uid, n):
    if not re.match(r'^[A-Za-z0-9-]+$', uid) or n < 0 or n > 50: abort(404)
    path = os.path.join(THUMBS_FOLDER, uid, f"thumb_{n}.jpg")
    if not os.path.exists(path): abort(404)
    return send_file(path, max_age=86400)


@app.route('/video/extract-audio', methods=['POST'])
def video_extract_audio():
    """Quick tool: extract the audio track from an uploaded video."""
    media_id = request.form.get('media_id', '')
    path = _media_path(media_id)
    if not path:
        return jsonify({"error": "Media not found — upload it first."}), 404

    fmt = request.form.get('format', 'mp3')
    if fmt not in ('mp3', 'wav'): fmt = 'mp3'
    bitrate = request.form.get('bitrate', '192k')
    if bitrate not in ('128k', '192k', '256k', '320k'): bitrate = '192k'

    out_name = f"extracted_{uuid.uuid4()}.{fmt}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
    try:
        video_processor.extract_audio(path, out_path, fmt=fmt, bitrate=bitrate)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    base = os.path.splitext(request.form.get('name', 'video'))[0] or 'video'
    return send_file(out_path, as_attachment=True,
                     download_name=f"{base}_audio.{fmt}")


@app.route('/video/quick', methods=['POST'])
def video_quick():
    """One-click beginner tools: gif / compress / convert / frame / mute."""
    media_id = request.form.get('media_id', '')
    path = _media_path(media_id)
    if not path:
        return jsonify({"error": "Media not found — upload it first."}), 404

    op = request.form.get('op', '')
    base = os.path.splitext(request.form.get('name', 'video'))[0] or 'video'
    uid = uuid.uuid4()

    def _num(key, default):
        try:
            return float(request.form.get(key, default))
        except (TypeError, ValueError):
            return default

    try:
        if op == 'gif':
            out = os.path.join(PROCESSED_FOLDER, f"gif_{uid}.gif")
            video_processor.quick_to_gif(
                path, out, start=_num('start', 0), duration=_num('duration', 5))
            return send_file(out, as_attachment=True, download_name=f"{base}.gif")

        if op == 'compress':
            level = request.form.get('level', 'balanced')
            out = os.path.join(PROCESSED_FOLDER, f"compressed_{uid}.mp4")
            video_processor.quick_compress(path, out, level=level)
            return send_file(out, as_attachment=True,
                             download_name=f"{base}_compressed.mp4")

        if op == 'convert':
            container = request.form.get('container', 'mp4')
            if container not in ('mp4', 'webm', 'mkv'): container = 'mp4'
            quality = request.form.get('quality', '720p')
            out = os.path.join(PROCESSED_FOLDER, f"converted_{uid}.{container}")
            video_processor.quick_convert(path, out, container=container, quality=quality)
            return send_file(out, as_attachment=True,
                             download_name=f"{base}.{container}")

        if op == 'frame':
            out = os.path.join(PROCESSED_FOLDER, f"frame_{uid}.jpg")
            video_processor.quick_extract_frame(path, out, t=_num('t', 0))
            return send_file(out, as_attachment=True, download_name=f"{base}_frame.jpg")

        if op == 'mute':
            out = os.path.join(PROCESSED_FOLDER, f"muted_{uid}.mp4")
            video_processor.quick_mute(path, out)
            return send_file(out, as_attachment=True, download_name=f"{base}_muted.mp4")

        return jsonify({"error": f"Unknown operation: {op}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


EXPORT_FORMATS = {
    'mp4': 'video/mp4',
    'webm': 'video/webm',
    'gif': 'image/gif',
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
}
GIF_EXPORT_MAX_SECONDS = 30.0


@app.route('/video/export', methods=['POST'])
def video_export():
    """Render the full timeline (clips + text + music) via FFmpeg."""
    spec = request.get_json(silent=True)
    if not spec:
        return jsonify({"error": "Invalid export request."}), 400

    fmt = str(spec.get('format', 'mp4')).lower()
    if fmt not in EXPORT_FORMATS:
        return jsonify({"error": "Unsupported export format. Choose MP4, WebM, GIF, MP3, or WAV."}), 400
    # WebM and GIF are transcoded from a master MP4 render.
    spec['format'] = 'mp4' if fmt in ('webm', 'gif') else fmt

    work_dir = tempfile.mkdtemp(prefix='vexport_', dir=app.config['PROCESSED_FOLDER'])
    out_name = f"video_export_{uuid.uuid4()}.{fmt}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)

    try:
        if fmt in ('webm', 'gif'):
            master_path = os.path.join(work_dir, 'master.mp4')
            video_processor.export_project(spec, _media_path, work_dir, master_path)
            if fmt == 'webm':
                video_processor.quick_convert(master_path, out_path, container='webm', quality='original')
            else:
                duration = video_processor.probe_media(master_path).get('duration') or GIF_EXPORT_MAX_SECONDS
                video_processor.quick_to_gif(master_path, out_path, start=0.0,
                                             duration=min(float(duration), GIF_EXPORT_MAX_SECONDS),
                                             fps=15, width=640)
        else:
            video_processor.export_project(spec, _media_path, work_dir, out_path)
        return send_file(out_path, as_attachment=True,
                         mimetype=EXPORT_FORMATS[fmt],
                         download_name=f"edited_video.{fmt}")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Master export failed (%s)", type(e).__name__)
        return jsonify({"error": "Export failed."}), 500
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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
