from flask import Blueprint, request, jsonify, send_file, current_app, abort, Response, stream_with_context, render_template

import os

import uuid

import time

import json

import shutil

import re

import zipfile

import tempfile

from werkzeug.utils import secure_filename

import audio_processor

import video_processor

import image_processor

import ai_processor

import separation_processor

import agent_processor

import agent_memory

import agent_skills

from app import (app, save_temp_upload, get_uploaded_file, _media_path, _processed_url, 

                 THUMBS_FOLDER, PROCESSED_FOLDER, EXPORT_FORMATS, GIF_EXPORT_MAX_SECONDS, 

                 logger, log_upload_details, _separation_base_name, allowed_file, 

                 _process_and_register_uploaded_media, _copilot_memory_ready, _memory_repeat_plan, 

                 _agent_public_url, _touch_in_use, _loudness_normalize_segment)

import threading
from app import _maybe_cleanup_temp_files, _bg_jobs_lock, _bg_jobs, _bg_job_worker



assistant_bp = Blueprint('assistant', __name__)



@assistant_bp.route('/agent')
@assistant_bp.route('/chat')
def agent_full_window():
    _maybe_cleanup_temp_files()
    return render_template('agent.html')

@assistant_bp.route('/api/agent/tools', methods=['GET'])
def agent_get_tools():
    """Returns registered AI agent tool schemas and active Sub-Agents."""
    return jsonify({
        "status": "success",
        "subagents": ["VisionSubAgent", "VideoSubAgent", "AudioSubAgent", "InspectorSubAgent", "MasterOrchestrator"],
        "tools": agent_processor.TOOL_DEFINITIONS
    })

@assistant_bp.route('/api/agent/samples', methods=['GET'])
def agent_get_samples():
    """Returns list of bundled demo media available for immediate testing."""
    samples = [
        {
            "id": "sample_clip",
            "type": "video",
            "title": "Sample Video Clip",
            "description": "1080p clip with speech and scene motion for trimming, speed, and GIF",
            "icon": "fa-film",
            "filename": "sample_clip.mp4"
        },
        {
            "id": "sample_portrait",
            "type": "image",
            "title": "Sample Portrait Photo",
            "description": "High-res portrait photo for background cutout, enhance, and 4x upscaling",
            "icon": "fa-image",
            "filename": "sample_portrait.png"
        },
        {
            "id": "sample_voiceover",
            "type": "audio",
            "title": "Sample Voice Recording",
            "description": "Spoken voice track with ambient noise for denoise, STT, and isolation",
            "icon": "fa-wave-square",
            "filename": "sample_voiceover.wav"
        }
    ]
    return jsonify({"status": "success", "samples": samples})

@assistant_bp.route('/api/agent/sample/load', methods=['POST'])
def agent_load_sample():
    """Loads a demo sample media item directly into the active upload workspace."""
    data = request.get_json(silent=True) or {}
    sample_id = str(data.get("sample_id") or "").strip()
    sample_map = {
        "sample_clip": "sample_clip.mp4",
        "sample_portrait": "sample_portrait.png",
        "sample_voiceover": "sample_voiceover.wav"
    }
    sample_file = sample_map.get(sample_id)
    if not sample_file:
        return jsonify({"error": "Invalid or unknown sample identifier."}), 400

    static_sample_path = os.path.join(app.root_path, "static", "samples", sample_file)
    if not os.path.exists(static_sample_path):
        return jsonify({"error": f"Sample asset '{sample_file}' not found on server."}), 404

    unique_id = str(uuid.uuid4())
    orig_ext = os.path.splitext(sample_file)[1].lower() or ".bin"
    saved_filename = f"agent_{unique_id}{orig_ext}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
    shutil.copyfile(static_sample_path, saved_path)

    return _process_and_register_uploaded_media(saved_path, sample_file)

@assistant_bp.route('/api/agent/upload', methods=['POST'])
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

    return _process_and_register_uploaded_media(saved_path, file.filename)

@assistant_bp.route('/api/agent/upload/chunk', methods=['POST'])
def agent_upload_chunk():
    """Receives an individual chunk of a large media upload for resilient network transfers."""
    if 'file' not in request.files:
        return jsonify({"error": "No chunk payload received"}), 400
    chunk = request.files['file']
    upload_id = request.form.get('upload_id', '').strip()
    if not upload_id or not re.match(r'^[a-zA-Z0-9_\-]+$', upload_id):
        return jsonify({"error": "Invalid upload session identifier"}), 400

    filename = request.form.get('filename', '').strip()
    if not filename or not allowed_file(filename):
        return jsonify({"error": "Unsupported media format"}), 400

    try:
        chunk_index = int(request.form.get('chunk_index', -1))
        total_chunks = int(request.form.get('total_chunks', 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid chunk coordinates"}), 400

    if chunk_index < 0 or total_chunks <= 0 or chunk_index >= total_chunks:
        return jsonify({"error": "Chunk index out of bounds"}), 400

    chunk_dir = os.path.join(app.config['UPLOAD_FOLDER'], '.chunks', upload_id)
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_path = os.path.join(chunk_dir, f"part_{chunk_index:05d}.chunk")
    chunk.save(chunk_path)
    received_bytes = os.path.getsize(chunk_path)

    return jsonify({
        "status": "chunk_received",
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "received_bytes": received_bytes
    })

@assistant_bp.route('/api/agent/upload/complete', methods=['POST'])
def agent_upload_complete():
    """Stitches verified upload chunks into final media file and runs probe/perception."""
    data = request.get_json(silent=True) or {}
    upload_id = str(data.get('upload_id') or '').strip()
    filename = str(data.get('filename') or '').strip()
    try:
        total_chunks = int(data.get('total_chunks', 0))
    except (ValueError, TypeError):
        total_chunks = 0

    if not upload_id or not re.match(r'^[a-zA-Z0-9_\-]+$', upload_id):
        return jsonify({"error": "Invalid upload session identifier"}), 400
    if not filename or not allowed_file(filename):
        return jsonify({"error": "Unsupported media file format"}), 400
    if total_chunks <= 0:
        return jsonify({"error": "Invalid total chunks count"}), 400

    chunk_dir = os.path.join(app.config['UPLOAD_FOLDER'], '.chunks', upload_id)
    if not os.path.isdir(chunk_dir):
        return jsonify({"error": "Upload session not found or already completed"}), 404

    # Verify all parts exist before stitching
    for i in range(total_chunks):
        part_path = os.path.join(chunk_dir, f"part_{i:05d}.chunk")
        if not os.path.exists(part_path):
            return jsonify({"error": f"Missing chunk part {i} of {total_chunks}. Please retry."}), 400

    orig_ext = os.path.splitext(filename)[1].lower() or '.bin'
    saved_filename = f"agent_{uuid.uuid4()}{orig_ext}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)

    try:
        with open(saved_path, 'wb') as outfile:
            for i in range(total_chunks):
                part_path = os.path.join(chunk_dir, f"part_{i:05d}.chunk")
                with open(part_path, 'rb') as infile:
                    while True:
                        buf = infile.read(1024 * 1024)
                        if not buf:
                            break
                        outfile.write(buf)
    except Exception as exc:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return jsonify({"error": f"Failed to assemble media upload: {exc}"}), 500
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    return _process_and_register_uploaded_media(saved_path, filename)

@assistant_bp.route('/api/agent/upload/abort', methods=['POST'])
def agent_upload_abort():
    """Cancels and purges temporary chunks for an aborted upload."""
    data = request.get_json(silent=True) or {}
    upload_id = str(data.get('upload_id') or '').strip()
    if upload_id and re.match(r'^[a-zA-Z0-9_\-]+$', upload_id):
        chunk_dir = os.path.join(app.config['UPLOAD_FOLDER'], '.chunks', upload_id)
        if os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)
    return jsonify({"status": "aborted", "upload_id": upload_id})

@assistant_bp.route('/api/agent/chat', methods=['POST'])
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

        if tools_to_run and target_filepath and os.path.exists(target_filepath):
            is_long = any(t.get('name') in ('reduce_noise', 'enhance_speech', 'isolate_voice', 'remove_vocals', 'separate_stems', 'upscale_image', 'generate_voiceover', 'transcribe_audio') for t in tools_to_run)
            if is_long and not app.config.get('TESTING'):
                return jsonify({
                    "status": "success",
                    "thought": agent_plan.get("thought", ""),
                    "delegated_subagent": delegated_subagent,
                    "clarification_needed": True,
                    "clarification_options": ["Yes, start the background job", "No, cancel"],
                    "reply": "This edit involves AI models and may take a few minutes to process. Should I start it in the background?",
                    "tools_planned": tools_to_run,
                    "execution_results": [],
                    "output_file": None,
                    "output_url": None,
                    "artifacts": [],
                    "plan_notes": agent_plan.get("plan_notes", []) if isinstance(agent_plan.get("plan_notes"), list) else [],
                    "skills_used": agent_plan.get("skills_used", []) if isinstance(agent_plan.get("skills_used"), list) else [],
                    "auto_skill": agent_plan.get("auto_skill") if isinstance(agent_plan.get("auto_skill"), str) else None,
                    "suggested_actions": []
                })

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
            "suggested_actions": suggested_actions,
            "elapsed_seconds": round(time.perf_counter() - execution_started, 2) if execution_results else 0
        })

    except Exception as err:
        return jsonify({"error": str(err)}), 500

@assistant_bp.route('/api/agent/skills', methods=['GET'])
def agent_skills_catalog():
    """Skill catalog for "@" autocomplete and the Skills library (capability language only)."""
    media_type = request.args.get('media_type')
    return jsonify({'status': 'success', **agent_skills.catalog(media_type)})

@assistant_bp.route('/api/agent/skills/import', methods=['POST'])
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

@assistant_bp.route('/api/agent/skills/<skill_id>', methods=['GET', 'DELETE'])
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

@assistant_bp.route('/api/agent/skills/<skill_id>/rename', methods=['POST'])
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

@assistant_bp.route('/api/agent/memory', methods=['GET', 'DELETE'])
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

@assistant_bp.route('/api/agent/job/submit', methods=['POST'])
def agent_job_submit():
    """Submit a long edit as a background job; returns a job_id for polling."""
    data = request.get_json(silent=True) or {}
    tools_to_run = data.get('tools_to_run') or []
    filename = (data.get('filename') or '').strip()
    media_context = data.get('context') or {}
    if not tools_to_run or not filename:
        return jsonify({'error': 'tools_to_run and filename are required.'}), 400
    safe = os.path.basename(filename)
    target = None
    for folder in (app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']):
        candidate = os.path.join(folder, safe)
        if os.path.isfile(candidate):
            target = candidate
            break
    if not target:
        return jsonify({'error': 'File not found. Upload it first.'}), 404
    job_id = uuid.uuid4().hex
    cancel_event = threading.Event()
    with _bg_jobs_lock:
        _bg_jobs[job_id] = {'status': 'running', 'progress': 0, 'message': 'Starting…',
                             'result': None, 'error': None, 'cancel_event': cancel_event}
    t = threading.Thread(target=_bg_job_worker,
                         args=(job_id, target, tools_to_run, media_context, app.config['PROCESSED_FOLDER']),
                         daemon=True)
    t.start()
    return jsonify({'status': 'accepted', 'job_id': job_id})

@assistant_bp.route('/api/agent/job/<job_id>', methods=['GET'])
def agent_job_status(job_id):
    """Poll a background job: returns status, progress (0-100), message, and result when done."""
    with _bg_jobs_lock:
        job = _bg_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found.'}), 404
    payload = {k: v for k, v in job.items() if k != 'cancel_event'}
    return jsonify(payload)

@assistant_bp.route('/api/agent/job/<job_id>/cancel', methods=['POST'])
def agent_job_cancel(job_id):
    """Request cancellation of a running background job."""
    with _bg_jobs_lock:
        job = _bg_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found.'}), 404
    job['cancel_event'].set()
    with _bg_jobs_lock:
        if _bg_jobs.get(job_id) and _bg_jobs[job_id]['status'] == 'running':
            _bg_jobs[job_id]['status'] = 'cancelling'
    return jsonify({'status': 'cancelling', 'job_id': job_id})

@assistant_bp.route('/ai/detect-silence', methods=['POST'])
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

@assistant_bp.route('/ai/auto-trim', methods=['POST'])
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

@assistant_bp.route('/ai/detect-beats', methods=['POST'])
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

@assistant_bp.route('/ai/detect-vad', methods=['POST'])
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

@assistant_bp.route('/ai/transcribe', methods=['POST'])
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

@assistant_bp.route('/ai/noise-reduce', methods=['POST'])
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

@assistant_bp.route('/ai/filler-words', methods=['POST'])
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

@assistant_bp.route('/ai/enhance-speech', methods=['POST'])
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
        if res.get('fallback'):
            resp.headers['X-Enhance-Fallback'] = 'true'
            resp.headers['X-Enhance-Note'] = res.get('note', '')
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@assistant_bp.route('/ai/separate-stems', methods=['POST'])
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

@assistant_bp.route('/ai/separate/capabilities', methods=['GET'])
def ai_separate_capabilities():
    return jsonify(separation_processor.capability_status())

@assistant_bp.route('/ai/separate', methods=['POST'])
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

@assistant_bp.route('/ai/lyrics', methods=['POST'])
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

@assistant_bp.route('/ai/auto-duck', methods=['POST'])
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

@assistant_bp.route('/ai/pitch-speed', methods=['POST'])
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

@assistant_bp.route('/ai/tts/voices', methods=['GET'])
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

@assistant_bp.route('/ai/tts', methods=['POST'])
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

@assistant_bp.route('/ai/youtube-chapters', methods=['POST'])
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

@assistant_bp.route('/ai/trim-silence', methods=['POST'])
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
