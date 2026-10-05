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





video_bp = Blueprint('video', __name__)



@video_bp.route('/video')
def video_editor():
    return render_template('video.html')

@video_bp.route('/video/detect-scenes', methods=['POST'])
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

@video_bp.route('/video/burn-subtitles', methods=['POST'])
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

@video_bp.route('/video/enhance-quality', methods=['POST'])
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

@video_bp.route('/video/interpolate', methods=['POST'])
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

@video_bp.route('/video/upload', methods=['POST'])
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

@video_bp.route('/media/<media_id>')
def serve_media(media_id):
    path = _media_path(media_id)
    if not path: abort(404)
    return send_file(path, conditional=True)  # range requests for <video> seek

@video_bp.route('/video/thumb/<uid>/<int:n>')
def serve_thumb(uid, n):
    if not re.match(r'^[A-Za-z0-9-]+$', uid) or n < 0 or n > 50: abort(404)
    path = os.path.join(THUMBS_FOLDER, uid, f"thumb_{n}.jpg")
    if not os.path.exists(path): abort(404)
    return send_file(path, max_age=86400)

@video_bp.route('/video/extract-audio', methods=['POST'])
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

@video_bp.route('/video/quick', methods=['POST'])
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

@video_bp.route('/video/export', methods=['POST'])
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
