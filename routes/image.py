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





image_bp = Blueprint('image', __name__)



@image_bp.route('/image')
def image_editor():
    # Editing is client-side (HTML Canvas). Only optional AI calls reach backend.
    return render_template('image.html')

@image_bp.route('/image/remove-bg', methods=['POST'])
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

@image_bp.route('/image/clarity', methods=['POST'])
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

@image_bp.route('/image/inpaint', methods=['POST'])
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

@image_bp.route('/image/restore-faces', methods=['POST'])
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

@image_bp.route('/image/color-match', methods=['POST'])
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

@image_bp.route('/image/enhance', methods=['POST'])
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
