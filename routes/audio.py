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

from pydub import AudioSegment
from io import BytesIO



audio_bp = Blueprint('audio', __name__)



@audio_bp.route('/audio/lufs', methods=['POST'])
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

@audio_bp.route('/audio/eq', methods=['POST'])
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

@audio_bp.route('/audio/multitrack-mix', methods=['POST'])
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

@audio_bp.route('/audio')
def audio_editor():
    return render_template('index.html')

@audio_bp.route('/cut', methods=['POST'])
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
