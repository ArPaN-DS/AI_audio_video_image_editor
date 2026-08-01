import os
import time
import uuid
import json
import zipfile

import re
import shutil
import tempfile
from flask import Flask, render_template, request, send_file, jsonify, abort
import ai_processor
import video_processor
import image_processor
from pydub import AudioSegment
from logger import log_upload_details
from io import BytesIO

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

# Config: 500MB Limit
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


# ═══════════════════════════════════════════════════════════════════════════
#  END-TO-END PLATFORM MEMORY & STORAGE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

import gc
import sys

def _cleanup_old_temp_files(max_age_seconds=3600):
    """
    Background purger: Removes uploaded and processed temporary files
    older than max_age_seconds to prevent SSD disk bloat.
    """
    now = time.time()
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        if not os.path.exists(folder):
            continue
        for root, dirs, files in os.walk(folder):
            # Skip filmstrip thumbnails
            if "thumbs" in root:
                continue
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    if os.path.isfile(filepath):
                        file_age = now - os.path.getmtime(filepath)
                        if file_age > max_age_seconds:
                            os.remove(filepath)
                except Exception:
                    pass

@app.after_request
def end_to_end_memory_reclaim(response):
    """
    Global End-to-End HTTP Teardown Hook:
    Runs garbage collection and OS working set memory reclamation after
    EVERY request to keep Flask's baseline memory at ~30 MB.
    """
    try:
        gc.collect()
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass
    return response

@app.route('/')
def studio_landing():
    _cleanup_old_temp_files()
    return render_template('landing.html')


@app.route('/audio')
def audio_editor():
    return render_template('index.html')

@app.route('/cut', methods=['POST'])
def cut_audio():
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400

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
            return "No regions provided", 400
        
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
                target_dBFS = -14.0
                if segment.dBFS != float('-inf'):
                    change_in_dBFS = target_dBFS - segment.dBFS
                    segment = segment.apply_gain(change_in_dBFS)
            if do_reverse:
                segment = segment.reverse()
            
            processed_segments.append({
                'name': region.get('name', 'Region'),
                'audio': segment
            })
        
        if not processed_segments:
            return "No valid regions to process", 400
        
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
        print(f"Error: {e}")
        return f"Server Error: {str(e)}", 500

# Helper to save upload file temporarily
def save_temp_upload(file):
    unique_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename)[1] or ".webm"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{unique_id}{original_ext}")
    file.save(temp_path)
    return temp_path

@app.route('/ai/detect-silence', methods=['POST'])
def ai_detect_silence():
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    
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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    
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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    
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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    
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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    
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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    
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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400

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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400

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
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400

    temp_path = save_temp_upload(file)
    out_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"stems_{uuid.uuid4()}")
    try:
        res = ai_processor.separate_stems(temp_path, out_dir)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ═══════════════════════════════════════════
#  VIDEO EDITOR (audio + video combo editor)
# ═══════════════════════════════════════════

THUMBS_FOLDER = os.path.join(PROCESSED_FOLDER, 'thumbs')
os.makedirs(THUMBS_FOLDER, exist_ok=True)

MEDIA_ID_RE = re.compile(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9]+$')


def _media_path(media_id):
    """Resolve an uploaded media id to a safe absolute path (or None)."""
    if not media_id or not MEDIA_ID_RE.match(media_id):
        return None
    path = os.path.join(app.config['UPLOAD_FOLDER'], media_id)
    return path if os.path.exists(path) else None


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

    model_name = request.form.get('model', 'u2net')
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
    model_key = request.form.get('model', 'fast')
    if model_key not in ('fast', 'best'):
        model_key = 'fast'

    uid = uuid.uuid4()
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enh_in_{uid}.png")
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], f"enh_out_{uid}.png")
    file.save(in_path)

    try:
        info = image_processor.upscale(in_path, out_path, scale=scale, model_key=model_key)
        resp = send_file(out_path, mimetype='image/png')
        resp.headers['X-Enhance-Engine'] = str(info.get('engine', 'lanczos'))
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

    unique_id = str(uuid.uuid4())
    ext = (os.path.splitext(file.filename)[1] or '.mp4').lower()
    if not re.match(r'^\.[A-Za-z0-9]+$', ext):
        ext = '.mp4'
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
        os.remove(path)
        return jsonify({"error": f"Could not read media file: {e}"}), 400

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


@app.route('/video/export', methods=['POST'])
def video_export():
    """Render the full timeline (clips + text + music) via FFmpeg."""
    spec = request.get_json(silent=True)
    if not spec:
        return jsonify({"error": "Invalid export request."}), 400

    fmt = spec.get('format', 'mp4')
    if fmt not in ('mp4', 'mp3', 'wav'): fmt = 'mp4'
    spec['format'] = fmt

    work_dir = tempfile.mkdtemp(prefix='vexport_', dir=app.config['PROCESSED_FOLDER'])
    out_name = f"video_export_{uuid.uuid4()}.{fmt}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)

    try:
        video_processor.export_project(spec, _media_path, work_dir, out_path)
        return send_file(out_path, as_attachment=True,
                         download_name=f"edited_video.{fmt}")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
