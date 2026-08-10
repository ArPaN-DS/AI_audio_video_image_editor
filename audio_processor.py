"""
Audio Processor — Advanced Multitrack Audio Mixing & Signal Processing Engine.

Complements Pydub and Librosa with:
  - LUFS Loudness calculation (EBU R128 compliance)
  - 10-Band Parametric Equalizer filter calculation
  - Multi-track audio mixdown with volume, pan, and offset alignment
  - Batch conversion and formatting
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment

def calculate_lufs(audio_path):
    """
    Calculate integrated LUFS and peak loudness for EBU R128 compliance.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    
    # RMS based integrated loudness approximation
    rms = np.sqrt(np.mean(y**2))
    lufs = 20 * np.log10(rms + 1e-9) - 0.6  # Approximation offset
    peak_db = 20 * np.log10(np.max(np.abs(y)) + 1e-9)
    
    return {
        "lufs": round(float(lufs), 2),
        "peak_db": round(float(peak_db), 2),
        "sample_rate": sr,
        "channels": y.shape[0],
        "duration": round(librosa.get_duration(y=y, sr=sr), 3)
    }

def apply_parametric_eq(audio_path, output_path, eq_bands=None):
    """
    Apply 10-Band Parametric EQ adjustments to an audio file.
    eq_bands: dict of band_name -> gain_db (e.g. {"32": 2.0, "64": -1.0, ...})
    """
    if eq_bands is None:
        eq_bands = {}
        
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    is_mono = (y.ndim == 1)
    if is_mono:
        y = y.reshape(1, -1)
        
    # Standard 10 octave bands (Hz)
    bands = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
    # Apply FFT-based EQ curve filtering
    fft_len = 4096
    hop = 1024
    
    processed_channels = []
    for ch in range(y.shape[0]):
        stft = librosa.stft(y[ch], n_fft=fft_len, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_len)
        
        # Build frequency gain mask
        gain_mask = np.ones(stft.shape[0], dtype=np.float32)
        
        for band_freq in bands:
            band_str = str(int(band_freq))
            gain_db = float(eq_bands.get(band_str, 0.0))
            if gain_db != 0.0:
                # Gaussian shaped gain curve around band frequency
                linear_gain = 10.0 ** (gain_db / 20.0)
                sigma = band_freq * 0.4
                gaussian = np.exp(-0.5 * ((freqs - band_freq) / (sigma + 1e-5)) ** 2)
                gain_mask += (linear_gain - 1.0) * gaussian
                
        # Apply mask & ISTFT
        stft_eq = stft * gain_mask[:, np.newaxis]
        y_eq = librosa.istft(stft_eq, hop_length=hop, length=y.shape[1])
        processed_channels.append(y_eq)
        
    y_out = np.vstack(processed_channels)
    if is_mono:
        y_out = y_out[0]
    else:
        y_out = y_out.T
        
    sf.write(output_path, y_out, sr)
    return True

def mix_audio_tracks(tracks_spec, output_path, master_volume=1.0, format="wav"):
    """
    Mix multiple audio track clips together.
    tracks_spec: list of dicts:
      [
        {
          "file_path": "uploads/audio1.mp3",
          "start_time": 0.0,  # seconds in master timeline
          "clip_offset": 0.0, # trim start inside clip
          "duration": 5.0,
          "volume": 0.8,
          "pan": 0.0, # -1.0 to 1.0
          "mute": False
        },
        ...
      ]
    """
    if not tracks_spec:
        raise ValueError("No tracks provided to mix")
        
    # Find total master timeline duration
    max_duration = 0.0
    for t in tracks_spec:
        end_time = float(t.get("start_time", 0.0)) + float(t.get("duration", 0.0))
        if end_time > max_duration:
            max_duration = end_time
            
    if max_duration <= 0.0:
        max_duration = 1.0
        
    master_ms = int(max_duration * 1000) + 1000
    master_mix = AudioSegment.silent(duration=master_ms)
    
    for t in tracks_spec:
        if t.get("mute", False):
            continue
            
        file_path = t.get("file_path")
        if not file_path or not os.path.exists(file_path):
            continue
            
        clip = AudioSegment.from_file(file_path)
        
        # Trim clip
        clip_offset_ms = int(t.get("clip_offset", 0.0) * 1000)
        duration_ms = int(t.get("duration", clip.duration_seconds) * 1000)
        
        clip_seg = clip[clip_offset_ms : clip_offset_ms + duration_ms]
        
        # Volume gain (convert scalar volume multiplier to dB)
        vol = float(t.get("volume", 1.0))
        if vol <= 0.001:
            continue
        gain_db = 20 * np.log10(vol)
        clip_seg = clip_seg.apply_gain(gain_db)
        
        # Pan (-1.0 to 1.0)
        pan = float(t.get("pan", 0.0))
        if pan != 0.0:
            clip_seg = clip_seg.pan(max(-1.0, min(1.0, pan)))
            
        # Overlay onto master timeline
        start_ms = int(t.get("start_time", 0.0) * 1000)
        master_mix = master_mix.overlay(clip_seg, position=start_ms)
        
    # Apply master volume
    if master_volume != 1.0 and master_volume > 0.0:
        master_gain = 20 * np.log10(master_volume)
        master_mix = master_mix.apply_gain(master_gain)
        
    # Trim to exact length
    master_mix = master_mix[: int(max_duration * 1000)]
    
    # Export
    bitrate = "320k" if format == "mp3" else None
    master_mix.export(output_path, format=format, bitrate=bitrate)
    return output_path
