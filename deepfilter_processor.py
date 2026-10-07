"""
DeepFilterNet3 SOTA Neural Speech Enhancement & Audio Noise Reduction (PyTorch + CUDA).

Provides market-standard voice isolation, studio speech cleanup, and dereverberation:
- Full-band 48 kHz neural filtering without underwater phase distortion.
- Removes stationary hiss, hum, room reverb, fan noise, and transient clatter.
- Automatic GPU VRAM cleanup after execution.
"""

import os
import gc
import logging
import soundfile as sf
import numpy as np
import scipy.signal

_log = logging.getLogger("deepfilter")


def is_available():
    """Return True if DeepFilterNet is importable."""
    try:
        import df
        return True
    except Exception:
        return False


def enhance_audio_deepfilter(src_path, out_path, post_filter=True):
    """
    Enhances audio speech clarity and removes background noise using DeepFilterNet3 on GPU.
    Preserves original channel count and sample rate.
    """
    import torch
    from df.enhance import init_df, enhance

    data, orig_sr = sf.read(src_path, dtype="float32", always_2d=True)
    orig_channels = data.shape[1]

    # Convert to mono for speech enhancement if multi-channel, or process per channel
    # DeepFilterNet expects 48 kHz mono
    target_sr = 48000
    if orig_sr != target_sr:
        data_48k = scipy.signal.resample_poly(data, target_sr, orig_sr, axis=0)
    else:
        data_48k = data

    # Model initialization on GPU
    model, df_state, _ = init_df(post_filter=post_filter)

    try:
        enhanced_channels = []
        for ch in range(data_48k.shape[1]):
            ch_data = data_48k[:, ch]
            ch_tensor = torch.from_numpy(ch_data).unsqueeze(0).to(df_state.device())  # [1, T]
            with torch.no_grad():
                enh_tensor = enhance(model, df_state, ch_tensor)
            enh_np = enh_tensor.squeeze(0).cpu().numpy()
            enhanced_channels.append(enh_np)

        out_data_48k = np.column_stack(enhanced_channels) if len(enhanced_channels) > 1 else enhanced_channels[0]

        # Resample back to original sample rate if needed
        if orig_sr != target_sr:
            final_data = scipy.signal.resample_poly(out_data_48k, orig_sr, target_sr, axis=0)
        else:
            final_data = out_data_48k

        # Clip and write output
        final_data = np.clip(final_data, -1.0, 1.0)
        sf.write(out_path, final_data, orig_sr)

        return {
            "status": "success",
            "engine": "deepfilternet3_gpu",
            "sample_rate": orig_sr,
            "channels": orig_channels
        }
    finally:
        del model
        del df_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
