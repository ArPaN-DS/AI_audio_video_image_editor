---
name: noise-removal
title: Noise removal
description: Use when a recording has hiss, hum, fan or room noise, or speech sits on background music. Chooses
  noise filtering or voice isolation from the measured condition.
category: cleanup
media_types:
- audio
- video
tags:
- noise
- hiss
- hum
- denoise
- music
example: '@noise-removal'
steps:
- tool: _auto_clean
---

# Noise removal

## When to use
Hiss, hum, fan, traffic or room tone under speech; or speech over a music bed.

## Inspection signals
- Background level (10th percentile frame level) and signal-to-noise gap (95th minus 10th percentile).
- Clean: background below -55 dBFS or gap above 32 dB -> nothing to remove.
- Broadband background (spectral flatness above 0.30) -> noise filtering.
- Tonal background below 180 Hz -> hum removal (noise filtering).
- Tonal background above 180 Hz -> music bed -> voice isolation (falls back to voice enhancement).

## Procedure
1. Measure the condition (bounded 30 s excerpt).
2. Pick exactly one cleanup path from the signals above; skip with a note when the recording is clean.
3. On video, edit a lossless copy of the soundtrack and remux once; the picture is untouched.

## Verification
The output must not be silent when the input had sound; otherwise the chain stops and the original stays available.
