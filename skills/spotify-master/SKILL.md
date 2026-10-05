---
name: spotify-master
title: Spotify master
description: Use to deliver a music track at Spotify loudness (-14 LUFS, -1 dBTP) as WAV or FLAC.
category: workflows
media_types:
- audio
tags:
- spotify
- music
- master
- streaming
params:
  format:
    type: string
    enum:
    - wav
    - flac
    default: wav
triggers:
- spotify master
- master for spotify
- master it for spotify
example: '@spotify-master flac'
steps:
- tool: normalize_audio
  args:
    preset: spotify
- tool: convert_audio_format
  args:
    target_format: '{{format}}'
---

# Spotify master

## Procedure
1. Measure integrated loudness and true peak; skip if already within 0.5 LU and under -1 dBTP.
2. Two-pass loudness normalization to -14 LUFS with a -1 dBTP ceiling.
3. Export lossless.
