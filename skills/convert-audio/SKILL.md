---
name: convert-audio
title: Convert audio
description: Use to export audio as MP3, WAV, FLAC or OGG, optionally choosing the MP3/OGG quality.
category: convert
media_types:
- audio
tags:
- mp3
- wav
- flac
- export
- format
params:
  format:
    type: string
    enum:
    - mp3
    - wav
    - flac
    - ogg
    default: mp3
  bitrate:
    type: string
    enum:
    - 128k
    - 192k
    - 256k
    - 320k
    description: MP3/OGG quality
example: '@convert-audio flac'
steps:
- tool: convert_audio_format
  args:
    target_format: '{{format}}'
    bitrate: '{{bitrate}}'
---

# Convert audio

## Procedure
Export once at the end of a chain. MP3 defaults to 192 kbps; FLAC and WAV are lossless.
