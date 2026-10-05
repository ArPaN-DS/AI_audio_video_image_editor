---
name: podcast-polish
title: Podcast polish
description: 'Use to turn a raw spoken-word recording into a publish-ready podcast MP3: voice cleanup, -16 LUFS,
  192 kbps or better.'
category: workflows
media_types:
- audio
- video
tags:
- podcast
- episode
- voice
- mp3
params:
  bitrate:
    type: string
    enum:
    - 192k
    - 256k
    - 320k
    default: 192k
triggers:
- podcast polish
- polish my podcast
- polish the podcast
- polish this episode
- clean up my podcast
example: '@podcast-polish 320k'
steps:
- tool: extract_audio
  args:
    format: wav
  when:
    media:
    - video
- tool: enhance_speech
- tool: normalize_audio
  args:
    preset: podcast
- tool: convert_audio_format
  args:
    target_format: mp3
    bitrate: '{{bitrate}}'
---

# Podcast polish

## Procedure
1. From video, take a lossless copy of the soundtrack.
2. Voice cleanup (presence and clarity).
3. Loudness to -16 LUFS integrated, -1 dBTP true peak.
4. Encode once to MP3 at the chosen bitrate (192 kbps default).

## Verification
Silent or unreadable output stops the chain; the last good result stays downloadable.
