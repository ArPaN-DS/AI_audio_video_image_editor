---
name: video-to-audio
title: Video to audio
description: Use to pull the soundtrack out of a video as MP3 or WAV, optionally cleaning and levelling it.
category: convert
media_types:
- video
tags:
- extract
- soundtrack
- mp3
- wav
params:
  format:
    type: string
    enum:
    - mp3
    - wav
    default: mp3
  clean:
    type: boolean
    default: false
    description: Clean and level the audio
triggers:
- video to audio
- video into audio
example: '@video-to-audio wav clean'
steps:
- tool: extract_audio
  args:
    format: wav
  when:
    param: clean
- tool: _auto_clean
  when:
    param: clean
- tool: normalize_audio
  when:
    param: clean
- tool: convert_audio_format
  args:
    target_format: '{{format}}'
  when:
    param: clean
- tool: extract_audio
  args:
    format: '{{format}}'
  when:
    not_param: clean
---

# Video to audio

## Procedure
- Plain: extract the soundtrack directly in the chosen format.
- With `clean`: extract lossless WAV, clean up (noise, hum or music aware), normalize to -14 LUFS, then
  encode once in the chosen format.

## Verification
Videos without a soundtrack stop with an explanation.
