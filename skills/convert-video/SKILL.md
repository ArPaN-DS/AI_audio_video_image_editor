---
name: convert-video
title: Convert video
description: Use to convert a video to MP4, WebM or MKV at full length.
category: convert
media_types:
- video
tags:
- mp4
- webm
- mkv
- convert
- format
params:
  format:
    type: string
    enum:
    - mp4
    - webm
    - mkv
    default: mp4
example: '@convert-video webm'
steps:
- tool: convert_video_format
  args:
    target_format: '{{format}}'
---

# Convert video

## Procedure
Re-encode the whole video into the chosen container (MP4 for compatibility, WebM for the web).
