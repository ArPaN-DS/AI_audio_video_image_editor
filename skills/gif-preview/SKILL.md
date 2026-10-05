---
name: gif-preview
title: GIF preview
description: Use to make a short animated GIF preview from part of a video.
category: convert
media_types:
- video
tags:
- gif
- preview
- animation
params:
  start:
    type: seconds
    minimum: 0
    default: 0
    description: Start time
  duration:
    type: seconds
    minimum: 0.5
    maximum: 30
    default: 5
    description: Length
triggers:
- gif preview
example: '@gif-preview 2 4'
steps:
- tool: convert_video_format
  args:
    target_format: gif
    start: '{{start}}'
    duration: '{{duration}}'
---

# GIF preview

## Procedure
Two-pass palette GIF at 12 fps and 480 px wide. Keep it short (under 10 s) for small files.
