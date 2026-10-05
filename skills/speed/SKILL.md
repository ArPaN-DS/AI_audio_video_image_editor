---
name: speed
title: Playback speed
description: Use to speed up or slow down audio or video (0.25x to 4x); audio keeps its pitch.
category: edit
media_types:
- audio
- video
tags:
- speed
- faster
- slower
- slow motion
params:
  speed:
    type: number
    minimum: 0.25
    maximum: 4
    required: true
    unit: x
    description: Speed multiplier
example: '@speed 1.25'
steps:
- tool: adjust_audio_speed
  args:
    speed: '{{speed}}'
  when:
    media:
    - audio
- tool: adjust_video_speed
  args:
    speed: '{{speed}}'
  when:
    media:
    - video
---

# Playback speed

## Procedure
Below 1 is slower, above 1 is faster. Audio speed changes preserve pitch.
