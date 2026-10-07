---
name: smooth-60fps
title: Smooth motion (60 FPS)
description: Smooth jerky, low-framerate video by synthesizing intermediate motion frames using AI frame interpolation.
category: enhance
media_types:
- video
tags:
- 60fps
- interpolate
- motion
- smooth
- high-framerate
params:
  fps:
    type: integer
    default: 60
example: '@smooth-60fps'
steps:
- tool: interpolate_video
  args:
    fps: '{{fps}}'
---

# Smooth motion (60 FPS)

## Procedure
Interpolates between consecutive video frames using optical flow and intermediate temporal synthesis to generate smooth, fluid 60 FPS playback.
