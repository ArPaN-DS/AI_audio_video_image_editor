---
name: video-enhance
title: Video enhance
description: Use when footage looks soft, grainy or low resolution; denoises, sharpens, reconstructs detail and resizes while keeping the aspect ratio.
category: enhance
media_types:
- video
tags:
- enhance
- sharpen
- denoise
- hd
- upscale
params:
  resolution:
    type: string
    enum:
    - 720p
    - 1080p
    - 1440p
    - 4k
    - original
    default: 1080p
  denoise:
    type: boolean
    default: true
  sharpen:
    type: boolean
    default: true
  ai:
    type: boolean
    default: false
example: '@video-enhance 1080p'
steps:
- tool: enhance_video
  args:
    mode: '{{resolution}}'
    denoise: '{{denoise}}'
    sharpen: '{{sharpen}}'
    ai: '{{ai}}'
---

# Video enhance

## Procedure
Gentle temporal grain reduction, micro-contrast sharpening and a mild contrast lift, or neural detail reconstruction, then resize so the short edge matches the target (vertical stays vertical). Use `original` to keep the size.
