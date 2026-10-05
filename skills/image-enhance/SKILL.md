---
name: image-enhance
title: Image enhance
description: Use when a photo looks grainy, soft or flat; tunes grain cleanup and sharpening to the measured condition,
  optionally upscaling.
category: enhance
media_types:
- image
tags:
- enhance
- clarity
- sharpen
- grain
- photo
params:
  scale:
    type: integer
    enum:
    - 1
    - 2
    - 4
    default: 1
    unit: x
    description: 1 keeps the size
example: '@image-enhance'
steps:
- tool: enhance_photo_clarity
  when:
    param_in:
      scale:
      - 1
- tool: upscale_image
  args:
    scale: '{{scale}}'
  when:
    param_in:
      scale:
      - 2
      - 4
---

# Image enhance

## Inspection signals
- Grain/compression: noise estimate above 6 -> stronger grain reduction.
- Soft focus: Laplacian variance under 60 -> stronger sharpening.

## Procedure
- scale 1: clarity polish tuned to the condition.
- scale 2 or 4: clean grain first if measured, upscale, sharpen after if soft.

## Verification
The output must be a readable image; otherwise the chain stops.
