---
name: upscale
title: Upscale
description: Use to enlarge an image 2x or 4x with reconstructed detail.
category: enhance
media_types:
- image
tags:
- upscale
- enlarge
- resolution
- 4k
params:
  scale:
    type: integer
    enum:
    - 2
    - 4
    default: 2
    unit: x
example: '@upscale 4'
steps:
- tool: upscale_image
  args:
    scale: '{{scale}}'
---

# Upscale

## Inspection signals
Grain or compression artifacts (noise estimate above 6) are cleaned before upscaling; soft focus
(Laplacian variance under 60) gets a sharpening pass after.
