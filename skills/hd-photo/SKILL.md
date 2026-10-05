---
name: hd-photo
title: HD photo
description: 'Use to make a small or soft photo crisp and high resolution: cleans grain if needed, then upscales
  2x.'
category: workflows
media_types:
- image
tags:
- hd
- photo
- sharp
- upscale
triggers:
- hd photo
- make this photo hd
example: '@hd-photo'
steps:
- tool: upscale_image
  args:
    scale: 2
---

# HD photo

## Procedure
1. Measure grain and focus.
2. Grainy -> clean first; soft -> sharpen after; clean and sharp -> upscale only.
3. Upscale 2x and verify the result is a readable image.
