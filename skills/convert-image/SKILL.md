---
name: convert-image
title: Convert image
description: Use to save an image as PNG, JPG or WebP.
category: convert
media_types:
- image
tags:
- png
- jpg
- webp
- convert
params:
  format:
    type: string
    enum:
    - png
    - jpg
    - webp
    default: png
example: '@convert-image jpg'
steps:
- tool: convert_image_format
  args:
    target_format: '{{format}}'
---

# Convert image

## Procedure
JPG fills transparent areas with white; use PNG or WebP to keep transparency.
