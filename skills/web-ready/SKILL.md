---
name: web-ready
title: Web ready
description: 'Use to prepare an image for a website: compact WebP that keeps transparency.'
category: workflows
media_types:
- image
tags:
- web
- webp
- website
- compress
triggers:
- web ready
- web-ready
- for my website
example: '@web-ready'
steps:
- tool: convert_image_format
  args:
    target_format: webp
---

# Web ready

## Procedure
Encode as high-quality WebP (smaller than PNG/JPG, keeps transparency).
