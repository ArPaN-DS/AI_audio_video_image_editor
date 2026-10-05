---
name: compress
title: Compress video
description: Use to make a video file smaller for sharing or uploading.
category: convert
media_types:
- video
tags:
- compress
- smaller
- share
- size
params:
  level:
    type: string
    enum:
    - balanced
    - high
    default: balanced
example: '@compress high'
steps:
- tool: compress_video
  args:
    level: '{{level}}'
---

# Compress video

## Procedure
Re-encode at 720p maximum; `high` compresses harder for the smallest files.
