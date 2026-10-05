---
name: thumbnail
title: Thumbnail
description: Use to save a still frame from a video as a separate image.
category: convert
media_types:
- video
tags:
- thumbnail
- still
- frame
- screenshot
params:
  time:
    type: seconds
    minimum: 0
    description: Time of the frame
  format:
    type: string
    enum:
    - jpg
    - png
    default: jpg
example: '@thumbnail 3'
steps:
- tool: extract_frame
  args:
    time_sec: '{{time}}'
    format: '{{format}}'
---

# Thumbnail

## Procedure
Grab one frame at the given time (default: a representative early frame). The video is unchanged.
