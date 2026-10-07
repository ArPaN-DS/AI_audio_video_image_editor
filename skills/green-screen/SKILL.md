---
name: green-screen
title: Cutout / Green screen
description: Extract subjects and remove video background, placing the subject against a broadcast green screen or transparency.
category: edit
media_types:
- video
tags:
- cutout
- greenscreen
- background
- matting
params:
  bg_type:
    type: string
    enum:
    - green
    - transparent
    - black
    - white
    default: green
example: '@green-screen'
steps:
- tool: remove_video_background
  args:
    bg_type: '{{bg_type}}'
---

# Cutout / Green screen

## Procedure
Extracts video foreground subjects with sub-pixel alpha matting and places them on a chroma green screen, transparent WebM, or solid color.
