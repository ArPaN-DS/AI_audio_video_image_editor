---
name: fade
title: Fade in and out
description: Use to add a smooth fade-in at the start and/or a fade-out at the end.
category: edit
media_types:
- audio
- video
tags:
- fade
- intro
- outro
params:
  fade_in:
    type: seconds
    minimum: 0
    maximum: 30
    default: 2
    description: Fade-in length
  fade_out:
    type: seconds
    minimum: 0
    maximum: 30
    default: 2
    description: Fade-out length
example: '@fade 1 3'
steps:
- tool: apply_audio_fade
  args:
    fade_in_sec: '{{fade_in}}'
    fade_out_sec: '{{fade_out}}'
---

# Fade in and out

## Procedure
Apply both fades in one pass (one encode). Fades longer than the clip are shortened proportionally.
