---
name: trim-intro
title: Remove the intro
description: Use to drop the first few seconds (for example a countdown or intro).
category: edit
media_types:
- audio
- video
tags:
- intro
- start
- trim
params:
  seconds:
    type: seconds
    minimum: 0
    required: true
    description: Seconds to remove
example: '@trim-intro 5'
steps:
- tool: trim_audio
  args:
    start_sec: '{{seconds}}'
  when:
    media:
    - audio
- tool: trim_video
  args:
    start_sec: '{{seconds}}'
  when:
    media:
    - video
---

# Remove the intro

## Procedure
Keep everything from the given time to the end.
