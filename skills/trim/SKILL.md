---
name: trim
title: Trim
description: Use to keep only the part between a start and an end time.
category: edit
media_types:
- audio
- video
tags:
- trim
- cut
- shorten
- clip
params:
  start:
    type: seconds
    minimum: 0
    required: true
    description: Start time
  end:
    type: seconds
    minimum: 0
    required: true
    description: End time
example: '@trim 2 8'
steps:
- tool: trim_audio
  args:
    start_sec: '{{start}}'
    end_sec: '{{end}}'
  when:
    media:
    - audio
- tool: trim_video
  args:
    start_sec: '{{start}}'
    end_sec: '{{end}}'
  when:
    media:
    - video
---

# Trim

## Procedure
Sub-second precision. Times accept seconds, milliseconds or mm:ss. An end past the media length is clamped.
