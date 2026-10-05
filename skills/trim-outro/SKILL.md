---
name: trim-outro
title: Remove the ending
description: Use to drop the last few seconds (for example an outro or trailing noise).
category: edit
media_types:
- audio
- video
tags:
- outro
- ending
- trim
params:
  seconds:
    type: seconds
    minimum: 0
    required: true
    description: Seconds to remove
example: '@trim-outro 3'
steps:
- tool: trim_audio
  args:
    drop_last_sec: '{{seconds}}'
  when:
    media:
    - audio
- tool: trim_video
  args:
    drop_last_sec: '{{seconds}}'
  when:
    media:
    - video
---

# Remove the ending

## Procedure
Keep everything up to the given number of seconds before the end (needs the media length).
