---
name: lyrics
title: Lyrics
description: Use to get timed lyrics from a song as SRT, VTT or plain text.
category: text
media_types:
- audio
- video
tags:
- lyrics
- song
- captions
- srt
requires_tools:
- extract_lyrics
params:
  format:
    type: string
    enum:
    - srt
    - vtt
    - txt
    default: srt
    description: Lyrics file format
example: '@lyrics srt'
steps:
- tool: extract_lyrics
  args:
    format: '{{format}}'
---

# Lyrics

## Procedure
Isolate the singing, then write out the words with timing. The lyrics file is delivered as a download.
Accuracy depends on how clearly the vocals can be heard.
