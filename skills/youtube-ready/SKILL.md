---
name: youtube-ready
title: YouTube ready
description: 'Use before uploading to YouTube: levels the soundtrack to -14 LUFS and delivers a compatible MP4.'
category: workflows
media_types:
- video
tags:
- youtube
- upload
- publish
- loudness
triggers:
- youtube ready
- youtube-ready
- ready for youtube
- prepare for youtube
- prep for youtube
example: '@youtube-ready'
steps:
- tool: normalize_audio
  args:
    preset: youtube
- tool: convert_video_format
  args:
    target_format: mp4
---

# YouTube ready

## Procedure
1. Measure soundtrack loudness; normalize to -14 LUFS with a -1 dBTP ceiling (skipped if already on target or
   there is no soundtrack).
2. Encode once to H.264/AAC MP4 with fast start.
