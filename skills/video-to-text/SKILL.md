---
name: video-to-text
title: Video to text
description: Use to turn a video into a transcript plus SRT and VTT subtitle files in one step.
category: text
media_types:
- video
tags:
- transcript
- video
- captions
- subtitles
- text
triggers:
- video to text
- transcript of this video
- transcribe this video
example: '@video-to-text'
steps:
- tool: transcribe_audio
---

# Video to text

## Procedure
1. Measure the soundtrack condition.
2. If it is noisy, music-heavy or very quiet, prepare a lossless copy of the soundtrack (cleanup or voice
   isolation); otherwise transcribe directly.
3. Transcribe with timestamps and export TXT, SRT and VTT.

## Verification and retry
Empty or low-confidence transcripts get one retry on a voice-enhanced copy. The video itself is never changed.
