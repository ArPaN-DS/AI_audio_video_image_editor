---
name: transcribe
title: Speech to text
description: Use to get a transcript with timestamps plus TXT, SRT and VTT files; cleans noisy or music-heavy audio
  first only when needed.
category: text
media_types:
- audio
- video
tags:
- transcript
- stt
- captions
- subtitles
- speech
example: '@transcribe'
steps:
- tool: transcribe_audio
---

# Speech to text

## When to use
Interviews, lectures, meetings, voice notes, videos that need captions.

## Inspection signals and preprocessing
- Music bed (tonal background, foreground under ~32 dB above it) -> isolate the voice first.
- Broadband noise or hum -> noise removal first.
- Very quiet (peak under -20 dBFS or active level under -38 dBFS) -> raise the level first.
- Clean -> transcribe directly. Silent -> skip with an explanation.
Preparation runs on a copy; your media is not changed. Say "as-is" to skip preparation.

## Verification and retry
An empty or low-confidence transcript (average word confidence under 0.45) is retried once on a
voice-enhanced copy when the audio contains speech-like activity; the better result is kept.
