---
name: voiceover
title: Voiceover
description: Use to generate a spoken voiceover audio file from a script or text.
category: voice
media_types:
- audio
- video
- image
tags:
- voiceover
- narration
- text to speech
- read aloud
requires_tools:
- generate_voiceover
params:
  text:
    type: string
    required: true
    description: The words to speak (up to 5000 characters)
example: '@voiceover "Welcome to the show"'
steps:
- tool: generate_voiceover
  args:
    text: '{{text}}'
---

# Voiceover

## Procedure
Turn the provided text into a spoken audio file on this computer. The source media is left unchanged and the
voiceover is delivered as a separate download. Voice, speed (0.5x to 2x) and format (WAV or MP3) can be set.
