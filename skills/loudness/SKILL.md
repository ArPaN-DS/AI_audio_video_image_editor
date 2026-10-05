---
name: loudness
title: Loudness for a platform
description: 'Use when audio should hit a delivery loudness target: YouTube or Spotify (-14 LUFS), podcasts (-16)
  or broadcast (-23).'
category: enhance
media_types:
- audio
- video
tags:
- lufs
- normalize
- volume
- loudness
params:
  target:
    type: string
    enum:
    - youtube
    - spotify
    - podcast
    - apple_podcasts
    - broadcast
    default: youtube
    description: Delivery platform
example: '@loudness podcast'
steps:
- tool: normalize_audio
  args:
    preset: '{{target}}'
---

# Loudness for a platform

## When to use
Before publishing, so platforms do not turn the audio down or leave it sounding quiet.

## Inspection signals
Integrated loudness (LUFS) and true peak are measured first. If the audio is already within 0.5 LU of the
target and peaks under -1 dBTP, the step is skipped.

## Procedure
Two-pass loudness normalization with a -1 dBTP true-peak ceiling. Put noise cleanup before this step.
