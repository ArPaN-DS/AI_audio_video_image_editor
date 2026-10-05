---
name: volume
title: Volume change
description: Use to make audio louder or quieter by a fixed number of decibels (positive is louder).
category: edit
media_types:
- audio
- video
tags:
- volume
- louder
- quieter
- gain
params:
  gain_db:
    type: number
    minimum: -30
    maximum: 30
    default: 6
    unit: dB
    description: Change in dB
example: '@volume 6'
steps:
- tool: adjust_volume
  args:
    gain_db: '{{gain_db}}'
---

# Volume change

## Procedure
Apply a fixed gain. Boosts are limited to the available headroom so the result never clips.
