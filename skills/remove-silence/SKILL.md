---
name: remove-silence
title: Remove silence
description: Use to cut dead air at the start and end of a recording.
category: cleanup
media_types:
- audio
tags:
- silence
- pauses
- dead air
params:
  threshold:
    type: integer
    minimum: 10
    maximum: 80
    default: 40
    unit: dB
    description: How far below the peak counts as silence
example: '@remove-silence'
steps:
- tool: auto_trim_silence
  args:
    threshold: '{{threshold}}'
---

# Remove silence

## Procedure
Detect the first and last audible moments (threshold in dB below the peak) and keep only that span.
If no audible content is found the step stops with an explanation.
