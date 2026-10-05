---
name: vocals
title: Vocals and instrumental
description: Use to split a song or recording into a vocals file and an instrumental file.
category: separate
media_types:
- audio
- video
tags:
- vocals
- instrumental
- acapella
- split
requires_tools:
- separate_stems
example: '@vocals'
steps:
- tool: separate_stems
  args:
    mode: vocals
---

# Vocals and instrumental

## Procedure
Separate the recording into two files, the vocals and the instrumental, each delivered as a download.
Quality adapts to this computer and the reply says which level was used.
