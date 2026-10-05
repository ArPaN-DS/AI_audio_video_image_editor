---
name: profile-pic
title: Profile picture
description: 'Use for an avatar: cuts out the person, enlarges 2x with restored detail and saves a transparent PNG.'
category: workflows
media_types:
- image
tags:
- avatar
- profile
- portrait
- headshot
triggers:
- profile picture
- profile pic
- make an avatar
example: '@profile-pic'
steps:
- tool: remove_background
  args:
    quality_profile: portrait
- tool: upscale_image
  args:
    scale: 2
- tool: convert_image_format
  args:
    target_format: png
---

# Profile picture

## Procedure
Portrait cutout, 2x upscale (grain is cleaned first when measured), transparent PNG. Square cropping is not
available from chat yet.
