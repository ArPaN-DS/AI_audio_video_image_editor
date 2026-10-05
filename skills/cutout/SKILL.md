---
name: cutout
title: Background removal
description: Use to remove the background and keep the subject on transparency; pick a profile for people, products
  or fine hair.
category: separate
media_types:
- image
tags:
- background
- cutout
- transparent
- remove bg
params:
  profile:
    type: string
    enum:
    - auto
    - portrait
    - product
    - hair
    - fast
    default: auto
    map:
      auto: detail
      product: studio
      hair: detail
    description: Subject type
example: '@cutout portrait'
steps:
- tool: remove_background
  args:
    quality_profile: '{{profile}}'
---

# Background removal

## Profiles
- auto: general subjects (highest-detail cutout).
- portrait: people and selfies.
- product: hard-edged objects on plain backgrounds.
- hair: fine detail such as hair and fur (highest-detail profile with soft alpha edges).

## Verification and retry
If the cutout removed almost nothing or almost everything, it is retried once with another profile.
