---
name: product-cutout
title: Product cutout
description: 'Use for shop listings: cuts out a product with crisp edges and saves a transparent PNG.'
category: workflows
media_types:
- image
tags:
- product
- shop
- ecommerce
- cutout
triggers:
- product cutout
- product photo cutout
- for my store
- for my shop
example: '@product-cutout'
steps:
- tool: remove_background
  args:
    quality_profile: studio
- tool: convert_image_format
  args:
    target_format: png
---

# Product cutout

## Procedure
Product profile cutout (hard edges), coverage check with one retry, then transparent PNG.
