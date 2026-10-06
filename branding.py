"""
Branding — the single source of truth for user-facing product names.

The final product and assistant names have not been chosen yet. Every page,
script and API reply reads them from here (templates via the `brand` context
variable, browser scripts via `window.APP_BRAND`), so renaming the product is
a one-line change in `.env`:

    APP_PRODUCT_NAME=Media Studio
    APP_ASSISTANT_NAME=AI Assistant
"""

import os
import re

DEFAULT_PRODUCT_NAME = "Media Studio"
DEFAULT_ASSISTANT_NAME = "Rini"
_MAX_NAME_LENGTH = 40
_SAFE_NAME = re.compile(r"[^\w .&'+-]", re.UNICODE)


def _clean(value, default):
    cleaned = _SAFE_NAME.sub("", str(value or "")).strip()[:_MAX_NAME_LENGTH].strip()
    return cleaned or default


def product_name():
    return _clean(os.environ.get("APP_PRODUCT_NAME"), DEFAULT_PRODUCT_NAME)


def assistant_name():
    return _clean(os.environ.get("APP_ASSISTANT_NAME"), DEFAULT_ASSISTANT_NAME)


def brand():
    """Names for templates and the browser (`window.APP_BRAND`)."""
    return {"product": product_name(), "assistant": assistant_name()}
