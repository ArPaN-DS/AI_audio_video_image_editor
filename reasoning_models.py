"""
Reasoning Model Registry — tiered local language models for the Media Copilot.

The Copilot can be backed by up to three local reasoning models of increasing
capacity (e.g. a ~2B small model today, a ~4B model and a ~12B model later).
Each tier is configured purely through environment variables, so adding a
bigger model later needs no code change:

    LOCAL_REASONING_URL              base endpoint for every tier (loopback only)
    LOCAL_REASONING_MODEL            baseline model id (always the last resort)
    LOCAL_REASONING_MODEL_BALANCED   optional mid-size model id
    LOCAL_REASONING_MODEL_MAX        optional largest model id
    LOCAL_REASONING_URL_BALANCED     optional separate endpoint for that tier
    LOCAL_REASONING_URL_MAX          optional separate endpoint for that tier
    LOCAL_REASONING_TIER             auto | lite | balanced | max  (ceiling)

Selection, per request:
  1. Discovery — each endpoint's OpenAI-compatible `/models` list is fetched
     (cached) so only models that are actually being served are attempted.
     If an endpoint does not support discovery, configured models are assumed.
  2. Capacity — the AdaptiveQualityGovernor's hardware tier (or the pinned
     LOCAL_REASONING_TIER) caps which tiers are eligible.
  3. Resilience — tiers are tried largest first; a timeout or failure falls
     through to the next tier, and a model that times out is paused with a
     cooldown so later requests do not keep waiting on it.

Only loopback endpoints are accepted, preserving the 100% local guarantee.
"""

import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from model_manager import (
    CapabilityVariant,
    ResourceExhaustedError,
    TIER_BALANCED,
    TIER_LITE,
    TIER_MAX,
    TIER_ORDER,
    global_quality_governor,
)

_log = logging.getLogger("reasoning_models")

CAPABILITY = "copilot.reasoning"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_ID = "local-media-copilot"
DISCOVERY_TTL_SEC = 60.0
DISCOVERY_OFFLINE_TTL_SEC = 15.0   # re-check a down server sooner
DISCOVERY_TIMEOUT_SEC = 5.0

# Larger models think longer; give them proportionally more time before
# treating the tier as too slow for this machine.
TIER_TIMEOUTS_SEC = {TIER_LITE: 30, TIER_BALANCED: 45, TIER_MAX: 60}
TIER_PUBLIC_LABELS = {
    TIER_LITE: "Standard Reasoning",
    TIER_BALANCED: "Enhanced Reasoning",
    TIER_MAX: "Extended Reasoning",
}


def _env(name, *fallbacks, default=""):
    for key in (name,) + fallbacks:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return default


DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def normalize_base_url(base_url):
    """Return the base URL if it points at a valid HTTP/HTTPS host, else None."""
    try:
        parsed = urlparse((base_url or "").strip())
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            return None
        return base_url.strip().rstrip("/")
    except ValueError:
        return None


def chat_completions_url(base_url):
    base = normalize_base_url(base_url)
    return f"{base}/chat/completions" if base else None


class TierConfig:
    __slots__ = ("tier", "model_id", "base_url")

    def __init__(self, tier, model_id, base_url):
        self.tier = tier
        self.model_id = model_id
        self.base_url = base_url

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"TierConfig({self.tier!r}, {self.model_id!r}, {self.base_url!r})"


def configured_tiers():
    """Configured tiers from smallest to largest; invalid endpoints are dropped."""
    base_url = _env("LOCAL_REASONING_URL", "VLLM_API_BASE", default=DEFAULT_BASE_URL)
    candidates = [
        TierConfig(TIER_LITE, _env("LOCAL_REASONING_MODEL", "VLLM_MODEL_NAME", default=DEFAULT_MODEL_ID),
                   base_url),
        TierConfig(TIER_BALANCED, _env("LOCAL_REASONING_MODEL_BALANCED"),
                   _env("LOCAL_REASONING_URL_BALANCED", default=base_url)),
        TierConfig(TIER_MAX, _env("LOCAL_REASONING_MODEL_MAX"),
                   _env("LOCAL_REASONING_URL_MAX", default=base_url)),
    ]
    tiers, seen = [], set()
    for candidate in candidates:
        if not candidate.model_id:
            continue
        candidate.base_url = normalize_base_url(candidate.base_url)
        if not candidate.base_url:
            _log.warning("A non-local reasoning endpoint was ignored.")
            continue
        key = (candidate.model_id, candidate.base_url)
        if key in seen:
            continue
        seen.add(key)
        tiers.append(candidate)
    return tiers


class ServedModelDiscovery:
    """Caches each endpoint's served model ids. None means 'unknown'."""

    def __init__(self, fetch=None, clock=time.monotonic, ttl=DISCOVERY_TTL_SEC):
        self._fetch = fetch or self._fetch_models
        self._clock = clock
        self._ttl = ttl
        self._lock = threading.Lock()
        self._cache = {}
        self._meta = {}     # base_url -> {model id: raw /models item} (context length, owner, params)

    def _fetch_models(self, base_url):
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        req = urllib.request.Request(f"{base_url}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=DISCOVERY_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return None
        ids, items = set(), {}
        for item in data:
            if isinstance(item, dict):
                for key in ("id", "root", "name", "model"):
                    if isinstance(item.get(key), str):
                        ids.add(item[key])
                        items.setdefault(item[key], item)
        with self._lock:
            self._meta[base_url] = items
        return ids

    def metadata(self, base_url, model_id):
        """Raw `/models` entry for a served model (None when unknown); used by slm_profiles."""
        with self._lock:
            return (self._meta.get(base_url) or {}).get(model_id)

    def served(self, base_url):
        now = self._clock()
        with self._lock:
            cached = self._cache.get(base_url)
            if cached:
                ttl = DISCOVERY_OFFLINE_TTL_SEC if cached[1] == frozenset() else self._ttl
                if now - cached[0] < ttl:
                    return cached[1]
        try:
            models = self._fetch(base_url)
            reachable = True
        except urllib.error.HTTPError:
            models, reachable = None, True      # server up, discovery unsupported
        except Exception:
            models, reachable = None, False     # server down / unreachable
        result = models if reachable else frozenset()
        with self._lock:
            self._cache[base_url] = (now, result)
        return result

    def invalidate(self):
        with self._lock:
            self._cache.clear()


discovery = ServedModelDiscovery()


def _tier_ceiling():
    pinned = _env("LOCAL_REASONING_TIER").lower()
    aliases = {"low": TIER_LITE, "standard": TIER_BALANCED, "medium": TIER_BALANCED, "high": TIER_MAX}
    pinned = aliases.get(pinned, pinned)
    if pinned in TIER_ORDER:
        return pinned
    return global_quality_governor.current_tier()


def reasoning_ladder(tiers=None, served_lookup=None):
    """
    Build the governor ladder (largest first, baseline last) from the tiers
    that are configured AND served. When an endpoint cannot list its models,
    its configured models are assumed to be available.
    """
    tiers = configured_tiers() if tiers is None else tiers
    served_lookup = served_lookup or discovery.served
    available = []
    for tier in tiers:
        served = served_lookup(tier.base_url)
        if served is None or tier.model_id in served:
            available.append(tier)
    if not available:
        return []

    ceiling = TIER_ORDER.index(_tier_ceiling())
    within = [tier for tier in available if TIER_ORDER.index(tier.tier) <= ceiling]
    # If the pinned/detected ceiling excludes every served model, still use the
    # smallest one that is actually running rather than failing outright.
    chosen = within or available[:1]

    ladder = []
    for tier in reversed(chosen):
        ladder.append(CapabilityVariant(
            tier.model_id,
            tier=TIER_LITE,  # eligibility already decided above
            public_label=TIER_PUBLIC_LABELS[tier.tier],
            options={"url": chat_completions_url(tier.base_url),
                     "timeout": TIER_TIMEOUTS_SEC[tier.tier],
                     "tier": tier.tier},
        ))
    return ladder


class ContextOverflowError(ValueError):
    """The server rejected the prompt as longer than its context window (`limit` when it said)."""

    def __init__(self, message, limit=None):
        super().__init__(message)
        self.limit = limit


class StructuredOutputUnsupportedError(ValueError):
    """The server rejected the structured-output (response_format) request fields."""


_OVERFLOW_RE = re.compile(r"context|too long|exceed|maximum.{0,20}length|n_ctx|max_model_len", re.IGNORECASE)
_LIMIT_RE = re.compile(r"(?:context (?:size|length|window)|maximum context length is|n_ctx|max_model_len)\D{0,30}?(\d{3,7})",
                       re.IGNORECASE)
_FORMAT_RE = re.compile(r"response_format|json_schema|grammar|guided|structured", re.IGNORECASE)


def _http_error(error):
    try:
        detail = error.read().decode("utf-8", "replace")[:2000]
    except Exception:
        detail = ""
    if error.code in (400, 413, 422):
        if _FORMAT_RE.search(detail):
            return StructuredOutputUnsupportedError("Structured output rejected by the reasoning service.")
        if _OVERFLOW_RE.search(detail):
            limit = _LIMIT_RE.search(detail)
            return ContextOverflowError("Prompt exceeds the reasoning context window.",
                                        int(limit.group(1)) if limit else None)
    return None


def attempt_payload(variant, payload):
    """
    The request body for one tier. `payload["_render"](variant)` (if present) builds a
    tier-specific body (prompt sized to that model's window); private `_` keys are never sent.
    """
    render = payload.get("_render")
    body = render(variant) if callable(render) else payload
    clean = {key: value for key, value in body.items() if not str(key).startswith("_")}
    clean["model"] = variant.variant_id
    return clean


def _post(variant, payload, _recovered=False):
    try:
        return _post_once(variant, payload)
    except (ContextOverflowError, StructuredOutputUnsupportedError) as error:
        # One in-tier retry after the caller adapts (smaller prompt / plainer output mode).
        recover = payload.get("_recover")
        if _recovered or not callable(recover) or not recover(variant, error):
            raise
        return _post(variant, payload, _recovered=True)


def _post_once(variant, payload):
    body_out = attempt_payload(variant, payload)
    headers = {"Content-Type": "application/json", "User-Agent": DEFAULT_USER_AGENT}
    req = urllib.request.Request(
        variant.options["url"],
        data=json.dumps(body_out).encode("utf-8"),
        headers=headers,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=variant.options["timeout"]) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (socket.timeout, TimeoutError) as error:
        # Too slow for this machine right now: pause this tier like an OOM.
        raise ResourceExhaustedError("Reasoning tier timed out.") from error
    except urllib.error.HTTPError as error:
        specific = _http_error(error)
        if specific is not None:
            raise specific from error
        raise
    except urllib.error.URLError as error:
        if isinstance(getattr(error, "reason", None), (socket.timeout, TimeoutError)):
            raise ResourceExhaustedError("Reasoning tier timed out.") from error
        raise
    observe = payload.get("_observe")
    if callable(observe):
        try:
            observe(variant, body_out, body, time.monotonic() - started)
        except Exception:
            _log.debug("Reasoning usage observer failed.", exc_info=True)
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Reasoning service returned no choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Reasoning service returned no message.")
    content = message.get("content")
    return content if isinstance(content, str) else ""


def request_completion(payload, ladder=None):
    """
    Send a chat-completions payload to the best available reasoning tier.
    Returns (content, public_tier_label). Raises if no tier could answer.
    """
    if ladder is None:
        ladder = payload.get("_ladder") if isinstance(payload.get("_ladder"), list) else reasoning_ladder()
    tiers = payload.get("_tiers")
    if tiers:
        ladder = [variant for variant in ladder if variant.options.get("tier") in tiers]
    if not ladder:
        raise ConnectionError("No local reasoning model is available.")
    try:
        content, variant = global_quality_governor.run(CAPABILITY, ladder, lambda v: _post(v, payload))
    except Exception:
        # Let the next request re-discover (server may have restarted).
        discovery.invalidate()
        raise
    return content, variant.public_label


def public_status():
    """Capability-language summary for diagnostics (no model ids)."""
    try:
        ladder = reasoning_ladder()
    except Exception:
        ladder = []
    tiers = [variant.options.get("tier") for variant in ladder]
    return {
        "reasoning_available": bool(ladder),
        "reasoning_mode": ladder[0].public_label if ladder else "Built-in Intent Routing",
        "reasoning_tiers_ready": len(ladder),
        "reasoning_tiers": [TIER_PUBLIC_LABELS[tier] for tier in tiers if tier in TIER_PUBLIC_LABELS],
    }
