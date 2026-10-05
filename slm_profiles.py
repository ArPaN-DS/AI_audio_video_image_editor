"""
Small-language-model profiles for the local reasoning tiers.

A *profile* describes how to talk to one served model on one endpoint:

  * effective context window (what the server actually allocated, not the
    model's training maximum), and the training context when known;
  * structured-output mechanism (JSON-schema constrained decoding, plain JSON
    mode, or none) — downgraded automatically when a server rejects it;
  * chat-template quirks (system-role support, thinking/reasoning switches);
  * sampling for planning (deterministic) and for conversational replies;
  * token counting: the server's /tokenize endpoint when it has one, else a
    calibrated characters-per-token estimator.

Profiles are built from (1) the model id (family rules), (2) the endpoint's
OpenAI `/models` metadata, (3) one cheap server-specific probe (llama.cpp
`/props`, Ollama `/api/show` + `/api/ps`, LM Studio `/api/v0/models`, vLLM
`max_model_len`) and (4) env overrides. Unknown models get safe defaults.
Everything stays on loopback; nothing here is ever shown to users.

Env knobs (all optional):
    LOCAL_REASONING_CONTEXT_TOKENS      force the context window for every tier
    LOCAL_REASONING_STRUCTURED_OUTPUT   auto | json_schema | json_object | off
    LOCAL_REASONING_SERVER              auto | llamacpp | ollama | lmstudio | vllm | generic
    LOCAL_REASONING_PROBE               on | off   (server metadata probes)
"""

import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request

_log = logging.getLogger("slm_profiles")

DEFAULT_CONTEXT_TOKENS = 4096        # conservative: what most local servers allocate by default
OLLAMA_DEFAULT_NUM_CTX = 4096
PROBE_TIMEOUT_SEC = 1.5
PROFILE_TTL_SEC = 300.0
DEFAULT_CHARS_PER_TOKEN = 3.0        # conservative for prompts mixing prose, JSON and numbers
SERVER_KINDS = ("llamacpp", "ollama", "lmstudio", "vllm", "generic")
STRUCTURED_MODES = ("json_schema", "json_object", "off")

# ── Family rules (first match wins). Keys:
#   system_role: chat template accepts a system message
#   thinking:    'none' | 'switchable' (hybrid; we switch it off for planning) | 'always'
#   chars:       starting chars/token estimate before calibration
#   chat:        recommended conversational sampling
#   train_ctx:   documented training context (used only when the server does not say)
FAMILY_RULES = [
    ("gemma3n", r"gemma[-_ ]?3n", {"system_role": False, "thinking": "none", "chars": 3.2, "train_ctx": 32768,
                                   "chat": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}),
    ("gemma4", r"gemma[-_ ]?4", {"system_role": True, "thinking": "none", "chars": 3.2, "train_ctx": 131072,
                                 "chat": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}),
    ("gemma", r"gemma", {"system_role": False, "thinking": "none", "chars": 3.2, "train_ctx": 32768,
                         "chat": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}),
    ("qwen3", r"qwen[-_ ]?3", {"system_role": True, "thinking": "switchable", "chars": 3.3, "train_ctx": 40960,
                               "chat": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}),
    ("deepseek-r1", r"deepseek[-_ ]?r1|r1[-_]distill", {"system_role": True, "thinking": "always", "chars": 3.3,
                                                       "chat": {"temperature": 0.6, "top_p": 0.95}}),
    ("qwen", r"qwen", {"system_role": True, "thinking": "none", "chars": 3.3,
                       "chat": {"temperature": 0.7, "top_p": 0.8}}),
    ("llama", r"llama", {"system_role": True, "thinking": "none", "chars": 3.5,
                         "chat": {"temperature": 0.6, "top_p": 0.9}}),
    ("phi", r"\bphi", {"system_role": True, "thinking": "none", "chars": 3.3,
                       "chat": {"temperature": 0.7, "top_p": 0.9}}),
    ("mistral-legacy", r"mistral[-_ ]?7b[-_ ]?instruct[-_ ]?v0", {"system_role": False, "thinking": "none",
                                                               "chars": 3.2, "chat": {"temperature": 0.7}}),
    ("mistral", r"mistral|ministral", {"system_role": True, "thinking": "none", "chars": 3.3,
                                       "chat": {"temperature": 0.7, "top_p": 0.95}}),
    ("granite", r"granite", {"system_role": True, "thinking": "none", "chars": 3.4, "chat": {"temperature": 0.7}}),
    ("smollm", r"smol", {"system_role": True, "thinking": "none", "chars": 3.4, "chat": {"temperature": 0.6}}),
]
GENERIC_FAMILY = {"system_role": True, "thinking": "none", "chars": DEFAULT_CHARS_PER_TOKEN,
                  "chat": {"temperature": 0.6, "top_p": 0.9}}

PLAN_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 7}
_PARAMS_RE = re.compile(r"(?<![a-z0-9])e?(\d+(?:\.\d+)?)\s*b(?![a-z])", re.IGNORECASE)


def _env(name, default=""):
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name):
    try:
        value = int(_env(name, "0"))
    except ValueError:
        return None
    return value if value > 0 else None


def family_for(model_id):
    text = (model_id or "").lower()
    for name, pattern, traits in FAMILY_RULES:
        if re.search(pattern, text):
            return name, traits
    return "generic", GENERIC_FAMILY


def params_from_id(model_id):
    """Parameter count in billions guessed from the id ('qwen3:4b', 'gemma-3n-E2B' -> 2), else None."""
    match = _PARAMS_RE.search(model_id or "")
    return float(match.group(1)) if match else None


def size_class(params_b):
    if params_b is None:
        return None
    return "lite" if params_b < 3.5 else "balanced" if params_b < 9.5 else "max"


class ModelProfile:
    """Everything the prompt builder needs to know about one served model (never user-facing)."""

    def __init__(self, model_id, base_url=None, server_kind="generic", context_tokens=None, train_context=None,
                 params_b=None, structured="json_schema", tokenize_url=None, tokenize_style=None):
        self.model_id = model_id or ""
        self.base_url = base_url
        self.family, traits = family_for(self.model_id)
        self.system_role = traits["system_role"]
        self.thinking = traits["thinking"]
        self.chars_per_token = traits["chars"]
        self.chat_sampling = dict(traits["chat"])
        self.server_kind = server_kind if server_kind in SERVER_KINDS else "generic"
        self.train_context = train_context or traits.get("train_ctx")
        self.context_tokens = int(context_tokens or DEFAULT_CONTEXT_TOKENS)
        self.params_b = params_b if params_b is not None else params_from_id(self.model_id)
        self.structured = structured if structured in STRUCTURED_MODES else "json_schema"
        self.tokenize_url = tokenize_url
        self.tokenize_style = tokenize_style
        self.context_source = "default"

    @property
    def size_class(self):
        return size_class(self.params_b)

    def plan_sampling(self):
        return dict(PLAN_SAMPLING)

    def thinking_fields(self):
        """Request fields that switch hybrid 'thinking' off (it would eat the whole output budget)."""
        if self.thinking != "switchable":
            return {}
        if self.server_kind == "ollama":
            return {"reasoning_effort": "none"}
        if self.server_kind in ("llamacpp", "vllm"):
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def thinking_suffix(self):
        """Soft switch appended to the user turn when no request field is known to work."""
        if self.thinking == "switchable" and self.family == "qwen3" and not self.thinking_fields():
            return " /no_think"
        return ""

    def output_multiplier(self):
        return 3 if self.thinking == "always" else 1

    def public_summary(self):
        """Name-free summary for logs/evaluation."""
        return {"server": self.server_kind, "context": self.context_tokens, "context_source": self.context_source,
                "structured": self.structured, "system_role": self.system_role, "thinking": self.thinking,
                "size_class": self.size_class, "exact_tokens": bool(self.tokenize_url)}

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"ModelProfile({self.public_summary()!r})"


# ── HTTP helpers (loopback only; callers pass base URLs already normalised) ──

def _get_json(url, timeout=PROBE_TIMEOUT_SEC, data=None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Accept": "application/json",
                                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_root(base_url):
    """'http://127.0.0.1:8080/v1' -> 'http://127.0.0.1:8080' (native endpoints live at the root)."""
    base = (base_url or "").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _guess_kind(base_url, item):
    owned = str((item or {}).get("owned_by", "")).lower()
    if "llamacpp" in owned or "llama.cpp" in owned or isinstance((item or {}).get("meta"), dict):
        return "llamacpp"
    if "vllm" in owned or "max_model_len" in (item or {}):
        return "vllm"
    if owned == "library" or ":11434" in (base_url or ""):
        return "ollama"
    if ":1234" in (base_url or "") or owned == "organization_owner":
        return "lmstudio"
    return "generic"


def _probe_llamacpp(root, model_id, item):
    out = {"tokenize_url": f"{root}/tokenize", "tokenize_style": "llamacpp"}
    meta = (item or {}).get("meta") or {}
    if isinstance(meta.get("n_ctx_train"), int):
        out["train_context"] = meta["n_ctx_train"]
    if isinstance(meta.get("n_params"), (int, float)) and meta["n_params"] > 0:
        out["params_b"] = meta["n_params"] / 1e9
    props = _get_json(f"{root}/props")
    settings = props.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx") or props.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        out["context_tokens"] = n_ctx
    template = props.get("chat_template")
    if isinstance(template, str) and template and "system" not in template:
        out["system_role"] = False      # template never renders a system turn
    return out


def _probe_ollama(root, model_id, item):
    out = {}
    show = _get_json(f"{root}/api/show", data={"model": model_id})
    info = show.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            out["train_context"] = value
        if key == "general.parameter_count" and isinstance(value, (int, float)) and value > 0:
            out["params_b"] = value / 1e9
    num_ctx = None
    match = re.search(r"(?m)^\s*num_ctx\s+(\d+)", str(show.get("parameters") or ""))
    if match:
        num_ctx = int(match.group(1))
    try:
        for loaded in (_get_json(f"{root}/api/ps").get("models") or []):
            if loaded.get("name") == model_id or loaded.get("model") == model_id:
                if isinstance(loaded.get("context_length"), int) and loaded["context_length"] > 0:
                    num_ctx = loaded["context_length"]
    except Exception:
        pass
    env_ctx = _env_int("OLLAMA_CONTEXT_LENGTH")
    context = num_ctx or env_ctx or OLLAMA_DEFAULT_NUM_CTX
    if out.get("train_context"):
        context = min(context, out["train_context"])
    out["context_tokens"] = context
    return out


def _probe_lmstudio(root, model_id, item):
    data = _get_json(f"{root}/api/v0/models/{urllib.request.quote(model_id, safe='')}")
    out = {}
    for key in ("loaded_context_length", "max_context_length"):
        if isinstance(data.get(key), int) and data[key] > 0:
            out.setdefault("context_tokens", data[key])
    if isinstance(data.get("max_context_length"), int):
        out["train_context"] = data["max_context_length"]
    return out


def _probe_vllm(root, model_id, item):
    out = {"tokenize_url": f"{root}/tokenize", "tokenize_style": "vllm"}
    if isinstance((item or {}).get("max_model_len"), int):
        out["context_tokens"] = item["max_model_len"]
    return out


PROBES = {"llamacpp": _probe_llamacpp, "ollama": _probe_ollama, "lmstudio": _probe_lmstudio, "vllm": _probe_vllm}


class ProfileRegistry:
    """Caches one profile per (endpoint, model); thread-safe; probes are best-effort."""

    def __init__(self, metadata_lookup=None, clock=time.monotonic, ttl=PROFILE_TTL_SEC, probes=None):
        self._metadata_lookup = metadata_lookup
        self._clock = clock
        self._ttl = ttl
        self._probes = PROBES if probes is None else probes
        self._lock = threading.Lock()
        self._cache = {}
        self._calibration = {}        # (base_url, model) -> chars/token EMA
        self._structured_downgrade = {}

    def _metadata(self, base_url, model_id):
        lookup = self._metadata_lookup
        if lookup is None:
            try:
                import reasoning_models
                lookup = reasoning_models.discovery.metadata
            except Exception:
                return None
        try:
            return lookup(base_url, model_id)
        except Exception:
            return None

    def profile(self, model_id, base_url=None, tier=None):
        key = (base_url, model_id)
        now = self._clock()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self._ttl:
                return self._apply_dynamic(cached[1], key)
        profile = self._build(model_id, base_url, tier)
        with self._lock:
            self._cache[key] = (now, profile)
        return self._apply_dynamic(profile, key)

    def _build(self, model_id, base_url, tier):
        item = self._metadata(base_url, model_id) if base_url else None
        kind = _env("LOCAL_REASONING_SERVER", "auto").lower()
        if kind not in SERVER_KINDS:
            kind = _guess_kind(base_url, item) if base_url else "generic"
        found = {}
        probe = self._probes.get(kind)
        if probe and base_url and _env("LOCAL_REASONING_PROBE", "on").lower() not in ("off", "0", "false", "no"):
            try:
                found = probe(server_root(base_url), model_id, item) or {}
            except Exception as err:
                _log.info("Reasoning server metadata probe unavailable (%s).", type(err).__name__)
                found = {}
        profile = ModelProfile(model_id, base_url, kind, found.get("context_tokens"), found.get("train_context"),
                               found.get("params_b"), tokenize_url=found.get("tokenize_url"),
                               tokenize_style=found.get("tokenize_style"))
        if found.get("context_tokens"):
            profile.context_source = "server"
        if found.get("system_role") is False:
            profile.system_role = False
        if profile.params_b is None and tier:
            profile.params_b = {"lite": 2.0, "balanced": 4.0, "max": 12.0}.get(tier)
        forced = _env_int("LOCAL_REASONING_CONTEXT_TOKENS")
        if forced:
            profile.context_tokens, profile.context_source = forced, "env"
        mode = _env("LOCAL_REASONING_STRUCTURED_OUTPUT", "auto").lower()
        if mode in STRUCTURED_MODES:
            profile.structured = mode
        return profile

    def _apply_dynamic(self, profile, key):
        calibrated = self._calibration.get(key)
        if calibrated:
            profile.chars_per_token = calibrated
        downgrade = self._structured_downgrade.get(key)
        if downgrade:
            profile.structured = downgrade
        return profile

    # ── learning from responses ──
    def calibrate(self, profile, prompt_chars, prompt_tokens, overhead_tokens=0):
        """Update the chars/token estimate from a server-reported usage.prompt_tokens (EMA, clamped, conservative)."""
        try:
            content_tokens = int(prompt_tokens) - int(overhead_tokens)
        except (TypeError, ValueError):
            return
        if content_tokens <= 16 or prompt_chars <= 64:
            return
        observed = max(2.0, min(5.0, prompt_chars / content_tokens * 0.92))   # 8% safety margin
        key = (profile.base_url, profile.model_id)
        with self._lock:
            previous = self._calibration.get(key)
            value = observed if previous is None else 0.7 * previous + 0.3 * observed
            self._calibration[key] = round(value, 3)
            profile.chars_per_token = self._calibration[key]

    def downgrade_structured(self, profile):
        """Server rejected the structured-output request: json_schema -> json_object -> off."""
        order = list(STRUCTURED_MODES)
        index = order.index(profile.structured) if profile.structured in order else 0
        new_mode = order[min(index + 1, len(order) - 1)]
        with self._lock:
            self._structured_downgrade[(profile.base_url, profile.model_id)] = new_mode
        profile.structured = new_mode
        _log.info("Structured output downgraded to %s for a reasoning tier.", new_mode)
        return new_mode

    def invalidate(self):
        with self._lock:
            self._cache.clear()


registry = ProfileRegistry()


def default_profile(tier="lite"):
    """Profile used when no endpoint is known (tests, offline): safe 4k window, lite sizing."""
    profile = ModelProfile("", None, "generic", _env_int("LOCAL_REASONING_CONTEXT_TOKENS") or DEFAULT_CONTEXT_TOKENS,
                           params_b={"lite": 2.0, "balanced": 4.0, "max": 12.0}.get(tier, 2.0))
    mode = _env("LOCAL_REASONING_STRUCTURED_OUTPUT", "auto").lower()
    if mode in STRUCTURED_MODES:
        profile.structured = mode
    return profile


# ── Token counting ──

class TokenCounter:
    """Exact counts via the server's /tokenize when available (cached), else the calibrated estimator."""

    def __init__(self, profile, timeout=1.0):
        self.profile = profile
        self.timeout = timeout
        self._cache = {}
        self.exact_failures = 0

    def estimate(self, text):
        if not text:
            return 0
        return int(math.ceil(len(text.encode("utf-8")) / self.profile.chars_per_token))

    def exact(self, text):
        """Exact token count, or None when the server cannot tokenize."""
        url = self.profile.tokenize_url
        if not url or not text or self.exact_failures >= 2:
            return None
        if text in self._cache:
            return self._cache[text]
        try:
            if self.profile.tokenize_style == "vllm":
                body = _get_json(url, self.timeout, {"model": self.profile.model_id, "prompt": text})
                count = body.get("count") if isinstance(body.get("count"), int) else len(body.get("tokens") or [])
            else:
                body = _get_json(url, self.timeout, {"content": text, "add_special": False})
                count = len(body.get("tokens") or [])
        except Exception:
            self.exact_failures += 1
            return None
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[text] = count
        return count

    def count(self, text):
        exact = self.exact(text)
        return exact if exact is not None else self.estimate(text)
