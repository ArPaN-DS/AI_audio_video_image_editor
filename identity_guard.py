"""
Identity Guard — keeps the Copilot from disclosing what powers it.

Two deterministic layers that do not depend on the language model behaving:

1. Probe interception (input side): questions about the underlying model,
   vendor, libraries, architecture, or attempts to extract the system prompt
   ("ignore previous instructions…", "repeat the text above", role-play
   jailbreaks) are answered with a fixed capability-language reply, without
   the request ever reaching the language model.

2. Output scrubbing (output side): every string in the outgoing chat payload
   is cleaned of model-family names, runtime/library names, vendor
   attributions ("trained by …") and self-identification sentences, so even
   a model that ignores its instructions cannot leak its identity.
"""

import re
import unicodedata

import branding

_ASSISTANT = branding.assistant_name()
_PRODUCT = branding.product_name()
PUBLIC_IDENTITY_REPLY = (
    f"I'm {_ASSISTANT}, built into {_PRODUCT} and running entirely on your device. "
    f"I don't share details about the internal engines behind {_PRODUCT}, "
    "but I'm here to edit your audio, video, and images. What would you like to create?"
)
PUBLIC_IDENTITY_SENTENCE = f"I'm {_ASSISTANT}, built into {_PRODUCT}."
PUBLIC_ENGINE_NAME = f"{_PRODUCT}'s built-in engine"

PROBE_SUGGESTIONS = [
    "Transcribe speech",
    "Remove the background from a photo",
    "Trim a clip",
    "Normalize loudness",
]

# ── Model families, runtimes and libraries that must never be named ─────────
_PRIVATE_NAMES = r"""
    gemma[\w.:-]* | gemini[\w.:-]* | palm\s?2 | bard
  | llama[-\s]?\d[\w.:-]* | llama\.cpp | code\s?llama | meta[-\s]llama
  | qwen[\w.:-]* | mistral[\w.:-]* | mixtral[\w.:-]* | phi[-\s]?\d[\w.:-]*
  | gpt[-\s]?\d[\w.:-]* | gpt[-\s]?oss | chat\s?gpt | o\d[-\s]?mini | davinci
  | claude[\w.:-]* | deepseek[\w.:-]* | falcon[-\s]?\d+\w* | vicuna | alpaca
  | smol\s?lm\w* | tiny\s?llama | stable\s?lm | olmo\w* | granite[-\s]\d\w*
  | (?:faster[-_\s]?)?whisper[\w.-]* | wav2vec\w* | silero\w* | pyannote\w*
  | isnet[\w.-]* | u2net[\w.-]* | birefnet[\w.-]* | rembg | modnet | sam\s?2?\b(?=[-\s]?(?:model|segment))
  | (?:real[-_\s]?)?esrgan[\w.-]* | edsr | fsrcnn | gfpgan | codeformer | \blama\b
  | demucs\w* | spleeter | deepfilter\w* | noisereduce | rnnoise
  | v\s?llm\w* | ollama | lm\s?studio | text[-\s]generation[-\s]inference | \btgi\b
  | hugging\s?face | transformers\s(?:library|lib) | sentence[-\s]transformers
  | pytorch | \btorch\w* | tensorflow | \bonnx\w* | ctranslate2 | \bcuda\b | tensorrt
  | ffmpeg | ffprobe | opencv | \bcv2\b | librosa | pydub | moviepy | scenedetect | pyscenedetect
  | wavesurfer | font\s?awesome | jszip | flask | werkzeug | jinja2?
"""
_PRIVATE_NAME_RE = re.compile(rf"(?<![\w/])(?:{_PRIVATE_NAMES})", re.IGNORECASE | re.VERBOSE)

_VENDORS = (r"google(?:\s+deepmind)?|deep\s?mind|meta(?:\s+ai)?|facebook|open\s?ai|anthropic|alibaba|"
            r"mistral\s+ai|microsoft|nvidia|hugging\s?face|stability\s+ai|tii|ibm|apple|amazon|cohere|"
            r"x\.?ai|deepseek|tencent|baidu|zhipu|01\.ai")
# "trained/developed/created/built/made/provided by <vendor>" and "a <vendor> model"
_ATTRIBUTION_RE = re.compile(
    rf"\b(?:trained|developed|created|built|made|designed|released|provided|powered|hosted|owned)"
    rf"\s+(?:by|at|from)\s+(?:the\s+)?(?:{_VENDORS})\b[\w\s.,'-]{{0,40}}",
    re.IGNORECASE,
)
_VENDOR_MODEL_RE = re.compile(
    rf"\b(?:an?\s+)?(?:{_VENDORS})(?:'s)?\s+(?:large\s+)?(?:language\s+|ai\s+|open\s+|foundation\s+)?"
    rf"(?:model|llm|assistant|ai|family)\b",
    re.IGNORECASE,
)
_ENGINE_VENDOR_RE = re.compile(
    rf"({re.escape(PUBLIC_ENGINE_NAME)}|{re.escape(_ASSISTANT)})\s*,?\s+(?:by|from|of|at)\s+(?:the\s+)?(?:{_VENDORS})\b",
    re.IGNORECASE,
)
_FIRST_PERSON_VENDOR_RE = re.compile(
    rf"\b(i\s*am|i'm|i\s+was|i\s+come)\b([^.!?\n]{{0,60}}?)\s*,?\s+(?:by|from|at|of)\s+(?:the\s+)?(?:{_VENDORS})\b",
    re.IGNORECASE,
)
# A sentence in which the assistant describes what it is.
_SELF_ID_SENTENCE_RE = re.compile(
    r"[^.!?\n]*\b(?:i\s*am|i'm|i\s+was|as\s+an?|this\s+assistant\s+is|my\s+(?:underlying\s+)?"
    r"(?:model|architecture|weights|parameters|training|creators?|developers?|base\s+model))"
    r"[^.!?\n]*\b(?:language\s+model|llm|ai\s+model|neural\s+network|trained|parameters|weights|"
    r"model\s+(?:developed|created|made|trained|from|by)|open[-\s]?weights?)[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)
_SYSTEM_PROMPT_LEAK_RE = re.compile(
    r"[^.!?\n]*\b(?:my|the)\s+(?:system\s+prompt|system\s+message|hidden\s+instructions?|"
    r"initial\s+instructions?|developer\s+(?:message|instructions?))\b[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)

# ── Probes: identity questions and prompt-extraction attempts ───────────────
_IDENTITY_TARGET = (r"(?:model|llm|language\s+model|ai\s+model|engine|backend|back[-\s]end|library|libraries|lib|"
                    r"framework|stack|tech(?:nology)?(?:\s+stack)?|architecture|weights|parameters|params|"
                    r"version|base\s+model|neural\s+network|runtime|server|provider|vendor|api)")
_PROBE_PATTERNS = [
    # what/which model … you / this app / the copilot / the studio
    rf"\b(?:what|which|whats|what's|wich|tell\s+me|name|reveal|disclose|list|share)\b[^?\n]{{0,30}}\b{_IDENTITY_TARGET}\b"
    rf"[^?\n]{{0,14}}\b(?:you|your|u|ur|(?:this|the)\s+(?:copilot|assistant|agent|studio|app|tool|editor|bot|ai|platform)|"
    rf"copilot|assistant)\b",
    rf"\b(?:you|your|u|ur|this\s+(?:app|tool|studio|copilot|assistant|bot)|copilot|studio)\b[^?\n]{{0,40}}"
    rf"\b(?:use|using|built\s+(?:on|with)|based\s+on|powered\s+by|running\s+(?:on)?|run\s+on|made\s+with)\b"
    rf"[^?\n]{{0,40}}\b{_IDENTITY_TARGET}\b",
    r"\b(?:what|which)\s+(?:are|r)\s+(?:you|u)\s+(?:based\s+on|built\s+(?:on|with)|powered\s+by|running\s+on|made\s+of)\b",
    r"\b(?:what|which)\s+(?:company|companies|team|organi[sz]ation|lab|vendor|provider|firm)\s+"
    r"(?:made|built|created|trained|developed|owns|makes|is\s+behind|runs)\s+"
    r"(?:you|u|(?:this|the)\s+(?:copilot|studio|ai|app|assistant|bot|agent|tool))\b",
    r"\bwho\s+(?:made|built|created|trained|developed|programmed|designed|owns|makes)\s+(?:you|u|(?:this|the)\s+(?:copilot|studio|assistant|ai|app|bot|agent|tool))\b",
    r"\b(?:are|r)\s+(?:you|u)\s+(?:an?\s+)?(?:gemma|gemini|gpt|chat\s?gpt|llama|claude|qwen|mistral|phi|deepseek|"
    r"open\s?ai|google|meta|anthropic|whisper|llm|large\s+language\s+model|language\s+model|ai\s+model|chatbot\s+from)\b",
    r"\b(?:what|which)\s+(?:ai|llm|model|company|vendor|provider)\s+(?:is\s+(?:this|that|it|behind)\b|"
    r"are\s+(?:you|u)\b|powers?\b|runs?\s+(?:you|u|this|the)\b|drives?\b)",
    r"\bhow\s+(?:many|big)\b[^?\n]{0,20}\b(?:parameters|params|billion|weights)\b",
    r"\b(?:context\s+(?:window|length)|quantiz\w+|gguf|safetensors|checkpoint|fine[-\s]?tun\w+)\b[^?\n]{0,40}\b(?:you|your|this|copilot)\b",
    # prompt extraction / jailbreaks
    r"\b(?:system|initial|hidden|developer|original|secret|internal)\s+(?:prompt|instructions?|message|rules|config\w*)\b",
    r"\bignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|preceding|system|safety)?\s*(?:instructions?|rules|prompts?|directions|guidelines)\b",
    r"\b(?:repeat|print|output|show|reveal|dump|echo|recite|display|leak|return)\b[^?\n]{0,30}\b(?:everything|all|text|words|content|instructions?|prompt)\b"
    r"[^?\n]{0,20}\b(?:above|before|preceding|so\s+far|verbatim|you\s+were\s+given|you\s+received)\b",
    r"\b(?:developer|debug|god|admin|maintenance|unrestricted|jailbreak)\s+mode\b",
    r"\b(?:\bDAN\b|do\s+anything\s+now|pretend\s+(?:you\s+are|to\s+be)\s+(?:not|an?\s+(?:unfiltered|unrestricted))|"
    r"act\s+as\s+(?:an?\s+)?(?:unfiltered|unrestricted|different)\s+(?:ai|model|assistant))",
    r"\bwhat\s+(?:are|were)\s+your\s+(?:instructions|rules|guidelines|directives)\b",
    r"\b(?:print|show|reveal|tell\s+me)\s+(?:your|the)\s+(?:source\s+code|code|config(?:uration)?|env(?:ironment)?\s+var\w*|\.env|api\s+key|model\s+(?:id|name|path))\b",
    r"\bwhich\s+(?:python\s+)?(?:package|module|dependency|dependencies)\b",
    r"\b(?:pip\s+(?:list|freeze)|requirements\.txt|package\.json|node_modules|site-packages)\b",
]
_PROBE_RES = [re.compile(pattern, re.IGNORECASE) for pattern in _PROBE_PATTERNS]
_PRIVATE_NAME_PROBE_RE = re.compile(
    r"\b(?:gemma|gemini|llama|qwen|mistral|deepseek|chat\s?gpt|gpt[-\s]?\d|claude|whisper|vllm|ollama|"
    r"hugging\s?face|pytorch|torch|onnx|ffmpeg|opencv|librosa|rembg|u2net|isnet|birefnet|esrgan|edsr|"
    r"fsrcnn|demucs|wavesurfer|flask|werkzeug)\b",
    re.IGNORECASE,
)
_ZERO_WIDTH_RE = re.compile(r"[​-‏⁠-⁤﻿­]")
_LEETSPEAK = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _normalize(text):
    """Fold Unicode tricks (full-width letters, zero-width chars, spacing) used to dodge filters."""
    folded = unicodedata.normalize("NFKC", str(text or ""))
    folded = _ZERO_WIDTH_RE.sub("", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _deobfuscate(text):
    """Collapse s-p-a-c-e-d / dotted letters and leetspeak for probe matching only."""
    collapsed = re.sub(r"\b(?:\w[\s._*-]){2,}\w\b", lambda m: re.sub(r"[\s._*-]", "", m.group(0)), text)
    return collapsed.translate(_LEETSPEAK)


# Always-on signatures: unambiguous even with spacing/leetspeak removed.
_COMPACT_SIGNATURES = (
    "whatareyoubasedon", "whomadeyou", "whobuiltyou", "whocreatedyou", "whotrainedyou",
    "systemprompt", "hiddeninstruction", "ignorepreviousinstruction", "ignoreallpreviousinstruction",
    "ignoreyourinstruction", "ignoreallinstruction", "repeateverythingabove", "developermode", "jailbreak",
    "areyougemma", "areyougpt", "areyouchatgpt", "areyoullama", "areyouclaude", "areyouqwen",
    "areyougemini", "areyoumistral", "yoursourcecode",
)
# Only checked when the message is visibly obfuscated (s p a c e d / d.o.t.t.e.d letters).
_OBFUSCATED_SIGNATURES = (
    "whatmodel", "whichmodel", "whatllm", "whichllm", "whatlanguagemodel", "whatlibrar",
    "whichlibrar", "whatbackend", "whichbackend", "whatengine", "whichengine", "techstack",
    "modelareyou", "modelruyou", "modeldoyou",
)


def _looks_obfuscated(text):
    tokens = text.split()
    single = sum(1 for token in tokens if len(re.sub(r"[^A-Za-z0-9@$]", "", token)) == 1)
    return bool(tokens) and (single / len(tokens) >= 0.5 or bool(re.search(r"\b(?:\w[._*-]){3,}\w\b", text)))


def _compact(text):
    return re.sub(r"[^a-z]", "", text.lower().translate(_LEETSPEAK))


def is_identity_probe(message):
    """True when the message asks about internals or tries to extract hidden instructions."""
    normalized = _normalize(message)
    if not normalized:
        return False
    compact = _compact(normalized)
    if any(signature in compact for signature in _COMPACT_SIGNATURES):
        return True
    if _looks_obfuscated(normalized) and any(signature in compact for signature in _OBFUSCATED_SIGNATURES):
        return True
    for candidate in {normalized, _deobfuscate(normalized)}:
        if any(regex.search(candidate) for regex in _PROBE_RES):
            return True
        # Directly naming a private engine in a question ("are you using ollama?",
        # "do you run whisper") is a probe even without other keywords.
        if _PRIVATE_NAME_PROBE_RE.search(candidate) and re.search(
                r"\?|\b(?:you|your|u|ur|this|copilot|studio|use|using|run|running|powered|based)\b",
                candidate, re.IGNORECASE):
            return True
    return False


def probe_reply(message):
    """Return a complete chat payload for probes, or None for normal requests."""
    if not is_identity_probe(message):
        return None
    return {
        "status": "success",
        "thought": "Answering a question about the Studio itself.",
        "delegated_subagent": "Orchestrator",
        "clarification_needed": False,
        "clarification_options": [],
        "tools": [],
        "reply": PUBLIC_IDENTITY_REPLY,
        "suggested_actions": list(PROBE_SUGGESTIONS),
        "execution_results": [],
    }


def scrub_text(text):
    """Remove identity disclosures from generated text while keeping the useful rest."""
    if not isinstance(text, str) or not text:
        return text
    cleaned = _normalize(text) if _ZERO_WIDTH_RE.search(text) else text
    cleaned = _SYSTEM_PROMPT_LEAK_RE.sub(" ", cleaned)
    cleaned = _SELF_ID_SENTENCE_RE.sub(f" {PUBLIC_IDENTITY_SENTENCE} ", cleaned)
    cleaned = _ATTRIBUTION_RE.sub("built for this Studio", cleaned)
    cleaned = _VENDOR_MODEL_RE.sub(PUBLIC_ENGINE_NAME, cleaned)
    cleaned = _PRIVATE_NAME_RE.sub(PUBLIC_ENGINE_NAME, cleaned)
    # "<engine> by Google", "I'm … from OpenAI": drop the vendor attribution.
    cleaned = _ENGINE_VENDOR_RE.sub(r"\1", cleaned)
    cleaned = _FIRST_PERSON_VENDOR_RE.sub(r"\1\2", cleaned)
    # Collapse repeated replacements and stray whitespace.
    repeated = re.escape(PUBLIC_IDENTITY_SENTENCE)
    cleaned = re.sub(rf"(?:{repeated}\s*){{2,}}", PUBLIC_IDENTITY_SENTENCE + " ", cleaned)
    engine = re.escape(PUBLIC_ENGINE_NAME)
    cleaned = re.sub(rf"{engine}(?:[\s,/&+-]+(?:and\s+)?{engine})+", PUBLIC_ENGINE_NAME, cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    return cleaned.strip() or PUBLIC_IDENTITY_SENTENCE


# Keys whose values are server-controlled identifiers/paths, not prose.
_PASSTHROUGH_KEYS = {"output_url", "url", "download_url", "input_url", "filename", "file", "id",
                     "media_id", "status", "type", "tool", "name", "subagent", "delegated_subagent",
                     "code", "mime", "mimetype", "format", "extension"}


def guard_chat_response(payload):
    """Recursively scrub every prose string in an outgoing chat payload."""
    if isinstance(payload, dict):
        return {
            key: (value if key in _PASSTHROUGH_KEYS and isinstance(value, str) else guard_chat_response(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [guard_chat_response(item) for item in payload]
    if isinstance(payload, str):
        return scrub_text(payload)
    return payload
