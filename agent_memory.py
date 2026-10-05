"""
Copilot Memory — 100% on-device long-term memory for the media copilot.

Why: the reasoning tier is a small model with a short context window, so the
prompt cannot carry raw chat history.  This module stores typed memories in a
local database and packs a compact, relevance-ranked context block that fits a
strict token budget.

Design
------
Storage   One SQLite file (stdlib, WAL, incremental auto-vacuum, secure
          delete), versioned schema with ordered migrations, B-tree indexes on
          (session_id, kind, created_at), (kind, slot), (media_fp),
          (expires_at), (importance, accessed_at) and vectors(space).  A bad or
          unreadable file is quarantined (renamed) and a fresh store created.
Collections (``kind``)
          conversation     turns of one session (never shared across sessions)
          session_summary  rolling extractive summary, one per session
          preference       user preferences, the only cross-session memories
          media_fact       inspection facts keyed by a content fingerprint
          edit_outcome     what ran in a session and whether it worked
          job              orchestration memory: request + media condition
                           signature -> plan -> per-step outcome/timing,
                           merged across runs (see ``recall_similar_jobs``)
Indexing  Lexical: full-text index with BM25 ranking when the bundled engine
          supports it (feature-detected), otherwise a built-in BM25.  Vector:
          a deterministic hashed word/bigram/char-n-gram + domain-concept
          embedding is stored for every row (zero model memory).  Optional
          semantic vectors live in a separate table tagged with their
          embedding space; spaces are never compared with each other, and
          missing rows are lazily re-embedded at query time.  Each space has
          an in-memory float32 matrix cache maintained incrementally, so
          cosine top-k is one matrix product (milliseconds for tens of
          thousands of rows; approximate search is not needed at this scale).
Retrieval Weighted fusion of normalized BM25 + cosine (semantic when
          available), importance and recency boosts, metadata filters,
          MMR diversity with hard near-duplicate suppression.
Packing   ``build_context`` keeps the latest turns verbatim, then preferences,
          media facts, the session summary, similar past jobs and related
          earlier memories, measured with a conservative token estimate and
          never exceeding the budget.

Every entry point used by the web app is failure-isolated: memory problems
degrade to "no memory", never to a failed chat.  Public text uses capability
language only.
"""

import contextlib
import contextvars
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_log = logging.getLogger("agent_processor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "copilot_memory", "memory.db")

SCHEMA_VERSION = 1
LEXICAL_SPACE = "lex:v1"
LEX_DIM = 384
CAPABILITY = "copilot_memory"
import branding
PUBLIC_NAME = f"{branding.assistant_name()} memory"

MAX_TEXT_CHARS = 1200
KIND_TEXT_CAPS = {"conversation": 800, "preference": 240, "media_fact": 600,
                  "edit_outcome": 400, "job": 400, "session_summary": 700}
KINDS = ("conversation", "session_summary", "preference", "media_fact", "edit_outcome", "job")
DAY = 86400.0
DEFAULT_TTL = {"conversation": 30 * DAY, "session_summary": 90 * DAY, "media_fact": 180 * DAY,
               "edit_outcome": 90 * DAY, "job": 365 * DAY, "preference": None}
DEFAULT_IMPORTANCE = {"conversation": 0.45, "session_summary": 0.6, "preference": 0.9,
                      "media_fact": 0.6, "edit_outcome": 0.55, "job": 0.6}
DEFAULT_MAX_ROWS = 20000
SESSION_KINDS = ("conversation", "edit_outcome")
ELIGIBLE_LIMIT = 5000
LAZY_EMBED_CAP = 64
MIN_RELEVANCE = 0.10
DUPLICATE_COSINE = 0.92
RECENCY_HALF_LIFE_H = 72.0
CHARS_PER_TOKEN = 3.5          # conservative vs the usual ~4 chars/token
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


# ─────────────────────────────────────────────────────────────────────────
#  Text utilities
# ─────────────────────────────────────────────────────────────────────────

def estimate_tokens(text):
    """Conservative token estimate (UTF-8 bytes / 3.5) — never under-counts typical text."""
    if not text:
        return 0
    return int(math.ceil(len(text.encode("utf-8")) / CHARS_PER_TOKEN))


def valid_session_id(value):
    """Return a safe session id or None."""
    if isinstance(value, str) and SESSION_ID_RE.match(value.strip()):
        return value.strip()
    return None


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|users|mnt|var|tmp|private|root)/)[^\s\"'<>|]*")


def sanitize_text(text, limit=MAX_TEXT_CHARS):
    """Strip control characters, collapse whitespace, reduce absolute paths to file names, cap length."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _CONTROL_RE.sub("", text)
    text = _PATH_RE.sub(lambda match: re.split(r"[\\/]", match.group(0).rstrip("\\/"))[-1], text)
    text = re.sub(r"\s+", " ", text).strip()
    limit = min(limit, MAX_TEXT_CHARS)
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing for from had has have i i'm if in into is it
it's its just let me my of on or our please so than that the their them then there these this those to too
us was we were what when where which while who why will with would you your yours make made want wanted
""".split())

_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?[a-z]*|[a-z][a-z0-9']*")


def _stem(word):
    if len(word) <= 3 or word[0].isdigit() or word[0] == "-":
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text):
    """Lower-case, stop-word filtered, lightly stemmed tokens (numbers keep their sign/units)."""
    tokens = []
    for raw in _TOKEN_RE.findall((text or "").lower().replace("’", "'")):
        raw = raw.strip("'")
        if not raw or raw in STOPWORDS:
            continue
        tokens.append(_stem(raw.replace("'", "")))
    return tokens


# Small domain lexicon: paraphrases of the same media concept share a feature.
_CONCEPT_WORDS = {
    "loudness": "loudness loud lufs volume normalize normalise normalization normalisation gain level levels quiet quietly louder",
    "noise": "noise noisy hiss hum buzz denoise static clicks crackle rumble",
    "cutout": "cutout cut-out transparent matte backdrop",
    "transcribe": "transcribe transcript transcription captions caption subtitles subtitle srt dictation",
    "upscale": "upscale upscaling enlarge bigger super-resolution 4x 2x",
    "resolution": "resolution 1080p 720p 4k 2160p 1440p pixels",
    "speed": "speed slow slower faster fast motion tempo half 1.5x 2x 0.5x",
    "gif": "gif animated animation",
    "compress": "compress compression shrink smaller size email lighter",
    "fade": "fade fades fading",
    "extract_audio": "extract soundtrack pull rip",
    "silence": "silence silences silent pauses pause gaps gap",
    "faces": "face faces restore",
    "clarity": "clarity clearer clear muffled intelligible vocals voice speech",
    "sharpen": "sharpen sharp sharper blurry blur focus crisp",
    "photo": "photo photos picture image pic photograph scan portrait",
    "video": "video clip footage film movie trailer screencast",
    "lossless": "flac lossless archive archiving archival",
    "format": "format webm mp4 mkv mov container convert",
    "trim": "trim cut shorten timestamps seconds",
    "export": "export exports render bitrate kbps 320k 256k 192k 128k mp3",
    "podcast": "podcast podcasts episode",
    "interview": "interview guest host",
}
CONCEPTS = {}
for _concept, _words in _CONCEPT_WORDS.items():
    for _word in _words.split():
        for _token in tokenize(_word) or [_word]:
            CONCEPTS.setdefault(_token, set()).add(_concept)


def concepts_for(tokens):
    found = []
    for token in tokens:
        for concept in sorted(CONCEPTS.get(token, ())):
            found.append(concept)
    return found


def _concept_terms(tokens):
    return ["kx" + concept.replace("_", "") for concept in dict.fromkeys(concepts_for(tokens))]


def search_text_for(text):
    """Text indexed by the full-text engine: the memory plus its concept terms."""
    terms = _concept_terms(tokenize(text))
    return f"{text} {' '.join(terms)}".strip()


def _hash_feature(feature):
    value = zlib.crc32(feature.encode("utf-8"))
    return value % LEX_DIM, 1.0 if (value >> 16) & 1 else -1.0


def lexical_embedding(text):
    """Deterministic hashed embedding: stems, stem bigrams, char trigrams and concepts (L2-normalized)."""
    tokens = tokenize(text)
    features = {}

    def bump(feature, weight):
        features[feature] = features.get(feature, 0.0) + weight

    for token in tokens:
        bump("w:" + token, 1.0)
        if len(token) >= 4 and not token[0].isdigit():
            padded = f"#{token}#"
            for index in range(len(padded) - 2):
                bump("c:" + padded[index:index + 3], 0.2)
    for left, right in zip(tokens, tokens[1:]):
        bump(f"b:{left}_{right}", 0.5)
    for concept in concepts_for(tokens):
        bump("k:" + concept, 0.8)
    vector = np.zeros(LEX_DIM, dtype=np.float32)
    for feature, weight in features.items():
        index, sign = _hash_feature(feature)
        vector[index] += sign * (1.0 + math.log(weight)) if weight >= 1.0 else sign * weight
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _to_blob(vector):
    return np.ascontiguousarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
#  Media helpers
# ─────────────────────────────────────────────────────────────────────────

_fingerprint_cache = OrderedDict()
_fingerprint_lock = threading.Lock()


def media_fingerprint(path, sample_bytes=65536):
    """Fast content fingerprint (size + head/middle/tail samples); None if unreadable."""
    try:
        if not path or not os.path.isfile(path):
            return None
        stat = os.stat(path)
        key = (os.path.realpath(path), stat.st_size, stat.st_mtime_ns)
        with _fingerprint_lock:
            if key in _fingerprint_cache:
                _fingerprint_cache.move_to_end(key)
                return _fingerprint_cache[key]
        digest = hashlib.sha256(str(stat.st_size).encode("ascii"))
        with open(path, "rb") as handle:
            for offset in (0, max(0, stat.st_size // 2 - sample_bytes // 2), max(0, stat.st_size - sample_bytes)):
                handle.seek(offset)
                digest.update(handle.read(sample_bytes))
        value = digest.hexdigest()[:32]
        with _fingerprint_lock:
            _fingerprint_cache[key] = value
            while len(_fingerprint_cache) > 256:
                _fingerprint_cache.popitem(last=False)
        return value
    except OSError:
        return None


def condition_signature(report=None):
    """
    Media condition tags from an inspector report (or pass-through for a tag
    string/list).  Example: ['bg:noise', 'speech', 'type:audio'].
    """
    if not report:
        return []
    if isinstance(report, str):
        return sorted({tag for tag in re.split(r"[\s,]+", report.strip()) if tag})
    if isinstance(report, (list, tuple, set)):
        return sorted({str(tag).strip() for tag in report if str(tag).strip()})
    if not isinstance(report, dict):
        return []
    tags = set()
    if report.get("type") in ("audio", "video", "image"):
        tags.add("type:" + report["type"])
    audio = report.get("audio") or {}
    if isinstance(audio, dict) and audio:
        if audio.get("silent"):
            tags.add("silent")
        elif audio.get("background") in ("clean", "music", "noise", "hum"):
            tags.add("bg:" + audio["background"])
        if audio.get("speech_likely"):
            tags.add("speech")
        if audio.get("quiet"):
            tags.add("quiet")
        if audio.get("clipping"):
            tags.add("clipping")
    image = report.get("image") or {}
    if isinstance(image, dict) and image:
        for key, tag in (("blurry", "img:blurry"), ("noisy", "img:noisy"), ("has_alpha", "img:alpha"),
                         ("small", "img:small")):
            if image.get(key):
                tags.add(tag)
    video = report.get("video") or {}
    if isinstance(video, dict) and video:
        if video.get("dark"):
            tags.add("vid:dark")
        if video.get("has_video") and not video.get("has_audio"):
            tags.add("vid:no-audio")
    return sorted(tags)


def _jaccard(left, right):
    left, right = set(left or ()), set(right or ())
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


# ─────────────────────────────────────────────────────────────────────────
#  Embedding tiers
# ─────────────────────────────────────────────────────────────────────────

def _hf_cache_roots():
    roots = []
    for value in (os.environ.get("HF_HUB_CACHE"),
                  os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
                  os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
                  os.path.join(BASE_DIR, "models")):
        if value and os.path.isdir(value) and value not in roots:
            roots.append(value)
    return roots


class OnDeviceSemanticEmbedder:
    """
    Semantic sentence vectors from an already-cached on-device model (never
    downloads).  Loaded only through the Single-Active-Model manager and only
    when the quality governor allows it.
    """

    MODEL_KEY = "copilot_memory_semantic"
    PREFERRED_FILES = ("model_quint8_avx2.onnx", "model.onnx")
    MAX_FILE_BYTES = 600 * 1024 * 1024
    MAX_TOKENS = 128
    _discovered = None
    _discover_lock = threading.Lock()

    def __init__(self, model_dir, model_file):
        self.model_dir = model_dir
        self.model_file = model_file
        identity = f"{os.path.basename(os.path.dirname(os.path.dirname(model_dir)))}/{os.path.basename(model_file)}"
        self.space = "dense:local-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        self.cls_pooling = False
        try:
            with open(os.path.join(model_dir, "1_Pooling", "config.json"), "r", encoding="utf-8") as handle:
                self.cls_pooling = bool(json.load(handle).get("pooling_mode_cls_token"))
        except (OSError, ValueError):
            pass

    @staticmethod
    def libraries_present():
        return all(importlib.util.find_spec(name) is not None for name in ("onnxruntime", "tokenizers"))

    @classmethod
    def discover(cls, refresh=False):
        """Find a cached sentence-embedding model with an exported graph; None if absent."""
        with cls._discover_lock:
            if cls._discovered is not None and not refresh:
                return cls._discovered or None
            found = None
            if cls.libraries_present():
                explicit = os.environ.get("COPILOT_MEMORY_EMBEDDER_DIR")
                candidates = []
                if explicit and os.path.isdir(explicit):
                    candidates.append(explicit)
                for root in _hf_cache_roots():
                    try:
                        repos = sorted(name for name in os.listdir(root) if name.startswith("models--"))
                    except OSError:
                        continue
                    for repo in repos:
                        snapshots = os.path.join(root, repo, "snapshots")
                        try:
                            candidates.extend(os.path.join(snapshots, snap) for snap in sorted(os.listdir(snapshots)))
                        except OSError:
                            continue
                for directory in candidates:
                    model_file = cls._usable_file(directory)
                    if model_file:
                        found = cls(directory, model_file)
                        break
            cls._discovered = found or False
            return found

    @classmethod
    def _usable_file(cls, directory):
        marker = any(os.path.isfile(os.path.join(directory, name))
                     for name in ("sentence_bert_config.json", "modules.json"))
        if not marker or not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            return None
        for name in cls.PREFERRED_FILES:
            path = os.path.join(directory, "onnx", name)
            try:
                if os.path.isfile(path) and 0 < os.path.getsize(path) <= cls.MAX_FILE_BYTES:
                    return path
            except OSError:
                continue
        return None

    def available(self):
        return os.path.isfile(self.model_file) and self.libraries_present()

    def _load(self):
        import onnxruntime
        import tokenizers
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        options.log_severity_level = 3
        session = onnxruntime.InferenceSession(self.model_file, options, providers=["CPUExecutionProvider"])
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
        tokenizer.enable_truncation(self.MAX_TOKENS)
        tokenizer.enable_padding()
        return (session, tokenizer), {}

    @staticmethod
    def _unload(_instance):
        return None

    def embed(self, texts):
        from model_manager import global_model_manager
        with global_model_manager.session(self.MODEL_KEY, self._load, self._unload) as (instance, _meta):
            session, tokenizer = instance
            input_names = {item.name for item in session.get_inputs()}
            outputs = []
            for start in range(0, len(texts), 32):
                batch = tokenizer.encode_batch([text or " " for text in texts[start:start + 32]])
                ids = np.array([item.ids for item in batch], dtype=np.int64)
                mask = np.array([item.attention_mask for item in batch], dtype=np.int64)
                feeds = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in input_names:
                    feeds["token_type_ids"] = np.zeros_like(ids)
                hidden = session.run(None, feeds)[0]
                if self.cls_pooling:
                    pooled = hidden[:, 0, :]
                else:
                    weights = mask[..., None].astype(np.float32)
                    pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-6)
                outputs.append(pooled.astype(np.float32))
            return np.vstack(outputs) if outputs else np.zeros((0, 1), dtype=np.float32)


class LocalServiceEmbedder:
    """Vectors from the loopback reasoning service's embeddings endpoint (if it serves one)."""

    PATTERN = re.compile(r"embed|bge|e5-|gte|minilm|nomic|mxbai", re.IGNORECASE)

    def __init__(self, base_url, model_id, timeout=3.0):
        self.base_url = base_url
        self.model_id = model_id
        self.timeout = timeout
        self.space = "dense:svc-" + hashlib.sha1(f"{base_url}|{model_id}".encode("utf-8")).hexdigest()[:12]

    @classmethod
    def discover(cls):
        try:
            import reasoning_models
        except Exception:
            return None
        base = reasoning_models.normalize_base_url(
            os.environ.get("LOCAL_EMBEDDING_URL") or os.environ.get("LOCAL_REASONING_URL")
            or os.environ.get("VLLM_API_BASE") or reasoning_models.DEFAULT_BASE_URL)
        if not base:
            return None
        model_id = (os.environ.get("LOCAL_EMBEDDING_MODEL") or "").strip()
        if not model_id:
            served = reasoning_models.discovery.served(base)
            matches = sorted(item for item in (served or ()) if cls.PATTERN.search(item))
            if not matches:
                return None
            model_id = matches[0]
        return cls(base, model_id)

    def available(self):
        return True

    def embed(self, texts):
        import urllib.request
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model_id, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        data = sorted(body.get("data") or [], key=lambda item: item.get("index", 0))
        return np.array([item["embedding"] for item in data], dtype=np.float32)


def _default_embedders():
    embedders = []
    try:
        service = LocalServiceEmbedder.discover()
        if service:
            embedders.append(service)
    except Exception:
        pass
    try:
        local = OnDeviceSemanticEmbedder.discover()
        if local:
            embedders.append(local)
    except Exception:
        pass
    return embedders


class _VectorIndex:
    """In-memory float32 matrix for one embedding space, updated incrementally."""

    def __init__(self, dim):
        self.dim = dim
        self.matrix = np.zeros((64, dim), dtype=np.float32)
        self.ids = np.full(64, -1, dtype=np.int64)
        self.position = {}
        self.size = 0
        self.free = 0

    def add(self, memory_id, vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            return False
        if memory_id in self.position:
            self.matrix[self.position[memory_id]] = vector
            return True
        if self.size >= self.matrix.shape[0]:
            grow = max(64, self.matrix.shape[0])
            self.matrix = np.vstack([self.matrix, np.zeros((grow, self.dim), dtype=np.float32)])
            self.ids = np.concatenate([self.ids, np.full(grow, -1, dtype=np.int64)])
        self.matrix[self.size] = vector
        self.ids[self.size] = memory_id
        self.position[memory_id] = self.size
        self.size += 1
        return True

    def remove(self, memory_ids):
        for memory_id in memory_ids:
            row = self.position.pop(memory_id, None)
            if row is not None:
                self.ids[row] = -1
                self.matrix[row] = 0.0
                self.free += 1
        if self.free > 256 and self.free > self.size // 4:
            self._compact()

    def _compact(self):
        keep = np.nonzero(self.ids[: self.size] >= 0)[0]
        self.matrix = np.ascontiguousarray(self.matrix[keep])
        self.ids = self.ids[keep]
        self.size = len(keep)
        self.free = 0
        self.position = {int(memory_id): row for row, memory_id in enumerate(self.ids)}

    def has(self, memory_id):
        return memory_id in self.position

    def vectors(self, memory_ids):
        """Rows for ``memory_ids`` (zero vectors for ids not in this space)."""
        out = np.zeros((len(memory_ids), self.dim), dtype=np.float32)
        for slot, memory_id in enumerate(memory_ids):
            row = self.position.get(memory_id)
            if row is not None:
                out[slot] = self.matrix[row]
        return out

    def similarities(self, query, memory_ids):
        """Cosine (vectors are unit length) for the ids present; returns {id: score}."""
        present = [memory_id for memory_id in memory_ids if memory_id in self.position]
        if not present:
            return {}
        scores = self.matrix[[self.position[memory_id] for memory_id in present]] @ np.asarray(query, dtype=np.float32)
        return dict(zip(present, scores.tolist()))


def _normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# ─────────────────────────────────────────────────────────────────────────
#  Preferences
# ─────────────────────────────────────────────────────────────────────────

_PREF_TRIGGER = re.compile(
    r"\b(always|by default|from now on|going forward|in future|in the future|every time|each time|"
    r"i (?:usually |generally |always )?prefer|my preference|i like (?:my|it|to|all)|i want (?:all|every|my)|"
    r"remember (?:that|to)|never|don't ever|do not ever|default to)\b", re.IGNORECASE)
_PREF_EXCLUDE = re.compile(r"\bnever\s*mind\b", re.IGNORECASE)
_LUFS_RE = re.compile(r"(-?\d{1,2}(?:\.\d)?)\s*lufs\b", re.IGNORECASE)
_AUDIO_FORMAT_RE = re.compile(r"\b(mp3|wav|flac|ogg|aac|m4a|opus)\b", re.IGNORECASE)
_BITRATE_RE = re.compile(r"\b(\d{2,3})\s*(?:k|kbps|kb/s)\b", re.IGNORECASE)
_VIDEO_FORMAT_RE = re.compile(r"\b(mp4|webm|mkv|mov|gif)\b", re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"\b(4k|2160p|1440p|1080p|720p|480p)\b", re.IGNORECASE)
_SCOPE_RE = re.compile(r"\b(podcasts?|music|songs?|voice ?overs?|videos?|youtube|spotify|audiobooks?)\b", re.IGNORECASE)


def parse_preference_slots(sentence):
    """Structured slots from one preference sentence, e.g. {'audio_format': 'mp3'}."""
    slots = {}
    lufs = _LUFS_RE.search(sentence)
    if lufs:
        value = float(lufs.group(1))
        scope_match = _SCOPE_RE.search(sentence)
        scope = _stem(scope_match.group(1).lower().replace(" ", "")) if scope_match else "default"
        slots[f"loudness_lufs:{scope}"] = -abs(value)
    audio_format = _AUDIO_FORMAT_RE.search(sentence)
    if audio_format:
        slots["audio_format"] = audio_format.group(1).lower()
    bitrate = _BITRATE_RE.search(sentence)
    if bitrate and (audio_format or re.search(r"\baudio|export|bitrate", sentence, re.IGNORECASE)):
        slots["audio_bitrate_kbps"] = int(bitrate.group(1))
    video_format = _VIDEO_FORMAT_RE.search(sentence)
    if video_format:
        slots["video_format"] = video_format.group(1).lower()
    resolution = _RESOLUTION_RE.search(sentence)
    if resolution:
        slots["video_resolution"] = resolution.group(1).lower()
    return slots


# ─────────────────────────────────────────────────────────────────────────
#  Repeat / edit detection
# ─────────────────────────────────────────────────────────────────────────

_REPEAT_RE = re.compile(
    r"^(?:ok(?:ay)?\s+|now\s+)?(?:please\s+)?(?:"
    r"(?:do|apply|run)\s+(?:the\s+)?same(?:\s+(?:thing|edit|edits|steps?))?(?:\s+(?:again|as before|as last time))?"
    r"|(?:do|apply|run)\s+(?:that|it|this)(?:\s+(?:thing|edit))?\s+(?:again|as before|one more time)"
    r"|(?:the\s+)?same\s+(?:again|as before|as last time|edit(?:\s+again)?|thing again)"
    r"|repeat(?:\s+(?:that|it|the\s+last(?:\s+(?:edit|step|steps|one))?|the\s+previous(?:\s+(?:edit|steps?))?|last\s+edit))?"
    r")(?:\s+please)?(?:\s+(?:on|to|for)\s+(?:this|it|the\s+new\s+(?:file|clip|one)|this\s+(?:file|clip|one)))?(?:\s+please)?$")

_EDIT_HINT_RE = re.compile(
    r"\b(trim|cut|remove|clean|denoise|noise|normali[sz]e|loud|upscale|enhance|sharpen|restore|fade|convert|"
    r"export|extract|compress|speed|slow|transcribe|caption|subtitle|silence|fix|improve|better|polish|edit)",
    re.IGNORECASE)


def is_repeat_request(message):
    text = re.sub(r"[^\w\s.-]", " ", (message or "").lower())
    text = re.sub(r"\s+", " ", text).strip(" .")
    return bool(text) and bool(_REPEAT_RE.match(text))


def _tool_list(tools):
    """Plan steps reduced to name/args (+ planner role) — enough to re-plan, nothing media-specific."""
    normalized = []
    for tool in tools or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            args = tool.get("args") if isinstance(tool.get("args"), dict) else {}
            step = {"name": tool["name"][:64], "args": json.loads(json.dumps(args, default=str))}
            if isinstance(tool.get("role"), str) and tool["role"] != "requested":
                step["role"] = tool["role"][:32]
            normalized.append(step)
    return normalized


def _plan_signature(tools):
    return json.dumps([{"name": tool["name"], "args": tool["args"]} for tool in tools], sort_keys=True)


def _step_statuses(results):
    statuses = []
    for step in results or []:
        if isinstance(step, dict):
            entry = {"tool": str(step.get("tool") or "")[:64], "status": str(step.get("status") or "")[:16]}
            for key in ("elapsed_ms", "duration_ms"):
                if isinstance(step.get(key), (int, float)):
                    entry["ms"] = round(float(step[key]), 1)
            statuses.append(entry)
    return statuses


def _run_status(tools, statuses):
    if any(item["status"] == "error" for item in statuses):
        return "failed"
    if tools and any(item["status"] == "success" for item in statuses):
        return "success"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────
#  Store
# ─────────────────────────────────────────────────────────────────────────

class MemoryUnavailable(RuntimeError):
    pass


_MIGRATIONS = {
    1: [
        """CREATE TABLE IF NOT EXISTS memories (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               kind TEXT NOT NULL,
               session_id TEXT NOT NULL DEFAULT '',
               role TEXT NOT NULL DEFAULT '',
               text TEXT NOT NULL,
               search_text TEXT NOT NULL,
               slot TEXT,
               data TEXT,
               media_fp TEXT,
               condition TEXT,
               importance REAL NOT NULL DEFAULT 0.5,
               created_at REAL NOT NULL,
               accessed_at REAL NOT NULL,
               expires_at REAL,
               hits INTEGER NOT NULL DEFAULT 0,
               lex BLOB NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id, kind, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_mem_kind_slot ON memories(kind, slot)",
        "CREATE INDEX IF NOT EXISTS idx_mem_media ON memories(media_fp)",
        "CREATE INDEX IF NOT EXISTS idx_mem_expiry ON memories(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_mem_retention ON memories(importance, accessed_at)",
        """CREATE TABLE IF NOT EXISTS vectors (
               memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
               space TEXT NOT NULL,
               vec BLOB NOT NULL,
               PRIMARY KEY (memory_id, space))""",
        "CREATE INDEX IF NOT EXISTS idx_vec_space ON vectors(space)",
    ],
}

_FTS_SQL = [
    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
           search_text, content='memories', content_rowid='id', tokenize='porter unicode61')""",
    """CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
           INSERT INTO memories_fts(rowid, search_text) VALUES (new.id, new.search_text); END""",
    """CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
           INSERT INTO memories_fts(memories_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text); END""",
    """CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF search_text ON memories BEGIN
           INSERT INTO memories_fts(memories_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
           INSERT INTO memories_fts(rowid, search_text) VALUES (new.id, new.search_text); END""",
]


def _fts5_supported():
    try:
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
            return True
        finally:
            probe.close()
    except sqlite3.Error:
        return False


class MemoryStore:
    """Thread-safe local memory store. See module docstring for the design."""

    def __init__(self, path=DEFAULT_DB_PATH, *, dense="auto", embedders=None, governor=None,
                 clock=time.time, max_rows=DEFAULT_MAX_ROWS, full_text_index=True):
        self.path = path
        self.clock = clock
        self.max_rows = max(10, int(max_rows))
        self.dense = dense if dense in ("auto", "off") else "auto"
        self._embedders = embedders
        self._governor = governor
        self._lock = threading.RLock()
        self._conn = None
        self._indexes = {}
        self._rows = 0
        self._evicted_total = 0
        self._latency = {}
        self.enabled = False
        self.fts = False
        self._want_fts = bool(full_text_index)
        self._open()

    # ── lifecycle ──────────────────────────────────────────────────────
    def _connect(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise sqlite3.DatabaseError("write-ahead mode unavailable")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            check = conn.execute("PRAGMA quick_check").fetchone()[0]
            if str(check).lower() != "ok":
                raise sqlite3.DatabaseError("integrity check failed")
            return conn
        except BaseException:
            conn.close()
            raise

    def _quarantine(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            source = self.path + suffix
            if os.path.exists(source):
                try:
                    os.replace(source, f"{self.path}.corrupt-{stamp}{suffix}")
                except OSError:
                    try:
                        os.remove(source)
                    except OSError:
                        pass

    def _open(self):
        with self._lock:
            for attempt in range(2):
                try:
                    self._conn = self._connect()
                    self._migrate()
                    self._rows = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] if self.enabled else 0
                    return
                except sqlite3.DatabaseError as error:
                    if self._conn is not None:
                        with contextlib.suppress(Exception):
                            self._conn.close()
                        self._conn = None
                    self.enabled = False
                    if attempt == 0:
                        _log.warning("%s store was unreadable (%s); starting a fresh store.",
                                     PUBLIC_NAME, type(error).__name__)
                        self._quarantine()
                except OSError as error:
                    _log.warning("%s storage unavailable (%s).", PUBLIC_NAME, type(error).__name__)
                    self.enabled = False
                    return

    def _migrate(self):
        conn = self._conn
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row[0]) if row else 0
        if version > SCHEMA_VERSION:
            _log.warning("%s was written by a newer version; memory is paused.", PUBLIC_NAME)
            self.enabled = False
            return
        for target in range(version + 1, SCHEMA_VERSION + 1):
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _MIGRATIONS[target]:
                    conn.execute(statement)
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(target),))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        has_fts = conn.execute("SELECT 1 FROM sqlite_master WHERE name='memories_fts'").fetchone() is not None
        if self._want_fts and not has_fts and _fts5_supported():
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _FTS_SQL:
                    conn.execute(statement)
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
                conn.execute("COMMIT")
                has_fts = True
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                has_fts = False
        self.fts = bool(has_fts and self._want_fts)
        self.enabled = True

    def close(self):
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None
            self.enabled = False
            self._indexes.clear()

    @contextlib.contextmanager
    def _tx(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _timed(self, name, started):
        elapsed = (time.perf_counter() - started) * 1000.0
        count, total, worst = self._latency.get(name, (0, 0.0, 0.0))
        self._latency[name] = (count + 1, total + elapsed, max(worst, elapsed))

    # ── diagnostics ────────────────────────────────────────────────────
    def schema_version(self):
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            return int(row[0]) if row else 0

    def journal_mode(self):
        with self._lock:
            return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def _force_schema_version(self, version):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(version),))

    def stats(self):
        """Internal diagnostics (counts, latency, index state). Not for user-facing payloads."""
        with self._lock:
            if not self.enabled:
                return {"enabled": False}
            by_kind = {row[0]: row[1] for row in self._conn.execute(
                "SELECT kind, COUNT(*) FROM memories GROUP BY kind")}
            spaces = {row[0]: row[1] for row in self._conn.execute(
                "SELECT space, COUNT(*) FROM vectors GROUP BY space")}
            try:
                size = sum(os.path.getsize(self.path + suffix) for suffix in ("", "-wal")
                           if os.path.exists(self.path + suffix))
            except OSError:
                size = 0
            return {
                "enabled": True,
                "rows": sum(by_kind.values()),
                "by_kind": by_kind,
                "vectors_by_space_internal": spaces,
                "full_text_index": self.fts,
                "evicted_total": self._evicted_total,
                "max_rows": self.max_rows,
                "storage_bytes": size,
                "latency_ms": {name: {"count": count, "avg": round(total / count, 2), "max": round(worst, 2)}
                               for name, (count, total, worst) in self._latency.items()},
            }

    def public_stats(self):
        """Capability-language status safe for API payloads."""
        stats = self.stats()
        if not stats.get("enabled"):
            return {"name": PUBLIC_NAME, "enabled": False, "memories": 0}
        semantic = "Lexical index"
        if self.dense == "auto" and self._resolved_embedders():
            semantic = "Lexical index with semantic ranking when resources allow"
        return {"name": PUBLIC_NAME, "enabled": True, "on_device": True, "memories": stats["rows"],
                "collections": stats["by_kind"], "retrieval": semantic,
                "latency_ms": {name: value["avg"] for name, value in stats["latency_ms"].items()}}

    # ── indexes ────────────────────────────────────────────────────────
    def _index(self, space):
        index = self._indexes.get(space)
        if index is not None:
            return index
        if space == LEXICAL_SPACE:
            index = _VectorIndex(LEX_DIM)
            for row in self._conn.execute("SELECT id, lex FROM memories"):
                index.add(row[0], _from_blob(row[1]))
        else:
            index = None
            for row in self._conn.execute("SELECT memory_id, vec FROM vectors WHERE space=?", (space,)):
                vector = _from_blob(row[1])
                if index is None:
                    index = _VectorIndex(vector.shape[0])
                index.add(row[0], vector)
            if index is None:
                return None
        self._indexes[space] = index
        return index

    def _forget_ids_in_indexes(self, memory_ids):
        for index in self._indexes.values():
            index.remove(memory_ids)

    def _resolved_embedders(self):
        if self.dense == "off":
            return []
        if self._embedders is None:
            self._embedders = _default_embedders()
        return [embedder for embedder in self._embedders if embedder is not None]

    # ── writes ─────────────────────────────────────────────────────────
    def add(self, kind, text, *, session_id="", role="", importance=None, ttl=..., slot=None, data=None,
            media_fp=None, condition=None, dedup=True):
        """Insert a memory (returns its id, or an existing id when it is a near-duplicate)."""
        if kind not in KINDS:
            raise ValueError("unknown memory kind")
        with self._lock:
            if not self.enabled:
                return None
            started = time.perf_counter()
            text = sanitize_text(text, KIND_TEXT_CAPS.get(kind, MAX_TEXT_CHARS))
            if not text:
                return None
            now = self.clock()
            session_id = valid_session_id(session_id) or ""
            vector = lexical_embedding(text)
            if dedup:
                existing = self._find_duplicate(kind, text, vector, session_id, media_fp, slot, role)
                if existing is not None:
                    self._conn.execute("UPDATE memories SET accessed_at=?, hits=hits+1, importance=MAX(importance, ?) "
                                       "WHERE id=?", (now, float(importance or 0.0), existing))
                    self._timed("write", started)
                    return existing
            ttl = DEFAULT_TTL.get(kind) if ttl is ... else ttl
            importance = DEFAULT_IMPORTANCE.get(kind, 0.5) if importance is None else float(importance)
            cursor = self._conn.execute(
                "INSERT INTO memories(kind, session_id, role, text, search_text, slot, data, media_fp, condition, "
                "importance, created_at, accessed_at, expires_at, lex) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (kind, session_id, role or "", text, search_text_for(text), slot,
                 json.dumps(data) if data is not None else None, media_fp,
                 " ".join(condition_signature(condition)) or None, max(0.0, min(1.0, importance)), now, now,
                 (now + ttl) if ttl else None, _to_blob(vector)))
            memory_id = cursor.lastrowid
            if LEXICAL_SPACE in self._indexes:
                self._indexes[LEXICAL_SPACE].add(memory_id, vector)
            self._rows += 1
            if self._rows > self.max_rows:
                self.evict()
            self._timed("write", started)
            return memory_id

    def _find_duplicate(self, kind, text, vector, session_id, media_fp, slot, role):
        if kind == "conversation":
            row = self._conn.execute(
                "SELECT id, text FROM memories WHERE session_id=? AND kind='conversation' AND role=? "
                "ORDER BY id DESC LIMIT 1", (session_id, role or "")).fetchone()
            return row[0] if row and row[1] == text else None
        if kind in ("edit_outcome", "session_summary", "job"):
            return None
        clauses, params = ["kind=?"], [kind]
        if kind == "media_fact":
            clauses.append("media_fp IS ?")
            params.append(media_fp)
        rows = self._conn.execute(f"SELECT id FROM memories WHERE {' AND '.join(clauses)} "
                                  "ORDER BY id DESC LIMIT 500", params).fetchall()
        ids = [row[0] for row in rows]
        if not ids:
            return None
        similarities = self._index(LEXICAL_SPACE).similarities(vector, ids)
        best = max(similarities.items(), key=lambda item: item[1], default=(None, 0.0))
        return best[0] if best[1] >= 0.95 else None

    def _update_text(self, memory_id, text, data=None, **columns):
        text = sanitize_text(text)
        vector = lexical_embedding(text)
        assignments = ["text=?", "search_text=?", "lex=?"]
        params = [text, search_text_for(text), _to_blob(vector)]
        if data is not None:
            assignments.append("data=?")
            params.append(json.dumps(data))
        for column, value in columns.items():
            assignments.append(f"{column}=?")
            params.append(value)
        params.append(memory_id)
        self._conn.execute(f"UPDATE memories SET {', '.join(assignments)} WHERE id=?", params)
        self._conn.execute("DELETE FROM vectors WHERE memory_id=?", (memory_id,))
        for space, index in self._indexes.items():
            if space == LEXICAL_SPACE:
                index.add(memory_id, vector)
            else:
                index.remove([memory_id])

    def get(self, memory_id):
        with self._lock:
            if not self.enabled:
                return None
            row = self._conn.execute("SELECT id, kind, session_id, role, text, slot, data, media_fp, condition, "
                                     "importance, created_at, accessed_at, expires_at FROM memories WHERE id=?",
                                     (memory_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            item["data"] = json.loads(item["data"]) if item["data"] else None
            return item

    def delete(self, memory_id):
        return self._delete_ids([memory_id]) > 0

    def _delete_ids(self, memory_ids):
        memory_ids = [int(memory_id) for memory_id in memory_ids]
        if not memory_ids:
            return 0
        with self._lock:
            if not self.enabled:
                return 0
            with self._tx() as conn:
                removed = conn.execute("DELETE FROM memories WHERE id IN (SELECT value FROM json_each(?))",
                                       (json.dumps(memory_ids),)).rowcount
            self._forget_ids_in_indexes(memory_ids)
            self._rows = max(0, self._rows - removed)
            return removed

    def count(self, kind=None, session_id=None):
        with self._lock:
            if not self.enabled:
                return 0
            clauses, params = [], []
            if kind:
                clauses.append("kind=?")
                params.append(kind)
            if session_id is not None:
                clauses.append("session_id=?")
                params.append(session_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            return self._conn.execute(f"SELECT COUNT(*) FROM memories {where}", params).fetchone()[0]

    def evict(self):
        """Drop expired rows, then enforce the size cap by importance/LRU; compacts freed pages."""
        with self._lock:
            if not self.enabled:
                return 0
            now = self.clock()
            expired = [row[0] for row in self._conn.execute(
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))]
            removed = self._delete_ids(expired)
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            self._rows = total
            if total > self.max_rows:
                excess = total - int(self.max_rows * 0.9)
                victims = [row[0] for row in self._conn.execute(
                    "SELECT id FROM memories WHERE kind != 'preference' "
                    "ORDER BY importance + CASE WHEN accessed_at > ? THEN 0.25 ELSE 0 END ASC, accessed_at ASC "
                    "LIMIT ?", (now - 7 * DAY, excess))]
                removed += self._delete_ids(victims)
            if removed:
                self._evicted_total += removed
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("PRAGMA incremental_vacuum(4096)").fetchall()
            return removed

    def forget_session(self, session_id):
        """Delete every memory recorded in a session (turns, summary, outcomes, its preferences and jobs)."""
        session_id = valid_session_id(session_id)
        if not session_id:
            return 0
        with self._lock:
            if not self.enabled:
                return 0
            ids = [row[0] for row in self._conn.execute("SELECT id FROM memories WHERE session_id=?", (session_id,))]
            removed = self._delete_ids(ids)
            self._compact_after_forget()
            return removed

    def forget_all(self):
        with self._lock:
            if not self.enabled:
                return 0
            with self._tx() as conn:
                removed = conn.execute("DELETE FROM memories").rowcount
                conn.execute("DELETE FROM vectors")
            self._indexes.clear()
            self._rows = 0
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("VACUUM")
            self._compact_after_forget()
            return removed

    def _compact_after_forget(self):
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA incremental_vacuum").fetchall()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ── typed recorders ────────────────────────────────────────────────
    def record_turn(self, session_id, role, text):
        session_id = valid_session_id(session_id)
        if not session_id or role not in ("user", "assistant"):
            return None
        importance = DEFAULT_IMPORTANCE["conversation"] + (0.05 if role == "user" else -0.1)
        limit = KIND_TEXT_CAPS["conversation"] if role == "user" else 600
        return self.add("conversation", sanitize_text(text, limit), session_id=session_id, role=role,
                        importance=importance)

    def extract_preferences(self, session_id, text):
        """Store preference statements found in a user message; returns their memory ids."""
        ids = []
        if not isinstance(text, str):
            return ids
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
            sentence = sentence.strip()
            if (not sentence or sentence.endswith("?") or _PREF_EXCLUDE.search(sentence)
                    or not _PREF_TRIGGER.search(sentence) or len(sentence) < 8):
                continue
            memory_id = self._store_preference(session_id, sentence)
            if memory_id is not None:
                ids.append(memory_id)
        return ids

    def _store_preference(self, session_id, sentence):
        with self._lock:
            if not self.enabled:
                return None
            slots = parse_preference_slots(sentence)
            slot_key = ",".join(sorted(slots)) if slots else None
            if slots:
                for row in self._conn.execute("SELECT id, slot, data FROM memories WHERE kind='preference' "
                                              "AND slot IS NOT NULL").fetchall():
                    existing_slots = set(row[1].split(","))
                    values = (json.loads(row[2]) or {}).get("values", {}) if row[2] else {}
                    if existing_slots == set(slots) and values == slots:
                        self._conn.execute("UPDATE memories SET accessed_at=?, hits=hits+1 WHERE id=?",
                                           (self.clock(), row[0]))
                        return row[0]
                    if existing_slots <= set(slots):
                        self._delete_ids([row[0]])
            return self.add("preference", sentence, session_id=session_id or "", slot=slot_key,
                            data={"values": slots} if slots else None)

    def preferences(self):
        with self._lock:
            if not self.enabled:
                return []
            rows = self._conn.execute("SELECT id, text, data, created_at FROM memories WHERE kind='preference' "
                                      "ORDER BY created_at ASC, id ASC").fetchall()
            return [{"id": row[0], "text": row[1], "values": (json.loads(row[2]) or {}).get("values", {})
                     if row[2] else {}, "created_at": row[3]} for row in rows]

    def preference_values(self):
        merged = {}
        for preference in self.preferences():
            merged.update(preference["values"])
        return merged

    def record_media_facts(self, media_fp, lines=None, condition=None, report=None, context=None):
        """Upsert inspection facts for one media fingerprint (no media bytes are stored)."""
        if not media_fp:
            return None
        lines = list(lines or [])
        if report is not None and not lines:
            try:
                import media_inspector
                lines = media_inspector.describe(report)
            except Exception:
                lines = []
        tags = condition_signature(condition if condition is not None else report)
        context = context or {}
        if isinstance(context.get("duration"), (int, float)) and context.get("duration"):
            lines.append(f"Duration {float(context['duration']):.1f}s.")
        if not lines:
            return None
        text = "Current media: " + " ".join(sanitize_text(line, 200) for line in lines[:6])
        with self._lock:
            if not self.enabled:
                return None
            row = self._conn.execute("SELECT id, text FROM memories WHERE kind='media_fact' AND media_fp=? "
                                     "ORDER BY id DESC LIMIT 1", (media_fp,)).fetchone()
            data = {"condition": tags, "lines": lines[:6]}
            if row is not None:
                if row[1] != sanitize_text(text, KIND_TEXT_CAPS["media_fact"]):
                    self._update_text(row[0], sanitize_text(text, KIND_TEXT_CAPS["media_fact"]), data,
                                      condition=" ".join(tags) or None, accessed_at=self.clock())
                return row[0]
            return self.add("media_fact", text, media_fp=media_fp, condition=tags, data=data, slot="facts",
                            dedup=False)

    def media_facts(self, media_fp):
        with self._lock:
            if not self.enabled or not media_fp:
                return None
            row = self._conn.execute("SELECT text, data FROM memories WHERE kind='media_fact' AND media_fp=? "
                                     "AND (expires_at IS NULL OR expires_at > ?) ORDER BY id DESC LIMIT 1",
                                     (media_fp, self.clock())).fetchone()
            if row is None:
                return None
            return {"text": row[0], **(json.loads(row[1]) if row[1] else {})}

    def record_outcome(self, session_id, request, tools, results, media_fp=None, condition=None, elapsed_ms=None):
        """Session-scoped edit outcome (what ran and whether it worked)."""
        tools = _tool_list(tools)
        if not tools:
            return None
        statuses = _step_statuses(results)
        status = _run_status(tools, statuses)
        by_tool = {item["tool"]: item["status"] for item in statuses}
        steps = ", ".join(f"{tool['name']} {'ok' if by_tool.get(tool['name']) == 'success' else by_tool.get(tool['name']) or 'skipped'}"
                          for tool in tools)
        text = f"Edit '{sanitize_text(request, 160)}': {steps}"
        return self.add("edit_outcome", text, session_id=session_id or "", media_fp=media_fp, condition=condition,
                        importance=0.65 if status == "failed" else 0.55,
                        data={"tools": tools, "steps": statuses, "status": status, "elapsed_ms": elapsed_ms})

    def record_job(self, session_id, request, tools, results, condition=None, elapsed_ms=None, media_fp=None):
        """
        Orchestration memory: merge this run into the job keyed by (plan, media condition).
        Tracks runs/successes/failures, average wall time and the last per-step outcome.
        """
        tools = _tool_list(tools)
        if not tools:
            return None
        tags = condition_signature(condition)
        statuses = _step_statuses(results)
        status = _run_status(tools, statuses)
        key = hashlib.sha1((_plan_signature(tools) + "|" + " ".join(tags)).encode("utf-8")).hexdigest()[:20]
        request = sanitize_text(request, 160)
        with self._lock:
            if not self.enabled:
                return None
            row = self._conn.execute("SELECT id, text, data FROM memories WHERE kind='job' AND slot=? "
                                     "ORDER BY id DESC LIMIT 1", (key,)).fetchone()
            now = self.clock()
            if row is None:
                data = {"tools": tools, "condition": tags, "runs": 0, "successes": 0, "failures": 0,
                        "timed_runs": 0, "total_ms": 0.0}
            else:
                data = json.loads(row[2]) if row[2] else {}
            data["runs"] = data.get("runs", 0) + 1
            data["successes"] = data.get("successes", 0) + (1 if status == "success" else 0)
            data["failures"] = data.get("failures", 0) + (1 if status == "failed" else 0)
            if isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0:
                data["timed_runs"] = data.get("timed_runs", 0) + 1
                data["total_ms"] = data.get("total_ms", 0.0) + float(elapsed_ms)
            data["last_status"] = status
            data["last_steps"] = statuses
            data["failed_tool"] = next((item["tool"] for item in statuses if item["status"] == "error"), None)
            success_rate = data["successes"] / max(1, data["runs"])
            importance = 0.45 + 0.4 * success_rate if data["failures"] <= data["successes"] else 0.6
            if row is None:
                return self.add("job", request, session_id=session_id or "", slot=key, data=data, condition=tags,
                                media_fp=media_fp, importance=importance, dedup=False)
            text = row[1]
            phrasings = [part.strip() for part in text.split(" | ")]
            vector = lexical_embedding(request)
            if request and all(float(np.dot(vector, lexical_embedding(part))) < 0.9 for part in phrasings):
                text = " | ".join(([request] + phrasings)[:4])
            self._update_text(row[0], text, data, accessed_at=now, importance=importance,
                              session_id=valid_session_id(session_id) or "")
            return row[0]

    def recall_similar_jobs(self, request, media_condition=None, k=3):
        """
        Past orchestration jobs similar to ``request`` on media with a similar
        condition signature.  Returns dicts:

            {"request", "condition", "tools" (names), "plan" (tool dicts),
             "runs", "successes", "failures", "success_rate", "avg_ms",
             "last_status", "failed_tool", "recommendation" ('reuse'|'avoid'|'neutral'),
             "score"}

        Planners can reuse ``plan`` when recommendation == 'reuse' and steer
        away from plans marked 'avoid' for similar media.
        """
        started = time.perf_counter()
        tags = condition_signature(media_condition)
        with self._lock:
            if not self.enabled:
                return []
            now = self.clock()
            rows = self._conn.execute(
                "SELECT id, text, data, condition, accessed_at FROM memories WHERE kind='job' "
                "AND (expires_at IS NULL OR expires_at > ?) ORDER BY accessed_at DESC LIMIT ?",
                (now, ELIGIBLE_LIMIT)).fetchall()
            if not rows:
                return []
            relevance = self._relevance(request, [row[0] for row in rows])
            jobs = []
            for row in rows:
                rel = relevance.get(row[0], 0.0)
                data = json.loads(row[2]) if row[2] else {}
                condition = (row[3] or "").split()
                similarity = _jaccard(tags, condition) if tags else 0.0
                runs = max(1, data.get("runs", 1))
                success_rate = data.get("successes", 0) / runs
                score = 0.55 * rel + 0.35 * similarity + 0.10 * success_rate
                if rel < 0.05 or score < 0.15:
                    continue
                if data.get("successes", 0) > data.get("failures", 0):
                    recommendation = "reuse"
                elif data.get("failures", 0) > data.get("successes", 0):
                    recommendation = "avoid"
                else:
                    recommendation = "neutral"
                timed = data.get("timed_runs", 0)
                jobs.append({
                    "id": row[0], "request": row[1].split(" | ")[0], "condition": sorted(condition),
                    "tools": [tool["name"] for tool in data.get("tools", [])], "plan": data.get("tools", []),
                    "runs": data.get("runs", 0), "successes": data.get("successes", 0),
                    "failures": data.get("failures", 0), "success_rate": round(success_rate, 3),
                    "avg_ms": round(data.get("total_ms", 0.0) / timed, 1) if timed else None,
                    "last_status": data.get("last_status"), "failed_tool": data.get("failed_tool"),
                    "recommendation": recommendation, "score": round(score, 4)})
            jobs.sort(key=lambda job: job["score"], reverse=True)
            self._timed("recall_jobs", started)
            return jobs[: max(0, int(k))]

    def repeat_plan(self, session_id, message):
        """
        Raw plan ({thought, tools:[{name,args}], reply}) reusing this session's last
        successful edit when the user explicitly asks to repeat it; None otherwise.
        Only user-requested steps are returned, so the planner re-applies its own
        condition policies (pass it through the planner's validation before use).
        """
        session_id = valid_session_id(session_id)
        if not session_id or not is_repeat_request(message):
            return None
        with self._lock:
            if not self.enabled:
                return None
            for row in self._conn.execute("SELECT data FROM memories WHERE session_id=? AND kind='edit_outcome' "
                                          "ORDER BY created_at DESC, id DESC LIMIT 25", (session_id,)):
                data = json.loads(row[0]) if row[0] else {}
                tools = [{"name": tool["name"], "args": copy.deepcopy(tool.get("args") or {})}
                         for tool in data.get("tools") or [] if not tool.get("role")]
                if data.get("status") == "success" and tools:
                    names = " → ".join(tool["name"].replace("_", " ") for tool in tools)
                    return {"thought": "Repeating the last successful edit from this conversation.",
                            "delegated_subagent": "Orchestrator", "clarification_needed": False,
                            "clarification_options": [], "tools": tools,
                            "reply": f"Repeating your previous edit: {names}.", "suggested_actions": []}
        return None

    def apply_preferences(self, plan, message):
        """
        Planner hook: fill export arguments a *raw* plan left unset from stored
        preferences (never overrides anything the user or planner set).  Call it
        before the planner validates/describes the plan so replies stay in sync.
        """
        if not isinstance(plan, dict) or not isinstance(plan.get("tools"), list):
            return plan
        values = self.preference_values()
        if not values:
            return plan
        mentions_audio = bool(_AUDIO_FORMAT_RE.search(message or ""))
        mentions_video = bool(_VIDEO_FORMAT_RE.search(message or ""))
        updated = copy.deepcopy(plan)
        for tool in updated["tools"]:
            if not isinstance(tool, dict) or not isinstance(tool.get("args", {}), dict):
                continue
            args = tool.setdefault("args", {})
            name = tool.get("name")
            audio_format = values.get("audio_format")
            video_format = values.get("video_format")
            if name == "convert_audio_format" and "target_format" not in args and not mentions_audio \
                    and audio_format in ("mp3", "wav", "flac", "ogg"):
                args["target_format"] = audio_format
            elif name == "extract_audio" and "format" not in args and not mentions_audio \
                    and audio_format in ("mp3", "wav"):
                args["format"] = audio_format
            elif name == "convert_video_format" and "target_format" not in args and not mentions_video \
                    and video_format in ("mp4", "gif", "webm", "mkv"):
                args["target_format"] = video_format
        return updated

    # ── summaries ──────────────────────────────────────────────────────
    def session_summary(self, session_id, summarizer=None):
        """Rolling extractive summary, regenerated only when the session gained new memories."""
        session_id = valid_session_id(session_id)
        if not session_id:
            return ""
        with self._lock:
            if not self.enabled:
                return ""
            latest = self._conn.execute("SELECT MAX(id) FROM memories WHERE session_id=? AND kind IN "
                                        "('conversation','edit_outcome')", (session_id,)).fetchone()[0]
            if latest is None:
                return ""
            row = self._conn.execute("SELECT id, text, data FROM memories WHERE session_id=? AND "
                                     "kind='session_summary' ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
            if row is not None and (json.loads(row[2]) if row[2] else {}).get("through_id") == latest:
                return row[1]
            rows = self._conn.execute("SELECT id, kind, role, text FROM memories WHERE session_id=? AND kind IN "
                                      "('conversation','edit_outcome') ORDER BY id DESC LIMIT 80",
                                      (session_id,)).fetchall()[::-1]
            text = ""
            if summarizer is not None:
                with contextlib.suppress(Exception):
                    text = sanitize_text(summarizer([dict(item) for item in rows]) or "",
                                         KIND_TEXT_CAPS["session_summary"])
            text = text or _extractive_summary(rows)
            if not text:
                return ""
            data = {"through_id": latest}
            if row is None:
                self.add("session_summary", text, session_id=session_id, slot="summary", data=data, dedup=False)
            else:
                self._update_text(row[0], text, data, accessed_at=self.clock())
            return text

    # ── retrieval ──────────────────────────────────────────────────────
    def _eligible(self, session_id, kinds, media_fp, include_preferences, now):
        session_kinds = tuple(kind for kind in (kinds or SESSION_KINDS) if kind in SESSION_KINDS)
        clauses, params = [], []
        if session_id and session_kinds:
            clauses.append(f"(session_id=? AND kind IN ({','.join('?' * len(session_kinds))}))")
            params.extend([session_id, *session_kinds])
        if include_preferences and (kinds is None or "preference" in kinds):
            clauses.append("kind='preference'")
        if media_fp and (kinds is None or "media_fact" in kinds):
            clauses.append("(kind='media_fact' AND media_fp=?)")
            params.append(media_fp)
        if not clauses:
            return []
        sql = (f"SELECT id, kind, role, text, importance, created_at FROM memories WHERE "
               f"(expires_at IS NULL OR expires_at > ?) AND ({' OR '.join(clauses)}) "
               f"ORDER BY created_at DESC LIMIT {ELIGIBLE_LIMIT}")
        return self._conn.execute(sql, [now, *params]).fetchall()

    def _bm25(self, query_tokens, memory_ids):
        """Normalized BM25 in [0,1] for the given ids (full-text index, or built-in fallback)."""
        terms = list(dict.fromkeys(query_tokens + _concept_terms(query_tokens)))
        if not terms or not memory_ids:
            return {}
        scores = {}
        if self.fts:
            expression = " OR ".join('"%s"' % term.replace('"', "") for term in terms if term.strip('"-'))
            try:
                for row in self._conn.execute(
                        "SELECT rowid, bm25(memories_fts) FROM memories_fts WHERE memories_fts MATCH ? "
                        "AND rowid IN (SELECT value FROM json_each(?))", (expression, json.dumps(memory_ids))):
                    scores[row[0]] = -float(row[1])
            except sqlite3.Error:
                scores = self._python_bm25(terms, memory_ids)
        else:
            scores = self._python_bm25(terms, memory_ids)
        top = max(scores.values(), default=0.0)
        return {key: max(0.0, value) / top for key, value in scores.items()} if top > 0 else {}

    def _python_bm25(self, terms, memory_ids, k1=1.2, b=0.75):
        rows = self._conn.execute("SELECT id, search_text FROM memories WHERE id IN (SELECT value FROM json_each(?))",
                                  (json.dumps(memory_ids[:2000]),)).fetchall()
        docs = {}
        for row in rows:
            tokens = tokenize(row[1])
            docs[row[0]] = (tokens, len(tokens))
        if not docs:
            return {}
        average = sum(length for _, length in docs.values()) / len(docs) or 1.0
        query = set(terms)
        frequency = {term: sum(1 for tokens, _ in docs.values() if term in tokens) for term in query}
        scores = {}
        for memory_id, (tokens, length) in docs.items():
            score = 0.0
            for term in query:
                tf = tokens.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
            if score > 0:
                scores[memory_id] = score
        return scores

    def _dense_scores(self, query, memory_ids, prelim):
        """Semantic cosine in one embedding space (lazy re-embed of candidates); {} when unavailable."""
        embedders = self._resolved_embedders()
        if not embedders or not memory_ids:
            return {}
        from model_manager import CapabilityVariant, global_quality_governor, TIER_BALANCED, TIER_LITE
        governor = self._governor or global_quality_governor
        ladder = []
        for embedder in embedders:
            on_device = isinstance(embedder, OnDeviceSemanticEmbedder)
            ladder.append(CapabilityVariant(
                embedder.space, tier=TIER_BALANCED if on_device else TIER_LITE,
                min_ram_gb=1.0 if on_device else 0.0,
                is_available=(lambda e=embedder, d=on_device: e.available() and (not d or _model_slot_free())),
                public_label="Semantic memory", options={"embedder": embedder}))
        ladder.append(CapabilityVariant("lexical", public_label="Lexical memory"))
        ranked = sorted(memory_ids, key=lambda memory_id: prelim.get(memory_id, 0.0), reverse=True)

        def operation(variant):
            embedder = variant.options.get("embedder")
            if embedder is None:
                return None
            index = self._index(embedder.space)
            missing = [memory_id for memory_id in ranked if index is None or not index.has(memory_id)]
            missing = missing[:LAZY_EMBED_CAP]
            texts = {}
            if missing:
                for row in self._conn.execute("SELECT id, text FROM memories WHERE id IN "
                                              "(SELECT value FROM json_each(?))", (json.dumps(missing),)):
                    texts[row[0]] = row[1]
                missing = [memory_id for memory_id in missing if memory_id in texts]
            vectors = np.asarray(embedder.embed([query] + [texts[memory_id] for memory_id in missing]),
                                 dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(missing) + 1 or not np.all(np.isfinite(vectors)):
                raise ValueError("semantic vectors were malformed")
            vectors = _normalize_rows(vectors)
            if index is not None and index.dim != vectors.shape[1]:
                raise ValueError("semantic dimensionality changed")
            if missing:
                with self._tx() as conn:
                    conn.executemany("INSERT OR REPLACE INTO vectors(memory_id, space, vec) VALUES (?,?,?)",
                                     [(memory_id, embedder.space, _to_blob(vector))
                                      for memory_id, vector in zip(missing, vectors[1:])])
                if index is None:
                    index = _VectorIndex(vectors.shape[1])
                    self._indexes[embedder.space] = index
                for memory_id, vector in zip(missing, vectors[1:]):
                    index.add(memory_id, vector)
            return index.similarities(vectors[0], memory_ids)

        try:
            result, _variant = governor.run(CAPABILITY, ladder, operation)
            return result or {}
        except Exception as error:
            _log.info("%s semantic ranking skipped (%s).", PUBLIC_NAME, type(error).__name__)
            return {}

    def _relevance(self, query, memory_ids):
        """Hybrid relevance in [0,1]: lexical BM25 + cosine (semantic when available)."""
        if not memory_ids:
            return {}
        query_tokens = tokenize(query)
        query_vector = lexical_embedding(query)
        lexical = self._index(LEXICAL_SPACE).similarities(query_vector, memory_ids)
        bm25 = self._bm25(query_tokens, memory_ids)
        prelim = {memory_id: 0.5 * max(0.0, lexical.get(memory_id, 0.0)) + 0.5 * bm25.get(memory_id, 0.0)
                  for memory_id in memory_ids}
        dense = self._dense_scores(query, memory_ids, prelim) if self.dense == "auto" else {}
        if not dense:
            return prelim
        relevance = {}
        for memory_id in memory_ids:
            lex_cos = max(0.0, lexical.get(memory_id, 0.0))
            semantic = dense.get(memory_id)
            semantic = lex_cos if semantic is None else min(1.0, max(0.0, (semantic - 0.2) / 0.6))
            relevance[memory_id] = 0.45 * semantic + 0.35 * bm25.get(memory_id, 0.0) + 0.20 * lex_cos
        return relevance

    def search(self, query, session_id=None, k=5, kinds=None, media_fp=None, exclude_ids=(),
               include_preferences=True, min_score=MIN_RELEVANCE, mmr_lambda=0.7):
        """
        Hybrid search with metadata filters: this session's turns/outcomes,
        global preferences and facts for ``media_fp``.  Returns ranked dicts
        {id, kind, role, text, score, relevance}.
        """
        started = time.perf_counter()
        with self._lock:
            if not self.enabled or not (query or "").strip():
                return []
            now = self.clock()
            session_id = valid_session_id(session_id)
            excluded = set(exclude_ids or ())
            rows = [row for row in self._eligible(session_id, kinds, media_fp, include_preferences, now)
                    if row[0] not in excluded]
            if not rows:
                return []
            meta = {row[0]: row for row in rows}
            relevance = self._relevance(query, list(meta))
            scored = []
            for memory_id, rel in relevance.items():
                if rel < min_score:
                    continue
                row = meta[memory_id]
                age_hours = max(0.0, now - row["created_at"]) / 3600.0
                recency = 0.5 ** (age_hours / RECENCY_HALF_LIFE_H)
                score = rel * (0.8 + 0.2 * row["importance"]) + 0.08 * recency
                scored.append((score, rel, memory_id))
            scored.sort(reverse=True)
            pool = scored[: max(k * 4, 12)]
            selected = self._mmr(pool, k, mmr_lambda)
            if selected:
                self._conn.execute("UPDATE memories SET accessed_at=?, hits=hits+1 WHERE id IN "
                                   "(SELECT value FROM json_each(?))",
                                   (now, json.dumps([memory_id for _, _, memory_id in selected])))
            self._timed("search", started)
            return [{"id": memory_id, "kind": meta[memory_id]["kind"], "role": meta[memory_id]["role"],
                     "text": meta[memory_id]["text"], "score": round(score, 4), "relevance": round(rel, 4)}
                    for score, rel, memory_id in selected]

    def _mmr(self, pool, k, mmr_lambda):
        if not pool:
            return []
        index = self._index(LEXICAL_SPACE)
        ids = [memory_id for _, _, memory_id in pool]
        vectors = {memory_id: vector for memory_id, vector in zip(ids, index.vectors(ids))}
        selected, remaining = [], list(pool)
        while remaining and len(selected) < k:
            best, best_value = None, -1e9
            for candidate in remaining:
                redundancy = max((float(vectors[candidate[2]] @ vectors[chosen[2]]) for chosen in selected),
                                 default=0.0)
                if redundancy >= DUPLICATE_COSINE:
                    continue
                value = mmr_lambda * candidate[0] - (1 - mmr_lambda) * redundancy
                if value > best_value:
                    best, best_value = candidate, value
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
        return selected

    def _recent_turns(self, session_id, limit=2):
        rows = self._conn.execute("SELECT id, role, text FROM memories WHERE session_id=? AND kind='conversation' "
                                  "ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        return [(row[0], row[1], row[2]) for row in rows[::-1]]

    def build_context(self, session_id, query, token_budget=600, recent_history=None, media_context=None,
                      media_fingerprint=None, media_condition=None, recent_turns=None):
        """
        Compact prompt block within ``token_budget`` (conservative estimate):
        latest turns verbatim, preferences, media facts, session summary,
        similar past jobs, then related earlier memories (MMR).
        """
        started = time.perf_counter()
        budget = int(token_budget or 0)
        if budget <= 0:
            return ""
        with self._lock:
            if not self.enabled:
                return ""
            session_id = valid_session_id(session_id)
            turns_wanted = recent_turns or (2 if session_id else 6)
            recent, recent_ids = [], []
            if recent_history:
                for item in list(recent_history)[-turns_wanted:]:
                    if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip():
                        role = "assistant" if item.get("role") == "assistant" else "user"
                        recent.append((role, sanitize_text(item["content"], 2000)))
            elif session_id:
                for memory_id, role, text in self._recent_turns(session_id, turns_wanted):
                    recent.append((role, text))
                    recent_ids.append(memory_id)
            recent_texts = {text for _, text in recent}

            preferences = [item for item in self.search(query or "preferences", None, k=4, kinds=("preference",),
                                                        min_score=0.0, mmr_lambda=0.85)]
            if len(preferences) < 4:
                chosen = {item["id"] for item in preferences}
                for item in reversed(self.preferences()):
                    if item["id"] not in chosen and len(preferences) < 4:
                        preferences.append({"id": item["id"], "text": item["text"]})
            facts = self.media_facts(media_fingerprint) if media_fingerprint else None
            condition = condition_signature(media_condition) or (facts or {}).get("condition") or []
            media_type = (media_context or {}).get("type") if isinstance(media_context, dict) else None
            if not condition and media_type in ("audio", "video", "image"):
                condition = ["type:" + media_type]
            summary = self.session_summary(session_id) if session_id else ""
            jobs = self.recall_similar_jobs(query, condition, k=2) if _EDIT_HINT_RE.search(query or "") else []
            related = []
            if session_id:
                related = [item for item in self.search(query, session_id, k=4, kinds=SESSION_KINDS,
                                                        exclude_ids=recent_ids, include_preferences=False)
                           if item["text"] not in recent_texts and item["text"] != summary]

            packer = _Packer(budget)
            for role, text in reversed(recent):
                packer.add("recent", f"{role.upper()}: {text}", front=True, shrink=True)
            for item in preferences:
                packer.add("preferences", f"- {item['text']}")
            if facts:
                packer.add("media", facts["text"], shrink=True)
            if summary:
                packer.add("summary", summary, shrink=True)
            for job in jobs:
                outcome = (f"worked {job['successes']}/{job['runs']}" if job["recommendation"] != "avoid"
                           else f"failed {job['failures']}/{job['runs']} — avoid")
                tags = f" [{' '.join(job['condition'])}]" if job["condition"] else ""
                packer.add("jobs", f"- \"{job['request']}\"{tags} -> {', '.join(job['tools'])}: {outcome}")
            for item in related:
                label = "edit" if item["kind"] == "edit_outcome" else item["role"] or "note"
                packer.add("related", f"- ({label}) {sanitize_text(item['text'], 240)}", shrink=True)
            text = packer.render()
            self._timed("build_context", started)
            return text


def _model_slot_free():
    """True when no other capability currently holds the single model slot."""
    try:
        from model_manager import global_model_manager
        lock = global_model_manager._lock
        if not lock.acquire(blocking=False):
            return False
        try:
            return not global_model_manager._session_depth
        finally:
            lock.release()
    except Exception:
        return False


_SECTION_ORDER = (("preferences", "Preferences:"), ("media", "Media facts:"), ("summary", "Session summary:"),
                  ("jobs", "Similar past jobs:"), ("related", "Related earlier:"), ("recent", "Recent turns:"))
_INLINE_SECTIONS = {"media", "summary"}


class _Packer:
    """Greedy budget-aware assembly; render order is fixed, admission order is priority."""

    HEADER = "[Memory]"

    def __init__(self, budget):
        self.budget = budget
        self.sections = {name: [] for name, _ in _SECTION_ORDER}

    def render(self, sections=None):
        sections = sections or self.sections
        if not any(sections.values()):
            return ""
        lines = [self.HEADER]
        for name, label in _SECTION_ORDER:
            items = sections[name]
            if not items:
                continue
            if name in _INLINE_SECTIONS:
                lines.append(f"{label} {' '.join(items)}")
            else:
                lines.append(label)
                lines.extend(items)
        return "\n".join(lines)

    def _fits(self, name, line, front):
        trial = {key: list(value) for key, value in self.sections.items()}
        trial[name].insert(0, line) if front else trial[name].append(line)
        return estimate_tokens(self.render(trial)) <= self.budget

    def add(self, name, line, front=False, shrink=False):
        line = line.strip()
        if not line:
            return False
        if self._fits(name, line, front):
            self.sections[name].insert(0, line) if front else self.sections[name].append(line)
            return True
        if not shrink:
            return False
        low, high, best = 0, len(line) - 1, None
        while low <= high:
            middle = (low + high) // 2
            candidate = line[:middle].rstrip() + "…"
            if middle >= 24 and self._fits(name, candidate, front):
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        if best is None:
            return False
        self.sections[name].insert(0, best) if front else self.sections[name].append(best)
        return True


_SUMMARY_SKIP = re.compile(r"^(?:hi|hello|hey|thanks|thank you|ok|okay|great|cool|sure|yes|no)\b[\s!.]*$",
                           re.IGNORECASE)


def _extractive_summary(rows, limit=420):
    """Heuristic extractive summary: salient user requests + recent edit outcomes."""
    sentences = []
    outcomes = []
    for row in rows:
        kind, role, text = row["kind"], row["role"], row["text"]
        if kind == "edit_outcome":
            outcomes.append(text.split(": ", 1)[-1])
            continue
        if role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence.split()) >= 3 and not _SUMMARY_SKIP.match(sentence):
                sentences.append(sentence)
    picked_text = ""
    if sentences:
        vectors = np.vstack([lexical_embedding(sentence) for sentence in sentences])
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid = centroid / norm if norm else centroid
        scored = []
        for position, sentence in enumerate(sentences):
            score = 1.0 + 0.4 * float(vectors[position] @ centroid) + 0.3 * position / max(1, len(sentences) - 1)
            if _EDIT_HINT_RE.search(sentence):
                score += 0.6
            if re.search(r"\d", sentence):
                score += 0.2
            if _PREF_TRIGGER.search(sentence):
                score += 0.4
            scored.append((score, position))
        chosen, used = [], 0
        for score, position in sorted(scored, reverse=True):
            sentence = sanitize_text(sentences[position], 160)
            if any(float(vectors[position] @ vectors[other]) > 0.8 for other in chosen):
                continue
            if used + len(sentence) > limit * 0.7:
                continue
            chosen.append(position)
            used += len(sentence) + 3
        picked_text = " / ".join(sanitize_text(sentences[position], 160) for position in sorted(chosen))
    parts = []
    if picked_text:
        parts.append(f"Requests: {picked_text}")
    if outcomes:
        parts.append("Edits: " + "; ".join(dict.fromkeys(outcomes[-4:])))
    return sanitize_text(". ".join(parts), limit + 120)


# ─────────────────────────────────────────────────────────────────────────
#  Process-wide default store + failure-isolated helpers for the web app
# ─────────────────────────────────────────────────────────────────────────

_default_store = None
_default_kwargs = {}
_default_path = None
_configured = False
_default_lock = threading.Lock()
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="copilot-memory")
_request_media = contextvars.ContextVar("copilot_memory_media", default=None)


def memory_disabled():
    return str(os.environ.get("COPILOT_MEMORY", "on")).strip().lower() in ("0", "off", "false", "disabled")


def configure(path=None, **kwargs):
    """Point the default store at ``path`` (None resets to the standard location, opened lazily)."""
    global _default_store, _default_kwargs, _default_path, _configured
    with _default_lock:
        if _default_store is not None:
            _default_store.close()
        _default_store = None
        _default_path = path
        _default_kwargs = dict(kwargs)
        _configured = path is not None
        if path is not None:
            _default_store = MemoryStore(path, **kwargs)
        return _default_store


def is_configured():
    return _configured


def _running_test_script():
    """True when the process entry point is a test script (they must never touch the user's memory)."""
    import sys
    entry = os.path.basename(str(getattr(sys.modules.get("__main__"), "__file__", "") or (sys.argv or [""])[0]))
    return bool(re.match(r"^(?:test_.*|.*_test)\.py$", entry))


def get_store():
    """The process-wide store (lazily opened), or None when memory is turned off."""
    global _default_store
    if memory_disabled():
        return None
    with _default_lock:
        if _default_store is None:
            path = _default_path or os.environ.get("COPILOT_MEMORY_PATH")
            kwargs = dict(_default_kwargs)
            if not path and _running_test_script():
                import tempfile
                path = os.path.join(tempfile.mkdtemp(prefix="copilot-memory-"), "memory.db")
                kwargs.setdefault("dense", "off")
            path = path or DEFAULT_DB_PATH
            if "dense" not in kwargs and os.environ.get("COPILOT_MEMORY_SEMANTIC", "auto").lower() in ("0", "off"):
                kwargs["dense"] = "off"
            _default_store = MemoryStore(path, **kwargs)
        return _default_store if _default_store.enabled else None


@contextlib.contextmanager
def request_scope(media_path=None):
    """Bind the active media file for this request so prompt packing can use its stored facts."""
    token = _request_media.set(media_path)
    try:
        yield
    finally:
        _request_media.reset(token)


def prompt_budget(system_prompt, user_content, max_tokens=1024, context_tokens=None, floor=96, ceiling=900):
    """Tokens left for memory in the reasoning context window (env LOCAL_REASONING_CONTEXT_TOKENS)."""
    if context_tokens is None:
        try:
            context_tokens = int(os.environ.get("LOCAL_REASONING_CONTEXT_TOKENS", "4096"))
        except ValueError:
            context_tokens = 4096
    available = context_tokens - estimate_tokens(system_prompt) - estimate_tokens(user_content) - max_tokens - 64
    return max(floor, min(ceiling, available))


def build_context(session_id, query, media_context=None, token_budget=600, recent_history=None,
                  media_condition=None):
    """
    Module-level entry for the orchestrator.  Returns the packed block, or
    None when memory is unavailable (callers then use their own fallback).
    """
    store = get_store()
    if store is None:
        return None
    media_path = _request_media.get()
    if not media_path and isinstance(media_context, dict) and isinstance(media_context.get("_path"), str):
        media_path = media_context["_path"]
    fingerprint = media_fingerprint(media_path) if media_path else None
    if media_condition is None and media_path:
        try:
            import media_inspector  # cached: the planner has usually perceived this file already
            media_condition = condition_signature(media_inspector.perceive(media_path, (media_context or {}).get("type")))
        except Exception:
            media_condition = None
    return store.build_context(session_id, query, token_budget=token_budget, recent_history=recent_history,
                               media_context=media_context, media_fingerprint=fingerprint,
                               media_condition=media_condition)


def recall_similar_jobs(request, media_condition=None, k=3):
    """Failure-isolated module-level wrapper of ``MemoryStore.recall_similar_jobs`` (returns [] on error)."""
    try:
        store = get_store()
        return store.recall_similar_jobs(request, media_condition, k) if store else []
    except Exception as error:
        _log.warning("%s recall unavailable (%s).", PUBLIC_NAME, type(error).__name__)
        return []


def safe_repeat_plan(session_id, message):
    try:
        store = get_store()
        return store.repeat_plan(session_id, message) if store else None
    except Exception as error:
        _log.warning("%s unavailable (%s).", PUBLIC_NAME, type(error).__name__)
        return None


def safe_apply_preferences(plan, message):
    try:
        store = get_store()
        return store.apply_preferences(plan, message) if store else plan
    except Exception as error:
        _log.warning("%s unavailable (%s).", PUBLIC_NAME, type(error).__name__)
        return plan


def observe_chat(session_id, message, reply=None, plan=None, results=None, media_path=None,
                 media_context=None, elapsed_ms=None, inspect_media=True):
    """Record one chat exchange: turns, media facts, edit outcome and orchestration job."""
    store = get_store()
    if store is None:
        return
    session_id = valid_session_id(session_id)
    tools = (plan or {}).get("tools") if isinstance(plan, dict) else None
    fingerprint = media_fingerprint(media_path) if media_path else None
    condition = []
    if fingerprint:
        report = None
        if inspect_media and results:
            try:
                import media_inspector
                report = media_inspector.perceive(media_path, (media_context or {}).get("type"))
            except Exception:
                report = None
        if report is not None or (media_context or {}).get("duration"):
            store.record_media_facts(fingerprint, report=report, context=media_context)
        facts = store.media_facts(fingerprint)
        condition = condition_signature(report) or (facts or {}).get("condition") or []
        if not condition and (media_context or {}).get("type") in ("audio", "video", "image"):
            condition = ["type:" + media_context["type"]]
    if session_id:
        store.record_turn(session_id, "user", message)
        if reply:
            store.record_turn(session_id, "assistant", reply)
    if tools and results:
        if session_id:
            store.record_outcome(session_id, message, tools, results, media_fp=fingerprint, condition=condition,
                                 elapsed_ms=elapsed_ms)
        store.record_job(session_id, message, tools, results, condition=condition, elapsed_ms=elapsed_ms,
                         media_fp=fingerprint)


def _observe_safely(kwargs):
    try:
        observe_chat(**kwargs)
    except Exception as error:
        _log.warning("%s could not record this exchange (%s).", PUBLIC_NAME, type(error).__name__)


def _remember_preferences_safely(session_id, message):
    try:
        store = get_store()
        if store is not None:
            store.extract_preferences(valid_session_id(session_id) or "", message)
    except Exception as error:
        _log.warning("%s could not record a preference (%s).", PUBLIC_NAME, type(error).__name__)


def remember_preferences_async(session_id, message):
    """Capture stated preferences ("always export podcasts at -16 LUFS") even if the chat then fails."""
    try:
        _writer.submit(_remember_preferences_safely, session_id, message)
    except Exception as error:
        _log.warning("%s could not queue a preference (%s).", PUBLIC_NAME, type(error).__name__)


def observe_chat_async(**kwargs):
    """Queue ``observe_chat`` on the single background writer; never raises."""
    try:
        _writer.submit(_observe_safely, kwargs)
    except Exception as error:
        _log.warning("%s could not queue this exchange (%s).", PUBLIC_NAME, type(error).__name__)


def flush(timeout=10.0):
    """Wait for queued memory writes (tests / shutdown)."""
    try:
        _writer.submit(lambda: None).result(timeout=timeout)
    except Exception:
        pass


def forget(session_id=None, everything=False):
    store = get_store()
    if store is None:
        return 0
    flush()
    return store.forget_all() if everything else store.forget_session(session_id)


def public_status():
    try:
        store = get_store()
        return store.public_stats() if store else {"name": PUBLIC_NAME, "enabled": False, "memories": 0}
    except Exception:
        return {"name": PUBLIC_NAME, "enabled": False, "memories": 0}
