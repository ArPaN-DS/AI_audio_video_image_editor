"""
Voiceover — 100% local text-to-speech.

Public surface (capability language only — never engine/library/vendor names):
  * list_voices()                    -> [{id, label, language, gender, quality, default}]
  * synthesize(text, out_path, ...)  -> report dict (duration, tier label, voice label, ...)
  * voiceover_tool(args, processed_dir) -> {output_file, message, data}  (Copilot hook)

Engine tiers (run through ``global_quality_governor`` best first, baseline last):
  * "Studio voice"   — Kokoro-82M v1.0, full-precision ONNX graph (~330 MB; needs ~1.5 GB
                       free RAM; ~0.3x real time on a 16-core laptop CPU)
  * "Natural voice"  — Kokoro-82M v1.0, 8-bit quantized graph (~90 MB; low-memory option;
                       on CPU it is ~3x slower than full precision, so it ranks second)
  * "Standard voice" — the operating system's built-in voices; zero download:
        Windows: System.Speech (SAPI5) driven through the in-box Windows PowerShell
        macOS:   the ``say`` command
        Linux:   an ``espeak-ng``/``espeak`` binary if the user installed one
    The baseline is invoked as a separate OS process only; nothing GPL is linked.

Licence decision (this project is MIT):
  * Kokoro-82M weights + voice styles: Apache-2.0 (hexgrad/Kokoro-82M; ONNX export
    from onnx-community/Kokoro-82M-v1.0-ONNX, also Apache-2.0).
  * Runtime: ONNX Runtime (MIT) — already installed as a dependency of rembg.
  * Pronunciation: the upstream Kokoro pipeline uses misaki, whose English
    out-of-vocabulary fallback is espeak-ng via phonemizer (both GPL-3.0). We do NOT
    use that stack. Instead ``LexiconPronouncer`` reads misaki's Apache-2.0
    pronunciation dictionaries (us/gb gold + silver JSON; ~180k entries) directly,
    adds simple suffix/compound morphology and a small rule-based fallback for
    unknown words. No GPL code or data is installed or linked.
  * Piper was considered and rejected: its phonemizer links espeak-ng (GPL) and the
    maintained fork is itself GPL-3.0; many voices carry mixed licences.
  * The pack is never downloaded implicitly: ``python download_models.py --voice``
    (studio graph) and/or ``--voice-lite`` (compact graph) fetches pinned revisions into
    ``models/voice/``. Without it, only the "Standard voice" tier is offered.
  * No voice cloning (consent and abuse risk, see FEATURE_ROADMAP A15).

Memory: the neural runtime and lexicons are loaded per request inside
``global_model_manager.session`` (Single-Active-Model, zero idle memory) and
released right after synthesis. All synthesis is serialised by a module lock.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass

import numpy as np
import soundfile as sf

_log = logging.getLogger("tts")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "models", "voice")

CAPABILITY = "audio.tts"
VARIANT_STUDIO = "voice-studio"
VARIANT_NATURAL = "voice-natural"
VARIANT_SYSTEM = "voice-system"
NEURAL_VARIANTS = (VARIANT_STUDIO, VARIANT_NATURAL)

TIER_STUDIO_LABEL = "Studio voice"
TIER_NATURAL_LABEL = "Natural voice"
TIER_STANDARD_LABEL = "Standard voice"
_VARIANT_LABELS = {
    VARIANT_STUDIO: TIER_STUDIO_LABEL,
    VARIANT_NATURAL: TIER_NATURAL_LABEL,
    VARIANT_SYSTEM: TIER_STANDARD_LABEL,
}

MAX_TEXT_CHARS = 5000
CHUNK_CHARS = 280
SPEED_MIN, SPEED_MAX = 0.5, 2.0
PITCH_MIN, PITCH_MAX = -6.0, 6.0
OUTPUT_FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg"}
QUALITY_CHOICES = ("auto", "studio", "natural", "standard")
VOICE_TARGET_LUFS = -16.0
VOICE_TRUE_PEAK_DBTP = -1.0

PAUSE_WORDS = 0.06
PAUSE_CLAUSE = 0.18
PAUSE_SENTENCE = 0.32
PAUSE_LINE = 0.45
PAUSE_PARAGRAPH = 0.70

_SYNTH_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()


class VoiceoverUnavailableError(ValueError):
    """Synthesis could not run on this computer (message is app-authored)."""


# ═══════════════════════════════════════════════════════════════════════════
#  Voice pack (opt-in download) — pinned upstream revisions
# ═══════════════════════════════════════════════════════════════════════════

_WEIGHTS_URL = ("https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/"
                "1939ad2a8e416c0acfeecc08a694d14ef25f2231/")
_LEXICON_URL = ("https://raw.githubusercontent.com/hexgrad/misaki/"
                "fba1236595f2d2bf21d414ba6e57d25256afada3/misaki/data/")

_MODEL_FILES = {
    VARIANT_NATURAL: ("speech_natural.onnx", "onnx/model_quantized.onnx"),
    VARIANT_STUDIO: ("speech_studio.onnx", "onnx/model.onnx"),
}
_VOCAB_FILE = "vocab.json"


@dataclass(frozen=True)
class _NaturalVoice:
    id: str
    label: str
    language: str
    gender: str
    style_key: str     # upstream style file (internal only)
    dialect: str       # 'us' or 'gb' lexicon


_NATURAL_VOICES = (
    _NaturalVoice("natural-us-warm", "Warm, female (US English)", "English (United States)", "female", "af_heart", "us"),
    _NaturalVoice("natural-us-bright", "Bright, female (US English)", "English (United States)", "female", "af_bella", "us"),
    _NaturalVoice("natural-us-calm", "Calm, male (US English)", "English (United States)", "male", "am_michael", "us"),
    _NaturalVoice("natural-us-deep", "Deep, male (US English)", "English (United States)", "male", "am_fenrir", "us"),
    _NaturalVoice("natural-uk-clear", "Clear, female (UK English)", "English (United Kingdom)", "female", "bf_emma", "gb"),
    _NaturalVoice("natural-uk-steady", "Steady, male (UK English)", "English (United Kingdom)", "male", "bm_george", "gb"),
)
_NATURAL_BY_ID = {v.id: v for v in _NATURAL_VOICES}


def _style_path(voice):
    return os.path.join(VOICE_DIR, "styles", f"{voice.id}.bin")


def _lexicon_paths(dialect):
    return [os.path.join(VOICE_DIR, "lexicon", f"{dialect}_{kind}.json") for kind in ("gold", "silver")]


def voice_pack_manifest(studio=True, compact=False):
    """[(relative path under models/voice, pinned URL)] for the opt-in download.

    studio: full-precision graph (~330 MB) — best quality and, on CPUs, also the
            fastest (~0.3x real time on 8+ cores).
    compact: 8-bit graph (~90 MB) — lower memory, slower and slightly rougher.
    """
    variants = ([VARIANT_STUDIO] if studio else []) + ([VARIANT_NATURAL] if compact else [])
    items = [(_MODEL_FILES[v][0], _WEIGHTS_URL + _MODEL_FILES[v][1]) for v in variants]
    items.append((_VOCAB_FILE, _WEIGHTS_URL + "tokenizer.json"))
    for voice in _NATURAL_VOICES:
        items.append((f"styles/{voice.id}.bin", _WEIGHTS_URL + f"voices/{voice.style_key}.bin"))
    for dialect in ("us", "gb"):
        for kind in ("gold", "silver"):
            items.append((f"lexicon/{dialect}_{kind}.json", _LEXICON_URL + f"{dialect}_{kind}.json"))
    return items


def write_voice_pack_notice():
    """Record third-party licence attributions next to the downloaded pack."""
    os.makedirs(VOICE_DIR, exist_ok=True)
    text = (
        "Local voiceover pack - third-party notices\n\n"
        "Speech weights and voice styles: Kokoro-82M v1.0 by hexgrad, ONNX export by\n"
        "onnx-community. Licence: Apache License 2.0.\n"
        "  https://huggingface.co/hexgrad/Kokoro-82M\n"
        "  https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX\n\n"
        "Pronunciation lexicons (lexicon/*.json): misaki by hexgrad.\n"
        "Licence: Apache License 2.0. https://github.com/hexgrad/misaki\n\n"
        "No GPL components are included in this pack.\n"
    )
    with open(os.path.join(VOICE_DIR, "NOTICE.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)


def _runtime_present():
    import importlib.util
    return importlib.util.find_spec("onnxruntime") is not None


def _file_ok(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _neural_variant_installed(variant_id):
    """True only when the runtime and every file this variant needs are local."""
    if variant_id not in _MODEL_FILES or not _runtime_present():
        return False
    if not (_file_ok(os.path.join(VOICE_DIR, _MODEL_FILES[variant_id][0]))
            and _file_ok(os.path.join(VOICE_DIR, _VOCAB_FILE))):
        return False
    return bool(_installed_natural_voices())


def _installed_natural_voices():
    return [v for v in _NATURAL_VOICES
            if _file_ok(_style_path(v)) and all(_file_ok(p) for p in _lexicon_paths(v.dialect))]


def neural_installed():
    return any(_neural_variant_installed(v) for v in NEURAL_VARIANTS)


# ═══════════════════════════════════════════════════════════════════════════
#  Text normalisation (cheap, English-oriented)
# ═══════════════════════════════════════════════════════════════════════════

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"))
_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                      "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def number_to_words(n):
    n = int(n)
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    if n >= 10 ** 15:
        return " ".join(_ONES[int(d)] for d in str(n))
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return f"{number_to_words(head)} {name}" + (f" {number_to_words(rest)}" if rest else "")
    return str(n)


def ordinal_to_words(n):
    words = number_to_words(n)
    head, sep, last = words.rpartition(" ")
    prefix, hyphen, tail = last.rpartition("-")
    if tail in _ORDINAL_IRREGULAR:
        tail = _ORDINAL_IRREGULAR[tail]
    elif tail.endswith("y"):
        tail = tail[:-1] + "ieth"
    else:
        tail += "th"
    return f"{head}{sep}{prefix}{hyphen}{tail}"


def year_to_words(n):
    if 2000 <= n <= 2009:
        return "two thousand" + (f" {_ONES[n % 10]}" if n % 10 else "")
    high, low = divmod(n, 100)
    if low == 0:
        return f"{number_to_words(high)} hundred"
    if low < 10:
        return f"{number_to_words(high)} oh {_ONES[low]}"
    return f"{number_to_words(high)} {number_to_words(low)}"


def _digits_to_words(digits):
    return " ".join(_ONES[int(d)] for d in digits)


_ABBREVIATIONS = (
    (r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"), (r"\bMs\.", "Miz"), (r"\bDr\.", "Doctor"),
    (r"\bProf\.", "Professor"), (r"\bJr\.", "Junior"), (r"\bSr\.", "Senior"), (r"\bSt\.(?=\s+[A-Z])", "Saint"),
    (r"\bvs\.?(?=\s)", "versus"), (r"\be\.g\.(?=[\s,])", "for example"), (r"\bi\.e\.(?=[\s,])", "that is"),
    (r"\bapprox\.", "approximately"), (r"\bNo\.(?=\s*\d)", "number"), (r"\bdept\.", "department"),
    (r"\betc\.(?=\s+[A-Z]|\s*$)", "et cetera."), (r"\betc\.", "et cetera"),
)
_ABBREVIATIONS = tuple((re.compile(p), r) for p, r in _ABBREVIATIONS)
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}
_CURRENCY_RE = re.compile(
    r"([$£€])\s?(\d[\d,]*)(?:\.(\d{1,2}))?(?!\d)(?:\s?(thousand|million|billion|trillion)\b)?",
    re.IGNORECASE)


def _currency_sub(match):
    singular, plural, sub_one, sub_many = _CURRENCY[match.group(1)]
    whole = int(match.group(2).replace(",", ""))
    fraction, scale = match.group(3), match.group(4)
    if scale:
        amount = number_to_words(whole)
        if fraction:
            amount += " point " + _digits_to_words(fraction)
        return f"{amount} {scale.lower()} {plural}"
    parts = []
    if whole or not fraction:
        parts.append(f"{number_to_words(whole)} {singular if whole == 1 else plural}")
    if fraction:
        cents = int(fraction.ljust(2, "0"))
        if cents:
            parts.append(f"{number_to_words(cents)} {sub_one if cents == 1 else sub_many}")
    return " and ".join(parts) if parts else f"zero {plural}"


def _time_sub(match):
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
    spoken = number_to_words(hour)
    if minute == 0:
        spoken += "" if meridiem else " o'clock"
    elif minute < 10:
        spoken += f" oh {_ONES[minute]}"
    else:
        spoken += f" {number_to_words(minute)}"
    if meridiem:
        spoken += " " + " ".join(ch.upper() for ch in meridiem if ch.isalpha())
    return spoken


def _plain_number_sub(match):
    raw = match.group(0)
    if len(raw) > 1 and raw.startswith("0"):
        return _digits_to_words(raw)
    value = int(raw)
    if len(raw) == 4 and 1100 <= value <= 2099:
        return year_to_words(value)
    return number_to_words(value)


def normalize_text(text):
    """Expand numbers, currency, percentages, times and common abbreviations."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\.{3,}", "\u2026", text)
    text = re.sub(r"(?<=\d)\s?\u2013\s?(?=\d)", " to ", text)
    text = re.sub(r"\s?[\u2013\u2014]\s?", " \u2014 ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "a link", text)
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = _CURRENCY_RE.sub(_currency_sub, text)
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", r"\1 percent", text)
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(\s?[AaPp]\.?[Mm]\.?)?",
                  lambda m: _time_sub(m), text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal_to_words(int(m.group(1))), text,
                  flags=re.IGNORECASE)
    text = re.sub(r"(?<![\w.])(\d+)\.(\d+)(?![\d.]*\d)",
                  lambda m: f"{number_to_words(int(m.group(1)))} point {_digits_to_words(m.group(2))}", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: number_to_words(int(m.group(0).replace(",", ""))), text)
    text = re.sub(r"(?<![\w-])-(?=\d)", "minus ", text)
    text = re.sub(r"\d+", _plain_number_sub, text)
    text = text.replace("&", " and ").replace("+", " plus ").replace("=", " equals ")
    text = re.sub(r"(?<=\w)@(?=\w)", " at ", text)
    text = re.sub(r"(?<=\w)/(?=\w)", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Sentence-aware chunking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    text: str
    pause_after: float


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u2026][\"'\u201d)\]])\s+|(?<=[.!?\u2026])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:\u2014])\s+")


def _pack(pieces, max_chars):
    packed, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            packed.append(current)
        current = piece
    if current:
        packed.append(current)
    return packed


def _split_words(text, max_chars):
    words = []
    for word in text.split():
        while len(word) > max_chars:
            words.append(word[:max_chars])
            word = word[max_chars:]
        if word:
            words.append(word)
    return _pack(words, max_chars)


def _split_long_sentence(sentence, max_chars):
    """[(text, pause_after_within_sentence)] — the caller sets the final pause."""
    if len(sentence) <= max_chars:
        return [(sentence, PAUSE_SENTENCE)]
    out = []
    for clause in _pack(_CLAUSE_SPLIT.split(sentence), max_chars):
        if len(clause) <= max_chars:
            out.append((clause, PAUSE_CLAUSE))
            continue
        pieces = _split_words(clause, max_chars)
        out.extend((piece, PAUSE_WORDS) for piece in pieces[:-1])
        out.append((pieces[-1], PAUSE_CLAUSE))
    return out


def split_into_chunks(text, max_chars=CHUNK_CHARS):
    """Split into sentence-sized chunks (long sentences at clauses, then words)."""
    max_chars = max(20, int(max_chars))
    chunks = []
    for paragraph in re.split(r"\n\s*\n", str(text).strip()):
        lines = [re.sub(r"\s+", " ", line).strip() for line in paragraph.split("\n")]
        lines = [line for line in lines if line]
        for line in lines:
            for sentence in _SENTENCE_SPLIT.split(line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                pieces = _split_long_sentence(sentence, max_chars)
                chunks.extend(Chunk(piece, pause) for piece, pause in pieces)
                chunks[-1].pause_after = PAUSE_SENTENCE
            if chunks:
                chunks[-1].pause_after = PAUSE_LINE
        if chunks:
            chunks[-1].pause_after = PAUSE_PARAGRAPH
    if chunks:
        chunks[-1].pause_after = 0.0
    return chunks


def _speakable(chunks):
    return any(re.search(r"[^\W_]", c.text) for c in chunks)


# ═══════════════════════════════════════════════════════════════════════════
#  Pronunciation front end (GPL-free; Apache-2.0 lexicons + light rules)
# ═══════════════════════════════════════════════════════════════════════════

_LETTER_NAMES = {
    "a": "ˈA", "b": "bˈi", "c": "sˈi", "d": "dˈi", "e": "ˈi", "f": "ˈɛf", "g": "ʤˈi", "h": "ˈAʧ",
    "i": "ˈI", "j": "ʤˈA", "k": "kˈA", "l": "ˈɛl", "m": "ˈɛm", "n": "ˈɛn", "o": "ˈO", "p": "pˈi",
    "q": "kjˈu", "r": "ˈɑɹ", "s": "ˈɛs", "t": "tˈi", "u": "jˈu", "v": "vˈi", "w": "dˈʌbᵊljˌu",
    "x": "ˈɛks", "y": "wˈI", "z": "zˈi",
}
_RULES = (
    ("tch", "ʧ"), ("sch", "sk"), ("igh", "I"), ("ough", "O"), ("augh", "ɔ"), ("eigh", "A"),
    ("th", "θ"), ("sh", "ʃ"), ("ch", "ʧ"), ("ph", "f"), ("wh", "w"), ("ck", "k"), ("ng", "ŋ"),
    ("qu", "kw"), ("gh", ""), ("kn", "n"), ("wr", "ɹ"),
    ("ee", "i"), ("ea", "i"), ("ie", "i"), ("ei", "A"), ("ai", "A"), ("ay", "A"), ("oa", "O"),
    ("oe", "O"), ("oo", "u"), ("ou", "W"), ("ow", "O"), ("oi", "Y"), ("oy", "Y"), ("au", "ɔ"),
    ("aw", "ɔ"), ("ew", "u"), ("ue", "u"),
    ("ar", "ɑɹ"), ("er", "əɹ"), ("ir", "ɜɹ"), ("ur", "ɜɹ"), ("or", "ɔɹ"),
    ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɑ"), ("u", "ʌ"), ("y", "i"),
    ("b", "b"), ("c", "k"), ("d", "d"), ("f", "f"), ("g", "ɡ"), ("h", "h"), ("j", "ʤ"), ("k", "k"),
    ("l", "l"), ("m", "m"), ("n", "n"), ("p", "p"), ("q", "k"), ("r", "ɹ"), ("s", "s"), ("t", "t"),
    ("v", "v"), ("w", "w"), ("x", "ks"), ("z", "z"),
)
_LONG_VOWEL = {"a": "A", "e": "i", "i": "I", "o": "O", "u": "ju"}
_PHONE_VOWELS = set("æɛɪɑʌiAIOWYuəɜɔ")
_VOICELESS = set("pkfθ")
_SIBILANT = set("szʃʒʧʤ")
_STRESS = "ˈˌ"
_ATTACH_PUNCT = set(";:,.!?\u2026\u201d)\"")
_OPEN_PUNCT = set("(\u201c")
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+|\S")


class LexiconPronouncer:
    """Word → phoneme string using dictionary lookup, morphology and fallbacks."""

    def __init__(self, lexicons):
        self._lexicons = [lex for lex in lexicons if lex]

    @classmethod
    def from_files(cls, paths):
        lexicons = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                lexicons.append(json.load(handle))
        return cls(lexicons)

    def _lookup(self, word):
        for lexicon in self._lexicons:
            entry = lexicon.get(word)
            if isinstance(entry, dict):
                entry = entry.get("DEFAULT")
            if entry:
                return entry
        return None

    def _known(self, word):
        for candidate in (word, word.lower(), word.capitalize()):
            found = self._lookup(candidate)
            if found:
                return found
        return None

    @staticmethod
    def _last_phone(phones):
        stripped = phones.rstrip(_STRESS)
        return stripped[-1] if stripped else ""

    def _plural(self, phones):
        last = self._last_phone(phones)
        if last in _SIBILANT:
            return phones + "ᵻz"
        return phones + ("s" if last in _VOICELESS or last == "t" else "z")

    def _past(self, phones):
        last = self._last_phone(phones)
        if last in "td":
            return phones + "ᵻd"
        return phones + ("t" if last in _VOICELESS or last in "ʃsʧ" else "d")

    def _morph(self, word):
        if word.endswith("'s"):
            stem = self._known(word[:-2])
            return self._plural(stem) if stem else None
        if word.endswith("ies") and len(word) > 4:
            stem = self._known(word[:-3] + "y")
            if stem:
                return stem + "z"
        if word.endswith("es") and len(word) > 3 and re.search(r"(?:s|x|z|ch|sh)es$", word):
            stem = self._known(word[:-2])
            if stem:
                return stem + "ᵻz"
        if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
            stem = self._known(word[:-1])
            if stem:
                return self._plural(stem)
        if word.endswith("ied") and len(word) > 4:
            stem = self._known(word[:-3] + "y")
            if stem:
                return stem + "d"
        if word.endswith("ed") and len(word) > 3:
            for stem_word in (word[:-2], word[:-1], word[:-3] if word[-3] == word[-4] else None):
                stem = self._known(stem_word) if stem_word else None
                if stem:
                    return self._past(stem)
        if word.endswith("ing") and len(word) > 4:
            for stem_word in (word[:-3], word[:-3] + "e", word[:-4] if word[-4] == word[-5] else None):
                stem = self._known(stem_word) if stem_word else None
                if stem:
                    return stem + "ɪŋ"
        if word.endswith("ly") and len(word) > 4:
            stem = self._known(word[:-2])
            if stem:
                return stem + "li"
        return None

    def _compound(self, word):
        for cut in range(len(word) - 3, 2, -1):
            head = self._known(word[:cut])
            if not head:
                continue
            tail = self._known(word[cut:]) or self._morph(word[cut:])
            if tail:
                return head + tail.replace("ˈ", "ˌ")
        return None

    @staticmethod
    def spell(word):
        return " ".join(_LETTER_NAMES[ch] for ch in word.lower() if ch in _LETTER_NAMES)

    @staticmethod
    def _rules(word):
        word = re.sub(r"([b-df-hj-np-tv-z])\1", r"\1", word)
        phones, i = [], 0
        while i < len(word):
            rest = word[i:]
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if rest[0] == "c" and nxt in "eiy" and nxt:
                phones.append("s"); i += 1; continue
            if rest[0] == "g" and nxt in "eiy" and nxt:
                phones.append("ʤ"); i += 1; continue
            if i == 0 and rest[0] == "y" and len(word) > 1:
                phones.append("j"); i += 1; continue
            if rest == "e" and i >= 2:
                break                                        # silent final e
            if (rest[0] in _LONG_VOWEL and len(rest) == 3 and rest[2] == "e"
                    and rest[1] not in "aeiouyrw"):
                phones.append(_LONG_VOWEL[rest[0]]); i += 1; continue
            if rest == "o" and i:
                phones.append("O"); i += 1; continue
            if rest == "a" and i:
                phones.append("ə"); i += 1; continue
            for graph, phone in _RULES:
                if rest.startswith(graph):
                    phones.append(phone)
                    i += len(graph)
                    break
            else:
                i += 1
        out = "".join(phones)
        for index, ch in enumerate(out):
            if ch in _PHONE_VOWELS:
                return out[:index] + "ˈ" + out[index:]
        return out

    def word(self, word):
        found = self._known(word)
        if found:
            return found
        if word.isupper() and 1 < len(word) <= 5:
            return self.spell(word)
        lower = word.lower()
        found = self._morph(lower) or self._compound(lower)
        if found:
            return found
        letters = re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", lower))
        if not letters:
            return ""
        if not re.search(r"[aeiouy]", letters) or len(letters) == 1:
            return self.spell(letters)
        return self._rules(letters)

    def phonemize(self, text):
        out = ""
        for token in _TOKEN_RE.findall(str(text).replace("\u2019", "'")):
            if token[0].isdigit():
                token_phones = " ".join(self.word(w) for w in number_to_words(int(token)).replace("-", " ").split())
            elif token[0].isalpha():
                token_phones = self.word(token)
            elif token in _ATTACH_PUNCT:
                out = out.rstrip() + token
                continue
            elif token in _OPEN_PUNCT or token == "\u2014":
                out = f"{out.rstrip()} {token}" if out else token
                if token == "\u2014":
                    out += " "
                continue
            else:
                out += " "                                   # hyphen, slash, symbols: word break
                continue
            if not token_phones:
                continue
            if out and not out.endswith(" ") and out[-1] not in _OPEN_PUNCT:
                out += " "
            out += token_phones
        return re.sub(r" {2,}", " ", out).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Neural tier runtime (ONNX)
# ═══════════════════════════════════════════════════════════════════════════

class _NeuralEngine:
    """One loaded speech graph + vocabulary; lexicons/styles load lazily per request."""

    sample_rate = 24000
    MAX_TOKENS = 510

    def __init__(self, variant_id):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        model_path = os.path.join(VOICE_DIR, _MODEL_FILES[variant_id][0])
        self._session = ort.InferenceSession(model_path, sess_options=options,
                                             providers=["CPUExecutionProvider"])
        names = [item.name for item in self._session.get_inputs()]
        self._ids_name = next((n for n in ("input_ids", "tokens") if n in names), names[0])
        self._style_name = "style" if "style" in names else names[1]
        self._speed_name = "speed" if "speed" in names else names[2]
        with open(os.path.join(VOICE_DIR, _VOCAB_FILE), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._vocab = data.get("model", {}).get("vocab", data) if isinstance(data, dict) else {}
        self._pronouncers = {}
        self._styles = {}

    def _pronouncer(self, dialect):
        if dialect not in self._pronouncers:
            self._pronouncers[dialect] = LexiconPronouncer.from_files(_lexicon_paths(dialect))
        return self._pronouncers[dialect]

    def _style(self, voice):
        if voice.id not in self._styles:
            self._styles[voice.id] = np.fromfile(_style_path(voice), dtype=np.float32).reshape(-1, 256)
        return self._styles[voice.id]

    def _infer(self, ids, voice, speed):
        styles = self._style(voice)
        style = styles[min(len(ids), len(styles) - 1)][np.newaxis, :]
        feeds = {
            self._ids_name: np.asarray([[0, *ids, 0]], dtype=np.int64),
            self._style_name: style.astype(np.float32),
            self._speed_name: np.asarray([speed], dtype=np.float32),
        }
        return np.asarray(self._session.run(None, feeds)[0], dtype=np.float32).reshape(-1)

    def synthesize_chunk(self, text, voice_id, speed):
        voice = _NATURAL_BY_ID[voice_id]
        phonemes = self._pronouncer(voice.dialect).phonemize(text)
        ids = [self._vocab[ch] for ch in phonemes if ch in self._vocab]
        if not ids:
            return np.zeros(0, dtype=np.float32)
        if len(ids) <= self.MAX_TOKENS:
            return self._infer(ids, voice, speed)
        words = text.split()
        if len(words) < 2:
            return self._infer(ids[:self.MAX_TOKENS], voice, speed)
        middle = len(words) // 2
        gap = np.zeros(int(PAUSE_WORDS * self.sample_rate), dtype=np.float32)
        return np.concatenate([self.synthesize_chunk(" ".join(words[:middle]), voice_id, speed), gap,
                               self.synthesize_chunk(" ".join(words[middle:]), voice_id, speed)])

    def close(self):
        self._session = None
        self._pronouncers.clear()
        self._styles.clear()
        self._vocab = {}


def _load_neural_engine(variant_id):
    return _NeuralEngine(variant_id)


def _close_engine(engine):
    close = getattr(engine, "close", None)
    if callable(close):
        close()


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline tier: operating-system voices (separate process, zero download)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _SystemVoice:
    id: str
    label: str
    language: str
    gender: str
    native: str        # OS voice name — internal only, never returned


_SYSTEM_CACHE = {"voices": None, "at": 0.0}
_SYSTEM_RETRY_SEC = 60.0


def _windows_shell():
    root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return candidate if os.path.exists(candidate) else shutil.which("powershell")


def _system_backend():
    if sys.platform == "win32":
        return ("windows", _windows_shell()) if _windows_shell() else (None, None)
    if sys.platform == "darwin":
        tool = shutil.which("say")
        return ("mac", tool) if tool else (None, None)
    for name in ("espeak-ng", "espeak"):
        tool = shutil.which(name)
        if tool:
            return "linux", tool
    return None, None


def _run_powershell(shell, script, timeout):
    import base64
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=creation,
    )


def _ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


_PS_LIST = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $list = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object {
    $v = $_.VoiceInfo
    [pscustomobject]@{ n = $v.Name; c = $v.Culture.Name; d = $v.Culture.EnglishName; g = [string]$v.Gender }
  })
  ConvertTo-Json -InputObject $list -Compress
} finally { $s.Dispose() }
"""

_PS_SPEAK = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$job = Get-Content -LiteralPath __JOB__ -Raw -Encoding UTF8 | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  if ($job.voice) { $s.SelectVoice([string]$job.voice) }
  $s.Rate = [int]$job.rate
  $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
  foreach ($item in $job.items) {
    $s.SetOutputToWaveFile([string]$item.path, $fmt)
    $s.Speak([string]$item.text)
    $s.SetOutputToNull()
  }
} finally { $s.Dispose() }
"""


def _public_system_voices(raw):
    """raw: [(native name, language label, gender)] → stable public voices."""
    voices, seen = [], {}
    for native, language, gender in raw:
        gender = (gender or "").strip().lower()
        gender = gender if gender in ("female", "male") else ""
        language = language or "Unknown language"
        base = f"{gender.capitalize()} ({language})" if gender else f"Voice ({language})"
        seen[base] = seen.get(base, 0) + 1
        label = base if seen[base] == 1 else f"{base} {seen[base]}"
        digest = hashlib.sha1(native.encode("utf-8")).hexdigest()[:10]
        voices.append(_SystemVoice(f"standard-{digest}", label, language, gender, native))
    return voices


def _query_system_voices():
    backend, tool = _system_backend()
    if backend == "windows":
        proc = _run_powershell(tool, _PS_LIST, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError("system voice listing failed")
        payload = json.loads(proc.stdout.strip() or "[]")
        payload = payload if isinstance(payload, list) else [payload]
        raw = [(p.get("n", ""), p.get("d") or p.get("c"), p.get("g")) for p in payload if p.get("n")]
        raw.sort(key=lambda item: (not str(item[1]).startswith("English"), item[1], item[2] != "Female"))
        return _public_system_voices(raw)
    if backend == "mac":
        proc = subprocess.run([tool, "-v", "?"], capture_output=True, text=True, timeout=30)
        raw = []
        for line in proc.stdout.splitlines():
            match = re.match(r"^(.+?)\s+([a-z]{2}[_-][A-Za-z0-9]+)\s+#", line)
            if match:
                raw.append((match.group(1).strip(), match.group(2).replace("_", "-"), ""))
        raw.sort(key=lambda item: (not item[1].startswith("en"), item[1]))
        return _public_system_voices(raw[:60])
    if backend == "linux":
        proc = subprocess.run([tool, "--voices"], capture_output=True, text=True, timeout=30)
        raw = []
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 4:
                gender = {"M": "male", "F": "female"}.get(parts[2].split("/")[-1][:1].upper(), "")
                raw.append((parts[1], parts[1], gender))
        raw.sort(key=lambda item: (not item[1].startswith("en"), item[1]))
        return _public_system_voices(raw[:60])
    return []


def _system_voices(refresh=False):
    with _CACHE_LOCK:
        cached, at = _SYSTEM_CACHE["voices"], _SYSTEM_CACHE["at"]
        if cached and not refresh:
            return cached
        if cached is not None and not cached and not refresh and time.monotonic() - at < _SYSTEM_RETRY_SEC:
            return cached
        try:
            voices = _query_system_voices()
        except Exception as error:
            _log.warning("System voices could not be listed (%s).", type(error).__name__)
            voices = []
        _SYSTEM_CACHE["voices"], _SYSTEM_CACHE["at"] = voices, time.monotonic()
        return voices


def baseline_available():
    return bool(_system_voices())


def _system_rate(speed):
    """Map a speed multiplier to the OS voice rate scale (-10..10 ≈ 1/3x..3x)."""
    return int(max(-10, min(10, round(10.0 * math.log(speed) / math.log(3.0)))))


def _system_speak(items, voice, speed, work_dir):
    backend, tool = _system_backend()
    timeout = 60 + sum(len(item["text"]) for item in items) * 0.2
    if backend == "windows":
        job_path = os.path.join(work_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as handle:
            json.dump({"voice": voice.native if voice else "", "rate": _system_rate(speed),
                       "items": items}, handle, ensure_ascii=False)
        proc = _run_powershell(tool, _PS_SPEAK.replace("__JOB__", _ps_literal(job_path)), timeout)
        if proc.returncode != 0:
            raise RuntimeError("system voice synthesis failed")
        return
    for item in items:
        text_path = item["path"] + ".txt"
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(item["text"])
        if backend == "mac":
            cmd = [tool, "-r", str(int(175 * speed)), "-o", item["path"],
                   "--file-format=WAVE", "--data-format=LEI16@22050", "-f", text_path]
            if voice:
                cmd[1:1] = ["-v", voice.native]
        elif backend == "linux":
            cmd = [tool, "-s", str(int(165 * speed)), "-w", item["path"], "-f", text_path]
            if voice:
                cmd[1:1] = ["-v", voice.native]
        else:
            raise VoiceoverUnavailableError(_UNAVAILABLE)
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError("system voice synthesis failed")


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

_UNAVAILABLE = ("Voiceover isn't available on this computer yet. Install the natural voice pack "
                "with 'python download_models.py --voice' and try again.")


def list_voices(refresh=False):
    """Public voices: natural (if the pack is installed) first, then system voices."""
    voices = []
    if neural_installed():
        for voice in _installed_natural_voices():
            voices.append({"id": voice.id, "label": voice.label, "language": voice.language,
                           "gender": voice.gender, "quality": TIER_NATURAL_LABEL, "default": False})
    for voice in _system_voices(refresh):
        voices.append({"id": voice.id, "label": voice.label, "language": voice.language,
                       "gender": voice.gender, "quality": TIER_STANDARD_LABEL, "default": False})
    if voices:
        voices[0]["default"] = True
    return voices


def default_voice_id():
    voices = list_voices()
    return voices[0]["id"] if voices else None


def _number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Choose a valid {name}.")
    if not math.isfinite(number):
        raise ValueError(f"Choose a valid {name}.")
    return number


def _validate(text, speed, pitch, fmt, quality):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to turn into speech.")
    length = len(text.strip())
    if length > MAX_TEXT_CHARS:
        raise ValueError(f"Voiceover text is limited to {MAX_TEXT_CHARS:,} characters "
                         f"(this text has {length:,}). Split it into shorter parts.")
    speed = _number(speed if speed is not None else 1.0, "speed")
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        raise ValueError(f"Choose a speed between {SPEED_MIN:g}x and {SPEED_MAX:g}x.")
    pitch = _number(pitch if pitch is not None else 0, "pitch")
    if not (PITCH_MIN <= pitch <= PITCH_MAX):
        raise ValueError(f"Choose a pitch between {PITCH_MIN:g} and +{PITCH_MAX:g} semitones.")
    fmt = str(fmt or "wav").strip().lower().lstrip(".")
    if fmt not in OUTPUT_FORMATS:
        raise ValueError("Choose WAV or MP3 for the voiceover file.")
    quality = str(quality or "auto").strip().lower()
    if quality not in QUALITY_CHOICES:
        raise ValueError("Choose auto, studio, natural or standard voice quality.")
    return text.strip(), speed, pitch, fmt, quality


def _build_ladder(quality, want_neural):
    from model_manager import CapabilityVariant, TIER_LITE

    studio = CapabilityVariant(
        VARIANT_STUDIO, tier=TIER_LITE, min_ram_gb=1.5, min_cpu_cores=2,
        is_available=lambda: _neural_variant_installed(VARIANT_STUDIO),
        public_label=TIER_STUDIO_LABEL, options={"min_free_ram_gb": 1.2})
    natural = CapabilityVariant(
        VARIANT_NATURAL, tier=TIER_LITE, min_ram_gb=0.8, min_cpu_cores=1,
        is_available=lambda: _neural_variant_installed(VARIANT_NATURAL),
        public_label=TIER_NATURAL_LABEL, options={"min_free_ram_gb": 0.6})
    baseline = CapabilityVariant(VARIANT_SYSTEM, public_label=TIER_STANDARD_LABEL)
    if not want_neural or quality == "standard":
        return [baseline]
    if quality == "studio":
        return [studio, baseline]
    if quality == "natural":
        return [natural, baseline]
    return [studio, natural, baseline]


def _trim_silence(y, sr, threshold_db=-42.0, keep_sec=0.03):
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-6:
        return y[:0]
    hop = max(1, int(0.01 * sr))
    frames = np.abs(y[: (len(y) // hop) * hop]).reshape(-1, hop).max(axis=1) if len(y) >= hop else np.abs(y)
    loud = np.nonzero(frames >= peak * (10.0 ** (threshold_db / 20.0)))[0]
    if loud.size == 0:
        return y[:0]
    keep = int(keep_sec * sr)
    start = max(0, loud[0] * hop - keep)
    end = min(len(y), (loud[-1] + 1) * hop + keep)
    return y[start:end]


def _assemble(segments, sr):
    parts = [np.zeros(int(0.12 * sr), dtype=np.float32)]
    for audio, pause in segments:
        audio = _trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
        if audio.size:
            parts.append(audio)
        if pause > 0:
            parts.append(np.zeros(int(pause * sr), dtype=np.float32))
    parts.append(np.zeros(int(0.2 * sr), dtype=np.float32))
    return np.concatenate(parts)


def _has_voice(segments):
    return any(np.size(audio) and float(np.max(np.abs(audio))) > 1e-4 for audio, _ in segments)


def _run_neural(variant, chunks, voice_id, speed):
    from model_manager import global_model_manager

    def _load():
        return _load_neural_engine(variant.variant_id), {}

    need = float(variant.options.get("min_free_ram_gb", 0.0))
    with global_model_manager.session(f"{CAPABILITY}.{variant.variant_id}", _load, _close_engine,
                                      min_free_ram_gb=need) as (engine, _meta):
        segments = [(engine.synthesize_chunk(chunk.text, voice_id, speed), chunk.pause_after)
                    for chunk in chunks]
        sample_rate = int(engine.sample_rate)
    if not _has_voice(segments):
        raise RuntimeError("the natural voice produced no audio")
    return segments, sample_rate


def _run_system(chunks, voice, speed):
    backend, _tool = _system_backend()
    if backend is None or not _system_voices():
        raise VoiceoverUnavailableError(_UNAVAILABLE)
    import librosa

    with tempfile.TemporaryDirectory(prefix="voiceover_") as work_dir:
        items = [{"text": chunk.text, "path": os.path.join(work_dir, f"part{index:04d}.wav")}
                 for index, chunk in enumerate(chunks)]
        try:
            _system_speak(items, voice, speed, work_dir)
        except subprocess.TimeoutExpired:
            raise VoiceoverUnavailableError("The voice took too long to read this text. Try a shorter script.")
        except RuntimeError:
            raise VoiceoverUnavailableError("The voice could not read this text. Try a different voice.")
        segments, sample_rate = [], None
        for item, chunk in zip(items, chunks):
            if not _file_ok(item["path"]):
                raise VoiceoverUnavailableError("The voice stopped before finishing. Try a shorter script.")
            audio, file_rate = sf.read(item["path"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sample_rate is None:
                sample_rate = file_rate
            elif file_rate != sample_rate:
                audio = librosa.resample(audio, orig_sr=file_rate, target_sr=sample_rate)
            segments.append((audio, chunk.pause_after))
    if not _has_voice(segments):
        raise VoiceoverUnavailableError("The voice produced no audio for this text. Try a different voice.")
    return segments, sample_rate


def _resolve_voice(voice):
    """Return (natural voice or None, system voice or None) for a public voice id."""
    voices = list_voices()
    if not voices:
        raise VoiceoverUnavailableError(_UNAVAILABLE)
    voice_id = str(voice).strip() if voice not in (None, "") else voices[0]["id"]
    if voice_id not in {v["id"] for v in voices}:
        raise ValueError("That voice isn't available on this computer. Choose a voice from the list.")
    if voice_id in _NATURAL_BY_ID:
        return _NATURAL_BY_ID[voice_id], None
    return None, next(v for v in _system_voices() if v.id == voice_id)


def synthesize(text, out_path, voice=None, speed=1.0, pitch=0, fmt="wav", quality="auto"):
    """
    Read ``text`` aloud into ``out_path`` (WAV or MP3), loudness-normalised to
    -16 LUFS / -1 dBTP. Returns a report dict with public labels only.
    Raises ValueError (app-authored message) for bad input or when no voice can run.
    """
    text, speed, pitch, fmt, quality = _validate(text, speed, pitch, fmt, quality)
    chunks = split_into_chunks(normalize_text(text), CHUNK_CHARS)
    if not chunks or not _speakable(chunks):
        raise ValueError("The text has nothing that can be read aloud. Add some words and try again.")
    natural_voice, system_voice = _resolve_voice(voice)

    from model_manager import global_quality_governor

    with _SYNTH_LOCK:
        ladder = _build_ladder(quality, want_neural=natural_voice is not None or quality in ("studio", "natural"))
        neural_voice = natural_voice or (_installed_natural_voices() or [None])[0]

        def _operate(variant):
            if variant.variant_id == VARIANT_SYSTEM:
                fallback = system_voice or (_system_voices() or [None])[0]
                segments, sr = _run_system(chunks, fallback, speed)
                return segments, sr, fallback
            if neural_voice is None:
                raise RuntimeError("no natural voice installed")
            segments, sr = _run_neural(variant, chunks, neural_voice.id, speed)
            return segments, sr, neural_voice

        (segments, sample_rate, used_voice), variant = global_quality_governor.run(CAPABILITY, ladder, _operate)
        audio = _assemble(segments, sample_rate)
        if pitch:
            import librosa
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=float(pitch)).astype(np.float32)
        duration = len(audio) / float(sample_rate)
        _write_output(audio, sample_rate, out_path, fmt)

    report = {
        "status": "success",
        "duration": round(duration, 2),
        "tier": _VARIANT_LABELS[variant.variant_id],
        "voice": used_voice.label if used_voice else TIER_STANDARD_LABEL,
        "voice_id": used_voice.id if used_voice else None,
        "chunks": len(chunks),
        "characters": len(text),
        "format": fmt,
        "sample_rate": sample_rate,
        "speed": speed,
        "pitch": pitch,
        "target_lufs": VOICE_TARGET_LUFS,
    }
    if natural_voice is not None and variant.variant_id == VARIANT_SYSTEM:
        report["notice"] = ("The natural voice couldn't run right now, so a standard voice "
                            "was used instead.")
    return report


def _write_output(audio, sample_rate, out_path, fmt):
    import audio_processor

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    raw_fd, raw_path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(raw_fd)
    final_path = os.path.join(out_dir, f".voiceover_{uuid.uuid4().hex}.{fmt}")
    try:
        sf.write(raw_path, audio, sample_rate, subtype="FLOAT", format="WAV")
        audio_processor.normalize_audio(raw_path, final_path, target_lufs=VOICE_TARGET_LUFS,
                                        true_peak_ceiling=VOICE_TRUE_PEAK_DBTP)
        if not _file_ok(final_path):
            raise VoiceoverUnavailableError("The voiceover file could not be saved.")
        os.replace(final_path, out_path)
    finally:
        for path in (raw_path, final_path):
            try:
                os.remove(path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  Copilot hook (owned by the orchestration engineer's registry)
# ═══════════════════════════════════════════════════════════════════════════

VOICEOVER_TOOL_SCHEMA = {
    "name": "generate_voiceover",
    "description": ("Turn a script into a spoken voiceover audio file on this computer. "
                    "Use for narration of slides, explainers or video intros. No voice cloning."),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": MAX_TEXT_CHARS,
                     "description": "The words to speak, plain text."},
            "voice": {"type": "string",
                      "description": "Voice id from GET /ai/tts/voices; omit for the default voice."},
            "speed": {"type": "number", "minimum": SPEED_MIN, "maximum": SPEED_MAX, "default": 1.0},
            "format": {"type": "string", "enum": sorted(OUTPUT_FORMATS), "default": "wav"},
        },
        "required": ["text"],
    },
}


def voiceover_tool(args, processed_dir):
    """Copilot tool: returns {output_file, message, data}; raises ValueError for bad args."""
    args = dict(args or {})
    fmt = str(args.get("format") or "wav").strip().lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError("Choose WAV or MP3 for the voiceover file.")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, f"voiceover_{uuid.uuid4().hex[:12]}.{fmt}")
    report = synthesize(args.get("text"), out_path, voice=args.get("voice") or None,
                        speed=args.get("speed", 1.0), pitch=args.get("pitch", 0), fmt=fmt,
                        quality=args.get("quality") or "auto")
    message = f"Voiceover ready: {report['duration']:.1f} seconds, {report['tier'].lower()}."
    if report.get("notice"):
        message += " " + report["notice"]
    return {"output_file": out_path, "message": message, "data": report}
