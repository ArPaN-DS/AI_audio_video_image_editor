"""
Agent NLU — rule-based decomposition of natural-language editing requests.

Used whenever the local reasoning service is offline (the common case), so it
aims to be precise rather than clever:

  1. normalise wording (units, number words, "between A and B", "fade in and out"),
  2. split into ordered clauses (lists, commas, "and", "then", "also", "finally"...),
  3. detect every operation in a clause, ordered by position, with its own
     arguments bound to the nearest keyword (so "upscale 4x and speed up 2x"
     gives scale 4 and speed 2),
  4. flag clauses that look like requests but match nothing (-> clarification)
     and known-but-unsupported operations (-> explanation, rest still runs).

Output is a *raw* plan; ``agent_planner.finalize_plan`` validates and repairs it
exactly like a plan from the reasoning service.
"""

import math
import re

import branding

from agent_planner import (Clarify, KEEP_ORDER_RE, RANGE_PATTERN, TIME_PATTERN, MULTIPLIER_PATTERN, NUMBER_RE, UNIT_RE, NOT_TIME_AFTER,
                           SUGGESTIONS_BY_TYPE, SOUNDTRACK_EFFECTS, time_to_seconds, MediaState, finalize_plan)

IMAGE_WORDS = r'\b(?:photo|image|picture|pic|selfie|portrait shot|png|jpe?g)\b'
AUDIO_WORDS = r'\b(?:voice|speech|vocals?|dialog(?:ue)?|audio|sound|narration|podcast)\b'
FORMATS = r'(mp3|wav|flac|ogg|mp4|webm|mkv|gif|png|jpe?g|webp|m4a|aac|mov|avi|wma|opus|tiff?|bmp|heic)'
UNSUPPORTED_FORMATS = {'m4a', 'aac', 'mov', 'avi', 'wma', 'opus', 'tif', 'tiff', 'bmp', 'heic'}

NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20, 'thirty': 30,
                'forty': 40, 'fifty': 50, 'sixty': 60}

GREETING_RE = re.compile(r"^(?:hi+|hello|hey+|howdy|sup|yo|hiya|greetings|good (?:morning|afternoon|evening))\b[\s!,.:-]*"
                         r"(?:there\b|copilot\b|team\b)?[\s!,.:-]*")
FILLER_WORDS = {
    'the', 'a', 'an', 'it', 'its', "it's", 'this', 'that', 'these', 'those', 'please', 'pls', 'plz', 'now', 'then',
    'also', 'and', 'to', 'for', 'me', 'my', 'of', 'in', 'on', 'with', 'just', 'can', 'you', 'could', 'would', 'will',
    'i', 'want', 'need', 'like', "i'd", "i'm", 'file', 'video', 'audio', 'clip', 'image', 'photo', 'picture', 'track',
    'song', 'recording', 'result', 'output', 'thanks', 'thank', 'ok', 'okay', 'so', 'too', 'as', 'well', 'at', 'be',
    'is', 'are', 'first', 'after', 'finally', 'lastly', 'really', 'very', 'quickly', 'right', 'away', 'all', 'one',
    'same', 'time', 'here', 'there', 'by', 'from', 'up', 'some', 'bit', 'little', 'little', 'more', 'version', 'copy',
    'next', 'step', 'go', 'ahead', 'yes', 'yeah', 'sure', 'great', 'cool', 'nice', 'perfect', 'thx', 'ty', 'when',
    'done', 'only', 'again', 'still', 'kindly', 'pic', 'mp3', 'wav', 'gif', 'mp4', 'format', 'quality', 'good',
}
GENERIC_VERBS = {
    'do', 'make', 'add', 'change', 'apply', 'create', 'turn', 'put', 'set', 'fix', 'edit', 'generate', 'give',
    'process', 'improve', 'adjust', 'modify', 'transform', 'replace', 'insert', 'move', 'color', 'colour', 'mix',
    'remove', 'delete', 'convert', 'enhance', 'boost', 'increase', 'decrease', 'reduce', 'raise', 'lower', 'restore',
    'swap', 'paint', 'draw', 'write', 'animate', 'erase', 'insert', 'shift', 'tweak', 'clean',
}

UNSUPPORTED = [
    (r'\b(?:rotat\w*|turn (?:it |this )?(?:sideways|upside down)|flip\w*|mirror\w*)\b', 'Rotating or flipping'),
    (r'\b(?:watermark\w*|add (?:a |some )?(?:text|title|logo|caption overlay|sticker)|text overlay|burn(?:ed|t)?[- ]in)\b',
     'Text, logo or watermark overlays'),
    (r'\b(?:black[- ]?(?:&|and)?[- ]?white|gr[ae]yscale|sepia|colou?r[- ]?grad\w*|colou?r[- ]?correct\w*|saturat\w*|'
     r'brightness|brighter|brighten\w*|darker|darken\w*|lighten\w*|exposure|vintage|lut|instagram filter)\b', 'Colour and brightness grading'),
    (r'\bstabili[sz]\w*', 'Video stabilisation'),
    (r'\b(?:reverse\w*|play (?:it |this )?backwards)\b', 'Reversing playback'),
    (r'\b(?:loop\w*|boomerang)\b', 'Looping'),
    (r'\b(?:merge|concatenat\w*|join|combine|stitch)\b', 'Merging several files'),
    (r'\b(?:add (?:some |background )?music|voice-?over|dub\w*)\b', 'Adding music or a voice-over'),
    (r'\btranslat\w*', 'Translation'),
    (r'\b(?:pitch|chipmunk|deeper voice|auto-?tune)\b', 'Pitch changes'),
    (r'\b(?:blur (?:the )?(?:faces?|background|plate)|censor\w*|pixelat\w*|bleep\w*)\b', 'Blurring or censoring'),
    (r'\badd (?:an? |some )?(?:echo|reverb)\b', 'Adding echo or reverb'),
    (r'\b(?:\d+\s*fps|frame ?rate|interpolat\w*)\b', 'Frame-rate changes'),
    (r'\b(?:vertical|9:16|16:9|aspect ratio|square format|portrait mode|landscape mode)\b', 'Aspect-ratio reframing'),
    (r'\bresiz\w*\b|\b\d{2,5}\s*[x×]\s*\d{2,5}\b', 'Resizing to custom dimensions'),
    (r'\bcrop\w*\b', 'Cropping the frame'),
    (r'\b(?:undo|revert)\b', 'Undo'),
]
UNSUPPORTED = [(re.compile(pattern), label) for pattern, label in UNSUPPORTED]
UNSUPPORTED_HINTS = {
    'Undo': 'To start over, select the original file in the media panel.',
    'Cropping the frame': 'To shorten media instead, say "trim from 2s to 8s".',
    'Rotating or flipping': 'The Image and Video studios have rotate tools.',
}


# ─────────────────────────────────────────────────────────────────────────
#  Normalisation & splitting
# ─────────────────────────────────────────────────────────────────────────

def normalize_text(prompt):
    text = str(prompt or '').strip().lower()
    text = (text.replace('’', "'").replace('‘', "'").replace('–', '-').replace('—', ' - ')
            .replace('×', 'x').replace('“', '"').replace('”', '"'))
    text = re.sub(r'(?m)^\s*(?:[-*•]|\(?\d{1,2}[.)])\s+', '\n', text)          # list markers at line start
    text = re.sub(r'(?<![\w.:])\(?\d{1,2}[.)]\s+(?=[a-z])', '\n', text)        # inline "1) trim 2) normalize"
    text = re.sub(r'\bhalf an? (?:second|sec)\b', '0.5 seconds', text)
    text = re.sub(r'\ba (?:half|1/2) (?:second|sec)\b', '0.5 seconds', text)
    text = re.sub(r'\b(?:a|one) (second|sec|minute|min)\b', r'1 \1', text)
    words = '|'.join(NUMBER_WORDS)
    text = re.sub(rf'\b({words})\b(?=\s*(?:-\s*)?(?:seconds?|secs?|minutes?|mins?|x\b|times\b|db\b|percent|%))',
                  lambda match: str(NUMBER_WORDS[match.group(1)]), text)
    text = re.sub(r'\b(\d+)\s*(?:minutes?|mins?|m)\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b',
                  lambda match: f'{int(match.group(1)) * 60 + float(match.group(2)):g}s', text)
    text = re.sub(r'\b(?:twice as fast|double (?:the )?speed|2x as fast)\b', '2x speed', text)
    text = re.sub(r'\b(?:half (?:the )?speed|half as fast|twice as slow)\b', '0.5x speed', text)
    text = re.sub(r'\b(?:double the volume|twice as loud)\b', '6 db louder', text)
    text = re.sub(r'\bhalf (?:the )?volume\b', '6 db quieter', text)
    text = re.sub(r'\bfade[- ]?in\s*/\s*(?:fade[- ]?)?out\b', 'fade in & fade out', text)
    text = re.sub(r'\bin\s*/\s*out\b', 'in & out', text)
    # "between 2 and 8 seconds" -> "from 2 to 8 seconds"
    text = re.sub(rf'\bbetween\s+({NUMBER_RE}\s*(?:{UNIT_RE})?)\s+and\s+({NUMBER_RE})', r'from \1 to \2', text)

    def _join_fades(match):
        second = match.group(2)
        if 'fade' not in second:
            second = re.sub(r'\bout\b', 'fade out', second)
        return f'{match.group(1)} & {second}'
    text = re.sub(r'(fad(?:e|ing)[ -]?in\b[^,;\n]{0,40}?)\s+and\s+((?:an?\s+)?(?:\d[\w.:]*\s*\w*\s+)?(?:fade[ -]?)?out\b)',
                  _join_fades, text)
    text = re.sub(r'\bin and out\b', 'in & out', text)
    edge_time = rf'{NUMBER_RE}\s*(?:{UNIT_RE})?'
    text = re.sub(rf'\b((?:first|opening|initial|beginning)\s+{edge_time})\s*(?:,\s*)?(?:and|&|plus)\s+'
                  r'(?=(?:the\s+)?(?:last|final|closing|ending)\b)', r'\1 & ', text)
    text = re.sub(rf'\b(off (?:the )?(?:start|beginning|front|top))\s*(?:,\s*)?(?:and|&)\s+(?={edge_time}\s+off\b)',
                  r'\1 & ', text)
    text = re.sub(r'\b(black)\s+and\s+(white)\b', r'\1-&-\2', text)
    return text


SPLIT_RE = re.compile(
    r'\s*(?:\n|;|\.(?=\s|$)|!|,|\band then\b|\bthen\b|\bafter that\b|\bafterwards\b|\bafter which\b|'
    r'\bfinally\b|\blastly\b|\bnext\b|\balso\b|\bplus\b|\bas well as\b|\bfollowed by\b|\band\b)\s*')


def split_clauses(text):
    return [clause.strip(' .') for clause in SPLIT_RE.split(text) if clause and clause.strip(' .')]


# ─────────────────────────────────────────────────────────────────────────
#  Argument helpers
# ─────────────────────────────────────────────────────────────────────────

def _time_tokens(text):
    tokens = []
    for match in TIME_PATTERN.finditer(text):
        value = time_to_seconds(match.group(1), match.group(2))
        if value is not None:
            tokens.append((match.start(), value, match.group(2)))
    return tokens


def find_range(text):
    """Return (start, end, span) from the first explicit A-to-B range in text, else None."""
    for match in RANGE_PATTERN.finditer(text):
        a, a_unit, b, b_unit = match.groups()
        if a_unit is None and b_unit is not None and ':' not in a:
            a_unit = b_unit
        if b_unit is None and a_unit is not None and ':' not in b:
            b_unit = a_unit
        start, end = time_to_seconds(a, a_unit), time_to_seconds(b, b_unit)
        if start is None or end is None:
            continue
        return round(start, 3), round(end, 3), match.span()
    return None


def find_trim_window(text):
    """
    Interpret trimming phrases. Returns dict(start_sec, end_sec[, keep_last_sec]) or None.
    Handles ranges, mm:ss, 'first N', 'last N', 'from N to the end', 'until N'.
    """
    window = find_range(text)
    if window:
        return {'start_sec': window[0], 'end_sec': window[1]}
    time_re = rf'({NUMBER_RE})\s*({UNIT_RE})?(?![\w%]|[.:]\d){NOT_TIME_AFTER}'
    remove = r'(?:remove|delete|drop|lose|skip|cut(?: off| out)?|chop(?: off)?|get rid of|strip|trim(?: off)?|shave(?: off)?)'
    head = r'(?:first|opening|initial|beginning|start)'
    tail = r'(?:last|final|closing|ending)'

    def seconds(value, unit, fallback_unit=None):
        return time_to_seconds(value, unit or (fallback_unit if ':' not in str(value) else None))

    both = re.search(rf'\b{remove}\s+(?:the\s+)?{head}\s+{time_re}\s*&\s*(?:the\s+)?{tail}\s+{time_re}', text)
    if both:
        return {'start_sec': seconds(both.group(1), both.group(2)), 'end_sec': None, '_removal': True,
                'drop_last_sec': seconds(both.group(3), both.group(4), both.group(2))}
    except_match = re.search(rf'\b(?:keep|want|leave)\s+(?:everything|all|it|the rest)\s+(?:except|but|minus|apart from|other than|without)\s+'
                             rf'(?:the\s+)?({head}|{tail})\s+{time_re}', text)
    if except_match:
        value = seconds(except_match.group(2), except_match.group(3))
        if re.match(head, except_match.group(1)):
            return {'start_sec': value, 'end_sec': None, '_removal': True}
        return {'start_sec': 0.0, 'end_sec': None, 'drop_last_sec': value, '_removal': True}
    minus = re.search(rf'\bfrom\s+{time_re}\s+(?:until|till|to|up to)\s+(?:the\s+)?end\s+(?:minus|less)\s+{time_re}', text) or \
        re.search(rf'\bfrom\s+{time_re}\s+(?:until|till|to|up to)\s+{time_re}\s+before\s+the\s+end', text)
    if minus:
        return {'start_sec': seconds(minus.group(1), minus.group(2)), 'end_sec': None, '_removal': True,
                'drop_last_sec': seconds(minus.group(3), minus.group(4), minus.group(2))}
    each_end = re.search(rf'\b{time_re}\s+(?:off\s+)?(?:(?:of|from)\s+)?(?:each|both)\s+ends?\b', text)
    if each_end:
        value = seconds(each_end.group(1), each_end.group(2))
        return {'start_sec': value, 'end_sec': None, 'drop_last_sec': value, '_removal': True}
    off_start = re.search(rf'\b{time_re}\s+off\s+(?:the\s+)?(?:start|beginning|front|top)\b', text)
    off_end = re.search(rf'\b{time_re}\s+off\s+(?:the\s+)?(?:end|ending|back|tail)\b', text)
    if off_start or off_end:
        window = {'start_sec': seconds(off_start.group(1), off_start.group(2)) if off_start else 0.0,
                  'end_sec': None, '_removal': True}
        if off_end:
            window['drop_last_sec'] = seconds(off_end.group(1), off_end.group(2), off_start.group(2) if off_start else None)
        return window
    drop_first = re.search(rf'\b{remove}\s+(?:the\s+)?(?:first|opening|initial|beginning)\s+{time_re}', text)
    if drop_first:
        return {'start_sec': time_to_seconds(drop_first.group(1), drop_first.group(2)), 'end_sec': None, '_removal': True}
    drop_last = re.search(rf'\b{remove}\s+(?:the\s+)?(?:last|final|ending)\s+{time_re}', text)
    if drop_last:
        return {'start_sec': 0.0, 'end_sec': None, '_removal': True,
                'drop_last_sec': time_to_seconds(drop_last.group(1), drop_last.group(2))}
    after = re.search(rf'\b{remove}\s+(?:everything|anything|all|the rest|what\'?s)\s+(?:after|past|beyond)\s+{time_re}', text)
    if after:
        return {'start_sec': 0.0, 'end_sec': time_to_seconds(after.group(1), after.group(2)), '_removal': True}
    before = re.search(rf'\b{remove}\s+(?:everything|anything|all|what\'?s)\s+(?:before|until|up to)\s+{time_re}', text)
    if before:
        return {'start_sec': time_to_seconds(before.group(1), before.group(2)), 'end_sec': None, '_removal': True}
    first = re.search(rf'\b(?:first|opening|initial|beginning)\s+{time_re}', text)
    if first:
        return {'start_sec': 0.0, 'end_sec': time_to_seconds(first.group(1), first.group(2))}
    last = re.search(rf'\b(?:last|final|ending|closing)\s+{time_re}', text)
    if last:
        return {'start_sec': 0.0, 'end_sec': None, 'keep_last_sec': time_to_seconds(last.group(1), last.group(2))}
    onward = re.search(rf'\b(?:from|after|starting (?:at|from)|start(?:ing)? at|beginning at)\s+{time_re}'
                       r'(?:\s*(?:to|till|until)\s+the\s+end|\s*on(?:wards?)?\b|\s*$|\s+(?=\w))', text)
    if onward and not re.search(r'\b(?:first|last)\b', text):
        value = time_to_seconds(onward.group(1), onward.group(2))
        if value is not None:
            return {'start_sec': value, 'end_sec': None}
    until = re.search(rf'\b(?:until|till|up to|upto|to)\s+{time_re}', text)
    if until:
        value = time_to_seconds(until.group(1), until.group(2))
        if value is not None and value > 0:
            return {'start_sec': 0.0, 'end_sec': value}
    return None


def _nearest(items, anchor):
    return min(items, key=lambda item: abs(item[0] - anchor)) if items else None


# ─────────────────────────────────────────────────────────────────────────
#  Clause detection
# ─────────────────────────────────────────────────────────────────────────

class Clause:
    def __init__(self, text, media_type):
        self.text = text
        self.media_type = media_type
        self.found = []          # (position, name, args)
        self.claimed = []        # (start, end)
        self.notes = []

    def claim(self, span):
        self.claimed.append(span)

    def free(self, match):
        start, end = match.span()
        return not any(start < c_end and end > c_start for c_start, c_end in self.claimed)

    def search(self, pattern, group=0):
        """First match whose span (or the span of ``group``, e.g. just the format word) is not claimed yet."""
        for match in re.finditer(pattern, self.text):
            start, end = match.span(group) if match.group(group) is not None else match.span()
            if not any(start < c_end and end > c_start for c_start, c_end in self.claimed):
                return match
        return None

    def add(self, match_or_pos, name, args=None, claim=True):
        position = match_or_pos if isinstance(match_or_pos, int) else match_or_pos.start()
        self.found.append((position, name, args or {}))
        if claim and not isinstance(match_or_pos, int):
            self.claim(match_or_pos.span())


NEGATION_RE = re.compile(
    r"\b(?:don'?t|do not|dont|no need to|never|without|skip(?:ping)?|but not|not)\s+(?:\w+\s+){0,2}?"
    r"(?:de-?nois\w+|noise[- ](?:removal|reduction|cleanup)|clean\w*|enhanc\w*|normali[sz]\w*|process\w*|"
    r"chang\w*|touch\w*|fad\w*|trim\w*|sharpen\w*|upscal\w*|compress\w*|transcri\w*|prep\w*|isolat\w*|"
    r"(?:remove|reduce|cut)\s+(?:the\s+)?(?:background\s+)?noise)\w*")


def detect_clause(clause, ctx):
    text = clause.text
    # Negated operations are constraints, not actions ("transcribe it, but don't denoise").
    for match in NEGATION_RE.finditer(text):
        clause.claim(match.span())
    if re.fullmatch(r"(?:as[- ]is|raw|untouched|no (?:pre-?)?processing|no cleanup)", text.strip()):
        clause.claim((0, len(text)))

    image_context = clause.media_type == 'image' or bool(re.search(IMAGE_WORDS, text))
    audio_words = bool(re.search(AUDIO_WORDS, text))

    # ── Removal of a middle section (unsupported, must win over "trim") ──
    middle = clause.search(r'\b(?:cut out|remove|delete|drop|skip|get rid of)\b(?:\s+(?:the|a))?(?:\s+(?:part|section|segment|bit|portion))?'
                           rf'\s+(?:from\s+|between\s+)?{NUMBER_RE}')
    if middle and find_range(text[middle.start():]):
        clause.notes.append('Removing a middle section is not available yet; I can keep a range instead '
                            '(for example "keep 0 to 5 seconds").')
        clause.claim((middle.start(), len(text)))

    # ── Soundtrack removal ──
    mute = clause.search(r"\b(?:mute|silence (?:the |this )?(?:video|clip|audio|sound|it)|"
                         r"remove (?:the |all )?(?:audio|sound|soundtrack)(?!\s*(?:noise|hiss|hum|quality))\b|"
                         r"strip (?:out )?(?:the )?(?:audio|sound)|(?:without|no) (?:any )?(?:audio|sound)\b(?!\s*noise)|"
                         r"kill the (?:audio|sound))")
    if mute:
        clause.add(mute, 'mute_video')

    # ── Voiceover, stem separation and lyrics ──
    voice_over = clause.search(r'\b(?:read (?:this|the following|my|the) (?:text |script )?(?:out )?aloud|read (?:this|it) out loud|'
                               r'(?:make|create|generate|record|give me|do)(?: me)? (?:a |an |the )?(?:voice[- ]?over|narration)|'
                               r'voice[- ]?over (?:of|for)|text[- ]to[- ]speech|narrate)\b')
    if voice_over:
        quoted = re.search(r'"([^"]{1,5000})"', text)
        spoken = quoted.group(1) if quoted else None
        if not spoken:
            tail = re.search(r'(?:aloud|out loud|voice[- ]?over|narration|narrate|speech)\s*(?:of|for|saying|:|-|that says)?\s*:?\s*(.{3,5000})$', text)
            spoken = tail.group(1).strip(' "\'') if tail and tail.group(1) not in ('this', 'it', 'this text') else None
        args = {'text': spoken} if spoken else {}
        if re.search(r'\bmp3\b', text):
            args['format'] = 'mp3'
        clause.add(voice_over, 'generate_voiceover', args)
    stems_mode = None
    stems = clause.search(r'\b(?:split|separate|break|divide)\b[^,;]{0,25}?\b(?:into|to)\b[^,;]{0,12}?\b(?:4|four)?[- ]?stems?\b|'
                          r'\b(?:4|four)[- ]?stems?\b|'
                          r'\bkaraoke (?:version|track|mix|backing)\b|\bmake (?:a |me a )?karaoke\b|'
                          r'\bseparate (?:the )?vocals?(?: (?:and|from|&) (?:the )?(?:music|instrumental|backing|accompaniment))?\b|'
                          r'\bvocals? (?:and|&) (?:the )?(?:music|instrumental)\b')
    if stems:
        found = stems.group(0)
        stems_mode = ('karaoke' if 'karaoke' in found else '4stem' if re.search(r'stems?\b', found) else 'vocals')
        args = {'mode': stems_mode}
        if re.search(r'\bmp3\b', text):
            args['format'] = 'mp3'
        clause.add(stems, 'separate_stems', args)
    lyrics = clause.search(r"\b(?:get|extract|fetch|find|write (?:out|down)|pull out|give me|show me|need|want)\b[^,;]{0,15}?\blyrics\b|"
                           r"\blyrics (?:of|from|for) (?:this|the) (?:song|track|audio|recording)\b|\bsong lyrics\b")
    if lyrics:
        args = {'format': fmt} if (fmt := next((f for f in ('vtt', 'txt') if re.search(rf'\b{f}\b', text)), None)) else {}
        clause.add(lyrics, 'extract_lyrics', args)

    # ── Stems ──
    isolate = clause.search(r'\b(?:isolate|extract|separate|keep only|get|pull out) (?:the |just the |only the )?'
                            r'(?:voice|vocals?|speech|dialog(?:ue)?|singer)\b|\bvocals? only\b|\ba ?c[ae]p+ell?a\b|'
                            r'\b(?:remove|get rid of|cut|drop|lose|kill|strip|delete|mute) (?:the )?(?:background )?music\b|\bstem separation\b|\bseparate (?:the )?stems\b|'
                            r'\bsplit (?:the )?stems\b')
    if isolate:
        clause.add(isolate, 'isolate_voice')
    vocals_off = clause.search(r'\b(?:remove|delete|strip|cut|take out|get rid of|mute) (?:the )?(?:vocals?|singing|lyrics)\b|'
                               r'\binstrumental\b|\bkaraoke\b|\bbacking track\b|\bno vocals\b')
    if vocals_off:
        clause.add(vocals_off, 'remove_vocals')

    # ── Vision ──
    if image_context:
        _detect_image_intents(clause, text)
    cutout = clause.search(r'\b(?:remove|delete|erase|cut ?out|strip|drop|get rid of|clear|isolate|replace|lose)\b[^,;]{0,25}?'
                           r'\b(?:background|bg|backdrop)\b(?!\s*(?:noise|music|sound|hum|hiss|audio|chatter|voices?|track))|'
                           r'\b(?:background remov\w*|bg remov\w*|remove ?bg|cut-?out|transparent(?: background)?|'
                           r'isolate (?:the )?subject|rembg|subject cutout)\b')
    if cutout:
        if clause.media_type in ('audio', 'video') and not image_context:
            clause.add(cutout, 'reduce_noise')
            clause.notes.append('Read "background" as background noise for this recording.')
        else:
            refine = bool(ctx.get('refine') or re.search(r'\b(?:refine\w*|retry|again|cleaner|better edges|smoother|'
                                                          r'another (?:model|method|try)|not good|no no|redo|improve)\b', text))
            profile = ('portrait' if re.search(r'\b(?:portrait|person|people|human|selfie|hair|man|woman|guy|girl|me|myself|him|her|them)\b', text)
                       else 'studio' if re.search(r'\b(?:product|studio|object)\b', text)
                       else 'fast' if re.search(r'\b(?:fast|quick)\b', text) else 'detail')
            clause.add(cutout, 'remove_background', {'quality_profile': profile, 'refine': refine})

    faces = clause.search(r'\b(?:restor\w*|fix\w*|enhanc\w*|repair\w*|improv\w*|retouch\w*|sharpen\w*|clean up)\b[^,;]{0,25}?\bfaces?\b|'
                          r'\bfaces? (?:restor\w*|enhanc\w*|fix\w*|repair\w*|retouch\w*)\b|\bportrait restor\w*')
    if faces:
        clause.add(faces, 'restore_faces')

    upscale_re = (r'\b(?:upscal\w*|up-scal\w*|upsampl\w*|super[- ]?res\w*|enlarg\w*|increase (?:the )?(?:resolution|quality)|'
                  r'higher resolution|hi-?res|high[- ]res(?:olution)?|make (?:it |the image |the photo |this )?(?:\d+(?:\.\d+)?\s*x\s+)?(?:bigger|larger)|4k|uhd'
                  + (r'|hd\b' if clause.media_type in ('image', None) else '') + r')')
    video_enhance = clause.search(
        r'\b(?:enhanc\w*|improv\w*|sharpen\w*|upscal\w*|denois\w*|restor\w*|fix\w*|boost\w*|clean up)\s+(?:the\s+|my\s+)?'
        r'(?:video|footage|picture|visuals?|video quality|picture quality|image quality|frames?)\b|'
        r'\bvideo (?:enhanc\w*|quality)\b|\b(?:make|turn|convert|upscale) (?:it|this|the video) (?:to |in(?:to)? )?'
        r'(?:1080p|720p|1440p|4k|full hd|hd)\b|\bsharper video\b') if clause.media_type == 'video' else None
    if video_enhance:
        resolution = re.search(r'\b(720p|1080p|1440p|4k|2160p|full hd|hd)\b', text)
        mode = {'2160p': '4k', 'full hd': '1080p', 'hd': '1080p'}.get(resolution.group(1), resolution.group(1)) if resolution else '1080p'
        clause.add(video_enhance, 'enhance_video', {'mode': mode})
        if resolution:
            clause.claim(resolution.span())
    upscale = clause.search(upscale_re) if clause.media_type != 'video' or not video_enhance else None
    if upscale and clause.media_type == 'video':
        resolution = re.search(r'\b(720p|1080p|1440p|4k|2160p)\b', text)
        clause.add(upscale, 'enhance_video', {'mode': {'2160p': '4k'}.get(resolution.group(1), resolution.group(1))
                                              if resolution else '1080p'})
        upscale = None
    if upscale:
        args = {}
        speed_anchors = [match.start() for match in re.finditer(SPEED_RE, text)]
        multiplier = _bind_multiplier(clause, upscale.start(), speed_anchors)
        if multiplier is not None:
            args['scale'] = multiplier
        elif re.search(r'\b(?:4k|uhd|quadruple)\b', text):
            args['scale'] = 4
        clause.add(upscale, 'upscale_image', args)

    noise_word = r'(?:de-?nois\w*|noise(?! gate)|noisy|hiss\w*|hum\b|humming|static|buzz\w*|grain\w*|background (?:sound|chatter|hum))'
    clarity = clause.search(r'\b(?:clarity|sharpen\w*|sharper|crisp\w*|de-?blur\w*|unblur\w*|blurry|out of focus|less blurry|'
                            r'clean up (?:the )?(?:photo|image|picture)|photo enhance\w*|enhance (?:the )?(?:photo|image|picture|details?)|'
                            r'improve (?:the )?(?:photo|image|picture)|more detail(?:ed)?|contrast|grainy|clean image|'
                            r'(?:make (?:it|the (?:photo|image)) )?look better)\b')
    if clarity and not audio_words:
        args = {}
        if re.search(r'\b(?:strong\w*|a lot|heav\w*|very|max\w*|extra)\b', text):
            args['sharpen_strength'] = 2.0
        elif re.search(r'\b(?:subtle|slight\w*|a bit|gentl\w*|light\w*|a little)\b', text):
            args['sharpen_strength'] = 0.8
        if re.search(noise_word, text):
            args['denoise_strength'] = 10
        clause.add(clarity, 'enhance_photo_clarity', args)
    elif image_context and not audio_words:
        image_noise = clause.search(rf'\b{noise_word}')
        if image_noise:
            clause.add(image_noise, 'enhance_photo_clarity', {'denoise_strength': 10})
            if not re.search(IMAGE_WORDS, text):
                clause.notes.append('Read "noise" as image grain for this photo.')

    # ── Thumbnails (before format detection: "thumbnail as png") ──
    frame = clause.search(r'\b(?:thumbnails?|screenshots?|screen ?grab|snapshots?|still(?: frame| image)?|frame grab|'
                          r'poster(?: frame| image)?|grab (?:a |one )?frame|(?:extract|save|capture|export|get|take) (?:a |one |the )?'
                          r'(?:still |single )?frame|freeze[- ]frame|cover (?:image|photo))\b')
    if frame:
        args = {}
        at = re.search(rf'(?:\b(?:at|from)\s+|@\s*)({NUMBER_RE})\s*({UNIT_RE})?(?![\w%]|[.:]\d)', text)
        if at:
            args['time_sec'] = time_to_seconds(at.group(1), at.group(2))
            clause.claim(at.span())
        fmt = re.search(r'\b(png|jpe?g)\b', text)
        if fmt:
            args['format'] = 'png' if fmt.group(1) == 'png' else 'jpg'
            clause.claim(fmt.span())
        if re.search(r'\b(?:of|from) the (?:result|trimmed|edited|new|final|output|processed|cut|shortened)\b', text):
            args['_from_result'] = True
        clause.add(frame, 'extract_frame', args)

    # ── Speech / audio ──
    transcribe = clause.search(r"\b(?:transcri\w*|speech[- ]to[- ]text|stt|subtitles?|captions?|srt|vtt|"
                               r"what (?:is|are) (?:they|he|she|it|people) saying|what does (?:it|he|she) say|"
                               r"write (?:down|out) what|dictation|lyrics)\b")
    if transcribe:
        clause.add(transcribe, 'transcribe_audio')

    silence = clause.search(r'\b(?:silen(?:ce|t (?:parts?|bits?|sections?))|dead (?:air|space)|pauses?|gaps?|quiet (?:parts?|bits?))\b')
    if silence and (re.search(r'\b(?:trim\w*|cut\w*|remove|delete|strip|skip|drop|get rid|clean|shorten)\b', text)
                    or re.fullmatch(r'\s*(?:the )?silences?\s*', text)):
        clause.add(silence, 'auto_trim_silence', {})
        trim_word = re.search(r'\b(?:trim\w*|cut\w*)\b', text)
        if trim_word and not find_range(text):
            clause.claim(trim_word.span())

    enhance_voice = clause.search(
        r'\b(?:(?:voice|speech|vocal|dialog(?:ue)?) (?:clarity|enhance\w*|boost|polish|clean ?up)|'
        r'(?:enhance|improve|boost|clarify|polish|clean up|fix) (?:the |my )?(?:voice|vocals?|speech|dialog(?:ue)?|'
        r'narration|audio quality|sound quality)|clear(?:er)? (?:voice|speech|dialog(?:ue)?|audio)|studio (?:voice|sound|quality)|'
        r'podcast (?:quality|sound)|de-?reverb|(?:the )?(?:voice|speech|dialog(?:ue)?|vocals?|narration) (?:clearer|cleaner|crisper|more clear)|(?:remove|reduce) (?:the )?(?:echo|reverb|room sound)|'
        r'(?:make (?:it|the voice|me) )?sound (?:better|clearer|professional))\b')
    if enhance_voice:
        clause.add(enhance_voice, 'enhance_speech')

    if not image_context:
        noise = clause.search(rf'\b(?:{noise_word}|clean (?:up )?(?:the )?(?:audio|sound|recording|track)|clean it up)')
        if noise:
            clause.add(noise, 'reduce_noise')

    normalize_verbs = r'normali[sz]\w*|level(?:l?ing)? (?:out )?(?:the )?(?:audio|volume|sound)|even (?:out )?(?:the )?volume|' \
                      r'consistent volume|broadcast (?:level|standard)|volume level|(?:youtube|spotify|podcast|streaming|broadcast)[- ]ready'
    inspect_re = (r"\b(?:inspect\w*|analy[sz]\w*|check\w*|measure\w*|probe|tell me (?:about|the)|"
                  r"what(?:'s| is) the (?:duration|length|resolution|sample rate|loudness|size|format|bitrate|frame rate)|"
                  r"how (?:long|loud|big) is|media info|properties|metadata|details|info about)\b")
    inspect = clause.search(inspect_re)
    normalize = clause.search(rf'\b(?:{normalize_verbs}' + ('' if inspect else r'|loudness|lufs') + r')\b')
    if normalize:
        args = {}
        lufs = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:lufs|lu)\b', text)
        dbfs = re.search(r'(-?\d+(?:\.\d+)?)\s*dbfs?\b', text)
        if lufs:
            args['target_lufs'] = -abs(float(lufs.group(1)))
            clause.claim(lufs.span())
        elif dbfs:
            args['target_dbfs'] = -abs(float(dbfs.group(1)))
            clause.claim(dbfs.span())
        elif re.search(r'\bspotify\b', text):
            args['preset'] = 'spotify'
        elif re.search(r'\b(?:youtube|tiktok|instagram|streaming|reels?|social)\b', text):
            args['preset'] = 'youtube'
        elif re.search(r'\bapple (?:podcasts?|music)\b', text):
            args['preset'] = 'apple_podcasts'
        elif re.search(r'\bpodcasts?\b', text):
            args['preset'] = 'podcast'
        elif re.search(r'\b(?:broadcast|ebu|tv|television)\b', text):
            args['preset'] = 'broadcast'
        clause.add(normalize, 'normalize_audio', args)

    volume = clause.search(r"\b(?:louder|quieter|softer|(?:turn|crank) (?:it |the volume |the audio |the sound )?(?:up|down)|"
                           r"(?:increase|raise|boost|lower|decrease|reduce|drop|cut) (?:the )?(?:volume|gain|level|sound|audio)|"
                           r"amplif\w*|volume (?:up|down|boost)|gain\b|bump (?:up )?(?:the |it )?(?:volume|level|it)(?: up)?|boost (?:it|the audio|the sound)|more volume|less volume|"
                           r"too (?:quiet|loud|soft)|can'?t hear)\b")
    if volume:
        lowering = bool(re.search(r"\b(?:quieter|softer|down|lower|decrease|reduce|drop|less volume|too loud)\b", text))
        gain = None
        db = re.search(r'([+-]?\d+(?:\.\d+)?)\s*db\b', text)
        pct = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', text)
        if db:
            gain = abs(float(db.group(1)))
            if db.group(1).startswith('-'):
                lowering = True
        elif pct:
            ratio = float(pct.group(1)) / 100.0
            gain = abs(20 * math.log10(1 + ratio)) if not lowering else abs(20 * math.log10(max(0.05, 1 - ratio)))
        elif re.search(r'\b(?:a (?:little )?bit|slight\w*|a little|a touch|tad)\b', text):
            gain = 3.0
        elif re.search(r'\b(?:a lot|much|way|significantly|really|very|lot)\b', text):
            gain = 10.0
        else:
            gain = 6.0
        clause.add(volume, 'adjust_volume', {'gain_db': round(-gain if lowering else gain, 2)})

    fade = clause.search(r'\bfad(?:e|es|ed|ing)\b')
    if fade:
        clause.add(fade, 'apply_audio_fade', _fade_args(text))

    # ── Video timeline ──
    gif = clause.search(r'\b(?:gif|animated (?:image|picture))\b')
    window = None
    if gif:
        args = {'target_format': 'gif'}
        window = find_trim_window(text)
        if window and window.get('end_sec') is not None:
            args['start'] = window['start_sec']
            args['duration'] = round(max(0.1, window['end_sec'] - window['start_sec']), 3)
        else:
            length = re.search(rf'({NUMBER_RE})\s*({UNIT_RE})[- ]?(?:long )?gif|gif (?:of|lasting) ({NUMBER_RE})\s*({UNIT_RE})?', text)
            if length:
                value = time_to_seconds(length.group(1) or length.group(3), length.group(2) or length.group(4))
                if value:
                    args['duration'] = value
            if window and window.get('start_sec'):
                args['start'] = window['start_sec']
        clause.add(gif, 'convert_video_format', args)

    trim_verb = clause.search(r'\b(?:trim\w*|cut|clip|keep|shorten|snip|chop|lose|drop|remove|delete|skip|crop(?= (?:the )?(?:audio|video|clip|song|track|it)?\s*'
                              r'(?:from|to|between|\d))|extract (?:the )?(?:part|section|segment|portion|bit)|'
                              r'only (?:keep|want)|isolate (?:the )?(?:part|section))\b')
    if not gif:
        window = find_trim_window(text)
    trim_needs_detail = False
    removal_verb = bool(trim_verb) and trim_verb.group(0) in ('lose', 'drop', 'remove', 'delete', 'skip')
    removal_window = False
    if window is not None:
        removal_window = window.pop('_removal', False)
        if removal_verb and not removal_window:
            window = None            # "remove the noise from 2 to 5" is not a trim
    if window and not trim_verb and not gif and removal_window:
        clause.add(0, 'trim', window)
    elif trim_verb and not gif:
        if window:
            clause.add(trim_verb, 'trim', window)
        elif trim_verb.group(0) in ('trim', 'trimming', 'cut', 'shorten', 'snip', 'chop') and not silence:
            trim_needs_detail = True
    elif window and not gif and not clause.found and (clause.search(RANGE_PATTERN.pattern)
                                                      or clause.search(r'^\s*(?:first|last)\b')) \
            and not clause.search(inspect_re):
        clause.add(0, 'trim', window)

    speed = clause.search(SPEED_RE)
    if speed:
        clause.add(speed, 'speed', {'speed': _speed_value(clause, speed)})

    extract = clause.search(r'\b(?:extract|rip|pull|get|separate|save|grab|export|take|keep)\b[^,;]{0,20}?\b(?:the )?'
                            r'(?:audio|sound|soundtrack|music)(?: track| only)?\b|\b(?:audio|sound) only\b|\bjust the (?:audio|sound)\b')
    if extract and not mute:
        fmt = re.search(r'\b(mp3|wav|flac|ogg)\b', text)
        if fmt and fmt.group(1) in ('flac', 'ogg'):
            clause.add(extract, 'convert', {'target_format': fmt.group(1)})
        else:
            args = {'format': fmt.group(1)} if fmt else {}
            if not fmt and clause.media_type != 'audio':
                args = {'format': 'mp3'}
            clause.add(extract, 'extract_audio', args)
        if fmt:
            clause.claim(fmt.span())

    compress = clause.search(r'\b(?:compress\w*|reduce (?:the )?(?:file )?size|smaller (?:file|size)|shrink|'
                             r'make (?:it|the file|the video) smaller|lower (?:the )?(?:file )?size|'
                             r'optimi[sz]e (?:it )?for (?:the )?(?:web|sharing|email|whatsapp|upload))\b')
    if compress:
        level = 'high' if re.search(r'\b(?:high|max\w*|strong\w*|a lot|heav\w*|hard|aggressive\w*|as small as possible|tiny)\b', text) else 'balanced'
        clause.add(compress, 'compress_video', {'level': level})

    if not gif:
        fmt_match = (clause.search(rf'\b(?:to|as|into|in)\s+(?:an?\s+)?(?:\w+\s+)?{FORMATS}\b', 1)
                     or clause.search(rf'\b(?:give me|get me|send me|i (?:want|need)|make (?:it |me )?)\s*(?:an?\s+)?(?:[a-z-]+\s+)?{FORMATS}\b', 1)
                     or clause.search(rf'\b{FORMATS}\s+(?:file|format|version|export|copy)\b')
                     or clause.search(rf'^\s*(?:export |convert |save )?{FORMATS}\s*$'))
        if fmt_match:
            target = next(group for group in fmt_match.groups() if group)
            target = 'jpg' if target in ('jpeg', 'jpg') else target
            if target in UNSUPPORTED_FORMATS:
                raise Clarify(f'{target.upper()} export is not available. Which format should I use?',
                              {'image': ['Export as PNG', 'Export as JPG', 'Export as WebP'],
                               'video': ['Convert to MP4', 'Convert to WebM', 'Extract the audio as MP3']}
                              .get(clause.media_type, ['Export as MP3', 'Export as WAV', 'Export as FLAC']))
            format_group = next(index for index, group in enumerate(fmt_match.groups(), 1) if group)
            clause.add(fmt_match.start(format_group), 'convert', {'target_format': target}, claim=False)
            clause.claim(fmt_match.span(format_group))

    generic = clause.search(r'\b(?:enhance|improve|polish|make (?:it|this) better|fix (?:it|this) up|touch (?:it )?up|'
                            r'beautify|auto[- ]?enhance|enhance (?:the )?quality|improve (?:the )?quality)\b')
    if generic and not clause.found:
        clause.add(generic, '_enhance')

    if inspect and not clause.found:
        clause.add(inspect, 'inspect_media')

    if trim_needs_detail and not clause.found and clause.media_type == 'image':
        raise Clarify('Images have no timeline to trim. Did you mean cutting out the subject (removing the background)?',
                      ['Remove the background', 'Cut out the person', 'Boost clarity'])
    if trim_needs_detail and not clause.found:
        raise Clarify("I can trim with sub-second precision. Which part should I keep? For example "
                      "'Trim from 2.3 to 5.3 seconds' or 'From 00:02 to 00:10'.",
                      ['Trim from 0 to 5 seconds', 'Trim from 2.3 to 5.3 seconds', 'Keep the first 10 seconds'])

    if not clause.found:
        for pattern, label in UNSUPPORTED:
            if pattern.search(text):
                hint = UNSUPPORTED_HINTS.get(label, '')
                clause.notes.append(f'{label} is not available from chat yet.' + (f' {hint}' if hint else ''))
                clause.claim((0, len(text)))
                break
    clause.found.sort(key=lambda item: item[0])
    return clause


ERASE_VERBS = r'(?:remove|erase|delete|get rid of|take out|clean up|wipe out|wipe|eliminate|lose|hide)'
NOT_OBJECTS = re.compile(r'^(?:the\s+|this\s+|that\s+|my\s+|a\s+|an\s+|all\s+(?:the\s+)?|any\s+|some\s+)?'
                         r'(?:[\w-]+\s+)?(?:background|bg|backdrop|noise|grain\w*|blur\w*|artifacts?|compression|'
                         r'jpeg|haze|red[- ]?eye|silence|audio|sound|music|vocals?|first|last|it|this|that|colou?r|'
                         r'saturation|filter|cutout|transparency|alpha)\b')
SUBJECT_CUTOUT = re.compile(
    r'\b(?:cut|clip)\s+(?:(?:me|him|her|them|it|myself|the|this|that|my|a|an)\s+)?(?:[\w-]+\s+)?out\b|'
    r'\b(?:pull|lift|take|get)\s+(?:me|him|her|them|myself|the (?:subject|person|product|object|main subject))\s+out\b|'
    r'\bisolate\s+(?:me|myself|him|her|them|the (?:subject|person|product|object|main subject|foreground)|the \w+ from)\b|'
    r'\bextract\s+(?:the\s+)?(?:subject|person|product|object|foreground|main subject)\b|'
    r'\bseparate\s+(?:me|him|her|them|the (?:subject|person|product))\s+from\b|'
    r'\bmake (?:a |me a )?cut-?out\b')
OBJECT_REMOVAL_REPLY = (
    'Removing a specific object or person needs the area to be painted over (a mask), which I cannot do from chat, '
    'and it is not available in the studio yet. If you want to keep only the main subject, I can remove the '
    'whole background instead.')


def _detect_image_intents(clause, text):
    """Image-only meanings: subject cutouts phrased with cut/isolate/extract, and object removal (unsupported)."""
    for match in re.finditer(rf'\b{ERASE_VERBS}\s+(?P<object>.+?)(?=\s+(?:and|then)\b|[,;.]|$)', text):
        target = match.group('object').strip()
        if NOT_OBJECTS.match(target) or not clause.free(match):
            continue
        raise Clarify(OBJECT_REMOVAL_REPLY, ['Remove the background', 'Boost clarity', 'Upscale 2x'])
    cut = SUBJECT_CUTOUT.search(text)
    if cut and clause.free(cut):
        refine = bool(re.search(r'\b(?:refine\w*|retry|again|cleaner|better edges)\b', text))
        profile = ('portrait' if re.search(r'\b(?:me|myself|him|her|them|person|people|guy|girl|man|woman|portrait|selfie)\b', text)
                   else 'studio' if re.search(r'\b(?:product|object|item)\b', text) else 'detail')
        clause.add(cut, 'remove_background', {'quality_profile': profile, 'refine': refine})


def _fade_args(text):
    time_re = rf'({NUMBER_RE})\s*({UNIT_RE})?'
    joiner = r'(?:of|over|for|by|at|lasting|=|:|with)?\s*(?:about\s+|around\s+|a\s+)?'

    def value(pattern):
        match = re.search(pattern, text)
        return time_to_seconds(match.group(1), match.group(2)) if match else None
    explicit_in = value(rf'fad(?:e|ing)[ -]?in\s*{joiner}{time_re}(?!\s*(?:x|%|db))') \
        or value(rf'{time_re}\s*(?:second |sec |s )?(?:long\s+)?(?:of\s+)?fad(?:e|ing)[ -]?in')
    explicit_out = value(rf'fad(?:e|ing)[ -]?out\s*{joiner}{time_re}(?!\s*(?:x|%|db))') \
        or value(rf'(?<!in ){time_re}\s*(?:second |sec |s )?(?:long\s+)?(?:of\s+)?fad(?:e|ing)[ -]?out')
    lead = value(rf'{time_re}\s*(?:long\s+)?(?:of\s+)?fad') or value(rf'fad(?:e|es|ing)\b[^,;]*?\b(?:over|of|for|lasting)\s+{time_re}')
    wants_in = bool(re.search(r'fad(?:e|ing)[ -]?in\b|\bin\s*&\s*(?:fade\s*)?out\b', text))
    wants_out = bool(re.search(r'fad(?:e|ing)[ -]?out\b|&\s*(?:fade\s*)?out\b', text))
    if not wants_in and not wants_out:
        wants_in = wants_out = True
    fade_in = explicit_in if explicit_in is not None else (lead if lead is not None else 2.0) if wants_in else 0.0
    fade_out = explicit_out if explicit_out is not None else (lead if lead is not None else 2.0) if wants_out else 0.0
    if explicit_in is not None and explicit_out is None and wants_out and lead == explicit_in and re.search(r'fade[ -]?in\s*&', text):
        fade_out = explicit_in
    return {'fade_in_sec': fade_in, 'fade_out_sec': fade_out}


SPEED_RE = (r'\b(?:speed(?:\s*(?:it\s*)?up)?|faster|quicker|slower|slow(?:\s*(?:it|this|the \w+))?(?:\s*back)?\s*down|'
            r'slow[- ]?mo(?:tion)?|playback (?:rate|speed)|tempo|time[- ]?lapse|accelerat\w*|decelerat\w*)\b')


def _bind_multiplier(clause, anchor_pos, rival_positions):
    """Claim and return the free 'Nx' multiplier closest to anchor_pos (and closer to it than to any rival keyword)."""
    candidates = []
    for match in MULTIPLIER_PATTERN.finditer(clause.text):
        if not clause.free(match):
            continue
        distance = abs(match.start() - anchor_pos)
        if rival_positions and any(abs(match.start() - rival) < distance for rival in rival_positions):
            continue
        candidates.append((distance, match))
    if not candidates:
        return None
    match = min(candidates, key=lambda item: item[0])[1]
    clause.claim(match.span())
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _speed_value(clause, anchor):
    text = clause.text
    slow_anchor = bool(re.search(r'slow', anchor.group(0)))
    candidates = [(abs(match.start() - anchor.start()), match) for match in MULTIPLIER_PATTERN.finditer(text)
                  if clause.free(match)]
    if candidates:
        match = min(candidates, key=lambda item: item[0])[1]
        clause.claim(match.span())
        try:
            value = float(match.group(1))
        except ValueError:
            value = None
        if value is not None:
            after = text[match.end():match.end() + 12]
            if (re.match(r'\s*(?:slower|as slow)', after) or slow_anchor) and value > 1:
                return round(1.0 / value, 4)
            return value
    percent = re.search(r'(?:(by|to|at)\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)(\s*(?:speed|of (?:the )?(?:original )?speed))?', text)
    if percent:
        number = float(percent.group(2)) / 100.0
        if percent.group(1) in ('to', 'at') or percent.group(3):
            return round(number, 4)
        return round(1 - number if (slow_anchor or 'slower' in text) else 1 + number, 4)
    bare = re.search(r'\bspeed\s*(?:to|of|at|=)?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:s|sec|seconds?|%))', text)
    if bare:
        return float(bare.group(1))
    if re.search(r'slow[- ]?mo', anchor.group(0)):
        return 0.5
    if re.search(r'\b(?:a (?:little )?bit|slightly|a little|a touch|a tad)\b', text):
        return 0.8 if (slow_anchor or 'slower' in text) else 1.25
    return None


# ─────────────────────────────────────────────────────────────────────────
#  Conversation, follow-ups & knowledge
# ─────────────────────────────────────────────────────────────────────────

CAPABILITIES_REPLY = (
    "Here is what I can do for you — in one message, in any order:\n\n"
    "• **Images (Vision)**: background removal (with refinement), 2x/4x upscaling, clarity boost, face restoration, PNG/JPG/WebP export.\n"
    "• **Video**: sub-second trimming, speed 0.25x–4x, mute, thumbnails, picture enhancement, audio extraction, "
    "GIF/MP4/WebM conversion, compression.\n"
    "• **Audio**: speech-to-text with subtitles, noise reduction, voice isolation, vocal removal, silence trimming, "
    "voice enhancement, loudness normalization (LUFS), volume, fades and format export.\n\n"
    "I also check the media's condition first — for example, a noisy recording is cleaned before transcription. "
    "Try: 'Trim from 2s to 8s, remove the background noise, normalize loudness, then export as MP3', "
    "or type @ to pick a ready-made skill such as @podcast-polish."
)

KNOWLEDGE = [
    (r'\b(?:mp3 (?:vs\.?|versus|or|and) wav|wav (?:vs\.?|versus|or|and) mp3|difference between (?:mp3 and wav|wav and mp3))\b',
     "WAV is uncompressed, lossless studio audio with bit-perfect fidelity, best for editing and master recordings. "
     "MP3 is compressed lossy audio that reduces file size by ~90% while keeping great listening quality, ideal for sharing and web playback.",
     ['Convert to WAV', 'Export as MP3']),
    (r'\b(?:mp4 (?:vs\.?|versus|or|and) webm|webm (?:vs\.?|versus|or|and) mp4|difference between (?:mp4 and webm|webm and mp4))\b',
     "MP4 offers universal compatibility across all devices, browsers, and video editors. WebM is open-source, royalty-free, "
     "and yields smaller file sizes for modern web streaming, but is less supported on older hardware.",
     ['Convert to WebM', 'Convert to MP4']),
    (r'\b(?:png (?:vs\.?|versus|or|and) jpg|jpg (?:vs\.?|versus|or|and) png|png (?:vs\.?|versus|or|and) jpeg|jpeg (?:vs\.?|versus|or|and) png)\b',
     "JPG uses lossy compression best for photos and rich color gradients. PNG uses lossless compression with alpha channel "
     "transparency, ideal for graphics, logos, and cutouts after background removal.",
     ['Remove image background', 'Export as PNG']),
    (r'\b(?:flac (?:vs\.?|versus|or|and) wav|wav (?:vs\.?|versus|or|and) flac)\b',
     "Both WAV and FLAC are 100% lossless. FLAC compresses the audio data by ~50% without discarding any sonic detail, "
     "while WAV stores raw uncompressed PCM data with faster sample seeking in DAWs.",
     ['Convert to WAV', 'Inspect this file']),
    (r'\b(?:aac (?:vs\.?|versus|or|and) mp3|mp3 (?:vs\.?|versus|or|and) aac)\b',
     "AAC is a newer, higher-efficiency codec than MP3. At identical bitrates, AAC provides cleaner high frequencies "
     "and less compression artifacting, and is the default audio codec for MP4 video and streaming.",
     ['Export as MP3', 'Inspect this file']),
    (r'\b(?:what is |what\'s )?mp3\b',
     "MP3 (MPEG-1 Audio Layer III) is the world's most widely recognized lossy audio format. It compresses audio files by ~90% "
     "relative to uncompressed CD audio (WAV) while retaining clear listening fidelity, typically encoded at 128 to 320 kbps.",
     ['Export as MP3', 'Convert to WAV']),
    (r'\b(?:what is |what\'s )?wav\b',
     "WAV (Waveform Audio File Format) is an uncompressed, lossless audio container developed by Microsoft and IBM. "
     "It preserves exact sample values with zero compression artifacts, making it the industry standard for sound recording and editing.",
     ['Convert to WAV', 'Export as MP3']),
    (r'\b(?:what is |what\'s )?flac\b',
     "FLAC (Free Lossless Audio Codec) provides bit-perfect audio compression, reducing file size by 40–60% compared to raw WAV "
     "without sacrificing any audio fidelity. It is the premier choice for archival audio and audiophile listening.",
     ['Convert to WAV', 'Inspect this file']),
    (r'\b(?:what is |what\'s )?aac\b',
     "AAC (Advanced Audio Coding) is a high-performance lossy audio compression standard designed as the successor to MP3. "
     "It delivers superior sound quality at equal or lower bitrates, and is universally supported in MP4 and modern streaming.",
     ['Export as MP3', 'Convert to MP4']),
    (r'\b(?:what is |what\'s )?(?:opus|ogg)\b',
     "Opus is a modern, royalty-free audio codec designed for interactive speech and music streaming over the Internet. "
     "It offers ultra-low latency and beats MP3, AAC, and Vorbis at comparable bitrates.",
     ['Inspect this file', 'Convert to WebM']),
    (r'\b(?:what is |what\'s )?mp4\b',
     "MP4 (MPEG-4 Part 14) is the universal multimedia container format for video, audio, and subtitles. "
     "It typically pairs H.264 video with AAC audio and is supported natively across virtually all devices and browsers.",
     ['Convert to MP4', 'Trim video from 2s to 8s']),
    (r'\b(?:what is |what\'s )?webm\b',
     "WebM is an open, royalty-free media container format designed for the web. It uses VP8/VP9 or AV1 video codecs "
     "with Opus audio, generating compact streamable video files optimized for HTML5.",
     ['Convert to WebM', 'Convert to MP4']),
    (r'\b(?:what is |what\'s )?webp\b',
     "WebP is a modern image format from Google providing superior lossless and lossy compression for web images, "
     "typically 25–34% smaller than PNG or JPEG at equivalent visual quality, with full transparency support.",
     ['Export as WebP', 'Remove image background']),
    (r'\b(?:what is |what\'s )?(?:h\.?264|avc)\b',
     "H.264 (Advanced Video Coding / AVC) is the most widely adopted video compression standard in history, "
     "featuring hardware-accelerated playback on virtually every modern computer, phone, TV, and browser.",
     ['Convert to MP4', 'Compress this video']),
    (r'\b(?:what is |what\'s )?(?:h\.?265|hevc)\b',
     "H.265 (High Efficiency Video Coding / HEVC) delivers ~50% better compression efficiency than H.264 at identical quality, "
     "making it ideal for 4K and 8K video, though requiring more recent hardware for decoding.",
     ['Compress this video', 'Convert to MP4']),
    (r'\bloudness\b(?! normali)', "Loudness is how loud audio is perceived over time. It is measured in LUFS; "
                                  "streaming platforms target about -14 LUFS, so normalizing to that keeps your "
                                  "audio from being turned down or sounding quiet.",
     ['Normalize loudness to -14 LUFS', 'Measure the loudness']),
    (r'\blufs\b', "LUFS (Loudness Units relative to Full Scale) measures perceived loudness over time. "
                  "Streaming platforms such as YouTube and Spotify target about -14 LUFS, Apple Music about -16 LUFS, "
                  "and broadcast about -23 LUFS.", ['Normalize loudness to -14 LUFS', 'Measure the loudness']),
    (r'\b(?:sample rate|khz)\b', "Sample rate is how many audio samples are stored per second. 44.1 kHz is the music "
                                 "standard and 48 kHz is the video standard.", ['Inspect this file']),
    (r'\bbit ?rate\b', "Bitrate is how much data is used per second of media; higher bitrates keep more detail but make "
                       "bigger files.", ['Compress this video', 'Export as MP3']),
    (r'\b(?:codec|codecs)\b', "A codec compresses media for storage. Common choices are H.264 for MP4 video, VP9 for WebM, "
                              "and MP3, AAC or Opus for audio.", ['Convert to MP4', 'Convert to WebM']),
    (r'\b(?:upscal\w*|super[- ]?resolution)\b', "Upscaling enlarges an image 2x or 4x and reconstructs fine detail instead "
                                                "of just stretching pixels.", ['Upscale 2x', 'Upscale 4x']),
    (r'\bdb\b|\bdecibels?\b', "Decibels (dB) describe level changes: +6 dB is roughly twice the amplitude; 0 dBFS is "
                              "the loudest digital level before clipping.", ['Make it 6 dB louder', 'Normalize loudness']),
    (r'\b(?:fps|frame rate|frames per second)\b',
     "Frame rate (FPS) measures how many consecutive images are shown per second. 24 fps is film standard, "
     "30 fps is video broadcast/web standard, and 60 fps is used for ultra-smooth action and gaming.",
     ['Inspect this file', 'Speed up video 1.5x']),
    (r'\b(?:resolution|1080p|4k|720p)\b',
     "Resolution defines the pixel dimensions of a video or image. 1080p (Full HD) is 1920×1080 pixels; "
     "4K (Ultra HD) is 3840×2160 pixels, offering 4x the pixel density of 1080p.",
     ['Inspect this file', 'Upscale 2x']),
    (r'\b(?:aspect ratio|16:9|9:16|4:3)\b',
     "Aspect ratio is the proportional relationship between width and height. Standard horizontal video is 16:9, "
     "vertical video for Reels/TikTok/Shorts is 9:16, and classic square format is 1:1.",
     ['Inspect this file']),
    (r'\b(?:noise reduction|remove background noise|denoise|clean audio)\b',
     "Noise reduction identifies and attenuates steady background hiss, hum, and fan rumble using spectral analysis "
     "while preserving the clarity of spoken vocal frequencies.",
     ['Clean audio noise', 'Transcribe speech to text']),
    (r'\b(?:voice isolation|vocal removal|extract vocals|remove vocals|isolate vocals)\b',
     "Voice isolation separates spoken or sung vocals from background music and sound effects. Vocal removal suppresses "
     "vocals to produce an instrumental backing track.",
     ['Isolate voice from background', 'Clean audio noise']),
    (r'\b(?:transcription|transcribe|speech to text|subtitles|srt|vtt)\b',
     "Transcription uses local neural speech-to-text models to convert spoken dialogue into timestamped text, "
     "which can be exported as TXT, SRT, or WebVTT subtitle files.",
     ['Transcribe speech to text', 'Clean audio noise']),
]


def _question_reply(text, media_type):
    # 1. Capabilities / Help queries
    if re.search(r"\b(?:what can you do|what do you do|help me|features|capabilit\w*|how does this work|who are you)\b", text) \
            or text.strip(' ?!.') in ('help', 'menu', 'options'):
        return {'reply': CAPABILITIES_REPLY, 'suggested_actions': SUGGESTIONS_BY_TYPE.get(media_type, SUGGESTIONS_BY_TYPE[None]),
                'thought': 'Capability overview requested.'}

    # 2. Check curated knowledge directly (supports "MP4 vs WebM", "what is mp3", "lufs", etc.)
    for pattern, reply, suggestions in KNOWLEDGE:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return {'reply': reply, 'suggested_actions': suggestions, 'thought': 'Media knowledge question.'}

    # 3. How-to instruction questions
    how_to = re.match(r"\s*how (?:do|can|would|should) i\s+(.*)", text, flags=re.IGNORECASE)
    if how_to:
        return {'reply': f"Just tell me what you want in plain words and I will do it — for example "
                         f"\"{how_to.group(1).strip(' ?.').capitalize() or 'Trim from 2s to 8s'}\". "
                         "You can chain several edits in one message.",
                'suggested_actions': SUGGESTIONS_BY_TYPE.get(media_type, SUGGESTIONS_BY_TYPE[None]),
                'thought': 'How-to question.'}

    # 4. General question fallback
    if re.match(r"\s*(?:what|why|which|explain|difference|define|tell me about|is it|does|should|when)\b", text, flags=re.IGNORECASE):
        return {'reply': (f"I'm {branding.assistant_name()}, built into {branding.product_name()}, so I'm best at editing and media questions — "
                          "for example \"What is LUFS?\", \"MP4 vs WebM\", or \"What is MP3?\". Tell me an edit and I'll do it."),
                'suggested_actions': SUGGESTIONS_BY_TYPE.get(media_type, SUGGESTIONS_BY_TYPE[None]),
                'thought': 'General question without an edit request.'}
    return None


REPLAY_RE = re.compile(r"^(?:please\s+)?(?:do (?:it|that|this|the same)(?: thing)?(?: again)?|again|same (?:again|thing)(?: again)?|"
                       r"repeat(?: that| it| the last (?:edit|step))?|one more time|once more|redo(?: it| that)?)[\s.!]*$")
REFINE_RE = re.compile(r"\b(?:not good|no no|try again|another (?:model|method|way|try)|cleaner(?: edges)?|better edges|"
                       r"refine|retry|smoother edges|not clean)\b")


def _previous_user_messages(history):
    return [item.get('content', '') for item in reversed(history or [])
            if isinstance(item, dict) and item.get('role') == 'user' and isinstance(item.get('content'), str)]


# ─────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────

def parse_request(prompt, media_context=None, history=None, _depth=0):
    """Return a raw plan dict for ``agent_planner.finalize_plan``."""
    media_type = MediaState.from_context(media_context).type
    prompt = re.sub(r"\(\s*no skill\s*\)|\bwithout (?:a |any )?skills?\b", ' ', str(prompt or ''), flags=re.IGNORECASE)
    text = normalize_text(prompt)
    text = GREETING_RE.sub('', text, count=1).strip(' ,.!') if GREETING_RE.match(text) else text
    previous = _previous_user_messages(history)

    if not text:
        return {'tools': [], 'reply': (f"Hello. I am {branding.assistant_name()}, with image, video, audio and inspection "
                                       "tools. Upload any audio, video or image and tell me what to do — you can "
                                       "chain several edits in one message, like 'Trim from 2s to 8s, remove the "
                                       "background noise, then export as MP3'."),
                'suggested_actions': ['Trim video from 2.3s to 5.3s', 'Remove image background', 'Clean audio noise',
                                      'Transcribe speech'],
                'thought': 'Greeting; ready for instructions.'}

    keep_order = bool(KEEP_ORDER_RE.search(text))
    if keep_order:
        text = KEEP_ORDER_RE.sub(' ', text).strip(' ,.!')
        if previous and re.fullmatch(r"(?:please\s+)?(?:(?:run|do)(?: it| that| this)?|just)?", text):
            replay = parse_request(previous[0], media_context, None, _depth + 1)
            if replay.get('tools'):
                replay['keep_order'] = True
                replay['thought'] = 'Running the previous request exactly in the order given.'
                return replay

    if _depth == 0 and previous:
        if REPLAY_RE.match(text):
            replay = parse_request(previous[0], media_context, None, _depth + 1)
            if replay.get('tools'):
                replay['thought'] = 'Repeating the previous request on the current result.'
                return replay
        if REFINE_RE.search(text) and not re.search(r'\b(?:background|cutout|bg)\b', text):
            earlier = next((message for message in previous
                            if re.search(r'\b(?:background|cutout|bg|subject)\b', message.lower())), None)
            if earlier and media_type in ('image', None):
                refined = parse_request(earlier + ' ' + text, media_context, None, _depth + 1)
                if refined.get('tools'):
                    for tool in refined['tools']:
                        if tool['name'] == 'remove_background':
                            tool['args']['refine'] = True
                    refined['thought'] = 'Refining the previous cutout with an alternative edge profile.'
                    return refined

    plan = _parse_text(text, media_type)
    question = _question_reply(text, media_type)
    explicit_question = (re.match(r"^(?:what|why|which|explain|difference|define|tell me about|is it|does|should|when|how)\b", text)
                         or re.fullmatch(r"\w+\s+(?:vs\.?|versus)\s+\w+[?!.]*", text))
    edit_request = plan.get('tools') or plan.get('clarification_needed')
    if question and (explicit_question or not edit_request) and not re.search(r"\b(?:what(?:'s| is) the (?:duration|length|resolution|sample rate|loudness|size|format)|"
                                  r"how (?:long|loud|big) is)\b", text):
        return {'tools': [], **question}

    if keep_order:
        plan['keep_order'] = True

    if _depth == 0 and previous and (plan.get('clarification_needed') or not plan.get('tools')) and len(text.split()) <= 10:
        earlier = parse_request(previous[0], media_context, None, _depth + 1)
        if earlier.get('clarification_needed') or (earlier.get('tools') and finalize_plan(
                earlier, previous[0], media_context, None, 'local').get('clarification_needed')):
            combined = parse_request(previous[0] + ' ' + text, media_context, None, _depth + 1)
            if combined.get('tools') and not combined.get('clarification_needed'):
                combined['thought'] = 'Completing the previous request with the details just provided.'
                return combined
    return plan


def _parse_text(text, media_type):
    tools, notes, unknown = [], [], []
    tracked = media_type
    for raw_clause in split_clauses(text):
        clause = Clause(raw_clause, tracked)
        try:
            detect_clause(clause, {})
        except Clarify as clarify:
            return {'tools': [], 'clarification_needed': True, 'reply': clarify.question,
                    'clarification_options': clarify.options, 'thought': 'Missing details; asking before editing.'}
        notes.extend(clause.notes)
        if not clause.found and not clause.notes:
            if _looks_like_request(raw_clause, clause):
                unknown.append(raw_clause)
            continue
        for _, name, args in clause.found:
            window = find_range(raw_clause) if name in SOUNDTRACK_EFFECTS else None
            if window:
                args = dict(args, range_start_sec=window[0], range_end_sec=window[1])
            tools.append({'name': name, 'args': dict(args)})
            tracked = _track(tracked, name, args)

    if unknown:
        options = SUGGESTIONS_BY_TYPE.get(media_type, SUGGESTIONS_BY_TYPE[None])
        quoted = '", "'.join(unknown[:2])
        return {'tools': [], 'clarification_needed': True,
                'reply': (f'I could not map "{quoted}" to an edit I can make. Please clarify that step '
                          '(I will not change your media until every step is clear).'),
                'clarification_options': options, 'thought': 'Part of the request is unclear; asking before editing.'}
    if not tools:
        if notes:
            return {'tools': [], 'notes': notes, 'reply': ' '.join(notes),
                    'suggested_actions': SUGGESTIONS_BY_TYPE.get(media_type, SUGGESTIONS_BY_TYPE[None]),
                    'thought': 'Requested operation is not available.'}
        return {'tools': [], 'reply': (
            'Tell me what you would like to do. For images: "Remove the background" or "Upscale 4x". '
            'For videos: "Trim from 2.3 to 5.3 seconds" or "Make a GIF". For audio: "Transcribe speech" or '
            '"Remove the background noise". You can chain several edits in one message.'),
            'suggested_actions': SUGGESTIONS_BY_TYPE.get(media_type, SUGGESTIONS_BY_TYPE[None]),
            'thought': 'General message without an edit request.'}
    return {'tools': tools, 'notes': notes, 'thought': ''}


def _track(media_type, name, args):
    if name == 'extract_audio' or (name == 'convert' and args.get('target_format') in ('mp3', 'wav', 'flac', 'ogg')):
        return 'audio'
    if name == 'convert_video_format' and args.get('target_format') == 'gif':
        return 'image'
    return media_type


def _looks_like_request(raw_clause, clause):
    remaining = raw_clause
    for start, end in sorted(clause.claimed, reverse=True):
        remaining = remaining[:start] + ' ' + remaining[end:]
    if re.fullmatch(r"\s*(?:export|download|save|render|send|share)(?: (?:it|this|that|the (?:result|file|video|audio|image)))?"
                    r"(?: for me| please)?\s*", remaining):
        return False
    words = re.findall(r"[a-z']+", remaining)
    meaningful = [word for word in words if word not in FILLER_WORDS]
    if not meaningful:
        return False
    return any(word in GENERIC_VERBS for word in meaningful) or len(meaningful) >= 3
