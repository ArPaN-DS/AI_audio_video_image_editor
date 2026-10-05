"""
Copilot Skills — expert-authored, tested editing recipes invoked with "@" (or picked automatically).

A skill is a package on disk:

    skills/<skill-id>/SKILL.md          (built-in, reviewed)
    user_skills/<skill-id>/SKILL.md     (saved or imported by the user; gitignored)

SKILL.md = YAML front matter (metadata, typed params, declarative ``steps`` that reference
ONLY registered Copilot tools) + a Markdown body with the expert procedure. No code is ever
executed from a skill. Every file is validated at load time; an invalid skill is skipped and
logged, never fatal.

A skill expands into ordinary plan steps which then go through
``agent_planner.finalize_plan`` (validation, media-type tracking, ordering rules, perception
policies, user overrides) and the normal executor — skills never bypass safety checks.

Progressive disclosure: only the ids + one-line descriptions of the few most relevant skills
are offered to the reasoning model; full bodies are read from disk only when requested.
"""

import copy
import difflib
import importlib.util
import logging
import os
import re
import threading
from collections import OrderedDict

import numpy as np
import yaml

import agent_planner

_log = logging.getLogger("agent_processor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(BASE_DIR, 'skills')
USER_SKILLS_DIR = os.environ.get('COPILOT_USER_SKILLS_DIR') or os.path.join(BASE_DIR, 'user_skills')

SKILL_ID_RE = re.compile(r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$')
PARAM_NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,23}$')
MAX_SKILL_BYTES = 32 * 1024
MAX_STEPS = 12
MAX_USER_SKILLS = 100
CATEGORIES = OrderedDict([
    ('cleanup', 'Clean up'), ('enhance', 'Enhance'), ('edit', 'Edit'), ('convert', 'Convert'),
    ('text', 'Transcribe & text'), ('separate', 'Separate'), ('voice', 'Voice'), ('workflows', 'Workflows'),
    ('mine', 'My skills'),
])
PARAM_TYPES = ('number', 'integer', 'seconds', 'string', 'boolean')
PSEUDO_TOOLS = {'_auto_clean': ('audio', 'video')}
FILLER_SEGMENT_RE = re.compile(
    r"(?:(?:hey|hi|ok|okay|please|pls|now|just|then|and)\s+)*(?:(?:can|could|would|will) you\s+)?(?:please\s+)?"
    r"(?:make|get|turn|do|run|apply|use|give me|try|with|using|via)?\s*(?:it|this|that|me|the (?:file|clip|video|audio|image|photo|track))?"
    r"\s*(?:into|to|a|an|the|for me|please|skill)?\s*(?:for me|please|thanks|thank you)?[\s.!?]*", re.IGNORECASE)
ARG_FILLER = {'please', 'pls', 'this', 'it', 'that', 'now', 'for', 'me', 'thanks', 'the', 'file', 'clip', 'audio',
              'video', 'image', 'photo', 'one', 'on', 'to', 'with', 'a', 'an', 'of', 'my', 'here', 'skill'}
NO_SKILL_RE = re.compile(r"\(\s*no skill\s*\)|\bwithout (?:a |any )?skills?\b|\bno skills?\b", re.IGNORECASE)
INVOCATION_RE = re.compile(r'(?<![\w@/.])@([a-z][a-z0-9-]{0,39})(?![\w-])')
ARG_SEPARATOR_RE = re.compile(r'\s*(?:,|;|\n|\band then\b|\bthen\b|\bafter that\b|\bfinally\b|\balso\b|\band\b|(?=@))\s*',
                              re.IGNORECASE)

_lock = threading.RLock()
_registry = None
_recent_plans = OrderedDict()


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════

class Skill:
    __slots__ = ('id', 'title', 'description', 'category', 'media_types', 'tags', 'params', 'steps', 'example',
                 'triggers', 'requires_tools', 'source', 'path', 'hidden', 'availability', '_vector')

    def __init__(self, **fields):
        for key in self.__slots__:
            setattr(self, key, fields.get(key))
        self._vector = None

    @property
    def tools(self):
        return [step['tool'] for step in self.steps]

    def public(self):
        return {
            'id': self.id, 'title': self.title, 'description': self.description,
            'category': self.category, 'category_label': CATEGORIES.get(self.category, self.category),
            'media_types': list(self.media_types), 'tags': list(self.tags),
            'params': [{'name': name, **{key: value for key, value in spec.items() if key != 'map'}}
                       for name, spec in self.params.items()],
            'example': self.example, 'steps': [_step_label(step) for step in self.steps],
            'availability': dict(self.availability), 'user': self.source == 'user',
        }


def _step_label(step):
    tool = step['tool']
    if tool == '_auto_clean':
        return 'Clean up the audio (noise, hum or music aware)'
    spec = agent_planner.TOOL_REGISTRY.get(tool)
    label = spec.label if spec else tool
    when = step.get('when') or {}
    if when.get('media'):
        label += f" (for {' or '.join(when['media'])})"
    return label


# ═══════════════════════════════════════════════════════════════════════════
#  LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class SkillError(ValueError):
    """Validation failure with a user-safe explanation."""


def split_front_matter(text):
    text = text.lstrip('﻿')
    match = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)$', text, re.DOTALL)
    if not match:
        raise SkillError('A skill file must start with a front matter block between two "---" lines.')
    return match.group(1), match.group(2)


def _check_private(value, where):
    texts = [value] if isinstance(value, str) else [str(item) for item in _flatten(value)]
    for text in texts:
        if agent_planner.scrub_private_terms(text) != text:
            raise SkillError(f'{where} must describe capabilities, not internal technology names.')


def _flatten(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _flatten(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif value is not None:
        yield value


def _validate_param(name, spec):
    if not PARAM_NAME_RE.match(str(name)):
        raise SkillError(f'Parameter "{name}" has an invalid name.')
    if not isinstance(spec, dict):
        raise SkillError(f'Parameter "{name}" must be a mapping.')
    kind = spec.get('type', 'string')
    if kind not in PARAM_TYPES:
        raise SkillError(f'Parameter "{name}" has an unknown type.')
    clean = {'type': kind}
    for key in ('description', 'default', 'enum', 'minimum', 'maximum', 'required', 'map', 'unit'):
        if key in spec:
            clean[key] = spec[key]
    if 'enum' in clean and (not isinstance(clean['enum'], list) or not clean['enum']):
        raise SkillError(f'Parameter "{name}" enum must be a non-empty list.')
    if 'map' in clean and not isinstance(clean['map'], dict):
        raise SkillError(f'Parameter "{name}" map must be a mapping.')
    clean['required'] = bool(clean.get('required', False))
    if clean.get('default') is not None:
        coerce_param(name, clean, clean['default'])
    return clean


def coerce_param(name, spec, value):
    """Coerce one user-supplied value; raises SkillError with a helpful message."""
    kind = spec['type']
    if kind == 'boolean':
        if isinstance(value, bool):
            result = value
        elif str(value).strip().lower() in ('true', 'yes', 'on', '1', name):
            result = True
        elif str(value).strip().lower() in ('false', 'no', 'off', '0'):
            result = False
        else:
            raise SkillError(f'"{name}" must be yes or no.')
    elif kind == 'seconds':
        result = agent_planner.coerce_seconds(value)
        if result is None:
            raise SkillError(f'"{name}" must be a time such as 5, 2.5s or 1:30.')
    elif kind in ('number', 'integer'):
        result = agent_planner.coerce_number(value)
        if result is None:
            raise SkillError(f'"{name}" must be a number.')
        if kind == 'integer':
            if float(result) != int(result):
                raise SkillError(f'"{name}" must be a whole number.')
            result = int(result)
    else:
        result = str(value).strip().lower()
    if 'enum' in spec:
        options = spec['enum']
        normalized = {str(option).lower(): option for option in options}
        if str(result).lower() not in normalized:
            raise SkillError(f'"{name}" must be one of: {", ".join(str(option) for option in options)}.')
        result = normalized[str(result).lower()]
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        if spec.get('minimum') is not None and result < spec['minimum']:
            raise SkillError(f'"{name}" must be at least {spec["minimum"]}.')
        if spec.get('maximum') is not None and result > spec['maximum']:
            raise SkillError(f'"{name}" must be at most {spec["maximum"]}.')
    return result


def _validate_step(step, params):
    if not isinstance(step, dict):
        raise SkillError('Each step must be a mapping with a "tool".')
    tool = step.get('tool')
    if tool not in agent_planner.TOOL_REGISTRY and tool not in PSEUDO_TOOLS:
        raise SkillError(f'Step tool "{tool}" is not an available editing tool.')
    args = step.get('args') or {}
    if not isinstance(args, dict):
        raise SkillError(f'Arguments for "{tool}" must be a mapping.')
    spec = agent_planner.TOOL_REGISTRY.get(tool)
    for key, value in args.items():
        if spec and key not in spec.params:
            raise SkillError(f'"{tool}" has no argument "{key}".')
        for reference in re.findall(r'\{\{\s*([a-z0-9_]+)\s*\}\}', str(value)):
            if reference not in params:
                raise SkillError(f'Step argument refers to unknown parameter "{reference}".')
        if isinstance(value, str) and '{{' in value and not re.fullmatch(r'\{\{\s*[a-z0-9_]+\s*\}\}', value):
            raise SkillError('A parameter reference must be the whole argument value, e.g. "{{scale}}".')
    when = step.get('when') or {}
    if not isinstance(when, dict) or set(when) - {'media', 'param', 'param_in', 'not_param'}:
        raise SkillError('"when" supports only media, param, param_in and not_param.')
    if when.get('media') and not set(when['media']) <= set(agent_planner.MEDIA_TYPES):
        raise SkillError('"when.media" lists an unknown media type.')
    for key in ('param', 'not_param'):
        if when.get(key) and when[key] not in params:
            raise SkillError(f'"when.{key}" refers to an unknown parameter.')
    for name in (when.get('param_in') or {}):
        if name not in params:
            raise SkillError('"when.param_in" refers to an unknown parameter.')
    return {'tool': tool, 'args': dict(args), 'when': dict(when)}


def parse_skill(text, source='builtin', path=None, expected_id=None):
    """Validate a SKILL.md document and return a Skill (raises SkillError)."""
    if len(text.encode('utf-8')) > MAX_SKILL_BYTES:
        raise SkillError('The skill file is too large (32 KB maximum).')
    head, _body = split_front_matter(text)
    try:
        meta = yaml.safe_load(head)
    except yaml.YAMLError:
        raise SkillError('The front matter is not valid YAML.')
    if not isinstance(meta, dict):
        raise SkillError('The front matter must be a mapping.')
    skill_id = str(meta.get('name') or '').strip().lower()
    if not SKILL_ID_RE.match(skill_id) or len(skill_id) > 40:
        raise SkillError('"name" must be a short lowercase id such as podcast-polish.')
    if expected_id and skill_id != expected_id:
        raise SkillError(f'"name" ({skill_id}) must match the folder name ({expected_id}).')
    title = str(meta.get('title') or skill_id.replace('-', ' ').capitalize()).strip()[:60]
    description = str(meta.get('description') or '').strip()
    if not 10 <= len(description) <= 320:
        raise SkillError('"description" must say when to use the skill (10-320 characters).')
    category = str(meta.get('category') or ('mine' if source == 'user' else 'workflows')).strip().lower()
    if category not in CATEGORIES:
        raise SkillError(f'"category" must be one of: {", ".join(CATEGORIES)}.')
    media_types = meta.get('media_types') or []
    if isinstance(media_types, str):
        media_types = [media_types]
    if not media_types or not set(media_types) <= set(agent_planner.MEDIA_TYPES):
        raise SkillError('"media_types" must list audio, video and/or image.')
    raw_params = meta.get('params') or {}
    if not isinstance(raw_params, dict) or len(raw_params) > 6:
        raise SkillError('"params" must be a mapping with at most 6 parameters.')
    params = OrderedDict((str(name), _validate_param(name, spec)) for name, spec in raw_params.items())
    raw_steps = meta.get('steps')
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_STEPS:
        raise SkillError(f'"steps" must list 1-{MAX_STEPS} tool steps.')
    steps = [_validate_step(step, params) for step in raw_steps]
    tags = [str(tag).strip().lower() for tag in (meta.get('tags') or []) if str(tag).strip()][:12]
    triggers = [str(item).strip().lower() for item in (meta.get('triggers') or []) if len(str(item).strip()) >= 6][:10]
    requires = [str(item) for item in (meta.get('requires_tools') or [])]
    example = str(meta.get('example') or f'@{skill_id}').strip()[:120]
    _check_private({'title': title, 'description': description, 'tags': tags, 'example': example,
                    'params': {name: spec.get('description', '') for name, spec in params.items()}}, 'Skill text')
    _check_private(_body, 'The skill procedure')
    return Skill(id=skill_id, title=title, description=description, category=category,
                 media_types=tuple(dict.fromkeys(media_types)), tags=tags, params=params, steps=steps,
                 example=example, triggers=triggers, requires_tools=requires, source=source, path=path,
                 hidden=False, availability={'state': 'ready', 'note': ''})


def _availability(skill):
    missing = [tool for tool in skill.requires_tools if tool not in agent_planner.TOOL_REGISTRY]
    if missing:
        return True, {'state': 'unavailable', 'note': 'Coming soon: this capability is not installed yet.'}
    if {'isolate_voice', 'remove_vocals'} & set(skill.tools) and importlib.util.find_spec('demucs') is None:
        return False, {'state': 'limited',
                       'note': 'Full voice/music separation is not installed; a lighter method is used on this computer.'}
    return False, {'state': 'ready', 'note': ''}


class SkillRegistry:
    def __init__(self, builtin_dir=None, user_dir=None):
        self.builtin_dir = builtin_dir if builtin_dir is not None else SKILLS_DIR
        self.user_dir = user_dir if user_dir is not None else USER_SKILLS_DIR
        self.skills = OrderedDict()
        self.errors = []
        self.load()

    def load(self):
        skills = OrderedDict()
        errors = []
        for directory, source in ((self.builtin_dir, 'builtin'), (self.user_dir, 'user')):
            if not directory or not os.path.isdir(directory):
                continue
            for folder in sorted(os.listdir(directory)):
                path = os.path.join(directory, folder, 'SKILL.md')
                if not os.path.isfile(path):
                    continue
                try:
                    with open(path, encoding='utf-8') as handle:
                        skill = parse_skill(handle.read(), source, path, expected_id=folder)
                    if skill.id in skills:
                        raise SkillError('A skill with this name already exists.')
                    skill.hidden, skill.availability = _availability(skill)
                    skills[skill.id] = skill
                except (SkillError, OSError, UnicodeDecodeError) as err:
                    errors.append((folder, str(err)))
                    _log.warning("Skipped invalid skill '%s': %s", folder, err)
        self.skills = skills
        self.errors = errors
        return self

    def get(self, skill_id, include_hidden=False):
        skill = self.skills.get(str(skill_id or '').lower())
        return skill if skill and (include_hidden or not skill.hidden) else None

    def visible(self, media_type=None):
        return [skill for skill in self.skills.values()
                if not skill.hidden and (not media_type or media_type in skill.media_types)]


def registry(reload=False):
    global _registry
    with _lock:
        if _registry is None:
            _registry = SkillRegistry()
        elif reload:
            _registry = SkillRegistry(_registry.builtin_dir, _registry.user_dir)
        return _registry


def configure(builtin_dir=SKILLS_DIR, user_dir=USER_SKILLS_DIR):
    """Point the registry at other folders (tests); returns the fresh registry."""
    global _registry, USER_SKILLS_DIR
    with _lock:
        USER_SKILLS_DIR = user_dir
        _registry = SkillRegistry(builtin_dir, user_dir)
        return _registry


def skill_body(skill_id):
    """Full procedure text of a skill, read from disk only when asked for (progressive disclosure)."""
    skill = registry().get(skill_id, include_hidden=False)
    if not skill or not skill.path:
        return None
    try:
        with open(skill.path, encoding='utf-8') as handle:
            return split_front_matter(handle.read())[1].strip()
    except (OSError, SkillError):
        return None


def catalog(media_type=None):
    media_type = media_type if media_type in agent_planner.MEDIA_TYPES else None
    return {'skills': [skill.public() for skill in registry().visible(media_type)],
            'categories': [{'id': key, 'label': label} for key, label in CATEGORIES.items()]}


def coverage():
    """Registered tool -> skills that use it (visible skills only)."""
    matrix = OrderedDict((name, []) for name in agent_planner.TOOL_REGISTRY)
    for skill in registry().visible():
        for tool in skill.tools:
            if tool == '_auto_clean':
                for target in ('reduce_noise', 'isolate_voice'):
                    matrix[target].append(skill.id)
            elif tool in matrix:
                matrix[tool].append(skill.id)
    return OrderedDict((tool, sorted(set(ids))) for tool, ids in matrix.items())


# ═══════════════════════════════════════════════════════════════════════════
#  RANKING (local hybrid lexical/semantic vectors, same features as Copilot Memory)
# ═══════════════════════════════════════════════════════════════════════════

def _embed(text):
    try:
        import agent_memory
        return agent_memory.lexical_embedding(text)
    except Exception:
        return None


def _skill_vector(skill):
    if skill._vector is None:
        skill._vector = _embed(' '.join([skill.id.replace('-', ' '), skill.title, skill.description,
                                         ' '.join(skill.tags), ' '.join(skill.triggers)]))
    return skill._vector


def rank_skills(query, media_type=None, k=3, min_score=0.12):
    """Most relevant visible skills for a request (ids + scores); empty when nothing is close."""
    query_vector = _embed(query or '')
    if query_vector is None:
        return []
    scored = []
    query_tokens = set(re.findall(r'[a-z0-9]+', (query or '').lower()))
    for skill in registry().visible(media_type if media_type in agent_planner.MEDIA_TYPES else None):
        vector = _skill_vector(skill)
        if vector is None:
            continue
        score = float(np.dot(query_vector, vector))
        score += 0.05 * len(query_tokens & set(skill.id.split('-')))
        scored.append((score, skill))
    scored.sort(key=lambda item: -item[0])
    return [(skill, round(score, 3)) for score, skill in scored[:k] if score >= min_score]


def search_skills(query, media_type=None, k=3):
    """Top-k relevant visible skills as [{'id', 'description'}] (one line each) for prompt budgeting."""
    return [{'id': skill.id, 'description': skill.description.split('. ')[0].rstrip('.')[:140]}
            for skill, _ in rank_skills(query, media_type, k)]


def prompt_hint(query, media_type=None, k=3):
    """Tiny prompt block for the reasoning model: ids + one-line descriptions of the top-k skills only."""
    if not rank_skills(query, media_type, k):
        return ''
    lines = [f"- @{item['id']}: {item['description']}" for item in search_skills(query, media_type, k)]
    return ('\nRelevant skills (a step may be {"skill": "<id>", "args": {}} to use one):\n' + '\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
#  INVOCATION PARSING & EXPANSION
# ═══════════════════════════════════════════════════════════════════════════

def find_invocations(prompt):
    """[(start, end, skill_id, arg_text)] for every "@skill args" in the prompt, in order."""
    text = prompt or ''
    found = []
    for match in INVOCATION_RE.finditer(text):
        rest = text[match.end():]
        separator = ARG_SEPARATOR_RE.search(rest)
        arg_text = rest[:separator.start()] if separator else rest
        end = match.end() + len(arg_text)
        found.append((match.start(), end, match.group(1), arg_text.strip()))
    return found


def parse_args(skill, arg_text, leftover=None):
    """
    Positional and key=value arguments -> validated params with defaults (raises SkillError).
    Words after all settings are filled go to ``leftover`` (free text such as "as-is") when a list is given.
    """
    values = {}
    names = list(skill.params)
    positional = [name for name in names if skill.params[name]['type'] != 'boolean']
    tokens = re.findall(r'[a-z_]+\s*=\s*"[^"]*"|[a-z_]+\s*=\s*\S+|"[^"]*"|\S+', arg_text or '', re.IGNORECASE)
    position = 0
    for token in tokens:
        key_value = re.match(r'([a-z_]+)\s*=\s*(.+)$', token, re.IGNORECASE)
        if key_value:
            name, raw = key_value.group(1).lower(), key_value.group(2).strip('"')
            if name not in skill.params:
                partial = [param for param in skill.params if name in param.split('_')]
                name = partial[0] if len(partial) == 1 else name
            if name not in skill.params:
                raise SkillError(f'@{skill.id} has no "{name}" setting.')
        else:
            raw = token.strip('"')
            if raw.lower().strip('.,!?') in ARG_FILLER:
                continue
            boolean = next((name for name in names if skill.params[name]['type'] == 'boolean'
                            and raw.lower() in (name, name.replace('_', '-'))), None)
            if boolean:
                values[boolean] = True
                continue
            enum_owner = next((name for name in positional if name not in values and 'enum' in skill.params[name]
                               and raw.lower() in [str(option).lower() for option in skill.params[name]['enum']]), None)
            if enum_owner:
                name = enum_owner
            else:
                while position < len(positional) and positional[position] in values:
                    position += 1
                if position >= len(positional):
                    if leftover is not None:
                        leftover.append(raw)
                        continue
                    raise SkillError(f'@{skill.id} got more values than it accepts ("{raw}").')
                name = positional[position]
        values[name] = coerce_param(name, skill.params[name], raw)
    for name, spec in skill.params.items():
        if name not in values:
            if spec.get('required'):
                raise SkillError(f'@{skill.id} needs a value for "{name}". Example: {skill.example}')
            values[name] = spec.get('default')
    return values


def _when_matches(when, params, media_type):
    if when.get('media') and media_type and media_type not in when['media']:
        return False
    if when.get('param') and not params.get(when['param']):
        return False
    if when.get('not_param') and params.get(when['not_param']):
        return False
    for name, allowed in (when.get('param_in') or {}).items():
        if params.get(name) not in allowed:
            return False
    return True


def expand(skill, params, media_type=None):
    """Concrete raw plan steps for a skill invocation (still to be validated by finalize_plan)."""
    steps = []
    tag = {'id': skill.id, 'title': skill.title}
    for step in skill.steps:
        if not _when_matches(step['when'], params, media_type):
            continue
        args = {}
        for key, value in step['args'].items():
            reference = re.fullmatch(r'\{\{\s*([a-z0-9_]+)\s*\}\}', value) if isinstance(value, str) else None
            if reference:
                name = reference.group(1)
                value = params.get(name)
                mapping = skill.params[name].get('map') or {}
                value = mapping.get(value, mapping.get(str(value), value))
                if value is None:
                    continue
            args[key] = copy.deepcopy(value)
        steps.append({'name': step['tool'], 'args': args, 'skill': dict(tag)})
    return steps


def _track(media_type, steps):
    for step in steps:
        spec = agent_planner.TOOL_REGISTRY.get(step['name'])
        if spec and spec.kind == 'edit':
            media_type = spec.output_type(step['args'], media_type) or media_type
        elif step['name'] in ('convert', 'extract_audio'):
            media_type = 'audio'
    return media_type


def _suggest_for_media(skill, media_type, limit=3):
    ranked = rank_skills(f'{skill.title} {skill.description}', media_type, k=limit, min_score=0.0)
    return [f'@{item.id}' for item, _ in ranked]


def close_matches(skill_id, media_type=None, limit=4):
    ids = [skill.id for skill in registry().visible(media_type)] or [skill.id for skill in registry().visible()]
    matches = difflib.get_close_matches(skill_id, ids, n=limit, cutoff=0.45)
    for skill, _ in rank_skills(skill_id.replace('-', ' '), media_type, k=limit, min_score=0.15):
        if skill.id not in matches:
            matches.append(skill.id)
    return matches[:limit]


def auto_select(prompt, media_type=None):
    """A skill whose trigger phrase appears in the request (longest trigger wins); None otherwise."""
    if not prompt or '@' in prompt or NO_SKILL_RE.search(prompt):
        return None
    text = prompt.lower()
    best = None
    for skill in registry().visible(media_type if media_type in agent_planner.MEDIA_TYPES else None):
        for trigger in skill.triggers:
            match = re.search(rf'(?<![\w-]){re.escape(trigger)}(?![\w-])', text)
            if match and (best is None or len(trigger) > best[1]):
                best = (skill, len(trigger), match.span())
    return best


def build_raw_plan(prompt, media_context=None, history=None, parse_text=None):
    """
    Raw plan for a message that uses skills (explicit "@id" or an automatic trigger), or None
    when no skill applies. Free text around invocations is planned by ``parse_text`` and kept
    in order, so "@youtube-ready then trim the first 5s" works.
    """
    media_type = agent_planner.MediaState.from_context(media_context).type
    prompt = prompt or ''
    automatic = None
    if not INVOCATION_RE.search(prompt):
        automatic = auto_select(prompt, media_type)
        if not automatic:
            return None
        skill, _, (start, end) = automatic
        prompt = f'{prompt[:start]}@{skill.id}{prompt[end:]}'
    invocations = find_invocations(prompt)
    if not invocations:
        return None

    tools, notes, thoughts = [], [], []
    cursor = 0
    tracked = media_type
    segments = []
    for start, end, skill_id, arg_text in invocations:
        segments.append(('text', prompt[cursor:start]))
        segments.append(('skill', (skill_id, arg_text)))
        cursor = end
    segments.append(('text', prompt[cursor:]))

    for kind, value in segments:
        if kind == 'text':
            text = re.sub(r'^\s*(?:,|;|and then|then|and|also|after that|finally)\s*|\s*(?:,|;|and then|then|and|also)\s*$',
                          '', value.strip(), flags=re.IGNORECASE).strip(' .,;')
            if not text or not parse_text or FILLER_SEGMENT_RE.fullmatch(text):
                continue
            sub_plan = parse_text(text, dict(media_context or {}, type=tracked) if tracked else media_context, history)
            if sub_plan.get('clarification_needed'):
                return sub_plan
            notes.extend(sub_plan.get('notes') or [])
            for step in sub_plan.get('tools') or []:
                tools.append(step)
                tracked = _track(tracked, [step])
            continue
        skill_id, arg_text = value
        skill = registry().get(skill_id)
        hidden = registry().get(skill_id, include_hidden=True)
        if not skill and hidden:
            return {'tools': [], 'clarification_needed': False,
                    'reply': f'@{hidden.id} ({hidden.title}) is coming soon: this capability is not installed yet.',
                    'suggested_actions': [], 'thought': 'Skill not available yet.'}
        if not skill:
            matches = close_matches(skill_id, media_type)
            listing = ', '.join(f'@{item}' for item in matches)
            return {'tools': [], 'clarification_needed': True,
                    'reply': (f'I do not know the skill @{skill_id}.' + (f' Did you mean {listing}?' if matches else '')
                              + ' Type @ to see the skills for this media.'),
                    'clarification_options': [f'@{item}' for item in matches][:3] or ['What can you do?'],
                    'thought': 'Unknown skill requested.'}
        if tracked and tracked not in skill.media_types:
            suggestions = _suggest_for_media(skill, tracked)
            return {'tools': [], 'clarification_needed': False,
                    'reply': (f'@{skill.id} works on {" or ".join(skill.media_types)}, but this is '
                              f'{"an image" if tracked == "image" else tracked}.'
                              + (f' For {tracked}, try {", ".join(suggestions)}.' if suggestions else '')),
                    'suggested_actions': suggestions or agent_planner.SUGGESTIONS_BY_TYPE.get(tracked, []),
                    'thought': 'Skill does not support this media type.'}
        extra_words = []
        try:
            params = parse_args(skill, arg_text, extra_words)
        except SkillError as err:
            return {'tools': [], 'clarification_needed': True, 'reply': str(err),
                    'clarification_options': [skill.example], 'thought': 'Skill settings need attention.'}
        expanded = expand(skill, params, tracked)
        tools.extend(expanded)
        tracked = _track(tracked, expanded)
        extra_text = ' '.join(extra_words).strip()
        if extra_text and parse_text and not FILLER_SEGMENT_RE.fullmatch(extra_text):
            sub_plan = parse_text(extra_text, dict(media_context or {}, type=tracked) if tracked else media_context, history)
            if sub_plan.get('clarification_needed'):
                return sub_plan
            for step in sub_plan.get('tools') or []:
                tools.append(step)
                tracked = _track(tracked, [step])
        thoughts.append(f'Skill @{skill.id}' + (' (picked automatically)' if automatic else ''))

    plan = {'tools': tools, 'notes': notes, 'thought': '; '.join(thoughts), 'keep_order': False}
    if automatic:
        plan['auto_skill'] = automatic[0].id
    return plan


def expand_model_steps(raw, media_context=None):
    """Replace {"skill": id, "args": {...}} steps proposed by the reasoning model with the skill's steps."""
    if not isinstance(raw, dict) or not isinstance(raw.get('tools'), list):
        return raw
    media_type = agent_planner.MediaState.from_context(media_context).type
    tools = []
    for step in raw['tools']:
        skill_id = step.get('skill') if isinstance(step, dict) and isinstance(step.get('skill'), str) else None
        skill = registry().get(skill_id.lstrip('@')) if skill_id else None
        if not skill:
            tools.append(step)
            continue
        arg_values = step.get('args') if isinstance(step.get('args'), dict) else {}
        try:
            params = parse_args(skill, ' '.join(f'{key}={value}' for key, value in arg_values.items()
                                                if not isinstance(value, (dict, list))))
        except SkillError:
            params = {name: spec.get('default') for name, spec in skill.params.items()}
        tools.extend(expand(skill, params, media_type))
    return {**raw, 'tools': tools}


# ═══════════════════════════════════════════════════════════════════════════
#  USER SKILLS: save / import / rename / delete
# ═══════════════════════════════════════════════════════════════════════════

SAVE_RE = re.compile(r'^\s*(?:please\s+)?save\s+(?:this|that|it|these steps|the (?:last )?(?:edit|plan|steps))\s+as\s+'
                     r'@?([a-z][a-z0-9-]{1,39})\s*(?:[-:—]\s*(.+))?$', re.IGNORECASE)
DELETE_RE = re.compile(r'^\s*(?:please\s+)?(?:delete|remove|forget)\s+(?:the\s+)?(?:my\s+)?(?:skill\s+)?@([a-z][a-z0-9-]{1,39})'
                       r'(?:\s+skill)?\s*\.?$', re.IGNORECASE)


def remember_plan(session_key, tools, media_type):
    """Keep the last successful validated plan per conversation (for "save this as @name")."""
    steps = [{'tool': step['name'], 'args': {key: value for key, value in (step.get('args') or {}).items()
                                             if value is not None}}
             for step in tools or [] if isinstance(step, dict) and step.get('name') in agent_planner.TOOL_REGISTRY
             and step.get('role', 'requested') in ('requested', 'post')]
    if not steps or media_type not in agent_planner.MEDIA_TYPES:
        return
    with _lock:
        _recent_plans[str(session_key or 'default')] = {'steps': steps, 'media_type': media_type}
        _recent_plans.move_to_end(str(session_key or 'default'))
        while len(_recent_plans) > 200:
            _recent_plans.popitem(last=False)


def _user_skill_path(skill_id):
    return os.path.join(USER_SKILLS_DIR, skill_id, 'SKILL.md')


def _write_user_skill(skill_id, text):
    folder = os.path.join(USER_SKILLS_DIR, skill_id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'SKILL.md'), 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def render_skill_md(meta, body):
    return '---\n' + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + '---\n\n' + body.strip() + '\n'


def save_user_skill(skill_id, steps, media_type, description=None, title=None):
    skill_id = (skill_id or '').strip().lower().lstrip('@')
    if not SKILL_ID_RE.match(skill_id) or len(skill_id) > 40:
        raise SkillError('Use a short lowercase name such as @my-podcast.')
    reg = registry()
    existing = reg.get(skill_id, include_hidden=True)
    if existing and existing.source != 'user':
        raise SkillError(f'@{skill_id} is a built-in skill; choose another name.')
    if not existing and len([s for s in reg.skills.values() if s.source == 'user']) >= MAX_USER_SKILLS:
        raise SkillError('You have reached the limit of saved skills; delete one first.')
    labels = ' → '.join(_step_label({'tool': step['tool']}) for step in steps)
    meta = {'name': skill_id, 'title': (title or skill_id.replace('-', ' ').capitalize())[:60],
            'description': (description or f'Your saved steps: {labels}.')[:300], 'category': 'mine',
            'media_types': [media_type], 'tags': ['saved'], 'example': f'@{skill_id}', 'steps': steps}
    text = render_skill_md(meta, f'Saved from a successful edit.\n\nSteps: {labels}.')
    parse_skill(text, 'user', expected_id=skill_id)            # validate before writing
    _write_user_skill(skill_id, text)
    registry(reload=True)
    return registry().get(skill_id, include_hidden=True)


def import_user_skill(text):
    skill = parse_skill(text, 'user')
    if skill.category != 'mine':
        meta_text, body = split_front_matter(text)
        meta = yaml.safe_load(meta_text)
        meta['category'] = 'mine'
        text = render_skill_md(meta, body)
        skill = parse_skill(text, 'user')
    existing = registry().get(skill.id, include_hidden=True)
    if existing and existing.source != 'user':
        raise SkillError(f'@{skill.id} is a built-in skill; rename the imported skill.')
    _write_user_skill(skill.id, text)
    registry(reload=True)
    return registry().get(skill.id, include_hidden=True)


def delete_user_skill(skill_id):
    skill = registry().get(skill_id, include_hidden=True)
    if not skill or skill.source != 'user':
        raise SkillError('Only your own saved skills can be deleted.')
    path = _user_skill_path(skill.id)
    if os.path.isfile(path):
        os.remove(path)
    try:
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass
    registry(reload=True)
    return True


def rename_user_skill(skill_id, new_id, title=None):
    skill = registry().get(skill_id, include_hidden=True)
    if not skill or skill.source != 'user':
        raise SkillError('Only your own saved skills can be renamed.')
    new_id = (new_id or '').strip().lower().lstrip('@')
    if not SKILL_ID_RE.match(new_id) or len(new_id) > 40:
        raise SkillError('Use a short lowercase name such as @my-podcast.')
    if new_id != skill.id and registry().get(new_id, include_hidden=True):
        raise SkillError(f'@{new_id} already exists.')
    with open(skill.path, encoding='utf-8') as handle:
        meta_text, body = split_front_matter(handle.read())
    meta = yaml.safe_load(meta_text)
    meta['name'] = new_id
    meta['example'] = f'@{new_id}'
    if title:
        meta['title'] = str(title)[:60]
    text = render_skill_md(meta, body)
    parse_skill(text, 'user', expected_id=new_id)
    _write_user_skill(new_id, text)
    if new_id != skill.id:
        delete_user_skill(skill.id)
    registry(reload=True)
    return registry().get(new_id, include_hidden=True)


def handle_command(message, session_key=None):
    """
    Chat commands for personal skills. Returns a chat-style response dict, or None when the
    message is not a skills command.
    """
    text = (message or '').strip()
    save = SAVE_RE.match(text)
    if save:
        with _lock:
            recent = _recent_plans.get(str(session_key or 'default'))
        if not recent:
            return {'reply': 'There is no finished edit in this conversation to save yet. Run an edit first, '
                             'then say "save this as @my-skill".', 'suggested_actions': []}
        try:
            skill = save_user_skill(save.group(1), recent['steps'], recent['media_type'], description=save.group(2))
        except SkillError as err:
            return {'reply': str(err), 'suggested_actions': []}
        return {'reply': f'Saved as @{skill.id}: {" → ".join(skill.public()["steps"])}. Type @{skill.id} to run it again.',
                'suggested_actions': [f'@{skill.id}'], 'skill': skill.public()}
    delete = DELETE_RE.match(text)
    if delete and registry().get(delete.group(1), include_hidden=True):
        try:
            delete_user_skill(delete.group(1))
        except SkillError as err:
            return {'reply': str(err), 'suggested_actions': []}
        return {'reply': f'Deleted @{delete.group(1).lower()}.', 'suggested_actions': []}
    return None
