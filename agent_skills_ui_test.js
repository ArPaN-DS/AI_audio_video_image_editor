// Tests for static/agent_skills.js: filtering, "@" trigger detection, keyboard navigation,
// insertion, media-type filtering and the Skills library (search, chips, Use, Esc).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class FakeElement {
    constructor(tag = 'div') {
        this.tagName = tag;
        this.attributes = {};
        this.listeners = {};
        this.children = [];
        this.html = '';
        this.hidden = false;
        this.value = '';
        this.selectionStart = 0;
        this.queries = {};
        this.classes = new Set();
        this.classList = {
            add: name => this.classes.add(name), remove: name => this.classes.delete(name),
            contains: name => this.classes.has(name)
        };
    }
    set innerHTML(value) { this.html = value; }
    get innerHTML() { return this.html; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return this.attributes[name] ?? null; }
    removeAttribute(name) { delete this.attributes[name]; }
    appendChild(child) { this.children.push(child); child.parentNode = this; }
    addEventListener(name, callback, capture) { (this.listeners[name] ||= []).push({ callback, capture: !!capture }); }
    dispatch(name, event) { (this.listeners[name] || []).forEach(item => item.callback(event)); }
    querySelector(selector) { return (this.queries[selector] ||= new FakeElement()); }
    querySelectorAll() { return []; }
    setSelectionRange(start) { this.selectionStart = start; }
    focus() { this.focused = true; }
}

const catalog = {
    skills: [
        { id: 'podcast-polish', title: 'Podcast polish', description: 'Publish-ready podcast MP3 with voice cleanup.',
          category: 'workflows', media_types: ['audio', 'video'], tags: ['podcast'], params: [], steps: ['Enhance voice'],
          example: '@podcast-polish', availability: { state: 'ready' } },
        { id: 'loudness', title: 'Loudness for a platform', description: 'Hit a loudness target such as -14 LUFS.',
          category: 'enhance', media_types: ['audio', 'video'], tags: ['lufs'],
          params: [{ name: 'target', type: 'string', enum: ['youtube', 'podcast'], default: 'youtube' }],
          steps: ['Normalize'], example: '@loudness podcast', availability: { state: 'ready' } },
        { id: 'cutout', title: 'Background removal', description: 'Remove the background from a photo.',
          category: 'separate', media_types: ['image'], tags: ['transparent'], params: [], steps: ['Remove background'],
          example: '@cutout', availability: { state: 'ready' } },
        { id: 'my-podcast', title: 'My podcast', description: 'Your saved steps: Trim then Normalize.',
          category: 'mine', media_types: ['audio'], tags: ['saved'], params: [], steps: ['Trim', 'Normalize'],
          example: '@my-podcast', availability: { state: 'ready' }, user: true },
        { id: 'BAD id', title: 'x', description: 'invalid' }
    ],
    categories: [{ id: 'enhance', label: 'Enhance' }, { id: 'workflows', label: 'Workflows' }, { id: 'mine', label: 'My skills' }]
};

const body = new FakeElement('body');
const sandbox = {
    console,
    document: {
        body, activeElement: null,
        createElement: tag => new FakeElement(tag)
    },
    fetch: async () => ({ json: async () => catalog }),
    setTimeout: () => 0,
    Promise
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('static/agent_skills.js', 'utf8'), sandbox);
const Skills = sandbox.AgentSkills;
const same = (actual, expected, message) => assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected, message);
const settle = () => new Promise(resolve => setImmediate(resolve));

async function run() {
    // 1. Pure helpers
    const skills = catalog.skills;
    same(Skills.filterSkills(skills, '').map(s => s.id), ['podcast-polish', 'loudness', 'cutout', 'my-podcast']);
    same(Skills.filterSkills(skills, 'pod').map(s => s.id).slice(0, 1), ['podcast-polish']);
    same(Skills.filterSkills(skills, '', { mediaType: 'image' }).map(s => s.id), ['cutout']);
    same(Skills.filterSkills(skills, 'lufs').map(s => s.id), ['loudness']);
    same(Skills.filterSkills(skills, '', { category: 'enhance' }).map(s => s.id), ['loudness']);
    assert.equal(Skills.findTrigger('make it @pod', 12).query, 'pod');
    assert.equal(Skills.findTrigger('email me@pod', 12), null, 'an @ inside a word is not a trigger');
    assert.equal(Skills.findTrigger('no trigger here', 15), null);
    const trigger = Skills.findTrigger('then @lou please', 9);
    same({ ...Skills.applyCompletion('then @lou please', trigger, 'loudness') },
                     { text: 'then @loudness please', caret: 15 });
    assert.equal(Skills.applyCompletion('trim 2 8', null, 'fade').text, 'trim 2 8 @fade ');
    same([...Skills.paramsSummary(catalog.skills[1])], ['target (youtube | podcast), default youtube']);

    // 2. Autocomplete: opens on "@", filters by media, keyboard navigation, Enter inserts, Esc closes.
    Skills.setCatalog(catalog);
    const host = new FakeElement();
    const textarea = new FakeElement('textarea');
    host.appendChild(textarea);
    let media = 'audio';
    let libraryOpened = 0;
    const auto = Skills.attachAutocomplete(textarea, { getMediaType: () => media, onOpenLibrary: () => libraryOpened++ });
    assert.ok(textarea.listeners.keydown.some(item => item.capture), 'keydown is handled in the capture phase');
    assert.equal(textarea.getAttribute('aria-autocomplete'), 'list');
    textarea.value = 'Please @';
    textarea.selectionStart = textarea.value.length;
    textarea.dispatch('input', {});
    await settle();
    assert.equal(auto.state.open, true);
    assert.equal(textarea.getAttribute('aria-expanded'), 'true');
    same(auto.state.items.map(s => s.id), ['podcast-polish', 'loudness', 'my-podcast'], 'image-only skills are hidden for audio');
    assert.match(auto.popup.innerHTML, /role="option"[^>]*data-skill-id="podcast-polish" aria-selected="true"/);
    assert.equal(textarea.getAttribute('aria-activedescendant'), `${auto.popup.id}_opt0`);

    const key = (name, extra = {}) => {
        const event = { key: name, shiftKey: false, isComposing: false, prevented: false, stopped: false,
                        preventDefault() { this.prevented = true; }, stopImmediatePropagation() { this.stopped = true; }, ...extra };
        textarea.listeners.keydown.forEach(item => item.callback(event));
        return event;
    };
    key('ArrowDown');
    assert.equal(auto.state.active, 1);
    assert.equal(textarea.getAttribute('aria-activedescendant'), `${auto.popup.id}_opt1`);
    key('ArrowUp'); key('ArrowUp');
    assert.equal(auto.state.active, 2, 'navigation wraps around');
    const enter = key('Enter');
    assert.equal(enter.prevented, true, 'Enter is consumed so the message is not sent');
    assert.equal(enter.stopped, true);
    assert.equal(textarea.value, 'Please @my-podcast ');
    assert.equal(auto.state.open, false);
    assert.equal(textarea.getAttribute('aria-expanded'), 'false');

    textarea.value = '@lo';
    textarea.selectionStart = 3;
    textarea.dispatch('input', {});
    await settle();
    same(auto.state.items.map(s => s.id), ['loudness']);
    const tab = key('Tab');
    assert.equal(tab.prevented, true);
    assert.equal(textarea.value, '@loudness ');

    textarea.value = '@';
    textarea.selectionStart = 1;
    textarea.dispatch('input', {});
    await settle();
    const escape = key('Escape');
    assert.equal(escape.prevented, true);
    assert.equal(auto.state.open, false);
    const plainEnter = key('Enter');
    assert.equal(plainEnter.prevented, false, 'a closed popup never swallows Enter');

    media = 'image';
    textarea.value = '@';
    textarea.dispatch('input', {});
    await settle();
    same(auto.state.items.map(s => s.id), ['cutout']);
    assert.match(auto.popup.innerHTML, /Browse all skills/);
    const browse = { closest: selector => (selector === '[data-skills-browse]' ? browse : null) };
    auto.popup.dispatch('mousedown', { target: browse, preventDefault() {} });
    assert.equal(libraryOpened, 1);

    // 3. Library: media preset from context, search, chips, "My skills" tab, Use inserts into the composer, Esc closes.
    const used = [];
    const composer = new FakeElement('textarea');
    composer.value = 'trim 1 2 then ';
    composer.selectionStart = composer.value.length;
    const results = await Skills.openLibrary({ getMediaType: () => 'audio', onUse: id => { used.push(id); Skills.insertSkill(composer, id); } });
    same(results.map(s => s.id), ['podcast-polish', 'loudness', 'my-podcast']);
    const state = Skills._libraryState;
    assert.equal(state.media, 'audio');
    state.query = 'lufs';
    same(Skills.libraryResults().map(s => s.id), ['loudness']);
    state.query = '';
    state.tab = 'mine';
    same(Skills.libraryResults().map(s => s.id), ['my-podcast']);
    state.tab = 'all';
    state.category = 'workflows';
    same(Skills.libraryResults().map(s => s.id), ['podcast-polish']);
    state.category = 'all';
    const backdrop = body.children.at(-1);
    assert.equal(backdrop.hidden, false);
    assert.match(backdrop.innerHTML, /role="dialog" aria-modal="true"/);
    const searchTarget = { getAttribute: name => (name === 'data-skills-search' ? '1' : null) };
    const keyEvent = (name) => ({ key: name, target: searchTarget, prevented: false, preventDefault() { this.prevented = true; } });
    backdrop.dispatch('keydown', keyEvent('ArrowDown'));
    assert.equal(state.selected, 'loudness');
    backdrop.dispatch('keydown', keyEvent('Enter'));
    same(used, ['loudness']);
    assert.equal(composer.value, 'trim 1 2 then @loudness ');
    assert.equal(backdrop.hidden, true, 'Use closes the library');
    await Skills.openLibrary({ getMediaType: () => null, onUse() {} });
    assert.equal(state.media, 'all');
    backdrop.dispatch('keydown', keyEvent('Escape'));
    assert.equal(backdrop.hidden, true);
    assert.doesNotMatch(JSON.stringify(Skills.filterSkills(catalog.skills, '')), /BAD id/);

    console.log('agent_skills_ui_test: all checks passed');
}

run().catch(error => {
    console.error(error);
    process.exit(1);
});
