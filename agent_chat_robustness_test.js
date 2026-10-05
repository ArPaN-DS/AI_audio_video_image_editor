// Regression tests for the Copilot workspace (static/agent.js):
//  * malformed server responses or stored sessions never break rendering,
//  * an empty prompt is never sent (even with media attached),
//  * Split Compare compares the edit's input with its output (not output with itself),
//  * side-output artifacts (e.g. thumbnails) get their own download link.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
    constructor() {
        this.style = {};
        this.value = '';
        this.textContent = '';
        this.children = [];
        this.listeners = {};
        this.classes = new Set();
        this.dataset = {};
        this.classList = {
            add: name => this.classes.add(name),
            remove: name => this.classes.delete(name),
            contains: name => this.classes.has(name),
            toggle: name => this.classes.has(name) ? this.classes.delete(name) : this.classes.add(name)
        };
    }
    set innerHTML(value) { this.html = value; this.children = []; }
    get innerHTML() { return this.html || ''; }
    appendChild(child) { this.children.push(child); }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    querySelector() { return null; }
    querySelectorAll() { return []; }
    focus() {}
    remove() {}
    click() { this.clicks = (this.clicks || 0) + 1; }
    setAttribute(name, value) { this[name] = value; }
    getAttribute(name) { return this[name]; }
}

const media = filename => ({ filename, original_name: filename, type: 'image', size: 10, url: `/media/${filename}` });

function createHarness(messages = []) {
    const elements = new Map();
    const storage = new Map([['avi_agent_sessions', JSON.stringify([
        { id: 'A', title: 'Chat A', messages, activeMedia: media('a.png') }
    ])]]);
    const requests = [];
    const errors = [];
    const sandbox = {
        console: { error: (...args) => errors.push(args), warn() {} },
        document: {
            readyState: 'complete',
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, new Element());
                return elements.get(id);
            },
            createElement() { return new Element(); },
            querySelectorAll() { return []; },
            addEventListener() {}
        },
        window: { addEventListener() {} },
        localStorage: {
            getItem: key => storage.get(key),
            setItem: (key, value) => storage.set(key, value),
            removeItem: key => storage.delete(key)
        },
        FormData: class { append() {} },
        setInterval() { return 1; },
        clearInterval() {},
        setTimeout() { return 1; },
        clearTimeout() {},
        fetch(url, options) {
            return new Promise(resolve => {
                requests.push({
                    url,
                    payload: url.endsWith('/chat') ? JSON.parse(options.body) : null,
                    respond(data) { resolve({ ok: true, status: 200, json: async () => data }); }
                });
            });
        }
    };
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync('static/agent.js', 'utf8'), sandbox);
    return {
        elements, requests, errors,
        session: () => JSON.parse(storage.get('avi_agent_sessions')).find(item => item.id === 'A'),
        send: text => sandbox.window.agentSendPrompt(text),
        el: id => sandbox.document.getElementById(id),
        key(options) {
            const event = { key: 'Enter', shiftKey: false, isComposing: false, defaultPrevented: false, prevented: false,
                            preventDefault() { this.prevented = true; }, ...options };
            elements.get('chatInput').listeners.keydown(event);
            return event;
        },
        lastMarkup: () => elements.get('messagesStream').children.at(-1).innerHTML
    };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

async function run() {
    // 1. Malformed response fields are normalized instead of crashing the renderer.
    const malformed = createHarness();
    malformed.send('Normalize');
    malformed.requests[0].respond({
        status: 'success', reply: { text: 'object reply' }, thought: ['x'], suggested_actions: 'Upscale 2x',
        clarification_needed: 'false', clarification_options: 7, execution_results: 'broken', artifacts: 'nope',
        delegated_subagent: '<img src=x onerror=alert(1)>'
    });
    await settle();
    const stored = malformed.session().messages.at(-1);
    assert.equal(stored.reply, '');
    assert.deepEqual(stored.suggested_actions, ['Upscale 2x']);
    assert.equal(stored.clarification_needed, false);
    assert.deepEqual(stored.execution_results, []);
    const markup = malformed.lastMarkup();
    assert.match(markup, /data-agent-prompt="Upscale 2x"/);
    assert.doesNotMatch(markup, /onerror=alert/);

    // 2. Sessions stored by older builds with bad shapes still render.
    const legacy = createHarness([
        { role: 'user', content: 'hi' },
        { role: 'agent', reply: 42, suggested_actions: 'Trim', clarification_needed: true, clarification_options: 'A',
          execution_results: { tool: 'x' }, output_url: 'javascript:alert(1)' },
        null,
        { role: 'agent', reply: 'ok', execution_results: [{ tool: 'transcribe_audio', data: { segments: [{ start: 'x', text: 5 }] } }],
          output_url: '/media/a.wav' }
    ]);
    assert.equal(legacy.errors.length, 0, JSON.stringify(legacy.errors));
    const legacyMarkup = legacy.elements.get('messagesStream').children.map(child => child.innerHTML).join('');
    assert.doesNotMatch(legacyMarkup, /javascript:alert/);
    assert.match(legacyMarkup, /data-agent-prompt="Trim"/);

    // 3. Empty prompts are never sent, even with media attached.
    const empty = createHarness();
    empty.send('   ');
    empty.send('');
    assert.equal(empty.requests.length, 0);

    // 4. Split Compare uses the edit's input media, not the newly selected output.
    const compare = createHarness();
    compare.send('Boost clarity');
    compare.requests[0].respond({
        status: 'success', reply: 'Done', output_file: 'clarity.png', output_url: '/processed/clarity.png',
        artifacts: [{ tool: 'extract_frame', output_file: 'thumb.jpg', output_url: '/processed/thumb.jpg' }]
    });
    await settle();
    assert.equal(compare.session().activeMedia.filename, 'clarity.png');
    const compareMarkup = compare.lastMarkup();
    assert.match(compareMarkup, /agentToggleCompare\(this, '\/media\/a\.png', '\/processed\/clarity\.png'\)/);
    assert.match(compareMarkup, /href="\/processed\/thumb\.jpg" download/);
    assert.equal(compare.session().messages.at(-1).input_url, '/media/a.png');

    // 5. Composer wiring: send button, Enter, Shift+Enter, IME composition, empty input, in-flight guard.
    const composer = createHarness();
    const input = composer.el('chatInput');
    assert.equal(typeof composer.el('sendBtn').listeners.click, 'function', 'send button must be wired');
    assert.equal(typeof input.listeners.keydown, 'function', 'Enter-to-send must be wired');
    input.value = 'Trim from 1 to 2 seconds';
    composer.el('sendBtn').listeners.click({ preventDefault() {} });
    assert.equal(composer.requests.length, 1);
    assert.equal(composer.requests[0].payload.message, 'Trim from 1 to 2 seconds');
    assert.equal(input.value, '');
    input.value = 'Second while busy';
    composer.key({});
    assert.equal(composer.requests.length, 1, 'no send while a request is processing');
    composer.requests[0].respond({ status: 'success', reply: 'Done' });
    await settle();
    input.value = 'Normalize';
    const shifted = composer.key({ shiftKey: true });
    assert.equal(shifted.prevented, false, 'Shift+Enter keeps the newline');
    assert.equal(composer.requests.length, 1);
    composer.key({ isComposing: true });
    assert.equal(composer.requests.length, 1, 'IME composition never sends');
    const sent = composer.key({});
    assert.equal(sent.prevented, true);
    assert.equal(composer.requests.length, 2);
    assert.equal(composer.requests[1].payload.message, 'Normalize');
    composer.requests[1].respond({ status: 'success', reply: 'Done' });
    await settle();
    input.value = '   ';
    composer.key({});
    composer.el('sendBtn').listeners.click({ preventDefault() {} });
    assert.equal(composer.requests.length, 2, 'empty input is never sent');

    // 6. Attach flow: attach button opens the picker; choosing a file uploads it; remove clears it.
    const attach = createHarness();
    const picker = attach.el('filePicker');
    attach.el('attachBtn').listeners.click();
    assert.equal(picker.clicks, 1);
    picker.files = [{ name: 'clip.wav', size: 2048 }];
    picker.listeners.change();
    assert.equal(attach.requests.length, 1);
    assert.match(attach.requests[0].url, /\/api\/agent\/upload$/);
    assert.equal(picker.value, '');
    attach.requests[0].respond({ ...media('clip.wav'), type: 'audio', status: 'success' });
    await settle();
    assert.equal(attach.session().activeMedia.filename, 'clip.wav');
    attach.el('removePendingBtn').listeners.click();
    assert.equal(attach.session().activeMedia, null);

    console.log('agent_chat_robustness_test: all checks passed');
}

run().catch(error => {
    console.error(error);
    process.exit(1);
});
