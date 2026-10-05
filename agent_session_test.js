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
}

function media(filename) {
    return { filename, original_name: filename, type: 'image', size: 10, url: `/media/${filename}` };
}

function createHarness(messages = []) {
    const elements = new Map();
    const windowListeners = {};
    const storage = new Map([['avi_agent_sessions', JSON.stringify([
        { id: 'A', title: 'Chat A', messages, activeMedia: media('a.png') },
        { id: 'B', title: 'Chat B', messages: [], activeMedia: media('b.png') }
    ])]]);
    const requests = [];
    const sandbox = {
        console: { error() {}, warn() {} },
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
        window: {
            addEventListener(name, callback) { windowListeners[name] = callback; }
        },
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
            return new Promise((resolve, reject) => {
                requests.push({
                    url,
                    payload: url.endsWith('/chat') ? JSON.parse(options.body) : null,
                    respond(data, ok = true) {
                        resolve({ ok, status: ok ? 200 : 400, json: async () => data });
                    },
                    reject
                });
            });
        }
    };
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync('static/agent.js', 'utf8'), sandbox);

    const readSessions = () => JSON.parse(storage.get('avi_agent_sessions'));
    return {
        elements,
        requests,
        session(id) { return readSessions().find(session => session.id === id); },
        switchSession(id) {
            const index = readSessions().findIndex(session => session.id === id);
            elements.get('sessionsList').children[index].onclick();
        },
        deleteSession(id) {
            const index = readSessions().findIndex(session => session.id === id);
            elements.get('sessionsList').children[index].children[1].onclick({ stopPropagation() {} });
        },
        upload(filename) {
            windowListeners.drop({ preventDefault() {}, dataTransfer: { files: [{ name: filename, size: 10 }] } });
        },
        send(text) { sandbox.window.agentSendPrompt(text); }
    };
}

function editResult(filename) {
    return { status: 'success', reply: 'Done', output_file: filename, output_url: `/processed/${filename}` };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

async function run() {
    const history = createHarness();
    history.send('Brighten');
    assert.deepEqual(history.requests[0].payload.history, []);
    history.requests[0].respond({ reply: 'How much brighter?', clarification_needed: true });
    await settle();
    history.send('By 20 percent');
    assert.deepEqual(history.requests[1].payload.history, [
        { role: 'user', content: 'Brighten' },
        { role: 'assistant', content: 'How much brighter?' }
    ]);
    history.requests[1].respond(editResult('brightened.png'));
    await settle();
    history.send('Now crop');
    assert.equal(history.requests[2].payload.filename, 'brightened.png');
    assert.equal(history.requests[2].payload.history.at(-1).content, 'Done');
    history.requests[2].respond({ error: 'Edit failed' }, false);
    await settle();
    history.send('Try again');
    assert.equal(history.requests[3].payload.history.at(-1).role, 'assistant');
    assert.match(history.requests[3].payload.history.at(-1).content, /Edit failed/);
    history.requests[3].respond({ reply: 'Retry complete' });
    await settle();

    const priorMessages = Array.from({ length: 10 }, (_, index) => index % 2 === 0
        ? { role: 'user', content: `User ${index}` }
        : { role: 'agent', reply: `Reply ${index}` });
    const boundedHistory = createHarness(priorMessages);
    boundedHistory.send('New request');
    assert.equal(boundedHistory.requests[0].payload.history.length, 8);
    assert.equal(boundedHistory.requests[0].payload.history[0].content, 'User 2');
    assert.equal(boundedHistory.requests[0].payload.history.at(-1).content, 'Reply 9');
    boundedHistory.requests[0].respond({ reply: 'Done' });
    await settle();

    const chat = createHarness();
    const transcriptChat = createHarness();
    transcriptChat.send('Transcribe');
    transcriptChat.requests[0].respond({ ...editResult('source.wav'),
        clarification_needed: true, clarification_options: ["Keep the creator's voice"],
        suggested_actions: ["Don't change the picture"], execution_results: [{
        tool: 'transcribe_audio', status: 'success', data: {
            text: "It's <script>unsafe()</script> & private", segments: [],
            exports: [{ format: 'SRT', url: '/processed/transcript.srt' }]
        }
    }] });
    await settle();
    const transcriptMarkup = transcriptChat.elements.get('messagesStream').children.at(-1).innerHTML;
    assert.match(transcriptMarkup, /&lt;script&gt;unsafe\(\)&lt;\/script&gt;/);
    assert.match(transcriptMarkup, /transcript-copy-btn/);
    assert.doesNotMatch(transcriptMarkup, /clipboard.writeText\(decodeURIComponent/);
    assert.match(transcriptMarkup, /href="\/processed\/transcript.srt"/);
    assert.match(transcriptMarkup, /data-agent-prompt="Keep the creator&#039;s voice"/);
    assert.doesNotMatch(transcriptMarkup, /onclick="window.agentSendPrompt/);
    chat.send('Edit A');
    chat.switchSession('B');
    chat.requests[0].respond(editResult('edited_a.png'));
    await settle();
    assert.equal(chat.session('A').activeMedia.filename, 'edited_a.png');
    assert.equal(chat.session('A').messages.at(-1).reply, 'Done');
    assert.equal(chat.session('B').activeMedia.filename, 'b.png');
    assert.equal(chat.session('B').messages.length, 0);
    assert.equal(chat.elements.get('activeMediaName').textContent, 'b.png');
    chat.switchSession('A');
    chat.send('Follow up');
    assert.equal(chat.requests[1].payload.filename, 'edited_a.png');
    chat.requests[1].respond({ reply: 'Done' });
    await settle();

    const failedChat = createHarness();
    const replacedMedia = createHarness();
    replacedMedia.send('Edit original');
    replacedMedia.upload('new-selection.png');
    replacedMedia.requests[1].respond(media('new-selection.png'));
    await settle();
    replacedMedia.requests[0].respond(editResult('older-edit.png'));
    await settle();
    assert.equal(replacedMedia.session('A').activeMedia.filename, 'new-selection.png');
    assert.equal(replacedMedia.session('A').messages.at(-1).output_file, 'older-edit.png');
    const audioExport = createHarness();
    audioExport.send('Export audio');
    audioExport.requests[0].respond(editResult('export.flac'));
    await settle();
    assert.equal(audioExport.session('A').activeMedia.type, 'audio');
    failedChat.send('Edit A');
    failedChat.switchSession('B');
    failedChat.requests[0].reject(new Error('Network unavailable'));
    await settle();
    assert.match(failedChat.session('A').messages.at(-1).reply, /Network unavailable/);
    assert.equal(failedChat.session('B').messages.length, 0);
    assert.equal(failedChat.elements.get('activeMediaName').textContent, 'b.png');

    const deletedChat = createHarness();
    deletedChat.send('Edit A');
    deletedChat.deleteSession('A');
    deletedChat.requests[0].respond(editResult('deleted_a.png'));
    await settle();
    assert.equal(deletedChat.session('A'), undefined);
    assert.equal(deletedChat.session('B').activeMedia.filename, 'b.png');

    const uploads = createHarness();
    uploads.upload('first.png');
    uploads.upload('second.png');
    uploads.requests[1].respond(media('second.png'));
    await settle();
    uploads.requests[0].respond(media('first.png'));
    await settle();
    assert.equal(uploads.session('A').activeMedia.filename, 'second.png');
    assert.equal(uploads.elements.get('pendingFileName').textContent, 'second.png');

    const staleFailure = createHarness();
    staleFailure.upload('first.png');
    staleFailure.upload('second.png');
    staleFailure.requests[0].reject(new Error('Old upload failed'));
    await settle();
    assert.equal(staleFailure.elements.get('pendingFileName').textContent, 'Uploading second.png...');
    assert.equal(staleFailure.elements.get('pendingAttachmentCard').style.display, 'flex');
    staleFailure.requests[1].respond(media('second.png'));
    await settle();

    const failedUpload = createHarness();
    failedUpload.upload('failed.png');
    failedUpload.requests[0].respond({ error: 'Upload rejected' }, false);
    await settle();
    assert.equal(failedUpload.session('A').activeMedia.filename, 'a.png');
    assert.equal(failedUpload.elements.get('pendingFileName').textContent, 'a.png');
    assert.equal(failedUpload.elements.get('pendingAttachmentCard').style.display, 'flex');

    const backgroundFailure = createHarness();
    backgroundFailure.upload('failed_a.png');
    backgroundFailure.switchSession('B');
    backgroundFailure.requests[0].reject(new Error('Upload failed'));
    await settle();
    assert.equal(backgroundFailure.elements.get('pendingFileName').textContent, 'b.png');
    assert.equal(backgroundFailure.elements.get('toastContainer')?.children.length || 0, 0);

    const returnedUpload = createHarness();
    returnedUpload.upload('returned_a.png');
    returnedUpload.switchSession('B');
    returnedUpload.switchSession('A');
    returnedUpload.requests[0].respond(media('returned_a.png'));
    await settle();
    assert.equal(returnedUpload.session('A').activeMedia.filename, 'returned_a.png');
    assert.equal(returnedUpload.elements.get('activeMediaName').textContent, 'returned_a.png');

    const sessionUploads = createHarness();
    sessionUploads.upload('new_a.png');
    sessionUploads.switchSession('B');
    sessionUploads.upload('new_b.png');
    sessionUploads.requests[0].respond(media('new_a.png'));
    await settle();
    assert.equal(sessionUploads.session('A').activeMedia.filename, 'new_a.png');
    assert.equal(sessionUploads.session('B').activeMedia.filename, 'b.png');
    assert.equal(sessionUploads.elements.get('pendingFileName').textContent, 'Uploading new_b.png...');
    sessionUploads.requests[1].respond(media('new_b.png'));
    await settle();
    assert.equal(sessionUploads.session('B').activeMedia.filename, 'new_b.png');

    const removedUpload = createHarness();
    removedUpload.upload('removed.png');
    removedUpload.elements.get('removeActiveMediaBtn').onclick();
    removedUpload.requests[0].respond(media('removed.png'));
    await settle();
    assert.equal(removedUpload.session('A').activeMedia, null);

    const deletedUpload = createHarness();
    deletedUpload.upload('deleted.png');
    deletedUpload.deleteSession('A');
    deletedUpload.requests[0].respond(media('deleted.png'));
    await settle();
    assert.equal(deletedUpload.session('A'), undefined);
    assert.equal(deletedUpload.session('B').activeMedia.filename, 'b.png');

    console.log('Standalone agent session, history, and upload regression checks passed.');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
