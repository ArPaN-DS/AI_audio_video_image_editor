const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const listeners = {};
function element() {
    return { value: '', children: [], classList: { add() {}, remove() {}, contains() { return false; } },
        addEventListener(name, handler) { this[name] = handler; },
        appendChild(child) { this.children.push(child); }, querySelector() { return null; }, remove() {} };
}
const elements = Object.fromEntries(['aiAgentDrawer', 'agentInputText', 'btnSendAgent', 'agentChatHistory']
    .map(id => [id, element()]));
const player = { load() {} };
const requests = [];
const sandbox = {
    window: {}, console, setInterval() {}, clearInterval() {},
    FormData: class { append() {} },
    document: {
        addEventListener(name, callback) { (listeners[name] ||= []).push(callback); },
        getElementById(id) { return id === 'masterVideoPlayer' ? player : elements[id] || null; },
        createElement: element,
        querySelectorAll: () => []
    },
    fetch: async (url, options) => {
        requests.push({ url, options });
        if (url === '/api/agent/chat') return { json: async () => ({ status: 'success', reply: 'Ready.' }) };
        if (url.startsWith('/processed/')) return { ok: true, blob: async () => ({ type: 'video/mp4' }) };
        return { ok: true, json: async () => ({ id: 'edited.mp4', url: '/media/edited.mp4', has_video: true, duration: 0.6 }) };
    }
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('static/js/studio_core.js', 'utf8') + '\nglobalThis.CoreClass = StudioCore;', sandbox);

(async () => {
    const core = Object.create(sandbox.CoreClass.prototype);
    const video = { id: 'source.mp4', has_video: true, duration: 1, name: 'Source' };
    const image = { id: 'photo.png', type: 'image' };
    core.project = { workspace: 'video', mediaBin: [video, image] };
    core.renderMediaBin = () => {};
    core.switchWorkspace = workspace => { core.project.workspace = workspace; };
    sandbox.window.studioCore = core;
    sandbox.window.videoStudio = {
        selectedClip: { mediaId: video.id }, clips: [],
        addClipToTimeline(media) { this.clips.push({ mediaId: media.id, start: 1 }); },
        selectClip(clip) { this.selectedClip = clip; },
        preview: { seek(time) { this.time = time; } }
    };
    sandbox.window.imageStudio = { selectedLayer: { mediaId: image.id } };
    sandbox.window.audioStudio = { currentAudioMedia: { id: 'audio.wav' } };
    assert.equal(core.getActiveMedia(), video);
    core.project.workspace = 'image';
    assert.equal(core.getActiveMedia(), image);
    core.project.workspace = 'audio';
    assert.equal(core.getActiveMedia().id, 'audio.wav');
    core.project.workspace = 'video';

    vm.runInContext(fs.readFileSync('static/js/ai_agent.js', 'utf8'), sandbox);
    listeners.DOMContentLoaded.at(-1)();
    elements.agentInputText.value = 'Trim from 0.2 to 0.8 seconds';
    await elements.btnSendAgent.click();
    const first = JSON.parse(requests.at(-1).options.body);
    assert.equal(first.filename, video.id);
    assert.equal(first.context.active_workspace, 'video');
    assert.equal(first.context.type, 'video');
    assert.deepEqual(first.history, []);
    elements.agentInputText.value = 'Now normalize';
    await elements.btnSendAgent.click();
    const second = JSON.parse(requests.at(-1).options.body);
    assert.equal(second.history[1].role, 'assistant');
    assert.equal(second.history[1].content, 'Ready.');
    assert.equal(second.history.length, 2);
    sandbox.fetch = async () => ({ json: async () => ({ status: 'success', reply: 'Transcribed.',
        execution_results: [{ tool: 'transcribe_audio', status: 'success', data: {
            text: "It's <script>unsafe()</script> & private", segments: [],
            exports: [{ format: 'TXT', url: '/processed/transcript.txt' }, { format: 'TXT', url: 'https://example.com/private.txt' }]
        } }]
    }) });
    elements.agentInputText.value = 'Transcribe';
    await elements.btnSendAgent.click();
    const transcriptMarkup = elements.agentChatHistory.children.at(-1).innerHTML;
    assert.match(transcriptMarkup, /&lt;script&gt;unsafe\(\)&lt;\/script&gt;/);
    assert.match(transcriptMarkup, /href="\/processed\/transcript.txt"/);
    assert.doesNotMatch(transcriptMarkup, /https:\/\/example.com/);
    sandbox.fetch = async (url, options) => {
        requests.push({ url, options });
        if (url.startsWith('/processed/')) return { ok: true, blob: async () => ({ type: 'video/mp4' }) };
        return { ok: true, json: async () => ({ id: 'edited.mp4', url: '/media/edited.mp4', has_video: true, duration: 0.6 }) };
    };

    await core.loadProcessedMedia('/processed/result.mp4');
    assert.equal(core.getActiveMedia().id, 'edited.mp4');
    assert.equal(player.src, '/media/edited.mp4');
    assert.equal(sandbox.window.videoStudio.preview.time, 1);
    const imports = requests.filter(request => request.url === '/video/upload').length;
    await core.loadProcessedMedia('/processed/result.mp4', false);
    assert.equal(requests.filter(request => request.url === '/video/upload').length, imports);
    await assert.rejects(core.loadProcessedMedia('https://example.com/private.mp4'));
    console.log('Studio Copilot selection, history, and preview regressions passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
