const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class MediaElement {
    constructor() {
        this.paused = true;
        this.readyState = 4;
        this.currentTime = 0;
        this.style = {};
        this.events = {};
    }
    addEventListener(name, callback) { this.events[name] = callback; }
    play() { this.paused = false; return Promise.resolve(); }
    pause() { this.paused = true; }
    load() {}
    removeAttribute(name) { delete this[name]; }
}

class AudioNode {
    constructor() { this.gain = { value: 1 }; this.pan = { value: 0 }; }
    connect() {}
    disconnect() { this.disconnected = true; }
}

class AudioContext {
    constructor() { this.state = 'running'; this.destination = {}; }
    createMediaElementSource() { return new AudioNode(); }
    createGain() { return new AudioNode(); }
    createStereoPanner() { return new AudioNode(); }
}

let frameCallback;
const tracks = [
    { id: 'v1', type: 'video', volume: 1 },
    { id: 'a1', type: 'audio', volume: 0.5, solo: false },
    { id: 't1', type: 'text' }
];
const sandbox = {
    window: {
        AudioContext,
        studioCore: { project: { tracks, mediaBin: [
            { id: 'first', url: '/media/first', has_video: true, has_audio: true },
            { id: 'second', url: '/media/second', has_video: true, has_audio: true },
            { id: 'music', url: '/media/music', has_video: false, has_audio: true }
        ] } }
    },
    document: {
        addEventListener() {},
        createElement() { return new MediaElement(); }
    },
    requestAnimationFrame(callback) { frameCallback = callback; return 1; },
    cancelAnimationFrame() { frameCallback = null; }
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('static/js/video_studio.js', 'utf8') +
    '\nglobalThis.PreviewClass = TimelinePreview; globalThis.StudioClass = VideoStudio;', sandbox);
const studio = {
    videoPlayer: new MediaElement(),
    currentTime: 0,
    isPlaying: false,
    clips: [
        { id: 'clip1', mediaId: 'first', trackId: 'v1', start: 0.5,
          duration: 1, offset: 2, speed: 2, volume: 0.8, pan: -0.2 },
        { id: 'clip2', mediaId: 'second', trackId: 'v1', start: 2,
          duration: 1, offset: 0, speed: 1 },
        { id: 'audio1', mediaId: 'music', trackId: 'a1', start: 0.5,
          duration: 2.5, offset: 0, volume: 0.6, pan: 0.4 }
    ],
    getProjectTrack(trackId) { return tracks.find(track => track.id === trackId); },
    updatePlayheadUI() {},
    renderTextOverlays() {},
    syncProjectState() {},
    setPlayState(playing) { this.isPlaying = playing; }
};
const preview = new sandbox.PreviewClass(studio);
assert.equal(studio.videoPlayer.muted, true);
assert.equal(preview.duration(), 3);
preview.seek(0.2);
assert.equal(studio.videoPlayer.style.visibility, 'hidden');
preview.seek(0.75);
assert.equal(studio.videoPlayer.src, '/media/first');
assert.equal(studio.videoPlayer.currentTime, 2.5);
assert.equal(studio.videoPlayer.playbackRate, 2);
preview.play();
assert.equal(preview.audioPlayers.size, 2);
const music = preview.audioPlayers.get('audio1');
assert.equal(music.gain.gain.value, 0.3);
assert.equal(music.panner.pan.value, 0.4);
assert.equal(music.element.paused, false);
tracks[1].solo = true;
preview.refresh();
assert.equal(preview.audioPlayers.get('clip1').element.paused, true);
assert.equal(music.element.paused, false);
tracks[1].solo = false;
tracks[1].muted = true;
preview.refresh();
assert.equal(music.element.paused, true);
tracks[1].muted = false;
preview.seek(1.75);
assert.equal(studio.videoPlayer.style.visibility, 'hidden');
assert.equal(music.element.paused, false);
preview.seek(2.25);
assert.equal(studio.videoPlayer.src, '/media/second');
assert.equal(studio.videoPlayer.currentTime, 0.25);
frameCallback(1000);
frameCallback(1500);
assert.equal(studio.currentTime, 2.75);
frameCallback(2000);
assert.equal(studio.currentTime, 3);
assert.equal(studio.isPlaying, false);
assert.equal(music.element.paused, true);
preview.play();
assert.equal(studio.currentTime, 0);
preview.pause();
const oldPlayers = [...preview.audioPlayers.values()];
let persisted = false;
studio.syncProjectState = () => { persisted = true; };
preview.reset();
assert.equal(persisted, false);
assert.equal(preview.audioPlayers.size, 0);
assert.ok(oldPlayers.every(player => player.source.disconnected && !player.element.src));

const splitStudio = Object.create(sandbox.StudioClass.prototype);
splitStudio.selectedClip = { id: 'fast', trackId: 'v1', start: 0,
    duration: 4, offset: 2, speed: 2 };
splitStudio.clips = [splitStudio.selectedClip];
splitStudio.currentTime = 1;
splitStudio.getProjectTrack = () => ({ locked: false });
splitStudio.syncProjectState = () => {};
splitStudio.renderTimelineTrackClips = () => {};
splitStudio.splitSelectedClip();
assert.equal(splitStudio.clips[1].offset, 4);
assert.equal(splitStudio.clips[1].duration, 3);
console.log('Studio playback regression checks passed.');
