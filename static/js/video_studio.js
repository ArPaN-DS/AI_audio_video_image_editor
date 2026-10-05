/**
 * Video Studio Module
 * Multi-track NLE timeline management, playhead synchronization, text canvas overlays & export.
 */

class TimelinePreview {
    constructor(studio) {
        this.studio = studio;
        this.audioPlayers = new Map();
        this.audioContext = null;
        this.frame = null;
        this.lastTimestamp = null;
        this.videoSource = null;
        studio.videoPlayer.muted = true;
        studio.videoPlayer.addEventListener('loadedmetadata', () => this.refresh(true));
    }

    duration() {
        return this.studio.clips.reduce((end, clip) =>
            Math.max(end, Number(clip.start || 0) + Number(clip.duration || 0)), 0);
    }

    activeClips() {
        const time = this.studio.currentTime;
        return this.studio.clips.filter(clip => clip.mediaId
            && time >= clip.start && time < clip.start + clip.duration);
    }

    mediaFor(clip) {
        return window.studioCore?.project.mediaBin?.find(media => media.id === clip.mediaId);
    }

    mixFor(clip) {
        const tracks = window.studioCore?.project.tracks || [];
        const track = tracks.find(item => item.id === clip.trackId) || {};
        const hasSolo = tracks.some(item => item.type === 'audio' && item.solo);
        return {
            muted: Boolean(clip.muted || track.muted || (hasSolo && !track.solo)),
            volume: Math.max(0, Math.min(2, (clip.volume ?? 1) * (track.volume ?? 1))),
            pan: Math.max(-1, Math.min(1, clip.pan ?? track.pan ?? 0))
        };
    }

    sourceTime(clip) {
        const speed = Math.max(0.25, Math.min(4, clip.speed || 1));
        return Math.max(0, (clip.offset || 0) + (this.studio.currentTime - clip.start) * speed);
    }

    syncMedia(element, clip, forceSeek) {
        const sourceTime = this.sourceTime(clip);
        element.playbackRate = Math.max(0.25, Math.min(4, clip.speed || 1));
        const drift = Math.abs(element.currentTime - sourceTime);
        if (element.readyState >= 1 && drift > (forceSeek ? 0.01 : 0.2)) {
            element.currentTime = sourceTime;
        }
        if (this.studio.isPlaying && element.paused && element.readyState >= 2) {
            element.play().catch(() => {});
        } else if (!this.studio.isPlaying) {
            element.pause();
        }
    }

    ensureAudioContext() {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!this.audioContext && AudioContextClass) {
            this.audioContext = new AudioContextClass();
        }
        if (this.audioContext?.state === 'suspended') this.audioContext.resume().catch(() => {});
    }

    createAudioPlayer(clip, media) {
        const element = document.createElement('audio');
        element.preload = 'auto';
        element.src = media.url;
        const player = { element, source: null, gain: null, panner: null };
        if (this.audioContext) {
            player.source = this.audioContext.createMediaElementSource(element);
            player.gain = this.audioContext.createGain();
            player.panner = this.audioContext.createStereoPanner();
            player.source.connect(player.gain);
            player.gain.connect(player.panner);
            player.panner.connect(this.audioContext.destination);
        }
        element.addEventListener('loadedmetadata', () => this.refresh(true));
        element.addEventListener('canplay', () => this.refresh());
        this.audioPlayers.set(clip.id, player);
        return player;
    }

    refresh(forceSeek = false) {
        const active = this.activeClips();
        const videoClip = active.filter(clip => this.mediaFor(clip)?.has_video
            && this.studio.getProjectTrack(clip.trackId)?.type === 'video').at(-1);
        const video = this.studio.videoPlayer;
        video.style.visibility = videoClip ? 'visible' : 'hidden';
        if (videoClip) {
            const media = this.mediaFor(videoClip);
            if (this.videoSource !== media.url) {
                video.pause();
                this.videoSource = media.url;
                video.src = media.url;
                video.load();
                forceSeek = true;
            }
            this.syncMedia(video, videoClip, forceSeek);
        } else {
            video.pause();
        }
        const audibleIds = new Set();
        active.forEach(clip => {
            const media = this.mediaFor(clip);
            if (!media?.url || !media.has_audio || clip.hasAudio === false) return;
            const mix = this.mixFor(clip);
            if (mix.muted) return;
            audibleIds.add(clip.id);
            let player = this.audioPlayers.get(clip.id);
            if (!player && this.studio.isPlaying) player = this.createAudioPlayer(clip, media);
            if (!player) return;
            if (player.gain) {
                player.gain.gain.value = mix.volume;
                player.panner.pan.value = mix.pan;
            } else {
                player.element.volume = Math.min(1, mix.volume);
            }
            this.syncMedia(player.element, clip, forceSeek);
        });
        this.audioPlayers.forEach((player, clipId) => {
            if (!audibleIds.has(clipId)) player.element.pause();
            if (!this.studio.clips.some(clip => clip.id === clipId)) {
                this.releaseAudioPlayer(player);
                this.audioPlayers.delete(clipId);
            }
        });
        this.studio.updatePlayheadUI();
        this.studio.renderTextOverlays();
    }

    seek(time) {
        this.studio.currentTime = Math.max(0, Math.min(this.duration(), Number(time) || 0));
        this.lastTimestamp = null;
        this.refresh(true);
        this.studio.syncProjectState();
    }

    play() {
        if (this.duration() <= 0) return;
        if (this.studio.currentTime >= this.duration()) this.seek(0);
        this.ensureAudioContext();
        window.audioStudio?.wavesurfer?.pause();
        this.studio.setPlayState(true);
        this.lastTimestamp = null;
        this.refresh(true);
        const tick = timestamp => {
            if (!this.studio.isPlaying) return;
            if (this.lastTimestamp !== null) {
                this.studio.currentTime += (timestamp - this.lastTimestamp) / 1000;
            }
            this.lastTimestamp = timestamp;
            if (this.studio.currentTime >= this.duration()) {
                this.studio.currentTime = this.duration();
                this.pause();
                return;
            }
            this.refresh();
            this.frame = requestAnimationFrame(tick);
        };
        this.frame = requestAnimationFrame(tick);
    }

    pause(persist = true) {
        if (this.frame !== null) cancelAnimationFrame(this.frame);
        this.frame = null;
        this.lastTimestamp = null;
        this.studio.setPlayState(false);
        this.refresh(true);
        this.audioPlayers.forEach(player => player.element.pause());
        if (persist) this.studio.syncProjectState();
    }

    releaseAudioPlayer(player) {
        player.element.pause();
        player.element.removeAttribute('src');
        player.element.load();
        player.source?.disconnect();
        player.gain?.disconnect();
        player.panner?.disconnect();
    }

    reset() {
        this.pause(false);
        this.audioPlayers.forEach(player => this.releaseAudioPlayer(player));
        this.audioPlayers.clear();
        this.videoSource = null;
        this.studio.videoPlayer.removeAttribute('src');
        this.studio.videoPlayer.load();
    }
}

class VideoStudio {
    constructor() {
        this.videoPlayer = document.getElementById('masterVideoPlayer');
        this.textCanvas = document.getElementById('textOverlayCanvas');
        this.ctx = this.textCanvas ? this.textCanvas.getContext('2d') : null;
        
        this.clips = []; // [{ id, trackId, mediaId, name, start, duration, offset }]
        this.isPlaying = false;
        this.currentTime = 0.0;
        this.selectedClip = null;
        this.pxPerSec = 40;
        this.preview = new TimelinePreview(this);

        this.initPlayerEvents();
        this.initTimelineEvents();
        this.initInspectorEvents();
        this.initTrackControls();
        this.loadProjectState(window.studioCore?.project);
    }

    initPlayerEvents() {
        if (!this.videoPlayer) return;

        // Transport Controls
        const btnPlay = document.getElementById('btnMasterPlayPause');
        if (btnPlay) btnPlay.addEventListener('click', () => this.togglePlayPause());

        const btnPrev = document.getElementById('btnPrevFrame');
        if (btnPrev) btnPrev.addEventListener('click', () => this.nudgeFrame(-1));

        const btnNext = document.getElementById('btnNextFrame');
        if (btnNext) btnNext.addEventListener('click', () => this.nudgeFrame(1));

        const btnStop = document.getElementById('btnStopPlay');
        if (btnStop) btnStop.addEventListener('click', () => this.stopPlayback());
    }

    initTimelineEvents() {
        const zoomSlider = document.getElementById('tlZoomSlider');
        if (zoomSlider) {
            zoomSlider.addEventListener('input', (e) => {
                this.pxPerSec = parseInt(e.target.value, 10) || 40;
                this.renderTimelineTrackClips();
                this.updatePlayheadUI();
            });
            this.pxPerSec = parseInt(zoomSlider.value, 10) || 40;
        }

        const btnSplit = document.getElementById('btnTlSplit');
        if (btnSplit) btnSplit.addEventListener('click', () => this.splitSelectedClip());

        const btnDelete = document.getElementById('btnTlDelete');
        if (btnDelete) btnDelete.addEventListener('click', () => this.deleteSelectedClip());

        const btnAddText = document.getElementById('btnTlAddTextTrack');
        if (btnAddText) btnAddText.addEventListener('click', () => this.addTextTrackClip());

        const btnDuplicate = document.getElementById('btnTlDuplicate');
        if (btnDuplicate) btnDuplicate.addEventListener('click', () => this.duplicateSelectedClip());

        document.querySelectorAll('.track-timeline-content').forEach(container => {
            container.addEventListener('click', event => {
                const bounds = container.getBoundingClientRect();
                this.preview.seek((event.clientX - bounds.left) / this.pxPerSec);
            });
            container.addEventListener('dragover', event => event.preventDefault());
            container.addEventListener('drop', event => {
                event.preventDefault();
                const trackId = container.closest('.track-row')?.dataset.trackId;
                const track = this.getProjectTrack(trackId);
                if (!trackId || track?.locked) return;

                const dropX = event.clientX - container.getBoundingClientRect().left;
                const startTime = Math.max(0, dropX / this.pxPerSec);

                const clipId = event.dataTransfer.getData('text/x-studio-clip');
                if (clipId) {
                    // Move existing clip
                    const clip = this.clips.find(item => item.id === clipId);
                    if (!clip || this.getProjectTrack(clip.trackId)?.locked) return;
                    if ((clip.trackId === 't1') !== (trackId === 't1')) return;
                    clip.trackId = trackId;
                    clip.start = startTime;
                    this.syncProjectState();
                    this.renderTimelineTrackClips();
                    return;
                }

                const jsonStr = event.dataTransfer.getData('application/json');
                if (jsonStr) {
                    try {
                        const mediaItem = JSON.parse(jsonStr);
                        // Make sure audio only drops on audio, video on video
                        if (trackId === 'a1' && !mediaItem.has_audio) return;
                        if (trackId === 'v1' && !mediaItem.has_video && !mediaItem.has_audio && !mediaItem.url.match(/\.(jpg|jpeg|png|webp|gif)/i)) return; // Allow images on video track too
                        if (trackId === 't1') return; // Cannot drop media on text track

                        this.clips.push({
                            id: 'c_' + Date.now() + Math.random().toString(36).substr(2, 5),
                            trackId: trackId,
                            mediaId: mediaItem.id,
                            name: mediaItem.name,
                            start: startTime,
                            duration: mediaItem.duration || 5, // Default 5s for images without duration
                            offset: 0,
                            speed: 1,
                            volume: 1.0,
                            pan: 0.0,
                            muted: false
                        });
                        this.syncProjectState();
                        this.renderTimelineTrackClips();
                    } catch (e) {
                        console.error('Failed to parse media drop', e);
                    }
                }
            });
        });
    }

    initInspectorEvents() {
        const volume = document.getElementById('propClipVolume');
        const pan = document.getElementById('propClipPan');
        const text = document.getElementById('propTextContent');
        const fontSize = document.getElementById('propFontSize');
        const textColor = document.getElementById('propTextColor');
        const textBg = document.getElementById('propTextBgColor');

        volume?.addEventListener('input', event => {
            if (!this.selectedClip || this.getProjectTrack(this.selectedClip.trackId)?.locked) return;
            this.selectedClip.volume = Number(event.target.value) / 100;
            document.getElementById('valClipVolume').textContent = event.target.value + '%';
            this.syncProjectState();
        });
        pan?.addEventListener('input', event => {
            if (!this.selectedClip || this.getProjectTrack(this.selectedClip.trackId)?.locked) return;
            this.selectedClip.pan = Number(event.target.value) / 100;
            document.getElementById('valClipPan').textContent = this.formatPan(this.selectedClip.pan);
            this.syncProjectState();
        });
        text?.addEventListener('input', event => this.updateTextProperty('text', event.target.value));
        fontSize?.addEventListener('input', event => this.updateTextProperty('fontSize', Number(event.target.value)));
        textColor?.addEventListener('input', event => this.updateTextProperty('color', event.target.value));
        textBg?.addEventListener('input', event => this.updateTextProperty('backgroundColor', event.target.value));
    }

    initTrackControls() {
        document.querySelectorAll('.track-row').forEach(row => {
            const trackId = row.dataset.trackId;
            row.querySelector('.btn-mute-track')?.addEventListener('click', () => {
                const track = this.getProjectTrack(trackId);
                if (!track) return;
                track.muted = !track.muted;
                this.syncTrackControlUI();
                this.syncProjectState();
            });
            row.querySelector('.btn-lock-track')?.addEventListener('click', () => {
                const track = this.getProjectTrack(trackId);
                if (!track) return;
                track.locked = !track.locked;
                this.syncTrackControlUI();
                this.syncProjectState();
            });
            row.querySelector('.btn-solo-track')?.addEventListener('click', () => {
                const track = this.getProjectTrack(trackId);
                if (!track) return;
                track.solo = !track.solo;
                this.syncTrackControlUI();
                this.syncProjectState();
            });
        });
    }

    getProjectTrack(trackId) {
        return window.studioCore?.project.tracks.find(track => track.id === trackId);
    }

    syncTrackControlUI() {
        document.querySelectorAll('.track-row').forEach(row => {
            const track = this.getProjectTrack(row.dataset.trackId);
            if (!track) return;
            row.classList.toggle('track-muted', track.muted);
            row.classList.toggle('track-locked', track.locked);
            row.classList.toggle('track-solo', track.solo);
            row.querySelector('.btn-mute-track')?.setAttribute('aria-pressed', String(track.muted));
            row.querySelector('.btn-lock-track')?.setAttribute('aria-pressed', String(track.locked));
            row.querySelector('.btn-solo-track')?.setAttribute('aria-pressed', String(track.solo));
        });
        this.syncTimelineToolbarUI();
    }

    loadProjectState(project) {
        if (!project || !Array.isArray(project.tracks)) return;
        this.preview.reset();
        this.clips = project.tracks.flatMap(track => {
            const clips = Array.isArray(track.clips) ? track.clips : [];
            return clips.map(clip => ({
                volume: 1,
                pan: 0,
                speed: 1,
                offset: 0,
                muted: false,
                ...clip,
                trackId: track.id
            }));
        });
        this.currentTime = Number(project.playhead ?? project.currentTime ?? 0) || 0;
        this.selectedClip = null;
        this.renderTimelineTrackClips();
        this.updatePlayheadUI();
        this.syncTrackControlUI();
        this.restorePreviewMedia(project);
        this.preview.seek(this.currentTime);
    }

    restorePreviewMedia(project) {
        const audioClip = this.clips.find(clip => clip.trackId === 'a1' && clip.mediaId);
        const audioMedia = project.mediaBin?.find(item => item.id === audioClip?.mediaId);
        if (audioMedia?.url) {
            window.audioStudio?.loadAudio(audioMedia.url, audioMedia);
        }
    }

    syncProjectState() {
        const project = window.studioCore?.project;
        if (!project || !Array.isArray(project.tracks)) return;

        project.tracks.forEach(track => {
            track.clips = this.clips
                .filter(clip => clip.trackId === track.id)
                .map(clip => ({ ...clip }));
        });
        project.playhead = this.currentTime;
        project.currentTime = this.currentTime;
        project.duration = this.clips.reduce(
            (maximum, clip) => Math.max(maximum, Number(clip.start || 0) + Number(clip.duration || 0)),
            0
        );
        this.preview.refresh();
    }

    updateTextProperty(property, value) {
        if (!this.selectedClip || this.selectedClip.trackId !== 't1') return;
        if (this.getProjectTrack(this.selectedClip.trackId)?.locked) return;
        this.selectedClip[property] = value;
        this.syncProjectState();
        this.renderTimelineTrackClips();
        this.renderTextOverlays();
    }

    formatPan(value) {
        if (Math.abs(value) < 0.01) return 'Center';
        return Math.round(Math.abs(value) * 100) + (value < 0 ? '% L' : '% R');
    }

    togglePlayPause() {
        if (this.isPlaying) {
            this.preview.pause();
        } else {
            this.preview.play();
        }
    }

    setPlayState(playing) {
        this.isPlaying = playing;
        const btn = document.getElementById('btnMasterPlayPause');
        if (btn) {
            btn.innerHTML = playing ? '<i class="fa-solid fa-pause" aria-hidden="true"></i>' : '<i class="fa-solid fa-play" aria-hidden="true"></i>';
            btn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
        }
    }

    nudgeFrame(framesDelta) {
        const fps = 30;
        this.preview.seek(this.currentTime + framesDelta / fps);
    }

    stopPlayback() {
        this.preview.pause();
        this.preview.seek(0);
    }

    updatePlayheadUI() {
        const timecodeEl = document.getElementById('timecodeDisplay');
        if (timecodeEl) {
            const hrs = Math.floor(this.currentTime / 3600);
            const mins = Math.floor((this.currentTime % 3600) / 60);
            const secs = Math.floor(this.currentTime % 60);
            const frames = Math.floor((this.currentTime % 1) * 30);
            timecodeEl.textContent = `${String(hrs).padStart(2,'0')}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}:${String(frames).padStart(2,'0')}`;
        }

        const playhead = document.getElementById('timelinePlayhead');
        if (playhead) {
            playhead.style.left = (160 + (this.currentTime * this.pxPerSec)) + 'px';
        }
    }

    addClipToTimeline(mediaItem) {
        const clipId = 'clip_' + Date.now();
        const duration = mediaItem.duration || 10;
        const trackId = mediaItem.has_audio && !mediaItem.has_video ? 'a1' : 'v1';
        const track = this.getProjectTrack(trackId);
        if (track?.locked) return;
        const start = this.clips
            .filter(clip => clip.trackId === trackId)
            .reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);

        // Load into master video player if it's the first clip
        if (this.clips.length === 0 && mediaItem.has_video) {
            this.videoPlayer.src = mediaItem.url;
            this.videoPlayer.load();
        }

        const clip = {
            id: clipId,
            trackId,
            mediaId: mediaItem.id,
            source: mediaItem.id,
            name: mediaItem.name,
            start,
            duration: duration,
            offset: 0.0,
            speed: 1.0,
            volume: 1.0,
            pan: 0.0,
            muted: false,
            hasAudio: Boolean(mediaItem.has_audio)
        };

        this.clips.push(clip);
        if (trackId === 'a1') {
            window.audioStudio?.loadAudio(mediaItem.url, mediaItem);
        }
        this.syncProjectState();
        this.renderTimelineTrackClips();
    }

    renderTimelineTrackClips() {
        ['v1', 't1', 'a1'].forEach(trId => {
            const container = document.getElementById(`trackContent${trId.toUpperCase()}`);
            if (!container) return;
            container.innerHTML = '';

            this.clips.filter(c => c.trackId === trId).forEach(c => {
                const div = document.createElement('div');
                div.className = 'timeline-clip';
                div.draggable = true;
                div.dataset.clipId = c.id;
                div.style.left = (c.start * this.pxPerSec) + 'px';
                div.style.width = (c.duration * this.pxPerSec) + 'px';
                const safeName = window.studioCore
                    ? StudioCore.escapeHtml(c.name || 'Untitled clip')
                    : 'Untitled clip';
                div.innerHTML = `<span class="clip-name"><i class="fa-solid fa-grip-lines-vertical" aria-hidden="true"></i> ${safeName}</span>`;
                div.classList.toggle('selected', this.selectedClip?.id === c.id);
                div.tabIndex = 0;
                div.setAttribute('role', 'button');
                div.setAttribute('aria-pressed', String(this.selectedClip?.id === c.id));
                div.title = (c.name || 'Untitled clip') + ' · ' + Number(c.start || 0).toFixed(1) + 's, '
                    + Number(c.duration || 0).toFixed(1) + 's long';

                div.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.selectClip(c, div);
                });
                div.addEventListener('keydown', (e) => {
                    if (e.key !== 'Enter') return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.selectClip(c, div);
                });
                div.addEventListener('dragstart', event => {
                    event.dataTransfer.setData('text/x-studio-clip', c.id);
                    event.dataTransfer.effectAllowed = 'move';
                });

                container.appendChild(div);
            });
        });
        this.syncTimelineToolbarUI();
    }

    syncTimelineToolbarUI() {
        if (typeof document.getElementById !== 'function') return;
        document.getElementById('timelineWorkspace')?.classList.toggle('has-clips', this.clips.length > 0);
        const locked = this.selectedClip ? Boolean(this.getProjectTrack(this.selectedClip.trackId)?.locked) : true;
        ['btnTlSplit', 'btnTlDelete', 'btnTlDuplicate'].forEach(id => {
            const button = document.getElementById(id);
            if (button) button.disabled = !this.selectedClip || locked;
        });
        // Inspector sections only apply to the matching kind of selected clip.
        const isCaption = this.selectedClip?.trackId === 't1';
        const clipAudio = document.getElementById('clipAudioFields');
        const captionFields = document.getElementById('captionFields');
        if (clipAudio) clipAudio.disabled = !this.selectedClip || isCaption || locked;
        if (captionFields) captionFields.disabled = !isCaption || locked;
    }

    selectClip(clip, element) {
        document.querySelectorAll('.timeline-clip').forEach(el => {
            const selected = el === element || el.dataset.clipId === clip?.id;
            el.classList.toggle('selected', selected);
            el.setAttribute('aria-pressed', String(selected));
        });
        this.selectedClip = clip;
        this.syncTimelineToolbarUI();
        const volume = document.getElementById('propClipVolume');
        const pan = document.getElementById('propClipPan');
        if (volume) volume.value = Math.round((clip.volume ?? 1) * 100);
        if (pan) pan.value = Math.round((clip.pan ?? 0) * 100);
        const volumeLabel = document.getElementById('valClipVolume');
        const panLabel = document.getElementById('valClipPan');
        if (volumeLabel) volumeLabel.textContent = Math.round((clip.volume ?? 1) * 100) + '%';
        if (panLabel) panLabel.textContent = this.formatPan(clip.pan ?? 0);

        if (clip.trackId === 't1') {
            document.getElementById('propTextContent').value = clip.text || '';
            document.getElementById('propFontSize').value = clip.fontSize || 36;
            document.getElementById('propTextColor').value = clip.color || '#ffffff';
            document.getElementById('propTextBgColor').value = clip.backgroundColor || '#000000';
        }
    }

    splitSelectedClip() {
        if (!this.selectedClip) return;
        const c = this.selectedClip;
        if (this.getProjectTrack(c.trackId)?.locked) return;
        const splitPoint = this.currentTime - c.start;

        if (splitPoint <= 0.5 || splitPoint >= c.duration - 0.5) return;

        const clip2 = {
            ...c,
            id: 'clip_' + Date.now(),
            start: c.start + splitPoint,
            duration: c.duration - splitPoint,
            offset: (c.offset || 0) + splitPoint * (c.speed || 1)
        };

        c.duration = splitPoint;
        this.clips.push(clip2);
        this.syncProjectState();
        this.renderTimelineTrackClips();
    }

    deleteSelectedClip() {
        if (!this.selectedClip) return;
        if (this.getProjectTrack(this.selectedClip.trackId)?.locked) return;
        this.clips = this.clips.filter(c => c.id !== this.selectedClip.id);
        this.selectedClip = null;
        this.syncProjectState();
        this.renderTimelineTrackClips();
    }

    duplicateSelectedClip() {
        const source = this.selectedClip;
        if (!source || this.getProjectTrack(source.trackId)?.locked) return;
        const trackEnd = this.clips
            .filter(clip => clip.trackId === source.trackId)
            .reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
        const copy = { ...source, id: (source.trackId === 't1' ? 'text_' : 'clip_') + Date.now(), start: trackEnd };
        this.clips.push(copy);
        this.syncProjectState();
        this.renderTimelineTrackClips();
        this.selectClip(copy, document.querySelector?.(`.timeline-clip[data-clip-id="${copy.id}"]`));
    }

    addTextTrackClip() {
        const textClip = {
            id: 'text_' + Date.now(),
            trackId: 't1',
            mediaId: null,
            name: 'Caption',
            start: this.currentTime,
            duration: 4.0,
            text: 'Caption text',
            fontSize: 36,
            color: '#ffffff',
            backgroundColor: '#000000',
            volume: 1,
            pan: 0
        };
        this.clips.push(textClip);
        this.syncProjectState();
        this.renderTimelineTrackClips();
        const element = document.querySelector?.(`.timeline-clip[data-clip-id="${textClip.id}"]`);
        this.selectClip(textClip, element);
    }

    renderTextOverlays() {
        if (!this.textCanvas || !this.ctx) return;
        this.textCanvas.width = this.videoPlayer.clientWidth || 1280;
        this.textCanvas.height = this.videoPlayer.clientHeight || 720;
        this.ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);

        const textTrack = this.getProjectTrack('t1');
        const activeTextClips = this.clips.filter(c => (
            c.trackId === 't1'
            && !textTrack?.muted
            && this.currentTime >= c.start
            && this.currentTime <= c.start + c.duration
        ));
        activeTextClips.forEach(tc => {
            this.ctx.font = `bold ${tc.fontSize || 32}px "Segoe UI", sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = tc.color || '#ffffff';
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 4;
            
            const x = this.textCanvas.width / 2;
            const y = this.textCanvas.height * 0.85;

            if (tc.backgroundColor) {
                const width = this.ctx.measureText(tc.text || 'Caption').width + 32;
                const height = (tc.fontSize || 32) + 18;
                this.ctx.fillStyle = tc.backgroundColor;
                this.ctx.fillRect(x - width / 2, y - height + 8, width, height);
                this.ctx.fillStyle = tc.color || '#ffffff';
            }
            this.ctx.strokeText(tc.text || 'Caption', x, y);
            this.ctx.fillText(tc.text || 'Caption', x, y);
        });
    }

    async exportVideoTimeline(format = 'mp4', quality = '1080p', aspectRatio = '16:9') {
        this.syncProjectState();
        const tracks = window.studioCore?.project.tracks || [];
        const hasSoloTrack = tracks.some(track => track.type === 'audio' && track.solo);
        const mediaClips = this.clips
            .filter(clip => clip.trackId !== 't1' && clip.mediaId)
            .sort((first, second) => first.start - second.start)
            .map(clip => {
                const track = tracks.find(item => item.id === clip.trackId) || {};
                return {
                    source: clip.mediaId,
                    start: clip.start,
                    in: clip.offset || 0,
                    out: (clip.offset || 0) + clip.duration * (clip.speed || 1),
                    speed: clip.speed || 1,
                    volume: (clip.volume ?? 1) * (track.volume ?? 1),
                    pan: clip.pan ?? track.pan ?? 0,
                    muted: Boolean(clip.muted || track.muted || (hasSoloTrack && !track.solo)),
                    hasAudio: clip.hasAudio !== false
                };
            });
        const spec = {
            format,
            resolution: quality,
            aspect_ratio: aspectRatio,
            clips: mediaClips,
            texts: this.clips
                .filter(clip => clip.trackId === 't1' && !this.getProjectTrack('t1')?.muted)
                .map(clip => ({
                    text: clip.text,
                    start: clip.start,
                    end: clip.start + clip.duration,
                    size: clip.fontSize,
                    color: clip.color,
                    position: 'bc',
                    backgroundColor: clip.backgroundColor,
                    bg: Boolean(clip.backgroundColor)
                }))
        };

        try {
            const resp = await fetch('/video/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(spec)
            });

            if (resp.ok) {
                const blob = await resp.blob();
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `rendered_video.${format}`;
                link.click();
            } else {
                const data = await resp.json();
                alert(`Export failed: ${data.error || 'Server error'}`);
            }
        } catch (err) {
            console.error(err);
        }
        hideProcessingOverlay();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.videoStudio = new VideoStudio();
});
