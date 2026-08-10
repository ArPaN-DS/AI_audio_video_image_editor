/**
 * Video Studio Module
 * Multi-track NLE timeline management, playhead synchronization, text canvas overlays & export.
 */

class VideoStudio {
    constructor() {
        this.videoPlayer = document.getElementById('masterVideoPlayer');
        this.textCanvas = document.getElementById('textOverlayCanvas');
        this.ctx = this.textCanvas ? this.textCanvas.getContext('2d') : null;
        
        this.clips = []; // [{ id, trackId, mediaId, name, start, duration, offset }]
        this.isPlaying = false;
        this.currentTime = 0.0;
        this.selectedClip = null;

        this.initPlayerEvents();
        this.initTimelineEvents();
    }

    initPlayerEvents() {
        if (!this.videoPlayer) return;

        this.videoPlayer.addEventListener('timeupdate', () => {
            this.currentTime = this.videoPlayer.currentTime;
            this.updatePlayheadUI();
            this.renderTextOverlays();
        });

        this.videoPlayer.addEventListener('ended', () => {
            this.setPlayState(false);
        });

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
        const btnSplit = document.getElementById('btnTlSplit');
        if (btnSplit) btnSplit.addEventListener('click', () => this.splitSelectedClip());

        const btnDelete = document.getElementById('btnTlDelete');
        if (btnDelete) btnDelete.addEventListener('click', () => this.deleteSelectedClip());

        const btnAddText = document.getElementById('btnTlAddTextTrack');
        if (btnAddText) btnAddText.addEventListener('click', () => this.addTextTrackClip());
    }

    togglePlayPause() {
        if (this.isPlaying) {
            this.videoPlayer.pause();
            this.setPlayState(false);
        } else {
            this.videoPlayer.play().catch(() => {});
            this.setPlayState(true);
        }
    }

    setPlayState(playing) {
        this.isPlaying = playing;
        const btn = document.getElementById('btnMasterPlayPause');
        if (btn) {
            btn.innerHTML = playing ? '<i class="fa-solid fa-pause"></i>' : '<i class="fa-solid fa-play"></i>';
        }
    }

    nudgeFrame(framesDelta) {
        const fps = 30;
        this.currentTime += framesDelta / fps;
        if (this.currentTime < 0) this.currentTime = 0;
        this.videoPlayer.currentTime = this.currentTime;
    }

    stopPlayback() {
        this.videoPlayer.pause();
        this.setPlayState(false);
        this.currentTime = 0;
        this.videoPlayer.currentTime = 0;
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
            const pps = 40; // pixels per second
            playhead.style.left = (160 + (this.currentTime * pps)) + 'px';
        }
    }

    addClipToTimeline(mediaItem) {
        const clipId = 'clip_' + Date.now();
        const duration = mediaItem.duration || 10;

        // Load into master video player if it's the first clip
        if (this.clips.length === 0 && mediaItem.has_video) {
            this.videoPlayer.src = mediaItem.url;
            this.videoPlayer.load();
        }

        const clip = {
            id: clipId,
            trackId: mediaItem.has_video ? 'v1' : 'a1',
            mediaId: mediaItem.id,
            name: mediaItem.name,
            start: 0.0,
            duration: duration,
            offset: 0.0
        };

        this.clips.push(clip);
        this.renderTimelineTrackClips();
    }

    renderTimelineTrackClips() {
        ['v1', 't1', 'a1'].forEach(trId => {
            const container = document.getElementById(`trackContent${trId.toUpperCase()}`);
            if (!container) return;
            container.innerHTML = '';

            const pps = 40;
            this.clips.filter(c => c.trackId === trId).forEach(c => {
                const div = document.createElement('div');
                div.className = 'timeline-clip';
                div.style.left = (c.start * pps) + 'px';
                div.style.width = (c.duration * pps) + 'px';
                div.innerHTML = `<span class="clip-name"><i class="fa-solid fa-grip-lines-vertical"></i> ${c.name}</span>`;

                div.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.selectClip(c, div);
                });

                container.appendChild(div);
            });
        });
    }

    selectClip(clip, element) {
        document.querySelectorAll('.timeline-clip').forEach(el => el.style.outline = 'none');
        if (element) element.style.outline = '2px solid var(--accent-secondary)';
        this.selectedClip = clip;
    }

    splitSelectedClip() {
        if (!this.selectedClip) return;
        const c = this.selectedClip;
        const splitPoint = this.currentTime - c.start;

        if (splitPoint <= 0.5 || splitPoint >= c.duration - 0.5) return;

        const clip2 = {
            ...c,
            id: 'clip_' + Date.now(),
            start: c.start + splitPoint,
            duration: c.duration - splitPoint,
            offset: c.offset + splitPoint
        };

        c.duration = splitPoint;
        this.clips.push(clip2);
        this.renderTimelineTrackClips();
    }

    deleteSelectedClip() {
        if (!this.selectedClip) return;
        this.clips = this.clips.filter(c => c.id !== this.selectedClip.id);
        this.selectedClip = null;
        this.renderTimelineTrackClips();
    }

    addTextTrackClip() {
        const textClip = {
            id: 'text_' + Date.now(),
            trackId: 't1',
            mediaId: null,
            name: 'Caption Overlay',
            start: this.currentTime,
            duration: 4.0,
            text: 'Sample Subtitle Caption',
            fontSize: 36,
            color: '#ffffff'
        };
        this.clips.push(textClip);
        this.renderTimelineTrackClips();
    }

    renderTextOverlays() {
        if (!this.textCanvas || !this.ctx) return;
        this.textCanvas.width = this.videoPlayer.clientWidth || 1280;
        this.textCanvas.height = this.videoPlayer.clientHeight || 720;
        this.ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);

        const activeTextClips = this.clips.filter(c => c.trackId === 't1' && this.currentTime >= c.start && this.currentTime <= c.start + c.duration);
        activeTextClips.forEach(tc => {
            this.ctx.font = `bold ${tc.fontSize || 32}px Inter, sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = tc.color || '#ffffff';
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 4;
            
            const x = this.textCanvas.width / 2;
            const y = this.textCanvas.height * 0.85;

            this.ctx.strokeText(tc.text || 'Caption', x, y);
            this.ctx.fillText(tc.text || 'Caption', x, y);
        });
    }

    async exportVideoTimeline(format = 'mp4', quality = '1080p') {
        const spec = {
            format: format,
            quality: quality,
            clips: this.clips,
            textOverlays: this.clips.filter(c => c.trackId === 't1')
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
