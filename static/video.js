/* ═══════════════════════════════════════
   Video Editor Pro — Timeline engine
   Audio + Video combo editor · local FFmpeg export
   ═══════════════════════════════════════ */

(() => {
    'use strict';

    // ─── STATE ───
    const media = {};        // id -> {id, name, url, duration, hasVideo, hasAudio, thumbs, width, height}
    let clips = [];          // ordered timeline clips
    let texts = [];          // text overlays
    let music = null;        // {source, name, volume, loop}
    let selected = null;     // {type:'clip'|'text'|'music', id}
    let clipSeq = 1, textSeq = 1;

    let project = { resolution: 'original', transition: 'none', format: 'mp4' };

    let pxPerSec = 40;       // timeline zoom
    let playhead = 0;        // seconds
    let playing = false;
    let rafId = null;
    let lastTick = 0;

    const undoStack = [];
    const MAX_UNDO = 30;

    // ─── DOM ───
    const $ = (id) => document.getElementById(id);
    const hero = $('veHero'), editor = $('veEditor');
    const stage = $('veStage'), stageVideos = $('veStageVideos'),
        stageTexts = $('veStageTexts'), stageEmpty = $('veStageEmpty');
    const trackVideo = $('veTrackVideo'), trackText = $('veTrackText'),
        trackMusic = $('veTrackMusic'), ruler = $('veRuler'),
        playheadEl = $('vePlayhead'), tlInner = $('veTimelineInner'),
        tlScroll = $('veTimelineScroll');
    const mediaList = $('veMediaList');

    const videoEls = {};     // clipId -> <video> element on the stage

    // ─── HELPERS ───
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fmtTime = (s) => {
        if (!s || isNaN(s)) s = 0;
        const m = Math.floor(s / 60), sec = Math.floor(s % 60), t = Math.floor((s % 1) * 10);
        return `${m}:${sec.toString().padStart(2, '0')}.${t}`;
    };
    const fmtSize = (b) => b < 1048576 ? (b / 1024).toFixed(0) + ' KB' : (b / 1048576).toFixed(1) + ' MB';
    const clipDur = (c) => Math.max(0.05, (c.out - c.in) / (c.speed || 1));
    const totalDur = () => clips.reduce((a, c) => a + clipDur(c), 0);

    function showLoading(txt, percent = 30) {
        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.show({
                title: 'Video Studio Engine',
                stageText: txt || 'Processing video clips & rendering timeline...',
                category: 'video'
            });
            window.ProcessingOverlay.updateProgress(percent, txt || 'Working on video project...');
        } else {
            $('loadingText').textContent = txt || 'Processing...';
            $('loadingOverlay').classList.remove('hidden');
        }
    }
    function hideLoading() {
        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.hide();
        } else {
            $('loadingOverlay').classList.add('hidden');
        }
    }

    function toast(msg, type = 'info') {
        document.querySelectorAll('.app-toast').forEach(t => t.remove());
        const el = document.createElement('div');
        el.className = 'app-toast';
        const colors = { error: '#e74c3c', warning: '#f39c12', success: '#27ae60', info: '#2c3e50' };
        el.textContent = msg;
        el.style.cssText = `position:fixed;bottom:30px;left:50%;transform:translateX(-50%) translateY(20px);
            background:${colors[type] || colors.info};color:#fff;padding:14px 28px;border-radius:12px;
            font-size:.88rem;font-weight:600;z-index:99999;box-shadow:0 10px 30px rgba(0,0,0,.25);
            opacity:0;transition:all .35s cubic-bezier(.16,1,.3,1);`;
        document.body.appendChild(el);
        requestAnimationFrame(() => { el.style.opacity = '1'; el.style.transform = 'translateX(-50%) translateY(0)'; });
        setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateX(-50%) translateY(20px)'; setTimeout(() => el.remove(), 400); }, 3200);
    }

    // ─── UNDO ───
    function snapshot() {
        undoStack.push(JSON.stringify({ clips, texts, music }));
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        $('veUndoBtn').disabled = false;
    }
    function undo() {
        if (!undoStack.length) return;
        const s = JSON.parse(undoStack.pop());
        clips = s.clips; texts = s.texts; music = s.music;
        $('veUndoBtn').disabled = !undoStack.length;
        selected = null;
        rebuildStageVideos();
        renderAll();
        toast('Undone', 'info');
    }

    // ─── UPLOAD ───
    async function uploadFiles(fileList) {
        const files = Array.from(fileList).filter(f =>
            f.type.startsWith('video/') || f.type.startsWith('audio/'));
        if (!files.length) { toast('Please choose video or audio files.', 'warning'); return; }

        hero.classList.add('hidden');
        editor.classList.remove('hidden');

        for (const file of files) {
            showLoading(`Importing ${file.name}...`);
            const fd = new FormData();
            fd.append('file', file);
            try {
                const res = await fetch('/video/upload', { method: 'POST', body: fd });
                const data = await res.json();
                if (!res.ok) { toast(data.error || 'Upload failed', 'error'); continue; }
                media[data.id] = data;
                renderMediaLibrary();
            } catch (e) {
                toast('Upload failed: ' + e.message, 'error');
            }
        }
        hideLoading();
        renderAll();
    }

    // ─── MEDIA LIBRARY ───
    function renderMediaLibrary() {
        mediaList.innerHTML = '';
        Object.values(media).forEach(m => {
            const item = document.createElement('div');
            item.className = 've-media-item';
            const thumb = m.thumbs && m.thumbs.length
                ? `<img src="${m.thumbs[Math.floor(m.thumbs.length / 2)]}" alt="">`
                : `<i class="fas fa-file-audio ve-audio-ico"></i>`;
            const kind = m.has_video ? 'VIDEO' : 'AUDIO';
            item.innerHTML = `
                <div class="ve-media-thumb">${thumb}<span class="ve-media-badge">${fmtTime(m.duration)} · ${kind}</span></div>
                <div class="ve-media-body">
                    <div class="ve-media-name" title="${m.name}">${m.name}</div>
                    <div class="ve-media-actions">
                        <button class="ve-add-clip" data-add="${m.id}"><i class="fas fa-plus"></i> Add</button>
                        ${m.has_audio ? `<button data-music="${m.id}" title="Use as background music"><i class="fas fa-music"></i> Music</button>` : ''}
                        ${m.has_video ? `<button data-tools="${m.id}" title="One-click quick tools"><i class="fas fa-wand-magic-sparkles"></i> Tools</button>` : ''}
                    </div>
                </div>`;
            mediaList.appendChild(item);
        });

        mediaList.querySelectorAll('[data-add]').forEach(b => b.onclick = () => addClip(b.dataset.add));
        mediaList.querySelectorAll('[data-music]').forEach(b => b.onclick = () => setMusic(b.dataset.music));
        mediaList.querySelectorAll('[data-tools]').forEach(b => b.onclick = () => openQuickTools(b.dataset.tools));
    }

    // ─── CLIPS ───
    function addClip(mediaId) {
        const m = media[mediaId];
        if (!m) return;
        snapshot();
        clips.push({
            id: 'c' + (clipSeq++),
            source: m.id, name: m.name,
            in: 0, out: m.duration,
            speed: 1, volume: 1, muted: false,
            hasAudio: m.has_audio, hasVideo: m.has_video,
            rotate: 0, fadeIn: false, fadeOut: false,
            filters: { brightness: 0, contrast: 1, saturation: 1, grayscale: false, sepia: false },
        });
        rebuildStageVideos();
        renderAll();
        toast(`Added “${m.name}” to timeline`, 'success');
    }

    function deleteClip(id) {
        snapshot();
        clips = clips.filter(c => c.id !== id);
        if (videoEls[id]) { videoEls[id].remove(); delete videoEls[id]; }
        if (selected && selected.id === id) selected = null;
        rebuildStageVideos();
        renderAll();
    }

    function splitAtPlayhead() {
        const hit = clipAtTime(playhead);
        if (!hit) { toast('Move the playhead over a clip to split it.', 'warning'); return; }
        const { clip, localOffset } = hit;
        const sourceOffset = clip.in + localOffset * (clip.speed || 1);
        if (sourceOffset <= clip.in + 0.05 || sourceOffset >= clip.out - 0.05) {
            toast('Playhead too close to the clip edge.', 'warning'); return;
        }
        snapshot();
        const idx = clips.indexOf(clip);
        const right = JSON.parse(JSON.stringify(clip));
        right.id = 'c' + (clipSeq++);
        clip.out = sourceOffset;
        right.in = sourceOffset;
        clips.splice(idx + 1, 0, right);
        rebuildStageVideos();
        renderAll();
        toast('Clip split', 'success');
    }

    // Which clip covers a given timeline time? Returns {clip, startTime, localOffset}
    function clipAtTime(t) {
        let acc = 0;
        for (const c of clips) {
            const d = clipDur(c);
            if (t >= acc && t < acc + d) return { clip: c, startTime: acc, localOffset: t - acc };
            acc += d;
        }
        // Snap to last clip if at the very end
        if (clips.length && t >= acc - 0.001) {
            const c = clips[clips.length - 1];
            return { clip: c, startTime: acc - clipDur(c), localOffset: clipDur(c) };
        }
        return null;
    }

    // ─── EXTRACT AUDIO (quick tool) ───
    async function extractAudio(mediaId) {
        const m = media[mediaId];
        if (!m) return;
        showLoading('Extracting audio...');
        try {
            const fd = new FormData();
            fd.append('media_id', m.id);
            fd.append('name', m.name);
            fd.append('format', 'mp3');
            fd.append('bitrate', '192k');
            const res = await fetch('/video/extract-audio', { method: 'POST', body: fd });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                toast(err.error || 'Extraction failed', 'error'); return;
            }
            downloadBlob(await res.blob(), m.name.replace(/\.[^.]+$/, '') + '_audio.mp3');
            toast('Audio extracted', 'success');
        } catch (e) {
            toast('Extraction failed: ' + e.message, 'error');
        } finally { hideLoading(); }
    }

    // ─── MUSIC ───
    function setMusic(mediaId) {
        const m = media[mediaId];
        if (!m) return;
        snapshot();
        music = { source: m.id, name: m.name, volume: 0.6, loop: true };
        renderAll();
        selectMusic();
        toast(`“${m.name}” set as background music`, 'success');
    }

    // ═══════════════════════════════════════
    //  RENDER
    // ═══════════════════════════════════════
    function renderAll() {
        layoutTimeline();
        renderClips();
        renderTexts();
        renderMusic();
        renderInspector();
        updateStageAspect();
        renderMusicStatus();
        $('veTotalTime').textContent = fmtTime(totalDur());
        stageEmpty.classList.toggle('hidden', clips.length > 0);
        seekTo(clamp(playhead, 0, totalDur()));
    }

    function layoutTimeline() {
        const dur = Math.max(totalDur(), 10);
        const width = dur * pxPerSec;
        tlInner.style.width = (width + 40) + 'px';

        // Ruler ticks
        ruler.innerHTML = '';
        const step = pxPerSec < 20 ? 10 : pxPerSec < 50 ? 5 : 1;
        for (let t = 0; t <= dur; t += step) {
            const tick = document.createElement('div');
            tick.className = 've-ruler-tick';
            tick.style.left = (t * pxPerSec) + 'px';
            tick.textContent = fmtTime(t);
            ruler.appendChild(tick);
        }
    }

    function renderClips() {
        trackVideo.querySelectorAll('.ve-clip').forEach(e => e.remove());
        trackVideo.dataset.filled = clips.length ? '1' : '0';
        let acc = 0;
        clips.forEach(c => {
            const d = clipDur(c);
            const el = document.createElement('div');
            el.className = 've-clip' + (selected && selected.id === c.id ? ' selected' : '');
            el.style.left = (acc * pxPerSec) + 'px';
            el.style.width = Math.max(24, d * pxPerSec) + 'px';
            el.dataset.id = c.id;

            const m = media[c.source];
            let thumbsHtml = '';
            if (m && m.thumbs && m.thumbs.length) {
                thumbsHtml = '<div class="ve-clip-thumbs">' +
                    m.thumbs.map(u => `<img src="${u}" alt="">`).join('') + '</div>';
            } else {
                el.style.background = 'linear-gradient(135deg,#0d9488,#14b8a6)';
            }
            const badges = [];
            if (c.speed !== 1) badges.push(`${c.speed}×`);
            if (c.muted) badges.push('<i class="fas fa-volume-xmark"></i>');
            if (c.filters.grayscale) badges.push('B&W');
            const badgeHtml = badges.length
                ? '<div class="ve-clip-badges">' + badges.map(b => `<span class="ve-clip-badge">${b}</span>`).join('') + '</div>' : '';

            el.innerHTML = thumbsHtml + badgeHtml +
                `<div class="ve-clip-label">${c.name}</div>` +
                `<div class="ve-handle ve-handle-l"></div><div class="ve-handle ve-handle-r"></div>`;
            trackVideo.appendChild(el);
            attachClipInteractions(el, c);
            acc += d;
        });
    }

    function renderTexts() {
        trackText.querySelectorAll('.ve-text-block').forEach(e => e.remove());
        trackText.dataset.filled = texts.length ? '1' : '0';
        texts.forEach(t => {
            const el = document.createElement('div');
            el.className = 've-text-block' + (selected && selected.id === t.id ? ' selected' : '');
            el.style.left = (t.start * pxPerSec) + 'px';
            el.style.width = Math.max(30, (t.end - t.start) * pxPerSec) + 'px';
            el.dataset.id = t.id;
            el.innerHTML = `<i class="fas fa-font"></i> ${t.text || 'Text'}`;
            trackText.appendChild(el);
            attachTextInteractions(el, t);
        });
    }

    function renderMusic() {
        trackMusic.querySelectorAll('.ve-music-block').forEach(e => e.remove());
        trackMusic.dataset.filled = music ? '1' : '0';
        if (!music) return;
        const dur = totalDur();
        const el = document.createElement('div');
        el.className = 've-music-block' + (selected && selected.type === 'music' ? ' selected' : '');
        el.style.width = Math.max(60, dur * pxPerSec) + 'px';
        el.innerHTML = `<i class="fas fa-music"></i> ${music.name}${music.loop ? ' · loop' : ''}`;
        el.onclick = selectMusic;
        trackMusic.appendChild(el);
    }

    function renderMusicStatus() {
        const box = $('veMusicStatus');
        if (!box) return;
        if (music) {
            box.innerHTML = `<i class="fas fa-music"></i><span>“${music.name}” · ${Math.round(music.volume * 100)}% volume${music.loop ? ' · looping' : ''}. Click the orange track to adjust.</span>`;
        } else {
            box.innerHTML = `<i class="fas fa-music"></i><span>No music added — import an audio file, then click <strong>“Music”</strong> on it.</span>`;
        }
    }

    function updateStageAspect() {
        stage.classList.remove('vertical', 'square');
        if (project.resolution === 'vertical') stage.classList.add('vertical');
        else if (project.resolution === 'square') stage.classList.add('square');
    }

    // ═══════════════════════════════════════
    //  INSPECTOR
    // ═══════════════════════════════════════
    function showPane(id) {
        ['paneProject', 'paneClip', 'paneText', 'paneMusic'].forEach(p =>
            $(p).classList.toggle('hidden', p !== id));
    }

    function renderInspector() {
        if (!selected) { showPane('paneProject'); return; }
        if (selected.type === 'clip') {
            const c = clips.find(x => x.id === selected.id);
            if (!c) { selected = null; showPane('paneProject'); return; }
            showPane('paneClip');
            $('veSpeedVal').textContent = c.speed + '×';
            document.querySelectorAll('#veSpeedRow .ve-chip').forEach(ch =>
                ch.classList.toggle('active', parseFloat(ch.dataset.speed) === c.speed));
            $('veClipVolume').value = Math.round(c.volume * 100);
            $('veClipVolVal').textContent = Math.round(c.volume * 100) + '%';
            $('veClipMuteBtn').classList.toggle('active', c.muted);
            $('veFadeInBtn').classList.toggle('active', c.fadeIn);
            $('veFadeOutBtn').classList.toggle('active', c.fadeOut);
            $('veGrayscaleBtn').classList.toggle('active', c.filters.grayscale);
            $('veSepiaBtn').classList.toggle('active', c.filters.sepia);
            $('veBrightness').value = Math.round(c.filters.brightness * 100);
            $('veContrast').value = Math.round(c.filters.contrast * 100);
            $('veSaturation').value = Math.round(c.filters.saturation * 100);
        } else if (selected.type === 'text') {
            const t = texts.find(x => x.id === selected.id);
            if (!t) { selected = null; showPane('paneProject'); return; }
            showPane('paneText');
            $('veTextContent').value = t.text;
            $('veTextSize').value = t.size;
            $('veTextColor').value = t.color;
            $('veTextBgBtn').classList.toggle('active', t.bg);
            document.querySelectorAll('#vePosGrid button').forEach(b =>
                b.classList.toggle('active', b.dataset.pos === t.position));
        } else if (selected.type === 'music') {
            if (!music) { selected = null; showPane('paneProject'); return; }
            showPane('paneMusic');
            $('veMusicName').querySelector('span').textContent = music.name;
            $('veMusicVolume').value = Math.round(music.volume * 100);
            $('veMusicVolVal').textContent = Math.round(music.volume * 100) + '%';
            $('veMusicLoopBtn').classList.toggle('active', music.loop);
        }
    }

    function selectClip(id) { selected = { type: 'clip', id }; renderClips(); renderTexts(); renderMusic(); renderInspector(); }
    function selectText(id) { selected = { type: 'text', id }; renderClips(); renderTexts(); renderMusic(); renderInspector(); }
    function selectMusic() { selected = { type: 'music', id: 'music' }; renderClips(); renderTexts(); renderMusic(); renderInspector(); }
    function selectedClip() { return selected && selected.type === 'clip' ? clips.find(c => c.id === selected.id) : null; }
    function selectedText() { return selected && selected.type === 'text' ? texts.find(t => t.id === selected.id) : null; }

    // ═══════════════════════════════════════
    //  STAGE PREVIEW (video sync)
    // ═══════════════════════════════════════
    function rebuildStageVideos() {
        // Remove stale
        Object.keys(videoEls).forEach(id => {
            if (!clips.find(c => c.id === id)) { videoEls[id].remove(); delete videoEls[id]; }
        });
        // Create missing (only for video clips)
        clips.forEach(c => {
            if (!c.hasVideo) return;
            if (videoEls[c.id]) return;
            const m = media[c.source];
            const v = document.createElement('video');
            v.src = m.url;
            v.preload = 'auto';
            v.playsInline = true;
            v.muted = true;   // stage playback muted; we manage a single active clip's audio separately if needed
            videoEls[c.id] = v;
            stageVideos.appendChild(v);
        });
    }

    let activeClipId = null;

    function renderStageAt(t) {
        const hit = clipAtTime(t);
        // videos
        let showId = hit && hit.clip.hasVideo ? hit.clip.id : null;
        Object.entries(videoEls).forEach(([id, v]) => {
            v.classList.toggle('active', id === showId);
        });
        if (hit) {
            const c = hit.clip;
            if (c.hasVideo) {
                const v = videoEls[c.id];
                const srcTime = c.in + hit.localOffset * (c.speed || 1);
                if (Math.abs(v.currentTime - srcTime) > 0.25 || !playing) {
                    try { v.currentTime = clamp(srcTime, 0, (media[c.source].duration || srcTime)); } catch (e) { }
                }
                applyFilterToVideo(v, c);
            }
            activeClipId = c.id;
        } else { activeClipId = null; }
        // texts
        renderStageTexts(t);
    }

    function applyFilterToVideo(v, c) {
        const f = c.filters;
        const parts = [];
        if (f.brightness) parts.push(`brightness(${1 + f.brightness})`);
        if (f.contrast !== 1) parts.push(`contrast(${f.contrast})`);
        if (f.saturation !== 1) parts.push(`saturate(${f.saturation})`);
        if (f.grayscale) parts.push('grayscale(1)');
        if (f.sepia) parts.push('sepia(0.7)');
        v.style.filter = parts.join(' ');
        let deg = c.rotate % 360;
        v.style.transform = deg ? `rotate(${deg}deg)` : '';
    }

    const posMap = {
        tl: [12, 10], tc: [50, 10], tr: [88, 10],
        ml: [12, 50], mc: [50, 50], mr: [88, 50],
        bl: [12, 88], bc: [50, 88], br: [88, 88],
    };
    function renderStageTexts(t) {
        stageTexts.innerHTML = '';
        const scale = stage.clientHeight / 1080;
        texts.forEach(tx => {
            if (t < tx.start || t > tx.end) return;
            const el = document.createElement('div');
            el.className = 've-stage-text';
            el.textContent = tx.text;
            const [x, y] = posMap[tx.position] || posMap.bc;
            el.style.left = x + '%';
            el.style.top = y + '%';
            el.style.fontSize = Math.max(10, tx.size * scale) + 'px';
            el.style.color = tx.color;
            if (tx.bg) { el.style.background = 'rgba(0,0,0,0.45)'; el.style.borderRadius = '6px'; }
            stageTexts.appendChild(el);
        });
    }

    function seekTo(t) {
        playhead = clamp(t, 0, Math.max(totalDur(), 0));
        playheadEl.style.left = (20 + playhead * pxPerSec) + 'px';
        $('veCurrentTime').textContent = fmtTime(playhead);
        renderStageAt(playhead);
    }

    // ─── PLAYBACK ───
    function play() {
        if (!clips.length) return;
        if (playhead >= totalDur() - 0.05) playhead = 0;
        playing = true;
        $('vePlayIcon').className = 'fas fa-pause';
        lastTick = performance.now();
        const active = activeClipId && videoEls[activeClipId];
        if (active) active.play().catch(() => { });
        rafId = requestAnimationFrame(tick);
    }
    function pause() {
        playing = false;
        $('vePlayIcon').className = 'fas fa-play';
        if (rafId) cancelAnimationFrame(rafId);
        Object.values(videoEls).forEach(v => v.pause());
    }
    function togglePlay() { playing ? pause() : play(); }

    function tick(now) {
        if (!playing) return;
        const dt = (now - lastTick) / 1000;
        lastTick = now;
        const prevClip = activeClipId;
        playhead += dt;
        if (playhead >= totalDur()) { seekTo(totalDur()); pause(); return; }
        seekTo(playhead);
        // start playing the newly active video, pause others
        if (activeClipId !== prevClip) {
            if (prevClip && videoEls[prevClip]) videoEls[prevClip].pause();
            if (activeClipId && videoEls[activeClipId]) videoEls[activeClipId].play().catch(() => { });
        }
        rafId = requestAnimationFrame(tick);
    }

    // ═══════════════════════════════════════
    //  TIMELINE INTERACTIONS (drag / trim / reorder)
    // ═══════════════════════════════════════
    function attachClipInteractions(el, clip) {
        el.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('ve-handle')) {
                startTrim(e, el, clip, e.target.classList.contains('ve-handle-l') ? 'l' : 'r');
            } else {
                startDrag(e, el, clip);
            }
        });
        el.addEventListener('click', (e) => {
            if (!el.dataset.moved) selectClip(clip.id);
            delete el.dataset.moved;
        });
    }

    function startTrim(e, el, clip, side) {
        e.preventDefault(); e.stopPropagation();
        const startX = e.clientX;
        const origIn = clip.in, origOut = clip.out;
        const speed = clip.speed || 1;
        let didChange = false;
        snapshot();
        function move(ev) {
            const deltaSec = ((ev.clientX - startX) / pxPerSec) * speed;
            if (side === 'l') clip.in = clamp(origIn + deltaSec, 0, clip.out - 0.1);
            else clip.out = clamp(origOut + deltaSec, clip.in + 0.1, media[clip.source].duration);
            didChange = true;
            renderClips(); layoutTimeline(); renderMusic();
            $('veTotalTime').textContent = fmtTime(totalDur());
        }
        function up() {
            document.removeEventListener('mousemove', move);
            document.removeEventListener('mouseup', up);
            if (!didChange) undoStack.pop();
            el.dataset.moved = '1';
            seekTo(playhead);
        }
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
    }

    function startDrag(e, el, clip) {
        e.preventDefault();
        const startX = e.clientX;
        let moved = false, captured = false;
        function move(ev) {
            if (Math.abs(ev.clientX - startX) < 5 && !moved) return;
            if (!captured) { snapshot(); captured = true; el.classList.add('dragging'); }
            moved = true;
            // Determine drop index by pointer x relative to other clips
            const idx = clips.indexOf(clip);
            const centers = [];
            let acc = 0;
            clips.forEach(c => { const d = clipDur(c); centers.push(acc + d / 2); acc += d; });
            const pointerSec = (ev.clientX - tlInner.getBoundingClientRect().left - 20) / pxPerSec;
            let target = clips.length - 1;
            for (let i = 0; i < centers.length; i++) { if (pointerSec < centers[i]) { target = i; break; } }
            if (target !== idx) {
                clips.splice(idx, 1);
                clips.splice(target, 0, clip);
                renderClips();
            }
        }
        function up() {
            document.removeEventListener('mousemove', move);
            document.removeEventListener('mouseup', up);
            el.classList.remove('dragging');
            if (moved) { el.dataset.moved = '1'; rebuildStageVideos(); renderAll(); }
            else if (captured) undoStack.pop();
        }
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
    }

    function attachTextInteractions(el, txt) {
        el.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const startX = e.clientX;
            const origStart = txt.start, origEnd = txt.end, dur = txt.end - txt.start;
            let moved = false, captured = false;
            function move(ev) {
                if (Math.abs(ev.clientX - startX) < 4 && !moved) return;
                if (!captured) { snapshot(); captured = true; }
                moved = true;
                const delta = (ev.clientX - startX) / pxPerSec;
                txt.start = clamp(origStart + delta, 0, Math.max(totalDur() - dur, 0));
                txt.end = txt.start + dur;
                renderTexts();
            }
            function up() {
                document.removeEventListener('mousemove', move);
                document.removeEventListener('mouseup', up);
                if (moved) { el.dataset.moved = '1'; seekTo(playhead); }
                else if (captured) undoStack.pop();
            }
            document.addEventListener('mousemove', move);
            document.addEventListener('mouseup', up);
        });
        el.addEventListener('click', () => {
            if (!el.dataset.moved) selectText(txt.id);
            delete el.dataset.moved;
        });
    }

    // Ruler scrub
    ruler.addEventListener('mousedown', (e) => {
        function scrub(ev) {
            const x = ev.clientX - tlInner.getBoundingClientRect().left - 20;
            seekTo(x / pxPerSec);
        }
        scrub(e);
        function up() { document.removeEventListener('mousemove', scrub); document.removeEventListener('mouseup', up); }
        document.addEventListener('mousemove', scrub);
        document.addEventListener('mouseup', up);
    });

    // ═══════════════════════════════════════
    //  TEXT OVERLAYS
    // ═══════════════════════════════════════
    function addText() {
        snapshot();
        const start = clamp(playhead, 0, Math.max(totalDur() - 0.5, 0));
        const t = {
            id: 't' + (textSeq++),
            text: 'Your text', start,
            end: Math.min(start + 3, totalDur() || start + 3),
            size: 48, color: '#FFFFFF', position: 'bc', bg: true,
        };
        texts.push(t);
        renderAll();
        selectText(t.id);
        toast('Text added — edit it on the right', 'success');
    }

    // ═══════════════════════════════════════
    //  EXPORT
    // ═══════════════════════════════════════
    function buildSpec(format) {
        return {
            format,
            resolution: project.resolution,
            transition: project.transition,
            clips: clips.map(c => ({
                source: c.source, in: c.in, out: c.out,
                speed: c.speed, volume: c.volume, muted: c.muted,
                hasAudio: c.hasAudio, rotate: c.rotate,
                fadeIn: c.fadeIn, fadeOut: c.fadeOut, filters: c.filters,
            })),
            texts: texts.map(t => ({
                text: t.text, start: t.start, end: t.end,
                size: t.size, color: t.color, position: t.position, bg: t.bg,
            })),
            music: music ? { source: music.source, volume: music.volume, loop: music.loop } : null,
        };
    }

    async function doExport(format) {
        if (!clips.length) { toast('Add at least one clip first.', 'warning'); return; }
        $('veExportOverlay').classList.add('hidden');
        showLoading('Rendering your ' + (format === 'mp4' ? 'video' : 'audio') + '... this can take a while.');
        try {
            const res = await fetch('/video/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(buildSpec(format)),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                toast(err.error || 'Export failed', 'error'); return;
            }
            downloadBlob(await res.blob(), 'edited_video.' + format);
            toast('Export complete!', 'success');
        } catch (e) {
            toast('Export failed: ' + e.message, 'error');
        } finally { hideLoading(); }
    }

    function downloadBlob(blob, name) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = name;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 4000);
    }

    // ═══════════════════════════════════════
    //  QUICK TOOLS (one-click, beginner-friendly)
    // ═══════════════════════════════════════
    let quickMediaId = null, quickOp = null;

    function openQuickTools(mediaId) {
        quickMediaId = mediaId; quickOp = null;
        $('veQuickFileName').textContent = media[mediaId] ? media[mediaId].name : 'file';
        $('veQuickOptions').classList.add('hidden');
        document.querySelectorAll('.ve-quick-tool').forEach(b => b.classList.remove('selected'));
        $('veQuickOverlay').classList.remove('hidden');
    }

    function pickQuickTool(op, btn) {
        quickOp = op;
        document.querySelectorAll('.ve-quick-tool').forEach(b => b.classList.toggle('selected', b === btn));
        // Extract & mute & frame have no options → run immediately
        if (op === 'extract') { $('veQuickOverlay').classList.add('hidden'); extractAudio(quickMediaId); return; }
        if (op === 'mute' || op === 'frame' || op === 'gif') { runQuick(); return; }
        // compress / convert show options
        $('veQuickOptions').classList.remove('hidden');
        document.querySelectorAll('.ve-quick-opt').forEach(o =>
            o.classList.toggle('hidden', o.dataset.for !== op));
        $('veRunQuickText').textContent = op === 'compress' ? 'Compress & Download' : 'Convert & Download';
    }

    async function runQuick() {
        const m = media[quickMediaId];
        if (!m || !quickOp) return;
        $('veQuickOverlay').classList.add('hidden');
        const labels = { gif: 'Making GIF', compress: 'Compressing', convert: 'Converting', frame: 'Grabbing frame', mute: 'Muting' };
        showLoading((labels[quickOp] || 'Processing') + '...');
        try {
            const fd = new FormData();
            fd.append('media_id', m.id);
            fd.append('name', m.name);
            fd.append('op', quickOp);
            if (quickOp === 'gif') { fd.append('start', 0); fd.append('duration', Math.min(5, m.duration)); }
            if (quickOp === 'frame') { fd.append('t', clamp(playhead, 0, m.duration)); }
            if (quickOp === 'compress') fd.append('level', document.querySelector('#veQuickOptions [data-for="compress"] .ve-chip.active').dataset.level);
            if (quickOp === 'convert') {
                fd.append('container', document.querySelector('#veConvContainer .ve-chip.active').dataset.container);
                fd.append('quality', document.querySelector('#veConvQuality .ve-chip.active').dataset.quality);
            }
            const res = await fetch('/video/quick', { method: 'POST', body: fd });
            if (!res.ok) { const err = await res.json().catch(() => ({})); toast(err.error || 'Failed', 'error'); return; }
            const ext = { gif: 'gif', compress: 'mp4', frame: 'jpg', mute: 'mp4', convert: (document.querySelector('#veConvContainer .ve-chip.active') || {}).dataset?.container || 'mp4' }[quickOp];
            downloadBlob(await res.blob(), m.name.replace(/\.[^.]+$/, '') + '_' + quickOp + '.' + ext);
            toast('Done!', 'success');
        } catch (e) { toast('Failed: ' + e.message, 'error'); }
        finally { hideLoading(); }
    }

    function initQuickTools() {
        $('veCloseQuick').onclick = () => $('veQuickOverlay').classList.add('hidden');
        document.querySelectorAll('.ve-quick-tool').forEach(b =>
            b.onclick = () => pickQuickTool(b.dataset.op, b));
        $('veRunQuickBtn').onclick = runQuick;
        // option chip groups
        document.querySelectorAll('#veQuickOptions .ve-chip-row').forEach(row => {
            row.addEventListener('click', e => {
                const b = e.target.closest('.ve-chip'); if (!b) return;
                row.querySelectorAll('.ve-chip').forEach(c => c.classList.toggle('active', c === b));
            });
        });
    }

    // ═══════════════════════════════════════
    //  WIRING
    // ═══════════════════════════════════════
    function init() {
        initQuickTools();
        // Upload
        $('veFileInput').addEventListener('change', e => uploadFiles(e.target.files));
        $('veFileInput2').addEventListener('change', e => { uploadFiles(e.target.files); e.target.value = ''; });
        $('veAddMediaBtn').onclick = () => $('veFileInput2').click();

        const ua = $('veUploadArea');
        ['dragover', 'dragenter'].forEach(ev => ua.addEventListener(ev, e => { e.preventDefault(); ua.classList.add('drag-active'); }));
        ['dragleave', 'drop'].forEach(ev => ua.addEventListener(ev, () => ua.classList.remove('drag-active')));
        ua.addEventListener('drop', e => { e.preventDefault(); uploadFiles(e.dataTransfer.files); });

        // Transport
        $('vePlayBtn').onclick = togglePlay;
        $('veSkipStartBtn').onclick = () => seekTo(0);
        $('veSkipEndBtn').onclick = () => seekTo(totalDur());
        $('veUndoBtn').onclick = undo;
        $('veSplitBtn').onclick = splitAtPlayhead;
        $('veAddTextBtn').onclick = addText;

        // Zoom
        $('veZoom').addEventListener('input', e => { pxPerSec = +e.target.value; renderAll(); });

        // Project chips
        $('veAspectRow').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); if (!b) return;
            project.resolution = b.dataset.res;
            document.querySelectorAll('#veAspectRow .ve-chip').forEach(c => c.classList.toggle('active', c === b));
            updateStageAspect(); renderStageAt(playhead);
        });
        $('veTransitionRow').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); if (!b) return;
            project.transition = b.dataset.tr;
            document.querySelectorAll('#veTransitionRow .ve-chip').forEach(c => c.classList.toggle('active', c === b));
        });
        $('veFormatRow').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); if (!b) return;
            project.format = b.dataset.fmt;
            document.querySelectorAll('#veFormatRow .ve-chip').forEach(c => c.classList.toggle('active', c === b));
        });

        // Clip inspector
        $('veDeleteClipBtn').onclick = () => selectedClip() && deleteClip(selected.id);
        $('veSpeedRow').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); const c = selectedClip(); if (!b || !c) return;
            snapshot(); c.speed = parseFloat(b.dataset.speed); rebuildStageVideos(); renderAll(); selectClip(c.id);
        });
        $('veClipVolume').addEventListener('input', e => {
            const c = selectedClip(); if (!c) return;
            c.volume = +e.target.value / 100; $('veClipVolVal').textContent = e.target.value + '%';
        });
        $('veClipVolume').addEventListener('change', () => snapshot());
        $('veClipMuteBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.muted = !c.muted; renderAll(); selectClip(c.id); };
        $('veFadeInBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.fadeIn = !c.fadeIn; renderInspector(); };
        $('veFadeOutBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.fadeOut = !c.fadeOut; renderInspector(); };
        $('veRotateBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.rotate = (c.rotate + 90) % 360; renderStageAt(playhead); };
        $('veGrayscaleBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.filters.grayscale = !c.filters.grayscale; c.filters.sepia = false; renderAll(); selectClip(c.id); };
        $('veSepiaBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.filters.sepia = !c.filters.sepia; c.filters.grayscale = false; renderAll(); selectClip(c.id); };
        $('veResetFiltersBtn').onclick = () => { const c = selectedClip(); if (!c) return; snapshot(); c.filters = { brightness: 0, contrast: 1, saturation: 1, grayscale: false, sepia: false }; c.rotate = 0; renderAll(); selectClip(c.id); };
        $('veBrightness').addEventListener('input', e => { const c = selectedClip(); if (c) { c.filters.brightness = +e.target.value / 100; renderStageAt(playhead); } });
        $('veContrast').addEventListener('input', e => { const c = selectedClip(); if (c) { c.filters.contrast = +e.target.value / 100; renderStageAt(playhead); } });
        $('veSaturation').addEventListener('input', e => { const c = selectedClip(); if (c) { c.filters.saturation = +e.target.value / 100; renderStageAt(playhead); } });

        // Text inspector
        $('veDeleteTextBtn').onclick = () => { const t = selectedText(); if (!t) return; snapshot(); texts = texts.filter(x => x.id !== t.id); selected = null; renderAll(); };
        $('veTextContent').addEventListener('input', e => { const t = selectedText(); if (t) { t.text = e.target.value; renderTexts(); renderStageAt(playhead); } });
        $('veTextContent').addEventListener('focus', () => snapshot());
        $('veTextSize').addEventListener('input', e => { const t = selectedText(); if (t) { t.size = +e.target.value; renderStageAt(playhead); } });
        $('veTextColor').addEventListener('input', e => { const t = selectedText(); if (t) { t.color = e.target.value; renderStageAt(playhead); } });
        $('veTextBgBtn').onclick = () => { const t = selectedText(); if (!t) return; t.bg = !t.bg; $('veTextBgBtn').classList.toggle('active', t.bg); renderStageAt(playhead); };
        $('vePosGrid').addEventListener('click', e => {
            const b = e.target.closest('button'); const t = selectedText(); if (!b || !t) return;
            t.position = b.dataset.pos;
            document.querySelectorAll('#vePosGrid button').forEach(x => x.classList.toggle('active', x === b));
            renderStageAt(playhead);
        });

        // Music inspector
        $('veDeleteMusicBtn').onclick = () => { snapshot(); music = null; selected = null; renderAll(); };
        $('veMusicVolume').addEventListener('input', e => { if (music) { music.volume = +e.target.value / 100; $('veMusicVolVal').textContent = e.target.value + '%'; renderMusicStatus(); } });
        $('veMusicVolume').addEventListener('change', () => snapshot());
        $('veMusicLoopBtn').onclick = () => { if (!music) return; snapshot(); music.loop = !music.loop; $('veMusicLoopBtn').classList.toggle('active', music.loop); renderMusic(); renderMusicStatus(); };

        // Export modal
        $('veExportBtn').onclick = openExport;
        $('veCloseExport').onclick = () => $('veExportOverlay').classList.add('hidden');
        $('veExportFormatRow').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); if (!b) return;
            document.querySelectorAll('#veExportFormatRow .ve-chip').forEach(c => c.classList.toggle('active', c === b));
        });
        $('veStartExportBtn').onclick = () => {
            const fmt = document.querySelector('#veExportFormatRow .ve-chip.active').dataset.fmt;
            doExport(fmt);
        };

        // Help
        $('veHelpBtn').onclick = () => $('veShortcutsOverlay').classList.remove('hidden');
        $('veCloseShortcuts').onclick = () => $('veShortcutsOverlay').classList.add('hidden');

        // Keyboard
        document.addEventListener('keydown', onKey);

        renderAll();
    }

    function openExport() {
        if (!clips.length) { toast('Add at least one clip first.', 'warning'); return; }
        const has = clips.some(c => c.hasVideo);
        $('veExportSummary').innerHTML =
            `<strong>${clips.length}</strong> clip(s) · <strong>${fmtTime(totalDur())}</strong> · ` +
            `${texts.length} text overlay(s)${music ? ' · background music' : ''}. ` +
            (has ? 'Choose MP4 for video, or MP3/WAV for audio-only.' : 'This is an audio-only timeline.');
        // preselect project format
        document.querySelectorAll('#veExportFormatRow .ve-chip').forEach(c =>
            c.classList.toggle('active', c.dataset.fmt === project.format));
        $('veExportOverlay').classList.remove('hidden');
    }

    function onKey(e) {
        const tag = (e.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;
        if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        else if (e.key === 's' || e.key === 'S') { e.preventDefault(); splitAtPlayhead(); }
        else if (e.key === 't' || e.key === 'T') { e.preventDefault(); addText(); }
        else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selected && selected.type === 'clip') deleteClip(selected.id);
            else if (selected && selected.type === 'text') { snapshot(); texts = texts.filter(x => x.id !== selected.id); selected = null; renderAll(); }
        }
        else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); undo(); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); seekTo(playhead - (e.shiftKey ? 5 : 0.5)); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); seekTo(playhead + (e.shiftKey ? 5 : 0.5)); }
        else if (e.key === '?') { $('veShortcutsOverlay').classList.toggle('hidden'); }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
