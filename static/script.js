/* ═══════════════════════════════════════
   Media Studio — Audio editor
   ═══════════════════════════════════════ */

let wavesurfer, wsRegions;
let allRegions = [];
let selectedRegion = null;
let recordedBlob = null;
let regionCounter = 1;
let currentVolume = 0.8;
let isResetting = false;

// ─── UNDO SYSTEM ───
let undoStack = [];
const MAX_UNDO = 20;

function pushUndo(action, data) {
    undoStack.push({ action, data, timestamp: Date.now() });
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    updateUndoBtn();
}

function updateUndoBtn() {
    const btn = document.getElementById('undoBtn');
    if (!btn) return;
    btn.disabled = undoStack.length === 0;
    btn.title = undoStack.length > 0
        ? `Undo: ${undoStack[undoStack.length - 1].action} (Ctrl+Z)`
        : 'Nothing to undo';
}

// Theme-aware waveform colours — resolved from the page's design tokens
// (--wave-color / --wave-progress / --wave-cursor in style.css) so the
// waveform matches whichever theme is active. A probe element resolves
// var() and color-mix() into a concrete rgb() the canvas can paint.
function resolveTokenColor(name, fallback) {
    try {
        const probe = document.createElement('span');
        probe.style.cssText = `position:absolute;width:0;height:0;overflow:hidden;color:var(${name}, ${fallback})`;
        document.body.appendChild(probe);
        const value = getComputedStyle(probe).color;
        probe.remove();
        return value || fallback;
    } catch (e) {
        return fallback;
    }
}

function themeWaveColors() {
    return {
        waveColor: resolveTokenColor('--wave-color', '#8A92A1'),
        progressColor: resolveTokenColor('--wave-progress', '#4F46E5'),
        cursorColor: resolveTokenColor('--wave-cursor', '#E11D48'),
    };
}

// Region / marker fills reference tokens directly. Regions render in the
// waveform's shadow DOM, which inherits custom properties, so these follow
// theme changes without any re-colouring.
const REGION_FILL = 'var(--region-fill)';
const MARKER_COLORS = {
    silence: 'var(--marker-silence)',
    speech: 'var(--marker-speech)',
    beat: 'var(--marker-beat)',
    filler: 'var(--marker-filler)',
};

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
    ));
}

// Range inputs paint an accent fill up to their value (see .styled-range).
function updateRangeFill(input) {
    if (!input) return;
    const min = Number(input.min) || 0;
    const max = Number(input.max) || 100;
    const pct = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
    input.style.setProperty('--fill', `${Math.max(0, Math.min(100, pct))}%`);
}

document.addEventListener('DOMContentLoaded', () => {

    // ─── 1. WAVESURFER SETUP ───
    wsRegions = WaveSurfer.Regions.create();

    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        ...themeWaveColors(),
        cursorWidth: 2,
        height: 140,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        responsive: true,
        normalize: true,
        plugins: [
            WaveSurfer.Timeline.create({ container: '#wave-timeline' }),
            wsRegions
        ]
    });

    // Recolor the waveform instantly when the user toggles light / dark.
    window.addEventListener('themechange', () => {
        if (!wavesurfer) return;
        try { wavesurfer.setOptions(themeWaveColors()); } catch (e) { /* older build */ }
    });

    // ─── 2. TABS ───
    const uploadSec = document.getElementById('uploadSection');
    const recordSec = document.getElementById('recordSection');
    const editorSec = document.getElementById('editor-interface');

    document.getElementById('btnTabUpload').onclick = () => switchTab('upload');
    document.getElementById('btnTabRecord').onclick = () => switchTab('record');

    function setTabState(mode) {
        const up = document.getElementById('btnTabUpload');
        const rec = document.getElementById('btnTabRecord');
        up.classList.toggle('active', mode === 'upload');
        rec.classList.toggle('active', mode === 'record');
        up.setAttribute('aria-selected', String(mode === 'upload'));
        rec.setAttribute('aria-selected', String(mode === 'record'));
    }

    function switchTab(mode) {
        setTabState(mode);
        // Clear any inline display left by a previous reset, then toggle panels.
        // (The editor wrapper also holds these tabs, so it must stay visible.)
        uploadSec.style.display = '';
        recordSec.style.display = '';
        uploadSec.classList.toggle('hidden', mode !== 'upload');
        recordSec.classList.toggle('hidden', mode !== 'record');
        editorSec.classList.remove('hidden');
    }

    // ─── 3. FILE UPLOAD with DRAG & DROP ───
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadSection');

    fileInput.addEventListener('change', function () {
        if (this.files[0]) handleFileLoad(this.files[0]);
    });

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-active');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-active');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-active');
        const file = e.dataTransfer.files[0];
        if (file && (file.type.startsWith('audio/') || file.type.startsWith('video/'))) {
            handleFileLoad(file);
        } else {
            showToast('That file type isn’t supported. Drop an audio or video file (MP3, WAV, M4A, MP4…).', 'warning');
        }
    });

    function handleFileLoad(file) {
        loadAudio(file);
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
    }

    // ─── 4. RECORDING ───
    let mediaRecorder;
    let chunks = [];
    const recBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const recStatus = document.getElementById('recordStatus');

    recBtn.onclick = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = () => {
                recordedBlob = new Blob(chunks, { type: 'audio/webm' });
                chunks = [];
                loadAudio(recordedBlob);
                document.getElementById('fileName').textContent = 'Recording';
                document.getElementById('fileSize').textContent = formatFileSize(recordedBlob.size);
            };
            mediaRecorder.start();
            recBtn.classList.add('recording');
            recBtn.setAttribute('aria-label', 'Recording');
            recStatus.textContent = 'Recording… Select Stop and edit when you’re done.';
            stopBtn.classList.remove('hidden');
            recBtn.disabled = true;
        } catch (err) {
            recStatus.textContent = 'Microphone access was blocked. Allow it in your browser’s site settings, then try again.';
        }
    };

    stopBtn.onclick = () => {
        mediaRecorder.stop();
        try { mediaRecorder.stream.getTracks().forEach(t => t.stop()); } catch (e) { /* already released */ }
        recBtn.classList.remove('recording');
        recBtn.setAttribute('aria-label', 'Start recording');
        recStatus.textContent = 'Recording finished. Loading the waveform…';
        recBtn.disabled = false;
        stopBtn.classList.add('hidden');
    };

    // ─── 5. LOAD AUDIO ───
    function loadAudio(source) {
        const url = URL.createObjectURL(source);
        wavesurfer.load(url);
    }

    wavesurfer.on('ready', () => {
        // Don't re-show editor if we're in the middle of a reset
        if (isResetting) return;

        const editorEl = document.getElementById('editor-interface');
        editorEl.classList.remove('hidden');
        editorEl.classList.remove('editor-pre-upload');

        // Hide waveform placeholder and show file info bar
        const wfPlaceholder = document.getElementById('waveformPlaceholder');
        if (wfPlaceholder) wfPlaceholder.style.display = 'none';
        document.getElementById('fileInfoBar')?.classList.remove('hidden');
        document.getElementById('waveform-container')?.classList.add('waveform-loaded');

        uploadSec.classList.add('hidden');
        recordSec.classList.add('hidden');
        document.getElementById('newFileBtn').classList.remove('hidden');

        wsRegions.clearRegions();
        allRegions = [];
        selectedRegion = null;
        regionCounter = 1;
        undoStack = [];
        updateUndoBtn();

        const duration = wavesurfer.getDuration();
        document.getElementById('totalTime').textContent = formatTimePrecise(duration);
        document.getElementById('fileDuration').textContent = formatTimePrecise(duration);
        document.getElementById('currentTime').textContent = '0:00';

        wavesurfer.setVolume(currentVolume);

        // Add default region
        addRegion(duration * 0.1, duration * 0.4, `Region ${regionCounter++}`, false);

        updateExportSummary();

        // Show helpful hint for first-time users
        showToast('Audio loaded. Double-click the waveform to add more regions.', 'info');
    });

    // ─── TIME UPDATE ───
    wavesurfer.on('audioprocess', () => {
        document.getElementById('currentTime').textContent = formatTimePrecise(wavesurfer.getCurrentTime());
    });

    wavesurfer.on('seeking', () => {
        document.getElementById('currentTime').textContent = formatTimePrecise(wavesurfer.getCurrentTime());
    });

    // Play icon toggle
    wavesurfer.on('play', () => {
        document.getElementById('playIcon').className = 'fas fa-pause';
        const pb = document.getElementById('playBtn');
        pb.classList.add('playing');
        pb.setAttribute('aria-label', 'Pause');
    });
    wavesurfer.on('pause', () => {
        document.getElementById('playIcon').className = 'fas fa-play';
        const pb = document.getElementById('playBtn');
        pb.classList.remove('playing');
        pb.setAttribute('aria-label', 'Play');
    });

    // ─── 6. REGION MANAGEMENT ───
    function addRegion(start, end, name, trackUndo = true) {
        const region = wsRegions.addRegion({
            start: start,
            end: end,
            color: REGION_FILL,
            drag: true,
            resize: true
        });

        const regionData = { id: region.id, name: name, region: region };
        allRegions.push(regionData);

        region.on('click', () => selectRegion(region.id));

        if (trackUndo) {
            pushUndo('Add Region', { id: region.id, name, start, end });
        }

        updateRegionList();

        // Pulse animation on the region list
        setTimeout(() => {
            const items = document.querySelectorAll('.region-item');
            const lastItem = items[items.length - 1];
            if (lastItem) {
                lastItem.classList.add('just-added');
                setTimeout(() => lastItem.classList.remove('just-added'), 800);
            }
        }, 50);

        return region;
    }

    function selectRegion(regionId) {
        // Clear all highlights
        allRegions.forEach(r => {
            if (!r.region.element) return;
            r.region.element.style.border = 'none';
            r.region.element.style.boxShadow = 'none';
        });
        document.querySelectorAll('.region-item').forEach(el => {
            el.classList.remove('selected');
            el.removeAttribute('aria-current');
        });

        const regionData = allRegions.find(r => r.id === regionId);
        if (regionData) {
            selectedRegion = regionData;
            if (regionData.region.element) {
                regionData.region.element.style.boxShadow = 'inset 0 0 0 2px var(--region-edge)';
            }

            const idx = allRegions.indexOf(regionData);
            const items = document.querySelectorAll('.region-item');
            if (items[idx]) {
                items[idx].classList.add('selected');
                items[idx].setAttribute('aria-current', 'true');
                items[idx].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    }

    function updateRegionList() {
        const list = document.getElementById('regionList');
        list.innerHTML = '';

        // Update region count badge
        const countBadge = document.getElementById('regionCount');
        if (countBadge) countBadge.textContent = allRegions.length;

        if (allRegions.length === 0) {
            list.innerHTML = '<p class="empty-state">No regions yet. Double-click the waveform or choose <strong>Add region</strong>.</p>';
            document.getElementById('exportModeGroup').style.display = 'none';
            updateExportSummary();
            return;
        }

        allRegions.forEach((r, index) => {
            const div = document.createElement('div');
            div.className = 'region-item';
            const dur = r.region.end - r.region.start;
            const safeName = escapeHtml(r.name);

            div.innerHTML = `
                <span class="region-index" aria-hidden="true">${index + 1}</span>
                <div class="region-info">
                    <div class="region-name">
                        <span class="region-name-text" data-id="${r.id}" title="Double-click to rename">${safeName}</span>
                    </div>
                    <div class="region-time">${formatTimePrecise(r.region.start)} – ${formatTimePrecise(r.region.end)}</div>
                </div>
                <span class="region-duration" title="Length">${formatTimePrecise(dur)}</span>
                <div class="region-actions">
                    <button type="button" class="region-play-btn" data-id="${r.id}" title="Play this region" aria-label="Play ${safeName}">
                        <i class="fas fa-play" aria-hidden="true"></i>
                    </button>
                    <button type="button" class="region-delete" data-id="${r.id}" title="Delete this region" aria-label="Delete ${safeName}">
                        <i class="fas fa-trash-can" aria-hidden="true"></i>
                    </button>
                </div>
            `;

            // Click to select
            div.onclick = (e) => {
                if (!e.target.closest('.region-delete') && !e.target.closest('.region-play-btn')) {
                    selectRegion(r.id);
                }
            };

            // Double-click region name to rename
            const nameSpan = div.querySelector('.region-name-text');
            nameSpan.addEventListener('dblclick', (e) => {
                e.stopPropagation();
                const currentName = r.name;
                const input = document.createElement('input');
                input.type = 'text';
                input.value = currentName;
                input.className = 'rename-input';
                input.maxLength = 30;
                nameSpan.replaceWith(input);
                input.focus();
                input.select();

                const saveName = () => {
                    const newName = input.value.trim() || currentName;
                    r.name = newName;
                    updateRegionList();
                };
                input.addEventListener('blur', saveName);
                input.addEventListener('keydown', (ke) => {
                    if (ke.key === 'Enter') input.blur();
                    if (ke.key === 'Escape') { r.name = currentName; input.blur(); }
                });
            });

            // Play region button
            div.querySelector('.region-play-btn').onclick = (e) => {
                e.stopPropagation();
                r.region.play();
                selectRegion(r.id);
            };

            // Delete button
            div.querySelector('.region-delete').onclick = (e) => {
                e.stopPropagation();
                deleteRegion(r.id);
            };

            list.appendChild(div);
        });

        document.getElementById('exportModeGroup').style.display = allRegions.length > 1 ? 'grid' : 'none';
        updateExportSummary();
    }

    // One-line summary under the export button: "2 regions · 0:12.4 · MP3"
    function updateExportSummary() {
        const el = document.getElementById('exportSummary');
        if (!el) return;
        if (allRegions.length === 0) {
            el.textContent = 'Add a region to export.';
            return;
        }
        const total = allRegions.reduce((sum, r) => sum + Math.max(0, r.region.end - r.region.start), 0);
        const fmt = (document.querySelector('input[name="format"]:checked')?.value || 'mp3').toUpperCase();
        const separate = allRegions.length > 1 && document.querySelector('input[name="export_mode"]:checked')?.value === 'separate';
        const count = `${allRegions.length} region${allRegions.length === 1 ? '' : 's'}`;
        el.textContent = `${count} · ${formatTimePrecise(total)} · ${fmt}${separate ? ' files in a ZIP' : ''}`;
    }
    document.querySelectorAll('input[name="format"], input[name="export_mode"]').forEach(input => {
        input.addEventListener('change', updateExportSummary);
    });

    window.deleteRegion = function (regionId) {
        const index = allRegions.findIndex(r => r.id === regionId);
        if (index !== -1) {
            const removed = allRegions[index];
            // Save undo data
            pushUndo('Delete Region', {
                name: removed.name,
                start: removed.region.start,
                end: removed.region.end
            });
            removed.region.remove();
            allRegions.splice(index, 1);
            selectedRegion = null;
            updateRegionList();
            showToast(`Deleted “${removed.name}”. Press Ctrl+Z to undo.`, 'info');
        }
    };

    // ─── UNDO ───
    document.getElementById('undoBtn').onclick = performUndo;

    function performUndo() {
        if (undoStack.length === 0) return;
        const last = undoStack.pop();
        updateUndoBtn();

        if (last.action === 'Delete Region') {
            // Re-add the deleted region
            addRegion(last.data.start, last.data.end, last.data.name, false);
            showToast(`Restored “${last.data.name}”.`, 'success');
        } else if (last.action === 'Add Region') {
            // Remove the last added region
            const idx = allRegions.findIndex(r => r.id === last.data.id);
            if (idx !== -1) {
                allRegions[idx].region.remove();
                allRegions.splice(idx, 1);
                selectedRegion = null;
                updateRegionList();
                showToast('Undid adding a region.', 'success');
            }
        } else if (last.action === 'Clear All') {
            // Re-add all cleared regions
            last.data.forEach(d => {
                addRegion(d.start, d.end, d.name, false);
            });
            showToast(`Restored ${last.data.length} region${last.data.length === 1 ? '' : 's'}.`, 'success');
        }
    }

    // Add Region Button
    document.getElementById('addRegionBtn').onclick = () => {
        const duration = wavesurfer.getDuration();
        const current = wavesurfer.getCurrentTime();
        const start = Math.max(0, current);
        const end = Math.min(duration, current + duration * 0.15);
        addRegion(start, end, `Region ${regionCounter++}`);
    };

    // Clear All — with undo support
    document.getElementById('clearAllBtn').onclick = () => {
        if (allRegions.length === 0) return;

        // Save all regions for undo
        const savedRegions = allRegions.map(r => ({
            name: r.name,
            start: r.region.start,
            end: r.region.end
        }));
        pushUndo('Clear All', savedRegions);

        allRegions.forEach(r => r.region.remove());
        allRegions = [];
        selectedRegion = null;
        updateRegionList();
        showToast('Cleared all regions. Press Ctrl+Z to undo.', 'info');
    };

    // Region update listener
    wsRegions.on('region-updated', () => updateRegionList());

    // ─── DOUBLE-CLICK waveform → add region ───
    // Remove old single-click handler, use dblclick instead
    let lastClickTime = 0;
    wavesurfer.on('click', (relativeX) => {
        const now = Date.now();
        if (now - lastClickTime < 350) {
            // Double click detected!
            const duration = wavesurfer.getDuration();
            const clickTime = relativeX * duration;
            const regionLen = Math.min(5, duration * 0.1);
            const start = Math.max(0, clickTime - regionLen / 2);
            const end = Math.min(duration, clickTime + regionLen / 2);
            addRegion(start, end, `Region ${regionCounter++}`);
        }
        lastClickTime = now;
    });

    // ─── 7. PLAY/PAUSE ───
    document.getElementById('playBtn').onclick = () => wavesurfer.playPause();

    // ─── 8. ZOOM ───
    const zoomSlider = document.getElementById('zoomSlider');
    zoomSlider.oninput = function () {
        updateRangeFill(this);
        try { wavesurfer.zoom(Number(this.value)); } catch (e) { /* no audio loaded yet */ }
    };
    updateRangeFill(zoomSlider);

    // ─── 9. VOLUME CONTROL ───
    const volumeSlider = document.getElementById('volumeSlider');
    volumeSlider.value = currentVolume * 100;
    updateRangeFill(volumeSlider);

    volumeSlider.oninput = function () {
        currentVolume = Number(this.value) / 100;
        wavesurfer.setVolume(currentVolume);
        updateVolumeIcon();
    };

    document.getElementById('muteBtn').onclick = () => {
        if (wavesurfer.getVolume() > 0) {
            wavesurfer.setVolume(0);
            volumeSlider.value = 0;
        } else {
            wavesurfer.setVolume(currentVolume || 0.8);
            volumeSlider.value = (currentVolume || 0.8) * 100;
        }
        updateVolumeIcon();
    };

    function updateVolumeIcon() {
        const vol = wavesurfer.getVolume();
        const icon = document.getElementById('volumeIcon');
        if (vol === 0) icon.className = 'fas fa-volume-mute';
        else if (vol < 0.5) icon.className = 'fas fa-volume-down';
        else icon.className = 'fas fa-volume-up';
        const muteBtn = document.getElementById('muteBtn');
        muteBtn.setAttribute('aria-pressed', String(vol === 0));
        muteBtn.setAttribute('aria-label', vol === 0 ? 'Unmute preview' : 'Mute preview');
        muteBtn.title = vol === 0 ? 'Unmute preview (M)' : 'Mute preview (M)';
        updateRangeFill(volumeSlider);
    }

    // ─── 10. SPEED CONTROL ───
    document.getElementById('speedControl').onchange = function () {
        wavesurfer.setPlaybackRate(Number(this.value));
    };

    // ─── 11. NEW FILE / REMOVE AUDIO BUTTONS ───

    function resetToUpload() {
        isResetting = true;

        // FIRST: Force-hide editor with both methods
        const ei = document.getElementById('editor-interface');
        const us = document.getElementById('uploadSection');
        const rs = document.getElementById('recordSection');

        ei.style.display = '';
        ei.classList.remove('hidden');
        ei.classList.add('editor-pre-upload');
        document.getElementById('fileInfoBar')?.classList.add('hidden');
        document.getElementById('aiResultsPanel')?.classList.add('hidden');
        const wfPlaceholder = document.getElementById('waveformPlaceholder');
        if (wfPlaceholder) wfPlaceholder.style.display = '';
        document.getElementById('waveform-container')?.classList.remove('waveform-loaded');

        us.style.display = '';
        us.classList.remove('hidden');

        rs.style.display = '';
        rs.classList.add('hidden');

        setTabState('upload');
        document.getElementById('newFileBtn').classList.add('hidden');
        document.getElementById('currentTime').textContent = '0:00';
        document.getElementById('totalTime').textContent = '0:00';

        // THEN: Try to clean up wavesurfer
        try {
            if (wavesurfer) {
                wavesurfer.pause();
                wsRegions.clearRegions();
                wavesurfer.empty();
            }
        } catch (e) {
            console.warn('WaveSurfer cleanup error:', e);
        }

        // Clear all data
        allRegions = [];
        selectedRegion = null;
        regionCounter = 1;
        undoStack = [];
        updateUndoBtn();
        recordedBlob = null;
        fileInput.value = '';
        updateRegionList();

        showToast('File closed. Open or record another to continue.', 'info');
        // Keep keyboard users oriented: move focus to the file picker.
        if (document.activeElement && document.activeElement.closest &&
            document.activeElement.closest('#fileInfoBar, .header-actions')) {
            fileInput.focus({ preventScroll: true });
        }

        // Allow future wavesurfer 'ready' events after a delay
        setTimeout(() => { isResetting = false; }, 500);
    }

    document.getElementById('newFileBtn').addEventListener('click', resetToUpload);
    document.getElementById('removeAudioBtn').addEventListener('click', resetToUpload);

    // ─── 12. KEYBOARD SHORTCUTS ───
    document.addEventListener('keydown', (e) => {
        const shortcutsEl = document.getElementById('shortcutsOverlay');
        if (!shortcutsEl.classList.contains('hidden')) {
            if (e.key === 'Escape') { e.preventDefault(); closeShortcuts(); }
            return;
        }
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        // Let Space / Enter activate a focused button or link as usual.
        if ((e.code === 'Space' || e.key === 'Enter') && e.target.closest && e.target.closest('button, a, [role="button"], [role="tab"]')) return;
        const hasAudio = !!(wavesurfer && wavesurfer.getDuration && wavesurfer.getDuration() > 0);
        if (!hasAudio && (e.code === 'Space' || e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'm' || e.key === 'M')) return;

        // Ctrl+Z = Undo
        if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
            e.preventDefault();
            performUndo();
            return;
        }

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                wavesurfer.playPause();
                break;
            case 'Delete':
            case 'Backspace':
                e.preventDefault();
                if (selectedRegion) deleteRegion(selectedRegion.id);
                break;
        }

        if (e.key === '?' || e.key === '/') {
            e.preventDefault();
            openShortcuts();
        }

        if (e.key === 'm' || e.key === 'M') {
            e.preventDefault();
            document.getElementById('muteBtn').click();
        }

        // Arrow keys: skip 5 seconds
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const t = Math.max(0, wavesurfer.getCurrentTime() - 5);
            wavesurfer.seekTo(t / wavesurfer.getDuration());
        }
        if (e.key === 'ArrowRight') {
            e.preventDefault();
            const t = Math.min(wavesurfer.getDuration(), wavesurfer.getCurrentTime() + 5);
            wavesurfer.seekTo(t / wavesurfer.getDuration());
        }
    });

    // Help overlay — modal dialog: focus moves in on open and back on close.
    let shortcutsReturnFocus = null;
    function openShortcuts() {
        const overlay = document.getElementById('shortcutsOverlay');
        shortcutsReturnFocus = document.activeElement;
        overlay.classList.remove('hidden');
        document.getElementById('closeShortcuts').focus({ preventScroll: true });
    }
    function closeShortcuts() {
        document.getElementById('shortcutsOverlay').classList.add('hidden');
        const target = shortcutsReturnFocus && document.contains(shortcutsReturnFocus)
            ? shortcutsReturnFocus : document.getElementById('helpBtn');
        if (target && target.focus) target.focus({ preventScroll: true });
        shortcutsReturnFocus = null;
    }

    document.getElementById('helpBtn').onclick = openShortcuts;
    document.getElementById('closeShortcuts').onclick = closeShortcuts;

    document.getElementById('shortcutsOverlay').onclick = (e) => {
        if (e.target.id === 'shortcutsOverlay') closeShortcuts();
    };
    // Trap Tab inside the dialog (it has a single focusable control).
    document.getElementById('shortcutsOverlay').addEventListener('keydown', (e) => {
        if (e.key === 'Tab') { e.preventDefault(); document.getElementById('closeShortcuts').focus(); }
    });

    // ─── 13. FORM SUBMISSION ───
    document.getElementById('cutForm').onsubmit = async (e) => {
        e.preventDefault();

        if (allRegions.length === 0) {
            showToast('Add at least one region before exporting. Double-click the waveform or choose Add region.', 'warning');
            return;
        }

        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.show({
                title: 'Exporting audio',
                stageText: `Preparing ${allRegions.length} region${allRegions.length === 1 ? '' : 's'}…`,
                category: 'audio'
            });
            window.ProcessingOverlay.updateProgress(35, 'Cutting regions and applying effects…');
        }

        const formData = new FormData(e.target);

        // Attach file
        if (recordedBlob && !fileInput.files[0]) {
            formData.append('file', recordedBlob, 'recording.webm');
        } else if (fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        } else {
            showToast('No audio is loaded. Open or record a file, then export again.', 'error');
            if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
            return;
        }

        // Attach regions
        const regionsData = allRegions.map(r => ({
            name: r.name,
            start: r.region.start,
            end: r.region.end
        }));
        formData.append('regions', JSON.stringify(regionsData));

        try {
            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(75, 'Encoding the file…');
            }

            const resp = await fetch('/cut', { method: 'POST', body: formData });

            if (resp.ok) {
                if (window.ProcessingOverlay) {
                    window.ProcessingOverlay.updateProgress(100, 'Export complete. Downloading…');
                }

                const contentType = resp.headers.get('content-type');
                const blob = await resp.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;

                if (contentType && contentType.includes('application/zip')) {
                    a.download = 'audio_cuts.zip';
                } else {
                    const ext = document.querySelector('input[name="format"]:checked').value;
                    a.download = `cut_audio.${ext}`;
                }

                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);

                showToast('Export complete. Check your downloads folder.', 'success');
                setTimeout(() => {
                    if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
                }, 1000);
            } else {
                const errText = await resp.text();
                showToast(`Export failed: ${cleanServerError(errText)}`, 'error');
                if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
            }
        } catch (err) {
            console.error(err);
            showToast('Couldn’t reach ' + ((window.APP_BRAND && window.APP_BRAND.product) || 'the app') + '. Make sure the app is still running, then try again.', 'error');
            if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
        }
    };

    // ─── HELPERS ───
    function formatTimePrecise(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 10);
        return `${min}:${sec.toString().padStart(2, '0')}.${ms}`;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Server errors can arrive as an HTML error page; keep only readable text.
    function cleanServerError(text) {
        const plain = String(text || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
        if (!plain) return 'Something went wrong. Try again.';
        return plain.length > 160 ? plain.slice(0, 157) + '…' : plain;
    }

    // Toasts: styled by .app-toast in style.css; one at a time, announced politely.
    const TOAST_ICONS = {
        info: 'fa-circle-info',
        success: 'fa-circle-check',
        warning: 'fa-triangle-exclamation',
        error: 'fa-circle-exclamation'
    };
    let toastTimers = [];
    function showToast(message, type = 'info') {
        toastTimers.forEach(clearTimeout);
        toastTimers = [];
        document.querySelectorAll('.app-toast').forEach(t => t.remove());

        const kind = TOAST_ICONS[type] ? type : 'info';
        const toast = document.createElement('div');
        toast.className = 'app-toast';
        toast.dataset.type = kind;
        toast.setAttribute('role', kind === 'error' ? 'alert' : 'status');
        toast.setAttribute('aria-live', kind === 'error' ? 'assertive' : 'polite');
        const icon = document.createElement('i');
        icon.className = `fas ${TOAST_ICONS[kind]}`;
        icon.setAttribute('aria-hidden', 'true');
        const text = document.createElement('span');
        text.textContent = message;
        toast.append(icon, text);
        document.body.appendChild(toast);

        requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('is-visible')));

        const visibleFor = kind === 'error' ? 6000 : 3200;
        toastTimers.push(setTimeout(() => toast.classList.remove('is-visible'), visibleFor));
        toastTimers.push(setTimeout(() => toast.remove(), visibleFor + 300));
    }

    // ─── 14. ANALYSIS & CLEAN-UP TOOLS ───
    let allAiRegions = [];
    let detectedBeats = [];
    let snapToBeatsEnabled = false;
    let isSnapping = false;

    // Helper to clear AI markers from the waveform
    function clearAiMarkers() {
        allAiRegions.forEach(r => {
            try { r.remove(); } catch(e) {}
        });
        allAiRegions = [];
    }

    // Results panel heading reflects the tool that produced it
    function setResultsTitle(title) {
        const el = document.getElementById('aiResultsTitle');
        if (el) el.textContent = title;
    }

    // Helper to add non-editable AI display regions
    function addAiMarker(start, end, color, content = '') {
        const reg = wsRegions.addRegion({
            start: start,
            end: end,
            color: color,
            drag: false,
            resize: false,
            content: content
        });
        allAiRegions.push(reg);
        return reg;
    }

    // Helper to get active file blob
    function getActiveAudioFile() {
        if (recordedBlob && !fileInput.files[0]) {
            return recordedBlob;
        } else if (fileInput.files[0]) {
            return fileInput.files[0];
        }
        return null;
    }

    // Main AI run helper
    async function runAIFeature(btnId, endpoint, extraParams = {}, onResponse) {
        const file = getActiveAudioFile();
        if (!file) {
            showToast('Open or record audio first.', 'warning');
            return;
        }

        const btn = document.getElementById(btnId);
        if (!btn) return;

        // Busy state: keep the label, swap the tool icon for a spinner.
        btn.classList.add('loading');
        btn.setAttribute('aria-busy', 'true');
        btn.disabled = true;
        const toolIcon = btn.querySelector('.ai-btn-icon i');
        const originalIconClass = toolIcon ? toolIcon.className : '';
        if (toolIcon) toolIcon.className = 'fas fa-circle-notch fa-spin';

        const formData = new FormData();
        formData.append('file', file, file.name || 'audio.webm');
        for (const [key, val] of Object.entries(extraParams)) {
            formData.append(key, val);
        }

        // Map endpoint to natural language titles & stage descriptions
        const aiMeta = {
            '/ai/transcribe': { title: 'Transcribing', stage: 'Converting speech to text…' },
            '/ai/detect-silence': { title: 'Finding pauses', stage: 'Scanning for silent gaps…' },
            '/ai/auto-trim': { title: 'Trimming silent edges', stage: 'Finding where sound starts and ends…' },
            '/ai/noise-reduce': { title: 'Removing noise', stage: 'Reducing hiss, hum and background noise…' },
            '/ai/detect-beats': { title: 'Detecting beats', stage: 'Measuring tempo and beat positions…' },
            '/ai/detect-vad': { title: 'Finding speech', stage: 'Marking sections with voice…' },
            '/ai/filler-words': { title: 'Finding filler words', stage: 'Listening for “um”, “uh” and similar…' },
            '/ai/enhance-speech': { title: 'Enhancing speech', stage: 'Reducing room echo and evening out the voice…' },
            '/ai/separate-stems': { title: 'Separating vocals', stage: 'Splitting vocals from music…' }
        };
        const meta = aiMeta[endpoint] || { title: 'Processing audio', stage: 'Working on your audio…' };

        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.show({
                title: meta.title,
                stageText: meta.stage,
                category: 'audio'
            });
            window.ProcessingOverlay.updateProgress(35, meta.stage);
        }

        try {
            const resp = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(80, 'Preparing results…');
            }

            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || 'Server error');
            }

            // Check if binary download (noise-reduce returns a file)
            const contentType = resp.headers.get('content-type');
            if (contentType && (contentType.includes('audio/') || contentType.includes('application/octet-stream') || endpoint.includes('noise-reduce'))) {
                const blob = await resp.blob();
                onResponse(blob, resp);
            } else {
                const json = await resp.json();
                if (json.error) {
                    throw new Error(json.error);
                }
                onResponse(json, resp);
            }

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(100, 'Done');
            }
        } catch (err) {
            console.error('[Audio tool error]', err);
            showToast(`${meta.title} failed: ${cleanServerError(err.message)}`, 'error');
        } finally {
            btn.classList.remove('loading');
            btn.removeAttribute('aria-busy');
            btn.disabled = false;
            if (toolIcon) toolIcon.className = originalIconClass;
            if (window.ProcessingOverlay) {
                setTimeout(() => window.ProcessingOverlay.hide(), 500);
            }
        }
    }

    // AI Silence detection
    document.getElementById('aiSilenceBtn').onclick = () => {
        runAIFeature('aiSilenceBtn', '/ai/detect-silence', { min_silence_len: 0.5, silence_thresh: 40 }, (data) => {
            clearAiMarkers();
            
            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            setResultsTitle('Pauses');
            if (!data || data.length === 0) {
                resultsContent.innerHTML = `
                    <p class="ai-result-empty"><i class="fas fa-circle-info" aria-hidden="true"></i>No pauses found. The audio has no silent gaps longer than half a second.</p>`;
                showToast('No pauses found.', 'info');
                return;
            }

            // Draw silence regions
            data.forEach((region, i) => {
                addAiMarker(region.start, region.end, MARKER_COLORS.silence, `Pause ${i+1}`);
            });

            // Populate results HTML
            let rowsHtml = '';
            data.forEach((region, i) => {
                rowsHtml += `
                    <div class="ai-stat-row">
                        <span class="ai-stat-label">Pause ${i+1} · ${formatTimePrecise(region.duration)}</span>
                        <span class="ai-stat-value">${formatTimePrecise(region.start)} – ${formatTimePrecise(region.end)}</span>
                    </div>`;
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p>Found <strong>${data.length}</strong> pause${data.length === 1 ? '' : 's'}, shaded grey on the waveform.</p>
                    <div class="ai-result-list">
                        ${rowsHtml}
                    </div>
                    <div class="ai-result-actions">
                        <button type="button" class="ai-action-btn" id="aiSplitSilencesBtn">
                            <i class="fas fa-scissors" aria-hidden="true"></i> Split at pauses
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiClearSilenceOverlayBtn">
                            <i class="fas fa-eraser" aria-hidden="true"></i> Clear markers
                        </button>
                    </div>
                </div>`;

            // Action: Clear overlay
            document.getElementById('aiClearSilenceOverlayBtn').onclick = () => {
                clearAiMarkers();
                resultsPanel.classList.add('hidden');
            };

            // Action: Auto-split at silences
            document.getElementById('aiSplitSilencesBtn').onclick = () => {
                if (data.length === 0) return;
                
                // Clear existing user regions
                allRegions.forEach(r => r.region.remove());
                allRegions = [];
                selectedRegion = null;

                const duration = wavesurfer.getDuration();
                
                // Calculate non-silent regions from silence regions
                let nonSilents = [];
                let current = 0.0;
                
                data.forEach(silence => {
                    if (silence.start - current >= 0.1) {
                        nonSilents.push({ start: current, end: silence.start });
                    }
                    current = silence.end;
                });
                if (duration - current >= 0.1) {
                    nonSilents.push({ start: current, end: duration });
                }

                // Add non-silent regions to the timeline
                nonSilents.forEach((ns, idx) => {
                    addRegion(ns.start, ns.end, `Part ${idx+1}`, false);
                });

                pushUndo('Split at Silences', nonSilents);
                updateRegionList();
                clearAiMarkers();
                resultsPanel.classList.add('hidden');
                showToast(`Split into ${nonSilents.length} regions.`, 'success');
            };

            showToast(`Found ${data.length} pause${data.length === 1 ? '' : 's'}.`, 'success');
        });
    };

    // AI Auto Trim silence
    document.getElementById('aiTrimBtn').onclick = () => {
        runAIFeature('aiTrimBtn', '/ai/auto-trim', { threshold: 40 }, (data) => {
            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            setResultsTitle('Trim silent edges');
            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p>Sound starts and ends at these points. Apply them to keep only the audible part.</p>
                    <div class="ai-result-list">
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">Original length</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.total_duration)}</span>
                        </div>
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">New start</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.trimmed_start)} (−${formatTimePrecise(data.removed_start_ms/1000)})</span>
                        </div>
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">New end</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.trimmed_end)} (−${formatTimePrecise(data.removed_end_ms/1000)})</span>
                        </div>
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">New length</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.trimmed_end - data.trimmed_start)}</span>
                        </div>
                    </div>
                    <div class="ai-result-actions">
                        <button type="button" class="ai-action-btn" id="aiApplyTrimBtn">
                            <i class="fas fa-crop-simple" aria-hidden="true"></i> Apply as region
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiCloseTrimBtn">
                            Dismiss
                        </button>
                    </div>
                </div>`;

            document.getElementById('aiCloseTrimBtn').onclick = () => {
                resultsPanel.classList.add('hidden');
            };

            // Action: Apply trim points
            document.getElementById('aiApplyTrimBtn').onclick = () => {
                // Clear user regions
                allRegions.forEach(r => r.region.remove());
                allRegions = [];
                selectedRegion = null;

                // Add trimmed region
                const name = "Trimmed";
                addRegion(data.trimmed_start, data.trimmed_end, name);
                
                // Highlight the new region
                if (allRegions.length > 0) {
                    selectRegion(allRegions[0].id);
                }

                resultsPanel.classList.add('hidden');
                showToast('Trim applied as a region.', 'success');
            };

            showToast('Trim points found.', 'success');
        });
    };

    // AI Beat & BPM detection
    document.getElementById('aiBeatBtn').onclick = () => {
        runAIFeature('aiBeatBtn', '/ai/detect-beats', {}, (data) => {
            clearAiMarkers();
            detectedBeats = data.beat_times || [];

            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            // Draw beat lines
            detectedBeats.forEach(t => {
                addAiMarker(t, t + 0.02, MARKER_COLORS.beat);
            });

            setResultsTitle('Tempo and beats');
            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <div class="ai-bpm-container">
                        <div class="ai-bpm-badge">
                            <span class="ai-bpm-number">${escapeHtml(data.bpm)}</span>
                            <span class="ai-bpm-label">BPM</span>
                        </div>
                        <div class="ai-bpm-details">
                            <div class="ai-bpm-detail-item">
                                <i class="fas fa-drum" aria-hidden="true"></i> <span><strong>${escapeHtml(data.total_beats)}</strong> beats</span>
                            </div>
                            <div class="ai-bpm-detail-item">
                                <i class="fas fa-clock" aria-hidden="true"></i> <span><strong>${(60 / data.bpm).toFixed(3)} s</strong> between beats</span>
                            </div>
                        </div>
                    </div>

                    <div class="ai-snap-container">
                        <input type="checkbox" id="aiSnapCheckbox" ${snapToBeatsEnabled ? 'checked' : ''}>
                        <label for="aiSnapCheckbox">Snap region edges to the nearest beat</label>
                    </div>

                    <div class="ai-result-actions">
                        <button type="button" class="ai-action-btn-outline" id="aiClearBeatsOverlayBtn">
                            <i class="fas fa-eraser" aria-hidden="true"></i> Clear markers
                        </button>
                    </div>
                </div>`;

            // Snap checkbox toggle
            document.getElementById('aiSnapCheckbox').onchange = (e) => {
                snapToBeatsEnabled = e.target.checked;
                showToast(snapToBeatsEnabled ? 'Region edges now snap to beats.' : 'Snap to beats turned off.', 'info');
            };

            document.getElementById('aiClearBeatsOverlayBtn').onclick = () => {
                clearAiMarkers();
                detectedBeats = [];
                snapToBeatsEnabled = false;
                resultsPanel.classList.add('hidden');
            };

            showToast(`Tempo: ${data.bpm} BPM.`, 'success');
        });
    };

    // Helper to find closest beat time
    function getClosestBeat(time) {
        if (detectedBeats.length === 0) return time;
        let closest = detectedBeats[0];
        let minDiff = Math.abs(time - closest);
        for (let i = 1; i < detectedBeats.length; i++) {
            let diff = Math.abs(time - detectedBeats[i]);
            if (diff < minDiff) {
                minDiff = diff;
                closest = detectedBeats[i];
            }
        }
        return closest;
    }

    // Modify region update logic in script.js to support snap to beats!
    // We will hook into wavesurfer region update events if snap is enabled
    wsRegions.on('region-updated', (region) => {
        if (isSnapping) return;
        if (snapToBeatsEnabled && detectedBeats.length > 0) {
            const currentStart = region.start;
            const currentEnd = region.end;
            const snappedStart = getClosestBeat(currentStart);
            const snappedEnd = getClosestBeat(currentEnd);

            // Avoid collapsing the region
            if (snappedEnd > snappedStart) {
                if (region.start !== snappedStart || region.end !== snappedEnd) {
                    isSnapping = true;
                    try {
                        region.setOptions({
                            start: snappedStart,
                            end: snappedEnd
                        });
                    } finally {
                        isSnapping = false;
                    }
                }
            }
        }
        updateRegionList();
    });

    // AI Noise Reduction
    document.getElementById('aiDenoiseBtn').onclick = () => {
        runAIFeature('aiDenoiseBtn', '/ai/noise-reduce', {}, (blob) => {
            // Re-load the denoised blob into Wavesurfer
            loadAudio(blob);

            // Keep reference to it so exporting works with the denoised audio
            recordedBlob = blob;

            // Clear fileInput value so we upload the new blob rather than the old local file input
            fileInput.value = '';

            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            setResultsTitle('Remove noise');
            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p class="ai-result-status"><i class="fas fa-circle-check" aria-hidden="true"></i> Noise removed</p>
                    <p>The cleaned audio replaced the original in the editor. Export as usual, or save the full cleaned file.</p>
                    <div class="ai-result-actions">
                        <button type="button" class="ai-action-btn" id="aiDownloadDenoisedBtn">
                            <i class="fas fa-download" aria-hidden="true"></i> Save cleaned file
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiDismissDenoiseBtn">
                            Dismiss
                        </button>
                    </div>
                </div>`;

            // Action: Save Denoised File directly
            document.getElementById('aiDownloadDenoisedBtn').onclick = () => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `denoised_audio.wav`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            };

            document.getElementById('aiDismissDenoiseBtn').onclick = () => {
                resultsPanel.classList.add('hidden');
            };

            showToast('Noise removed. The cleaned audio is now loaded.', 'success');
        });
    };

    // AI Filler Word Detection
    if (document.getElementById('aiFillerWordsBtn')) {
        document.getElementById('aiFillerWordsBtn').onclick = () => {
            runAIFeature('aiFillerWordsBtn', '/ai/filler-words', {}, (data) => {
                clearAiMarkers();
                const resultsPanel = document.getElementById('aiResultsPanel');
                const resultsContent = document.getElementById('aiResultsContent');
                resultsPanel.classList.remove('hidden');
                setResultsTitle('Filler words');

                const fillers = data.fillers || [];
                if (fillers.length === 0) {
                    resultsContent.innerHTML = `<p class="ai-result-empty"><i class="fas fa-circle-info" aria-hidden="true"></i>No filler words found (“um”, “uh”, “like”).</p>`;
                    showToast('No filler words found.', 'info');
                    return;
                }

                fillers.forEach(f => {
                    addAiMarker(f.start, f.end, MARKER_COLORS.filler, escapeHtml(f.word));
                });

                resultsContent.innerHTML = `
                    <div class="transcript-container">
                        <p>Found <strong>${fillers.length}</strong> filler word${fillers.length === 1 ? '' : 's'}, marked in red on the waveform.</p>
                        <div class="ai-result-actions">
                            <button type="button" class="ai-action-btn" id="aiCutFillersBtn">
                                <i class="fas fa-scissors" aria-hidden="true"></i> Turn into regions
                            </button>
                            <button type="button" class="ai-action-btn-outline" id="aiClearFillersBtn">
                                <i class="fas fa-eraser" aria-hidden="true"></i> Clear markers
                            </button>
                        </div>
                    </div>`;

                document.getElementById('aiClearFillersBtn').onclick = () => {
                    clearAiMarkers();
                    resultsPanel.classList.add('hidden');
                };

                document.getElementById('aiCutFillersBtn').onclick = () => {
                    allRegions.forEach(r => r.region.remove());
                    allRegions = [];
                    fillers.forEach(f => addRegion(f.start, f.end, `Filler: ${f.word}`));
                    resultsPanel.classList.add('hidden');
                    showToast(`Marked ${fillers.length} filler word${fillers.length === 1 ? '' : 's'} as regions.`, 'success');
                };
            });
        };
    }

    // AI Speech Studio Enhancement
    if (document.getElementById('aiEnhanceSpeechBtn')) {
        document.getElementById('aiEnhanceSpeechBtn').onclick = () => {
            runAIFeature('aiEnhanceSpeechBtn', '/ai/enhance-speech', {}, (blob, resp) => {
                loadAudio(blob);
                recordedBlob = blob;
                fileInput.value = '';

                const resultsPanel = document.getElementById('aiResultsPanel');
                const resultsContent = document.getElementById('aiResultsContent');
                resultsPanel.classList.remove('hidden');

                const note = (resp && resp.headers && resp.headers.get('x-enhance-note')) || '';
                const noteHtml = note
                    ? `<p class="separate-quality-note" style="margin-top: var(--space-2);"><i class="fas fa-info-circle" aria-hidden="true"></i> ${note.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>`
                    : '';

                setResultsTitle('Enhance speech');
                resultsContent.innerHTML = `
                    <div class="transcript-container">
                        <p class="ai-result-status"><i class="fas fa-circle-check" aria-hidden="true"></i> Speech enhanced</p>
                        <p>Room echo was reduced and the voice evened out. The enhanced audio replaced the original in the editor.</p>
                        ${noteHtml}
                        <div class="ai-result-actions">
                            <button type="button" class="ai-action-btn" id="aiSaveEnhancedBtn">
                                <i class="fas fa-download" aria-hidden="true"></i> Save enhanced file
                            </button>
                        </div>
                    </div>`;

                document.getElementById('aiSaveEnhancedBtn').onclick = () => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'enhanced_speech.wav';
                    a.click();
                    a.remove();
                };
                showToast('Speech enhanced. The new audio is now loaded.', 'success');
            });
        };
    }

    // Separate vocals tool: opens the Separate panel (vocals / stems / voice / lyrics).
    if (document.getElementById('aiSeparateStemsBtn')) {
        document.getElementById('aiSeparateStemsBtn').onclick = () => {
            const panel = document.getElementById('separatePanel');
            if (!panel) return;
            const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            panel.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
            const active = panel.querySelector('#separateModes .tab-btn.active');
            if (active) active.focus({ preventScroll: true });
        };
    }

    // ─── SEPARATE PANEL ───
    (function initSeparatePanel() {
        const panel = document.getElementById('separatePanel');
        if (!panel) return;
        const modeButtons = Array.from(panel.querySelectorAll('#separateModes .tab-btn'));
        const desc = document.getElementById('separateModeDesc');
        const formatSel = document.getElementById('separateFormat');
        const runBtn = document.getElementById('separateRunBtn');
        const note = document.getElementById('separateQualityNote');
        const progress = document.getElementById('separateProgress');
        const progressText = document.getElementById('separateProgressText');
        const results = document.getElementById('separateResults');
        const MODE_TEXT = {
            vocals: 'Two tracks: the singing voice and everything else, for karaoke or remixing.',
            '4stem': 'Four tracks: vocals, drums, bass and other instruments.',
            voice: 'Keeps the spoken or sung voice and removes music and background noise.',
            lyrics: 'Isolates the vocals, then writes down the words with timings you can save as subtitles.'
        };
        const FAILED = 'Separation could not be completed. Please try again.';
        let mode = 'vocals';
        let busy = false;

        function selectMode(btn, focus) {
            modeButtons.forEach(b => {
                const on = b === btn;
                b.classList.toggle('active', on);
                b.setAttribute('aria-checked', on ? 'true' : 'false');
                b.tabIndex = on ? 0 : -1;
            });
            mode = btn.dataset.mode;
            desc.textContent = MODE_TEXT[mode] || '';
            const lyrics = mode === 'lyrics';
            formatSel.disabled = lyrics;
            const formatWrap = formatSel.closest('.separate-format');
            if (formatWrap) formatWrap.classList.toggle('is-disabled', lyrics);
            runBtn.querySelector('span').textContent = lyrics ? 'Get lyrics' : 'Separate';
            if (focus) btn.focus();
        }
        modeButtons.forEach((btn, i) => {
            btn.addEventListener('click', () => selectMode(btn, false));
            btn.addEventListener('keydown', (e) => {
                const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
                if (!step) return;
                e.preventDefault();
                selectMode(modeButtons[(i + step + modeButtons.length) % modeButtons.length], true);
            });
        });

        fetch('/ai/separate/capabilities').then(r => (r.ok ? r.json() : null)).then(info => {
            if (!info) { note.textContent = ''; return; }
            note.textContent = info.studio_stems_installed
                ? 'Studio stems quality is available. Long files are processed in sections.'
                : 'Using Quick separation, which works best on stereo songs with centred vocals. Studio stems give cleaner results and are an optional download.';
        }).catch(() => { note.textContent = ''; });

        function fmtTime(t) {
            const s = Math.max(0, Math.floor(Number(t) || 0));
            return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
        }

        function setBusy(on, text) {
            busy = on;
            runBtn.disabled = on;
            runBtn.setAttribute('aria-busy', on ? 'true' : 'false');
            progress.classList.toggle('hidden', !on);
            if (text) progressText.textContent = text;
        }

        async function errorMessage(resp) {
            try {
                const body = await resp.json();
                return body.error || FAILED;
            } catch (e) {
                return FAILED;
            }
        }

        function warningsHtml(list) {
            if (!list || !list.length) return '';
            return `<ul class="separate-warnings">${list.map(w => `<li>${escapeHtml(w)}</li>`).join('')}</ul>`;
        }

        async function openInEditor(url, name) {
            try {
                const resp = await fetch(url);
                if (!resp.ok) throw new Error('fetch failed');
                const blob = await resp.blob();
                const file = new File([blob], name, { type: blob.type || 'audio/wav' });
                loadAudio(file);
                recordedBlob = file;
                fileInput.value = '';
                showToast(`${name} is now open in the editor.`, 'success');
            } catch (e) {
                showToast('That stem could not be opened. Try downloading it instead.', 'error');
            }
        }

        function renderStems(data) {
            const rows = data.stems.map((s, i) => {
                const loud = s.loudness || {};
                const meta = [];
                if (typeof loud.integrated_lufs === 'number') meta.push(`${loud.integrated_lufs.toFixed(1)} LUFS`);
                if (typeof loud.true_peak_dbtp === 'number') meta.push(`peak ${loud.true_peak_dbtp.toFixed(1)} dBTP`);
                return `
                    <div class="separate-stem" role="group" aria-labelledby="sepStem${i}">
                        <div class="separate-stem-head">
                            <strong id="sepStem${i}">${escapeHtml(s.label)}</strong>
                            <span class="separate-stem-meta">${escapeHtml(meta.join(' · '))}</span>
                        </div>
                        <audio controls preload="none" src="${escapeHtml(s.url)}" aria-label="Preview ${escapeHtml(s.label)}"></audio>
                        <div class="ai-result-actions">
                            <a class="ai-action-btn-outline" href="${escapeHtml(s.url)}" download="${escapeHtml(s.download_name)}">
                                <i class="fas fa-download" aria-hidden="true"></i> Download
                            </a>
                            <button type="button" class="ai-action-btn-outline" data-open-url="${escapeHtml(s.url)}" data-open-name="${escapeHtml(s.download_name)}">
                                <i class="fas fa-pen-to-square" aria-hidden="true"></i> Open in editor
                            </button>
                        </div>
                    </div>`;
            }).join('');
            const zip = data.zip_url ? `
                <div class="ai-result-actions">
                    <a class="ai-action-btn" href="${escapeHtml(data.zip_url)}" download>
                        <i class="fas fa-file-zipper" aria-hidden="true"></i> Download all (ZIP)
                    </a>
                </div>` : '';
            results.innerHTML = `
                <p class="ai-result-status"><i class="fas fa-circle-check" aria-hidden="true"></i> ${escapeHtml(data.mode_label)} finished</p>
                <p class="separate-summary">${escapeHtml(data.quality.label)} · ${escapeHtml(data.quality_note || '')}</p>
                ${warningsHtml(data.warnings)}${rows}${zip}`;
            results.querySelectorAll('[data-open-url]').forEach(btn => {
                btn.addEventListener('click', () => openInEditor(btn.dataset.openUrl, btn.dataset.openName));
            });
        }

        function renderLyrics(data) {
            const lines = (data.lines || []).map(l =>
                `<p><time>${fmtTime(l.start)}</time>${escapeHtml(l.text)}</p>`).join('');
            results.innerHTML = `
                <p class="ai-result-status"><i class="fas fa-circle-check" aria-hidden="true"></i> Lyrics ready</p>
                <p class="separate-summary">The vocals were isolated first (${escapeHtml(data.quality.label)}), then transcribed. Check the words before publishing.</p>
                ${warningsHtml(data.warnings)}
                <div class="separate-lyrics" tabindex="0" role="region" aria-label="Lyrics">${lines || '<p>No words were recognised.</p>'}</div>
                <div class="ai-result-actions">
                    <a class="ai-action-btn" href="${escapeHtml(data.srt_url)}" download><i class="fas fa-closed-captioning" aria-hidden="true"></i> Subtitles (SRT)</a>
                    <a class="ai-action-btn-outline" href="${escapeHtml(data.vtt_url)}" download><i class="fas fa-file-lines" aria-hidden="true"></i> Web subtitles (VTT)</a>
                    <a class="ai-action-btn-outline" href="${escapeHtml(data.txt_url)}" download><i class="fas fa-align-left" aria-hidden="true"></i> Text</a>
                </div>`;
        }

        runBtn.addEventListener('click', async () => {
            if (busy) return;
            const file = getActiveAudioFile();
            if (!file) { showToast('Open or record audio first.', 'warning'); return; }
            const lyrics = mode === 'lyrics';
            const form = new FormData();
            form.append('file', file, file.name || 'audio.webm');
            if (!lyrics) {
                form.append('mode', mode);
                form.append('format', formatSel.value);
                form.append('delivery', 'json');
            }
            results.classList.add('hidden');
            setBusy(true, lyrics ? 'Isolating the vocals and transcribing. This can take a few minutes.'
                                 : 'Separating. This can take up to about the length of the audio.');
            try {
                const resp = await fetch(lyrics ? '/ai/lyrics' : '/ai/separate', { method: 'POST', body: form });
                if (!resp.ok) throw new Error(await errorMessage(resp));
                const data = await resp.json();
                if (lyrics) renderLyrics(data); else renderStems(data);
                results.classList.remove('hidden');
                showToast(lyrics ? 'Lyrics are ready.' : 'Separation finished.', 'success');
            } catch (e) {
                const message = (e && e.message) || FAILED;
                results.innerHTML = `<p class="separate-summary" role="alert">${escapeHtml(message)}</p>`;
                results.classList.remove('hidden');
                showToast(message, 'error');
            } finally {
                setBusy(false);
            }
        });
    })();

    // AI Voice Activity Detection (VAD)
    document.getElementById('aiVadBtn').onclick = () => {
        runAIFeature('aiVadBtn', '/ai/detect-vad', { threshold_db: -35.0 }, (data) => {
            clearAiMarkers();

            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            setResultsTitle('Speech');
            if (!data || data.length === 0) {
                resultsContent.innerHTML = `<p class="ai-result-empty"><i class="fas fa-circle-info" aria-hidden="true"></i>No speech found in this audio.</p>`;
                showToast('No speech found.', 'info');
                return;
            }

            // Draw VAD regions
            let speechCount = 0;
            data.forEach(seg => {
                if (seg.type === 'speech') {
                    speechCount++;
                    addAiMarker(seg.start, seg.end, MARKER_COLORS.speech, `Speech`);
                } else {
                    addAiMarker(seg.start, seg.end, MARKER_COLORS.silence);
                }
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p>Found <strong>${speechCount}</strong> speech section${speechCount === 1 ? '' : 's'}, shaded green on the waveform.</p>
                    <div class="ai-result-actions">
                        <button type="button" class="ai-action-btn" id="aiExtractVocalsBtn">
                            <i class="fas fa-scissors" aria-hidden="true"></i> Turn into regions
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiClearVadOverlayBtn">
                            <i class="fas fa-eraser" aria-hidden="true"></i> Clear markers
                        </button>
                    </div>
                </div>`;

            document.getElementById('aiClearVadOverlayBtn').onclick = () => {
                clearAiMarkers();
                resultsPanel.classList.add('hidden');
            };

            // Action: Auto-create regions for vocal speech blocks only
            document.getElementById('aiExtractVocalsBtn').onclick = () => {
                // Clear user regions
                allRegions.forEach(r => r.region.remove());
                allRegions = [];
                selectedRegion = null;

                let count = 1;
                data.forEach(seg => {
                    if (seg.type === 'speech') {
                        addRegion(seg.start, seg.end, `Speech ${count++}`, false);
                    }
                });

                pushUndo('Extract Speech Regions', data);
                updateRegionList();
                clearAiMarkers();
                resultsPanel.classList.add('hidden');
                showToast(`Created ${count - 1} speech region${count - 1 === 1 ? '' : 's'}.`, 'success');
            };

            showToast(`Found ${speechCount} speech section${speechCount === 1 ? '' : 's'}.`, 'success');
        });
    };

    // AI Speech-to-Text Transcription
    const transcribeBtn = document.getElementById('aiTranscribeBtn');
    if (transcribeBtn) {
        transcribeBtn.onclick = () => {
            runAIFeature('aiTranscribeBtn', '/ai/transcribe', {}, (data) => {
            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            setResultsTitle('Transcript');
            if (!data.available) {
                resultsContent.innerHTML = `
                    <div class="ai-result-error" role="alert">
                        <h4><i class="fas fa-triangle-exclamation" aria-hidden="true"></i> Transcription unavailable</h4>
                        <p>${escapeHtml(cleanServerError(data.error))}</p>
                    </div>`;
                showToast('Transcription isn’t available on this computer.', 'warning');
                return;
            }

            // Draw segment markers as gray markers on wavesurfer
            clearAiMarkers();

            let linesHtml = '';
            data.segments.forEach((seg, i) => {
                linesHtml += `
                    <div class="transcript-line" data-start="${seg.start}" data-end="${seg.end}" id="transcriptLine_${i}" tabindex="0" role="button" aria-label="Jump to ${formatTimePrecise(seg.start)}: ${escapeHtml(seg.text)}">
                        <span class="transcript-time">${formatTimePrecise(seg.start)}</span>
                        <span class="transcript-text">${escapeHtml(seg.text)}</span>
                    </div>`;
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <div class="transcript-meta">
                        <span>Language: <strong>${escapeHtml(String(data.language || '').toUpperCase())}</strong> · ${data.segments.length} line${data.segments.length === 1 ? '' : 's'}</span>
                        <button type="button" class="mini-btn" id="copyTranscriptBtn" title="Copy the full transcript">
                            <i class="fas fa-copy" aria-hidden="true"></i> Copy
                        </button>
                    </div>
                    <div class="transcript-panel" id="transcriptLinesContainer">
                        ${linesHtml}
                    </div>
                </div>`;

            // Action: Click a line in transcript to jump playhead + highlight
            const lines = document.getElementById('transcriptLinesContainer').querySelectorAll('.transcript-line');
            lines.forEach(line => {
                line.onclick = () => {
                    const start = parseFloat(line.dataset.start);
                    wavesurfer.setTime(start);

                    // Highlight active
                    lines.forEach(l => l.classList.remove('active'));
                    line.classList.add('active');
                };
                line.onkeydown = (ev) => {
                    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); line.click(); }
                };
            });

            // Playhead highlight sync is handled by the global audio process listener

            // Action: Copy Transcript to clipboard
            document.getElementById('copyTranscriptBtn').onclick = () => {
                navigator.clipboard.writeText(data.full_text).then(() => {
                    showToast('Transcript copied.', 'success');
                }).catch(() => {
                    showToast('Couldn’t copy the transcript. Select the text and copy it manually.', 'error');
                });
            };

            showToast('Transcript ready. Click a line to jump there.', 'success');
        });
    };
}

    // Results panel close button
    document.getElementById('closeAiResults').onclick = () => {
        document.getElementById('aiResultsPanel').classList.add('hidden');
        clearAiMarkers();
    };

    // Global playhead tracking for transcript sync (called on play and seek)
    function syncTranscriptPlayhead() {
        const cur = wavesurfer.getCurrentTime();
        const container = document.getElementById('transcriptLinesContainer');
        if (!container) return;
        const lines = container.querySelectorAll('.transcript-line');
        if (lines.length === 0) return;
        
        let activeIdx = -1;
        lines.forEach((line, idx) => {
            const start = parseFloat(line.dataset.start);
            const end = parseFloat(line.dataset.end);
            if (cur >= start && cur <= end) {
                activeIdx = idx;
            }
        });

        if (activeIdx !== -1) {
            lines.forEach((l, idx) => {
                if (idx === activeIdx) {
                    if (!l.classList.contains('active')) {
                        l.classList.add('active');
                        l.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                } else {
                    l.classList.remove('active');
                }
            });
        }
    }

    wavesurfer.on('audioprocess', syncTranscriptPlayhead);
    wavesurfer.on('seeking', syncTranscriptPlayhead);

    const autoTrimSilenceBtn = document.getElementById('aiAutoTrimSilenceBtn');
    if (autoTrimSilenceBtn) {
        autoTrimSilenceBtn.onclick = async () => {
            if (!fileInput.files[0] && !recordedBlob) {
                showToast('Open or record audio first.', 'warning');
                return;
            }

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.show({
                    title: 'Removing long pauses',
                    stageText: 'Cutting silent gaps longer than 1 second…',
                    category: 'audio'
                });
            } else {
                showToast('Removing long pauses…', 'info');
            }

            try {
                const fd = new FormData();
                if (recordedBlob && !fileInput.files[0]) {
                    fd.append('file', recordedBlob, 'audio.webm');
                } else {
                    fd.append('file', fileInput.files[0]);
                }
                fd.append('min_silence_len', '1.0');

                const res = await fetch('/ai/trim-silence', { method: 'POST', body: fd });
                if (!res.ok) {
                    const e = await res.json().catch(() => ({}));
                    showToast(`Removing pauses failed: ${e.error || 'the server returned an error. Try again.'}`, 'error');
                    return;
                }

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'silence_trimmed_audio.wav';
                a.click();
                URL.revokeObjectURL(url);
                showToast('Long pauses removed. The new file was downloaded.', 'success');
            } catch (e) {
                showToast('Removing pauses failed: ' + e.message, 'error');
            } finally {
                if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
            }
        };
    }

    // ─── VOICEOVER (text to speech) — self-contained panel ───
    function initVoiceoverPanel() {
        const panel = document.getElementById('voiceoverPanel');
        if (!panel) return;
        const form = document.getElementById('voiceoverForm');
        const textEl = document.getElementById('voiceoverText');
        const countEl = document.getElementById('voiceoverCount');
        const voiceEl = document.getElementById('voiceoverVoice');
        const speedEl = document.getElementById('voiceoverSpeed');
        const speedValueEl = document.getElementById('voiceoverSpeedValue');
        const formatEl = document.getElementById('voiceoverFormat');
        const generateBtn = document.getElementById('voiceoverGenerateBtn');
        const statusEl = document.getElementById('voiceoverStatus');
        const resultEl = document.getElementById('voiceoverResult');
        const audioEl = document.getElementById('voiceoverAudio');
        const openBtn = document.getElementById('voiceoverOpenBtn');
        const downloadEl = document.getElementById('voiceoverDownload');
        let maxChars = Number(textEl.getAttribute('maxlength')) || 5000;
        let voicesReady = false;
        let busy = false;
        let lastBlob = null;
        let lastUrl = null;

        const setStatus = (message, tone) => {
            statusEl.textContent = message || '';
            statusEl.classList.toggle('is-error', tone === 'error');
            statusEl.classList.toggle('is-warning', tone === 'warning');
        };
        const updateCount = () => {
            const length = textEl.value.length;
            countEl.textContent = `${length.toLocaleString()} / ${maxChars.toLocaleString()} characters`;
            countEl.classList.toggle('is-error', length > maxChars);
            generateBtn.disabled = busy || !voicesReady || !textEl.value.trim() || length > maxChars;
        };
        const updateSpeed = () => {
            const value = Number(speedEl.value).toFixed(2).replace(/0$/, '');
            speedValueEl.textContent = `${value}x`;
            speedEl.setAttribute('aria-valuetext', `${value} times`);
        };

        async function loadVoices() {
            try {
                const res = await fetch('/ai/tts/voices');
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Voices could not be loaded.');
                maxChars = data.max_chars || maxChars;
                textEl.setAttribute('maxlength', String(maxChars));
                voiceEl.innerHTML = '';
                const groups = {};
                (data.voices || []).forEach((voice) => {
                    if (!groups[voice.quality]) {
                        groups[voice.quality] = document.createElement('optgroup');
                        groups[voice.quality].label = voice.quality;
                        voiceEl.appendChild(groups[voice.quality]);
                    }
                    const option = document.createElement('option');
                    option.value = voice.id;
                    option.textContent = voice.label;
                    option.selected = Boolean(voice.default);
                    groups[voice.quality].appendChild(option);
                });
                voicesReady = Boolean(data.voices && data.voices.length);
                voiceEl.disabled = !voicesReady;
                if (!voicesReady) {
                    voiceEl.innerHTML = '<option value="">No voices available</option>';
                    setStatus('Voiceover isn’t available on this computer yet.', 'warning');
                }
            } catch (err) {
                voiceEl.innerHTML = '<option value="">Voices unavailable</option>';
                setStatus('Voices could not be loaded. Reload the page to try again.', 'error');
            }
            updateCount();
        }

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (generateBtn.disabled) return;
            busy = true;
            updateCount();
            generateBtn.setAttribute('aria-busy', 'true');
            setStatus('Creating voiceover…');
            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.show({ title: 'Creating voiceover', stageText: 'Reading your script aloud…', category: 'audio' });
                window.ProcessingOverlay.updateProgress(35, 'Reading your script aloud…');
            }
            try {
                const format = formatEl.value;
                const res = await fetch('/ai/tts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textEl.value, voice: voiceEl.value, speed: Number(speedEl.value), format })
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.error || 'The voiceover could not be created. Try again.');
                }
                lastBlob = await res.blob();
                if (lastUrl) URL.revokeObjectURL(lastUrl);
                lastUrl = URL.createObjectURL(lastBlob);
                audioEl.src = lastUrl;
                downloadEl.href = lastUrl;
                downloadEl.setAttribute('download', `voiceover.${format}`);
                resultEl.classList.remove('hidden');
                const seconds = Number(res.headers.get('X-Audio-Duration') || 0);
                const quality = res.headers.get('X-Voice-Quality') || 'Voice';
                const notice = res.headers.get('X-Voice-Notice');
                setStatus(`${quality}, ${seconds.toFixed(1)} seconds.${notice ? ' ' + notice : ''}`, notice ? 'warning' : null);
                showToast('Voiceover ready.', 'success');
            } catch (err) {
                setStatus(err.message, 'error');
                showToast(`Voiceover failed: ${err.message}`, 'error');
            } finally {
                busy = false;
                generateBtn.removeAttribute('aria-busy');
                updateCount();
                if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
            }
        });

        openBtn.addEventListener('click', () => {
            if (!lastBlob) return;
            const ext = (formatEl.value === 'mp3') ? 'mp3' : 'wav';
            const file = new File([lastBlob], `voiceover.${ext}`, { type: lastBlob.type || `audio/${ext}` });
            recordedBlob = file;
            fileInput.value = '';
            handleFileLoad(file);
            document.getElementById('main')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            showToast('Voiceover opened in the editor.', 'success');
        });

        textEl.addEventListener('input', updateCount);
        speedEl.addEventListener('input', updateSpeed);
        updateSpeed();
        updateCount();
        loadVoices();
    }
    initVoiceoverPanel();

    // Initialize undo button state
    updateUndoBtn();
});