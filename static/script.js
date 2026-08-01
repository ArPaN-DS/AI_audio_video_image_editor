/* ═══════════════════════════════════════
   Audio Cutter Pro — Main Script
   Next-Level UX Edition
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

// Theme-aware waveform colors — read live design tokens so the
// waveform matches whichever theme (light / dark) is active.
function themeWaveColors() {
    const cs = getComputedStyle(document.documentElement);
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        waveColor: dark ? '#3A4A68' : '#CBD5E1',
        progressColor: (cs.getPropertyValue('--primary').trim() || '#2563EB'),
        cursorColor: (cs.getPropertyValue('--text').trim() || '#1E293B'),
    };
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

    function switchTab(mode) {
        document.getElementById('btnTabUpload').classList.toggle('active', mode === 'upload');
        document.getElementById('btnTabRecord').classList.toggle('active', mode === 'record');
        uploadSec.classList.toggle('hidden', mode !== 'upload');
        recordSec.classList.toggle('hidden', mode !== 'record');
        editorSec.classList.add('hidden');
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
            showToast('Please drop an audio or video file.', 'warning');
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
            recStatus.textContent = 'Recording... Click Stop when done.';
            stopBtn.classList.remove('hidden');
            recBtn.disabled = true;
        } catch (err) {
            recStatus.textContent = 'Microphone access denied.';
        }
    };

    stopBtn.onclick = () => {
        mediaRecorder.stop();
        recBtn.classList.remove('recording');
        recStatus.textContent = 'Recording finished!';
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

        editorSec.classList.remove('hidden');
        editorSec.style.display = '';
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

        // Show helpful hint for first-time users
        showToast('💡 Double-click the waveform to add cut regions!', 'info');
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
        document.getElementById('playBtn').classList.add('playing');
    });
    wavesurfer.on('pause', () => {
        document.getElementById('playIcon').className = 'fas fa-play';
        document.getElementById('playBtn').classList.remove('playing');
    });

    // ─── 6. REGION MANAGEMENT ───
    function addRegion(start, end, name, trackUndo = true) {
        const alpha = 0.12 + Math.random() * 0.13;
        const region = wsRegions.addRegion({
            start: start,
            end: end,
            color: `rgba(37, 99, 235, ${alpha})`,
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
            r.region.element.style.border = 'none';
            r.region.element.style.boxShadow = 'none';
        });
        document.querySelectorAll('.region-item').forEach(el => el.classList.remove('selected'));

        const regionData = allRegions.find(r => r.id === regionId);
        if (regionData) {
            selectedRegion = regionData;
            regionData.region.element.style.border = '2px solid #2563EB';
            regionData.region.element.style.boxShadow = '0 0 12px rgba(37, 99, 235, 0.35)';

            const idx = allRegions.indexOf(regionData);
            const items = document.querySelectorAll('.region-item');
            if (items[idx]) {
                items[idx].classList.add('selected');
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
            list.innerHTML = '<p class="empty-state"><i class="fas fa-info-circle"></i> No regions yet. Double-click the waveform or press "Add Region" to start cutting.</p>';
            return;
        }

        allRegions.forEach((r, index) => {
            const div = document.createElement('div');
            div.className = 'region-item';
            const dur = r.region.end - r.region.start;

            div.innerHTML = `
                <div class="region-info">
                    <div class="region-name">
                        <i class="fas fa-wave-square" style="color:var(--primary);margin-right:6px;font-size:0.8rem;"></i>
                        <span class="region-name-text" data-id="${r.id}">${r.name}</span>
                    </div>
                    <div class="region-time">${formatTimePrecise(r.region.start)} → ${formatTimePrecise(r.region.end)}  ·  ${formatTimePrecise(dur)}</div>
                </div>
                <div class="region-actions">
                    <button class="region-play-btn" data-id="${r.id}" title="Play just this region">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="region-delete" data-id="${r.id}" title="Delete this region">
                        <i class="fas fa-trash-alt"></i>
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
    }

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
            showToast(`"${removed.name}" removed. Press Ctrl+Z to undo.`, 'info');
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
            showToast(`↩ Restored "${last.data.name}"`, 'success');
        } else if (last.action === 'Add Region') {
            // Remove the last added region
            const idx = allRegions.findIndex(r => r.id === last.data.id);
            if (idx !== -1) {
                allRegions[idx].region.remove();
                allRegions.splice(idx, 1);
                selectedRegion = null;
                updateRegionList();
                showToast('↩ Region removed', 'success');
            }
        } else if (last.action === 'Clear All') {
            // Re-add all cleared regions
            last.data.forEach(d => {
                addRegion(d.start, d.end, d.name, false);
            });
            showToast(`↩ Restored ${last.data.length} region(s)`, 'success');
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
        showToast(`Cleared all regions. Press Ctrl+Z to undo.`, 'warning');
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
        wavesurfer.zoom(Number(this.value));
    };

    // ─── 9. VOLUME CONTROL ───
    const volumeSlider = document.getElementById('volumeSlider');
    volumeSlider.value = currentVolume * 100;

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
    }

    // ─── 10. SPEED CONTROL ───
    document.getElementById('speedControl').onchange = function () {
        wavesurfer.setPlaybackRate(Number(this.value));
    };

    // ─── 11. NEW FILE / REMOVE AUDIO BUTTONS ───

    function resetToUpload() {
        console.log('[AudioCutter] resetToUpload called');
        isResetting = true;

        // FIRST: Force-hide editor with both methods
        const ei = document.getElementById('editor-interface');
        const us = document.getElementById('uploadSection');
        const rs = document.getElementById('recordSection');

        ei.style.display = 'none';
        ei.classList.add('hidden');

        us.style.display = '';
        us.classList.remove('hidden');

        rs.style.display = 'none';
        rs.classList.add('hidden');

        document.getElementById('btnTabUpload').classList.add('active');
        document.getElementById('btnTabRecord').classList.remove('active');
        document.getElementById('newFileBtn').classList.add('hidden');

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

        showToast('Audio removed. Upload a new file to continue.', 'info');

        // Allow future wavesurfer 'ready' events after a delay
        setTimeout(() => { isResetting = false; }, 500);
    }

    document.getElementById('newFileBtn').addEventListener('click', resetToUpload);
    document.getElementById('removeAudioBtn').addEventListener('click', resetToUpload);

    // ─── 12. KEYBOARD SHORTCUTS ───
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

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
            document.getElementById('shortcutsOverlay').classList.remove('hidden');
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

    // Help overlay
    document.getElementById('helpBtn').onclick = () =>
        document.getElementById('shortcutsOverlay').classList.remove('hidden');

    document.getElementById('closeShortcuts').onclick = () =>
        document.getElementById('shortcutsOverlay').classList.add('hidden');

    document.getElementById('shortcutsOverlay').onclick = (e) => {
        if (e.target.id === 'shortcutsOverlay') e.target.classList.add('hidden');
    };

    // ─── 13. FORM SUBMISSION ───
    document.getElementById('cutForm').onsubmit = async (e) => {
        e.preventDefault();

        if (allRegions.length === 0) {
            showToast('Please add at least one region before exporting.', 'warning');
            return;
        }

        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.show({
                title: 'Audio Export Engine',
                stageText: `Processing ${allRegions.length} audio region(s) & preparing export...`,
                category: 'audio'
            });
            window.ProcessingOverlay.updateProgress(35, 'Cutting waveform regions & rendering audio output...');
        }

        const formData = new FormData(e.target);

        // Attach file
        if (recordedBlob && !fileInput.files[0]) {
            formData.append('file', recordedBlob, 'recording.webm');
        } else if (fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        } else {
            showToast('No audio source found.', 'error');
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
                window.ProcessingOverlay.updateProgress(75, 'Encoding final audio file...');
            }

            const resp = await fetch('/cut', { method: 'POST', body: formData });

            if (resp.ok) {
                if (window.ProcessingOverlay) {
                    window.ProcessingOverlay.updateProgress(100, '✅ Export complete! Downloading...');
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

                showToast('✅ Your audio has been exported successfully!', 'success');
                setTimeout(() => {
                    if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
                }, 1000);
            } else {
                const errText = await resp.text();
                showToast(`Error: ${errText}`, 'error');
                if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
            }
        } catch (err) {
            console.error(err);
            showToast('Network error. Check your connection.', 'error');
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

    function showToast(message, type = 'info') {
        // Remove old toasts
        document.querySelectorAll('.app-toast').forEach(t => t.remove());

        const toast = document.createElement('div');
        toast.className = 'app-toast';
        const colors = {
            error: '#e74c3c',
            warning: '#f39c12',
            success: '#27ae60',
            info: '#2c3e50'
        };
        toast.innerHTML = `<span>${message}</span>`;
        toast.style.cssText = `
            position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%) translateY(20px);
            background: ${colors[type] || colors.info};
            color: white; padding: 14px 28px; border-radius: 12px; font-size: 0.88rem;
            font-weight: 600; z-index: 99999; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            opacity: 0; transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex; align-items: center; gap: 8px; font-family: 'Inter', sans-serif;
            max-width: 480px;
        `;
        document.body.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(-50%) translateY(0)';
        });

        // Animate out
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(20px)';
        }, 3000);
        setTimeout(() => toast.remove(), 3500);
    }

    // ─── 14. 🤖 AI AUDIO PROCESSING UX ───
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
            showToast('Please upload or record an audio file first.', 'warning');
            return;
        }

        const btn = document.getElementById(btnId);
        if (!btn) return;

        btn.classList.add('loading');
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Running...`;

        const formData = new FormData();
        formData.append('file', file, file.name || 'audio.webm');
        for (const [key, val] of Object.entries(extraParams)) {
            formData.append(key, val);
        }

        // Map endpoint to natural language titles & stage descriptions
        const aiMeta = {
            '/ai/transcribe': { title: 'AI Speech-to-Text Studio', stage: 'Transcribing spoken voice using local Whisper AI model...' },
            '/ai/detect-silence': { title: 'AI Silence Detection', stage: 'Scanning audio waveform to locate silent gaps & pauses...' },
            '/ai/auto-trim': { title: 'AI Auto-Trimmer', stage: 'Analyzing audio thresholds & auto-trimming silence...' },
            '/ai/noise-reduce': { title: 'AI Noise Reduction Engine', stage: 'Filtering background noise & leveling audio voice clarity...' },
            '/ai/bpm': { title: 'BPM & Beat Tracker', stage: 'Tracking rhythm beats & calculating tempo (BPM)...' }
        };
        const meta = aiMeta[endpoint] || { title: 'AI Audio Processor', stage: 'Processing audio file using AI models...' };

        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.show({
                title: meta.title,
                stageText: meta.stage,
                category: 'audio'
            });
            window.ProcessingOverlay.updateProgress(35, meta.stage);
        }

        try {
            showToast('🤖 AI processing started...', 'info');
            const resp = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(80, 'Finalizing results & generating output...');
            }

            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || 'Server error');
            }

            // Check if binary download (noise-reduce returns a file)
            const contentType = resp.headers.get('content-type');
            if (contentType && (contentType.includes('audio/') || contentType.includes('application/octet-stream') || endpoint.includes('noise-reduce'))) {
                const blob = await resp.blob();
                onResponse(blob);
            } else {
                const json = await resp.json();
                if (json.error) {
                    throw new Error(json.error);
                }
                onResponse(json);
            }

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(100, '✅ Complete!');
            }
        } catch (err) {
            console.error('[AI Error]', err);
            showToast(`🤖 AI Error: ${err.message}`, 'error');
        } finally {
            btn.classList.remove('loading');
            btn.innerHTML = originalHtml;
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

            if (!data || data.length === 0) {
                resultsContent.innerHTML = `
                    <div style="text-align:center; padding:12px; color:var(--text-secondary);">
                        <i class="fas fa-info-circle" style="font-size:1.2rem; margin-bottom:6px; color:var(--ai-accent);"></i>
                        <p>No silent regions found in this audio file.</p>
                    </div>`;
                showToast('No silence detected.', 'info');
                return;
            }

            // Draw silence regions
            data.forEach((region, i) => {
                addAiMarker(region.start, region.end, 'rgba(100, 100, 100, 0.25)', `Silence ${i+1}`);
            });

            // Populate results HTML
            let rowsHtml = '';
            data.forEach((region, i) => {
                rowsHtml += `
                    <div class="ai-stat-row">
                        <span class="ai-stat-label">Silence #${i+1} (${formatTimePrecise(region.duration)}s)</span>
                        <span class="ai-stat-value">${formatTimePrecise(region.start)} → ${formatTimePrecise(region.end)}</span>
                    </div>`;
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p style="margin-bottom:8px; font-weight:500;">Detected <strong>${data.length}</strong> silent region(s) (highlighted in gray):</p>
                    <div style="max-height:160px; overflow-y:auto; border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--surface);">
                        ${rowsHtml}
                    </div>
                    <div style="display:flex; gap:10px;">
                        <button type="button" class="ai-action-btn" id="aiSplitSilencesBtn">
                            <i class="fas fa-cut"></i> Split at Silences
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiClearSilenceOverlayBtn">
                            <i class="fas fa-eraser"></i> Clear Overlay
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
                    addRegion(ns.start, ns.end, `Speech ${idx+1}`, false);
                });

                pushUndo('Split at Silences', nonSilents);
                updateRegionList();
                clearAiMarkers();
                resultsPanel.classList.add('hidden');
                showToast(`✂️ Created ${nonSilents.length} speech regions!`, 'success');
            };

            showToast(`Smart Silence: Detected ${data.length} regions.`, 'success');
        });
    };

    // AI Auto Trim silence
    document.getElementById('aiTrimBtn').onclick = () => {
        runAIFeature('aiTrimBtn', '/ai/auto-trim', { threshold: 40 }, (data) => {
            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p style="margin-bottom:8px; font-weight:500;">Calculated trim points to remove start/end silence:</p>
                    <div style="border:1px solid var(--border); border-radius:8px; padding:12px; background:var(--surface); margin-bottom:12px;">
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">Original Duration</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.total_duration)}s</span>
                        </div>
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">Trimmed Start Time</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.trimmed_start)}s (${formatTimePrecise(data.removed_start_ms/1000)}s removed)</span>
                        </div>
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">Trimmed End Time</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.trimmed_end)}s (${formatTimePrecise(data.removed_end_ms/1000)}s removed)</span>
                        </div>
                        <div class="ai-stat-row">
                            <span class="ai-stat-label">New Duration</span>
                            <span class="ai-stat-value">${formatTimePrecise(data.trimmed_end - data.trimmed_start)}s</span>
                        </div>
                    </div>
                    <div style="display:flex; gap:10px;">
                        <button type="button" class="ai-action-btn" id="aiApplyTrimBtn">
                            <i class="fas fa-crop-alt"></i> Apply Trim Region
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiCloseTrimBtn">
                            <i class="fas fa-times"></i> Dismiss
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
                const name = "Trimmed Audio";
                addRegion(data.trimmed_start, data.trimmed_end, name);
                
                // Highlight the new region
                if (allRegions.length > 0) {
                    selectRegion(allRegions[0].id);
                }

                resultsPanel.classList.add('hidden');
                showToast('✂️ Applied trim points to editor!', 'success');
            };

            showToast('Auto Trim calculation complete!', 'success');
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
                addAiMarker(t, t + 0.02, 'rgba(249, 115, 22, 0.45)');
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <div class="ai-bpm-container">
                        <div class="ai-bpm-badge">
                            <span class="ai-bpm-number">${data.bpm}</span>
                            <span class="ai-bpm-label">BPM (Tempo)</span>
                        </div>
                        <div class="ai-bpm-details">
                            <div class="ai-bpm-detail-item">
                                <i class="fas fa-drum"></i> Total Beats: <strong>${data.total_beats}</strong>
                            </div>
                            <div class="ai-bpm-detail-item">
                                <i class="fas fa-clock"></i> Average Interval: <strong>${(60 / data.bpm).toFixed(3)}s</strong>
                            </div>
                        </div>
                    </div>
                    
                    <div class="ai-snap-container">
                        <input type="checkbox" id="aiSnapCheckbox" ${snapToBeatsEnabled ? 'checked' : ''}>
                        <label for="aiSnapCheckbox">🧲 Snap region adjustments to closest beat markers</label>
                    </div>

                    <div style="display:flex; gap:10px; margin-top:8px;">
                        <button type="button" class="ai-action-btn-outline" id="aiClearBeatsOverlayBtn">
                            <i class="fas fa-eraser"></i> Clear Overlay
                        </button>
                    </div>
                </div>`;

            // Snap checkbox toggle
            document.getElementById('aiSnapCheckbox').onchange = (e) => {
                snapToBeatsEnabled = e.target.checked;
                showToast(snapToBeatsEnabled ? '🧲 Snap-to-beats enabled' : 'Snap-to-beats disabled', 'info');
            };

            document.getElementById('aiClearBeatsOverlayBtn').onclick = () => {
                clearAiMarkers();
                detectedBeats = [];
                snapToBeatsEnabled = false;
                resultsPanel.classList.add('hidden');
            };

            showToast(`🎵 Beat detection success: ${data.bpm} BPM`, 'success');
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

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p style="color:#27ae60; font-weight:600;"><i class="fas fa-check-circle"></i> Local Noise Reduction Complete!</p>
                    <p style="margin-top:6px;">Background hum, hiss, and noise removed. The processed audio has been successfully loaded into the editor timeline.</p>
                    
                    <div style="display:flex; gap:10px; margin-top:12px;">
                        <button type="button" class="ai-action-btn" id="aiDownloadDenoisedBtn">
                            <i class="fas fa-download"></i> Save Cleaned Audio File
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiDismissDenoiseBtn">
                            <i class="fas fa-times"></i> Dismiss
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

            showToast('🔊 Noise reduction applied!', 'success');
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

                const fillers = data.fillers || [];
                if (fillers.length === 0) {
                    resultsContent.innerHTML = `<p style="padding:10px;">Clean speech! No filler words ("um", "uh", "like") found.</p>`;
                    showToast('No filler words found!', 'info');
                    return;
                }

                fillers.forEach(f => {
                    addAiMarker(f.start, f.end, 'rgba(239, 68, 68, 0.35)', f.word);
                });

                resultsContent.innerHTML = `
                    <div class="transcript-container">
                        <p style="margin-bottom:8px; font-weight:500;">Found <strong>${fillers.length}</strong> filler word(s) ('um', 'uh', 'like'):</p>
                        <div style="display:flex; gap:10px; margin-top:8px;">
                            <button type="button" class="ai-action-btn" id="aiCutFillersBtn">
                                <i class="fas fa-cut"></i> Auto-Cut Filler Words
                            </button>
                            <button type="button" class="ai-action-btn-outline" id="aiClearFillersBtn">
                                <i class="fas fa-eraser"></i> Clear Markers
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
                    fillers.forEach(f => addRegion(f.start, f.end, `Cut: ${f.word}`));
                    resultsPanel.classList.add('hidden');
                    showToast(`Marked ${fillers.length} filler words for removal!`, 'success');
                };
            });
        };
    }

    // AI Speech Studio Enhancement
    if (document.getElementById('aiEnhanceSpeechBtn')) {
        document.getElementById('aiEnhanceSpeechBtn').onclick = () => {
            runAIFeature('aiEnhanceSpeechBtn', '/ai/enhance-speech', {}, (blob) => {
                loadAudio(blob);
                recordedBlob = blob;
                fileInput.value = '';

                const resultsPanel = document.getElementById('aiResultsPanel');
                const resultsContent = document.getElementById('aiResultsContent');
                resultsPanel.classList.remove('hidden');

                resultsContent.innerHTML = `
                    <div class="transcript-container">
                        <p style="color:#27ae60; font-weight:600;"><i class="fas fa-magic"></i> AI Speech Studio Enhancement Complete!</p>
                        <p style="margin-top:6px;">Room echo and acoustic reverberation removed. Enhanced voice loaded into timeline.</p>
                        <div style="display:flex; gap:10px; margin-top:12px;">
                            <button type="button" class="ai-action-btn" id="aiSaveEnhancedBtn">
                                <i class="fas fa-download"></i> Save Enhanced Speech
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
                showToast('✨ AI Speech Enhancement complete!', 'success');
            });
        };
    }

    // AI Vocal / Stem Separator
    if (document.getElementById('aiSeparateStemsBtn')) {
        document.getElementById('aiSeparateStemsBtn').onclick = () => {
            runAIFeature('aiSeparateStemsBtn', '/ai/separate-stems', {}, (data) => {
                const resultsPanel = document.getElementById('aiResultsPanel');
                const resultsContent = document.getElementById('aiResultsContent');
                resultsPanel.classList.remove('hidden');

                resultsContent.innerHTML = `
                    <div class="transcript-container">
                        <p style="color:#8b5cf6; font-weight:600;"><i class="fas fa-sliders"></i> Stem Separation Complete!</p>
                        <p style="margin-top:6px;">Separated vocals and instrumental tracks into high quality stems.</p>
                        <div style="display:flex; gap:10px; margin-top:12px;">
                            <a href="${data.vocals || '#'}" download="vocals.wav" class="ai-action-btn" style="text-decoration:none;">
                                <i class="fas fa-microphone"></i> Download Vocals
                            </a>
                            <a href="${data.no_vocals || '#'}" download="instrumental.wav" class="ai-action-btn-outline" style="text-decoration:none;">
                                <i class="fas fa-music"></i> Download Music
                            </a>
                        </div>
                    </div>`;
                showToast('🎤 Stems separated successfully!', 'success');
            });
        };
    }

    // AI Voice Activity Detection (VAD)
    document.getElementById('aiVadBtn').onclick = () => {
        runAIFeature('aiVadBtn', '/ai/detect-vad', { threshold_db: -35.0 }, (data) => {
            clearAiMarkers();

            const resultsPanel = document.getElementById('aiResultsPanel');
            const resultsContent = document.getElementById('aiResultsContent');
            resultsPanel.classList.remove('hidden');

            if (!data || data.length === 0) {
                resultsContent.innerHTML = `<p>No voice segments found.</p>`;
                return;
            }

            // Draw VAD regions
            let speechCount = 0;
            data.forEach(seg => {
                if (seg.type === 'speech') {
                    speechCount++;
                    addAiMarker(seg.start, seg.end, 'rgba(46, 204, 113, 0.22)', `Speech`);
                } else {
                    addAiMarker(seg.start, seg.end, 'rgba(100, 100, 100, 0.15)', `Silence`);
                }
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <p style="margin-bottom:8px; font-weight:500;">Isolated <strong>${speechCount}</strong> vocal sections (highlighted in green):</p>
                    <div style="display:flex; gap:10px; margin-top:8px;">
                        <button type="button" class="ai-action-btn" id="aiExtractVocalsBtn">
                            <i class="fas fa-external-link-alt"></i> Extract Speech Sections
                        </button>
                        <button type="button" class="ai-action-btn-outline" id="aiClearVadOverlayBtn">
                            <i class="fas fa-eraser"></i> Clear Overlay
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
                        addRegion(seg.start, seg.end, `Vocal Section ${count++}`, false);
                    }
                });

                pushUndo('Extract Speech Regions', data);
                updateRegionList();
                clearAiMarkers();
                resultsPanel.classList.add('hidden');
                showToast(`🗣️ Extracted ${count - 1} speech regions!`, 'success');
            };

            showToast(`Voice Activity: Identified ${speechCount} vocal section(s).`, 'success');
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

            if (!data.available) {
                resultsContent.innerHTML = `
                    <div style="border:1px solid #ffccd5; background:#fff5f6; border-radius:8px; padding:14px; color:#c92a2a;">
                        <h6 style="margin:0 0 6px; font-weight:600;"><i class="fas fa-exclamation-triangle"></i> Transcription Unavailable</h6>
                        <p style="font-size:0.8rem; line-height:1.4;">${data.error}</p>
                    </div>`;
                showToast('Whisper model not installed.', 'warning');
                return;
            }

            // Draw segment markers as gray markers on wavesurfer
            clearAiMarkers();

            let linesHtml = '';
            data.segments.forEach((seg, i) => {
                linesHtml += `
                    <div class="transcript-line" data-start="${seg.start}" data-end="${seg.end}" id="transcriptLine_${i}">
                        <span class="transcript-time">${formatTimePrecise(seg.start)}</span>
                        <span class="transcript-text">${seg.text}</span>
                    </div>`;
            });

            resultsContent.innerHTML = `
                <div class="transcript-container">
                    <div class="transcript-meta">
                        <span>Language detected: <strong>${data.language.toUpperCase()}</strong></span>
                        <button type="button" class="mini-btn" id="copyTranscriptBtn" title="Copy full transcript text">
                            <i class="fas fa-copy"></i> Copy Text
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
            });

            // Playhead highlight sync is handled by the global audio process listener

            // Action: Copy Transcript to clipboard
            document.getElementById('copyTranscriptBtn').onclick = () => {
                navigator.clipboard.writeText(data.full_text).then(() => {
                    showToast('📋 Transcript copied to clipboard!', 'success');
                }).catch(() => {
                    showToast('Failed to copy transcript text.', 'error');
                });
            };

            showToast('📝 Transcription timeline loaded!', 'success');
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
                showToast('Please upload or record audio first.', 'warning');
                return;
            }

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.show({
                    title: 'AI Audio Silence Trimmer',
                    message: 'Detecting dead silence gaps (>1s) & jump-cutting audio...'
                });
            } else {
                showToast('Auto-trimming silence gaps...', 'info');
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
                    showToast(e.error || 'Silence trimming failed', 'error');
                    return;
                }

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'silence_trimmed_audio.wav';
                a.click();
                URL.revokeObjectURL(url);
                showToast('⚡ Silence gaps trimmed and audio downloaded!', 'success');
            } catch (e) {
                showToast('Silence trimming failed: ' + e.message, 'error');
            } finally {
                if (window.ProcessingOverlay) window.ProcessingOverlay.hide();
            }
        };
    }

    // Initialize undo button state
    updateUndoBtn();
});