/**
 * Universal Studio Core Manager
 * Coordinates project state, file uploads, workspace switching, and serialization (.aviproject).
 */

function showProcessingOverlay(title, stageText) {
    window.ProcessingOverlay.show({ title, stageText, category: window.studioCore?.project.workspace || 'general' });
}

function hideProcessingOverlay() {
    window.ProcessingOverlay.hide();
}

class StudioCore {
    constructor() {
        this.project = {
            version: "3.0",
            id: null,
            name: "Untitled Project",
            aspectRatio: "16:9",
            duration: 0.0,
            currentTime: 0.0,
            playhead: 0.0,
            zoom: 1.0,
            mode: "pro", // "personal" or "pro"
            workspace: "combo", // "combo", "video", "audio", "image"
            mediaBin: [],
            tracks: [
                { id: "v1", type: "video", name: "Video V1", clips: [], volume: 1, pan: 0, muted: false, locked: false, solo: false },
                { id: "t1", type: "text", name: "Text / Subs", clips: [], volume: 1, pan: 0, muted: false, locked: false, solo: false },
                { id: "a1", type: "audio", name: "Audio A1", clips: [], volume: 1, pan: 0, muted: false, locked: false, solo: false }
            ],
            layers: []
        };
        
        this.undoStack = [];
        this.redoStack = [];
        this.initUI();
    }

    initUI() {
        // Workspace Tab Switcher
        document.querySelectorAll('.ws-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const target = e.currentTarget.dataset.workspace;
                this.switchWorkspace(target);
            });
        });

        // Personal vs Pro Mode Toggle ("Quick actions bar" switch: on = personal mode)
        const modeCheck = document.getElementById('modeToggleCheck');
        if (modeCheck) {
            modeCheck.addEventListener('change', (e) => {
                this.setMode(e.target.checked ? "personal" : "pro");
            });
        }

        // Panel Inner Tabs (ARIA tabs: click, arrow keys, Home/End)
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const btn = e.currentTarget;
                const parentHeader = btn.closest('.panel-header');
                const parentPanel = btn.closest('.studio-panel');
                if (!parentHeader || !parentPanel) return;

                parentHeader.querySelectorAll('.panel-tab').forEach(t => {
                    t.classList.remove('active');
                    t.setAttribute('aria-selected', 'false');
                    t.tabIndex = -1;
                });
                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
                btn.tabIndex = 0;

                const targetId = btn.dataset.target;
                parentPanel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                const targetEl = targetId ? parentPanel.querySelector(targetId) : null;
                if (targetEl) targetEl.classList.add('active');
            });
            tab.addEventListener('keydown', (e) => {
                const tabs = Array.from(tab.closest('.panel-header')?.querySelectorAll('.panel-tab') || []);
                const index = tabs.indexOf(tab);
                const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 }[e.key];
                if (next === undefined || !tabs.length) return;
                e.preventDefault();
                const target = tabs[(next + tabs.length) % tabs.length];
                target.focus();
                target.click();
            });
        });

        // File Dropzone & Browse
        const dropzone = document.getElementById('mediaDropzone');
        const fileInput = document.getElementById('mediaFileInput');
        if (dropzone && fileInput) {
            dropzone.addEventListener('click', () => fileInput.click());
            dropzone.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInput.click();
                }
            });
            dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
            dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropzone.classList.remove('drag-over');
                if (e.dataTransfer.files.length) this.handleFileUpload(e.dataTransfer.files);
            });
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) this.handleFileUpload(e.target.files);
            });
        }

        // Project Save & Load
        const btnSave = document.getElementById('btnSaveProject');
        if (btnSave) btnSave.addEventListener('click', () => this.saveProject());
        
        const btnOpen = document.getElementById('btnOpenProject');
        const inputOpen = document.getElementById('inputProjectFile');
        if (btnOpen && inputOpen) {
            btnOpen.addEventListener('click', () => inputOpen.click());
            inputOpen.addEventListener('change', (e) => this.loadProjectFile(e.target.files[0]));
        }

        // Master Export Modal
        const btnExport = document.getElementById('btnMasterExport');
        const exportModal = document.getElementById('exportModal');
        const btnCloseExport = document.getElementById('btnCloseExportModal');
        const btnCancelExport = document.getElementById('btnCancelExport');
        const btnStartExport = document.getElementById('btnStartExport');

        if (btnExport && exportModal) {
            btnExport.addEventListener('click', () => {
                // Pre-fill aspect ratio from current project setting
                const aspectSel = document.getElementById('exportAspectSelect');
                if (aspectSel) aspectSel.value = this.project.aspectRatio || '16:9';
                exportModal.classList.add('active');
            });
            [btnCloseExport, btnCancelExport].forEach(b => b?.addEventListener('click', () => exportModal.classList.remove('active')));
            btnStartExport?.addEventListener('click', () => this.executeMasterExport());
        }

        this.initStudioChrome();
    }

    /**
     * UI-only wiring for the studio shell: tool shortcuts into Copilot, new project,
     * preview framing, full screen, slider readouts, timeline ruler and drawer state.
     */
    initStudioChrome() {
        // Tools that are carried out by Copilot: open the drawer with the request filled in.
        document.querySelectorAll('[data-copilot-prompt]').forEach(button => {
            button.addEventListener('click', () => this.openCopilotWithPrompt(button.dataset.copilotPrompt));
        });
        // Shortcut buttons that mirror another control.
        document.querySelectorAll('[data-proxy-click]').forEach(button => {
            button.addEventListener('click', () => document.getElementById(button.dataset.proxyClick)?.click());
        });

        document.getElementById('btnNewProject')?.addEventListener('click', () => {
            const hasWork = this.project.mediaBin.length
                || this.project.tracks.some(track => track.clips?.length)
                || this.project.layers.length
                || window.videoStudio?.clips?.length
                || window.imageStudio?.layers?.length;
            if (hasWork && !confirm('Start a new project? Unsaved changes in this project will be lost.')) return;
            window.location.reload();
        });

        const aspectSelect = document.getElementById('aspectRatioSelect');
        aspectSelect?.addEventListener('change', () => this.applyAspectRatio(aspectSelect.value));
        this.applyAspectRatio(this.project.aspectRatio);

        const viewport = document.getElementById('viewportContainer');
        const btnFullscreen = document.getElementById('btnToggleFullscreen');
        if (viewport && btnFullscreen) {
            btnFullscreen.addEventListener('click', () => {
                if (document.fullscreenElement) document.exitFullscreen?.();
                else viewport.requestFullscreen?.().catch(() => {});
            });
            document.addEventListener('fullscreenchange', () => {
                const active = document.fullscreenElement === viewport;
                const label = active ? 'Exit full-screen preview' : 'Full-screen preview';
                btnFullscreen.setAttribute('aria-label', label);
                btnFullscreen.title = label;
                const icon = btnFullscreen.querySelector('i');
                if (icon) icon.className = 'fa-solid ' + (active ? 'fa-compress' : 'fa-expand');
            });
        }

        document.querySelectorAll('input[type="range"][data-readout]').forEach(slider => {
            slider.addEventListener('input', () => this.updateRangeReadout(slider));
        });

        this.renderTimeRuler();

        // Keep the Copilot toggle's expanded state in sync with the drawer (opened by ai_agent.js).
        const drawer = document.getElementById('aiAgentDrawer');
        const toggle = document.getElementById('btnToggleAiAgent');
        if (drawer && toggle && typeof MutationObserver !== 'undefined') {
            const sync = () => toggle.setAttribute('aria-expanded', String(drawer.classList.contains('open')));
            new MutationObserver(sync).observe(drawer, { attributes: true, attributeFilter: ['class'] });
            sync();
        }
    }

    openCopilotWithPrompt(prompt) {
        const drawer = document.getElementById('aiAgentDrawer');
        const input = document.getElementById('agentInputText');
        if (!drawer || !input) return;
        if (!drawer.classList.contains('open')) document.getElementById('btnToggleAiAgent')?.click();
        drawer.classList.add('open');
        input.value = prompt || '';
        input.focus();
        input.setSelectionRange?.(input.value.length, input.value.length);
    }

    applyAspectRatio(value) {
        const ratios = { '16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1, '4:5': 4 / 5, '21:9': 21 / 9 };
        const key = ratios[value] ? value : '16:9';
        this.project.aspectRatio = key;
        document.getElementById('viewportContainer')?.style.setProperty('--stage-ratio', ratios[key].toFixed(4));
        const select = document.getElementById('aspectRatioSelect');
        if (select && select.value !== key) select.value = key;
    }

    updateRangeReadout(slider) {
        const output = document.getElementById(slider.dataset.readout);
        if (!output) return;
        const value = Number(slider.value) || 0;
        const format = slider.dataset.format;
        output.textContent = format === 'percent' ? value + '%'
            : format === 'degrees' ? value + '°'
                : format === 'signed' ? (value > 0 ? '+' : value < 0 ? '−' : '') + Math.abs(value)
                    : String(value);
    }

    syncRangeReadouts() {
        document.querySelectorAll('input[type="range"][data-readout]').forEach(slider => this.updateRangeReadout(slider));
    }

    renderTimeRuler() {
        const ruler = document.getElementById('timelineRuler');
        if (!ruler) return;
        const pixelsPerSecond = 40; // mirrors video_studio.js
        const seconds = 120;
        ruler.innerHTML = '';
        for (let second = 0; second <= seconds; second += 5) {
            const label = document.createElement('span');
            label.className = 'ruler-label';
            label.style.left = (second * pixelsPerSecond) + 'px';
            label.textContent = Math.floor(second / 60) + ':' + String(second % 60).padStart(2, '0');
            ruler.appendChild(label);
        }
        ruler.addEventListener('click', (e) => {
            const bounds = ruler.getBoundingClientRect();
            window.videoStudio?.preview?.seek((e.clientX - bounds.left) / pixelsPerSecond);
        });
    }

    static formatDuration(seconds) {
        const total = Math.max(0, Math.round(Number(seconds) || 0));
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const secs = String(total % 60).padStart(2, '0');
        return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${secs}` : `${minutes}:${secs}`;
    }

    switchWorkspace(wsName) {
        this.project.workspace = wsName;
        document.querySelectorAll('.ws-tab').forEach(t => {
            const active = t.dataset.workspace === wsName;
            t.classList.toggle('active', active);
            t.setAttribute('aria-pressed', String(active));
        });

        const timelineWorkspace = document.getElementById('timelineWorkspace');
        const masterVideo = document.getElementById('masterVideoPlayer');
        const masterCanvas = document.getElementById('masterImageCanvas');
        const audioWaveformBox = document.getElementById('audioWaveformContainer');
        const viewport = document.getElementById('viewportContainer');
        const viewportTitle = document.getElementById('viewportTitle');

        // Hide all monitors
        [masterVideo, masterCanvas, audioWaveformBox].forEach(el => el?.classList.remove('active'));

        if (wsName === 'video' || wsName === 'combo') {
            masterVideo?.classList.add('active');
            if (timelineWorkspace) timelineWorkspace.style.display = 'flex';
        } else if (wsName === 'audio') {
            audioWaveformBox?.classList.add('active');
            if (timelineWorkspace) timelineWorkspace.style.display = 'flex';
        } else if (wsName === 'image') {
            masterCanvas?.classList.add('active');
            if (timelineWorkspace) timelineWorkspace.style.display = 'none';
        }
        if (viewport) viewport.dataset.workspace = wsName;
        if (document.body?.dataset) document.body.dataset.workspace = wsName;
        if (viewportTitle) {
            viewportTitle.textContent = { combo: 'Program', video: 'Program', audio: 'Waveform', image: 'Canvas' }[wsName] || 'Program';
        }
    }

    setMode(mode) {
        this.project.mode = mode;
        document.body.classList.toggle('mode-pro', mode === 'pro');
        document.body.classList.toggle('mode-personal', mode === 'personal');

        const personalQuickBar = document.getElementById('personalQuickBar');
        if (personalQuickBar) {
            personalQuickBar.style.display = (mode === 'personal') ? 'flex' : 'none';
        }
        const modeCheck = document.getElementById('modeToggleCheck');
        if (modeCheck) modeCheck.checked = mode === 'personal';
    }

    async handleFileUpload(files) {
        showProcessingOverlay("Importing media", "Reading duration and streams…");
        const failed = [];

        for (let file of files) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const isImage = file.type.startsWith('image/');
                const resp = await fetch(isImage ? '/api/agent/upload' : '/video/upload', {
                    method: 'POST', body: formData
                });
                const data = await resp.json();
                if (data.id) {
                    if (isImage) Object.assign(data, { name: file.name, has_video: false, has_audio: false });
                    this.project.mediaBin.push(data);
                    this.renderMediaBin();
                } else {
                    failed.push(file.name);
                }
            } catch (err) {
                console.error("File upload error:", err);
                failed.push(file.name);
            }
        }
        hideProcessingOverlay();
        if (failed.length) {
            alert((failed.length === 1 ? `"${failed[0]}" couldn't be imported.` : `${failed.length} files couldn't be imported.`)
                + ' Check that the file is a supported video, audio or image format and try again.');
        }
    }

    renderMediaBin() {
        const binList = document.getElementById('mediaBinList');
        if (!binList) return;
        document.body?.classList.toggle('studio-has-media', this.project.mediaBin.length > 0);

        if (this.project.mediaBin.length === 0) {
            binList.innerHTML = `
                <div class="empty-bin-msg">
                    <p class="empty-title">No media yet</p>
                    <p>Imported files appear here. Use <strong>Add</strong> or double-click a file to place it on the timeline.</p>
                </div>`;
            return;
        }

        binList.innerHTML = '';
        this.project.mediaBin.forEach(item => {
            const card = document.createElement('div');
            card.className = 'media-card';
            card.draggable = true;
            card.tabIndex = 0;
            card.dataset.id = item.id;

            const kind = item.has_video ? 'Video' : (item.has_audio ? 'Audio' : 'Image');
            const iconClass = item.has_video ? 'fa-film' : (item.has_audio ? 'fa-wave-square' : 'fa-image');
            const safeName = StudioCore.escapeHtml(item.name || 'Untitled media');
            const thumbHtml = item.thumbs && item.thumbs.length
                ? `<img src="${StudioCore.escapeHtml(item.thumbs[0])}" alt="">`
                : `<i class="fa-solid ${iconClass}" aria-hidden="true"></i>`;
            const meta = [kind];
            if (item.has_video || item.has_audio) meta.push(StudioCore.formatDuration(item.duration));
            if (item.width && item.height) meta.push(item.width + '×' + item.height);

            card.innerHTML = `
                <div class="media-thumb-box">${thumbHtml}</div>
                <div class="media-info">
                    <div class="media-name" title="${safeName}">${safeName}</div>
                    <div class="media-meta">${meta.join(' · ')}</div>
                </div>
                <button type="button" class="icon-btn media-add" aria-label="Add ${safeName} to ${kind === 'Image' ? 'canvas' : 'timeline'}" title="Add to ${kind === 'Image' ? 'canvas' : 'timeline'}">
                    <i class="fa-solid fa-plus" aria-hidden="true"></i>
                </button>
            `;

            card.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('application/json', JSON.stringify(item));
            });

            card.addEventListener('dblclick', (e) => {
                if (e.target.closest('.media-add')) return;
                this.addMediaToActiveTrack(item);
            });
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.target === card) {
                    e.preventDefault();
                    this.addMediaToActiveTrack(item);
                }
            });
            card.querySelector('.media-add')?.addEventListener('click', () => this.addMediaToActiveTrack(item));

            binList.appendChild(card);
        });
    }

    addMediaToActiveTrack(item) {
        if (!item.has_video && !item.has_audio && window.imageStudio) {
            this.switchWorkspace('image');
            window.imageStudio.importMedia(item).catch(() => alert('The image could not be opened.'));
            return;
        }
        if (window.videoStudio) {
            window.videoStudio.addClipToTimeline(item);
        }
    }

    getActiveMedia() {
        const workspace = this.project.workspace;
        if (workspace === 'image') {
            return this.project.mediaBin.find(media => media.id === window.imageStudio?.selectedLayer?.mediaId) || null;
        }
        const selected = this.project.mediaBin.find(media => media.id === window.videoStudio?.selectedClip?.mediaId);
        if (workspace === 'audio') {
            return selected?.has_audio && !selected.has_video ? selected : window.audioStudio?.currentAudioMedia || null;
        }
        return selected || null;
    }

    async loadProcessedMedia(url, activate = true) {
        if (!/^\/(processed|media)\/[^/?#]+$/.test(url)) throw new Error('Only local studio media can be opened.');
        const project = this.project;
        let media = project.mediaBin.find(item => item.agentOutputUrl === url || item.url === url);
        if (!media) {
            const response = await fetch(url);
            if (!response.ok) throw new Error('The edited media could not be opened.');
            const blob = await response.blob();
            const filename = decodeURIComponent(url.split('/').pop());
            const isImage = blob.type.startsWith('image/') || /\.(png|jpe?g|webp|gif)$/i.test(filename);
            const form = new FormData();
            form.append('file', blob, filename);
            const upload = await fetch(isImage ? '/api/agent/upload' : '/video/upload', { method: 'POST', body: form });
            const data = await upload.json();
            if (!upload.ok || !data.id) throw new Error(data.error || 'The edited media could not be imported.');
            media = { ...data, name: filename, filename: data.filename || data.id, agentOutputUrl: url };
            if (isImage) Object.assign(media, { has_video: false, has_audio: false });
            if (this.project !== project) return null;
            project.mediaBin.push(media);
            this.renderMediaBin();
        }
        if (!activate) return media;
        if (!media.has_video && !media.has_audio) {
            this.switchWorkspace('image');
            await window.imageStudio?.importMedia(media);
        } else {
            this.switchWorkspace(media.has_video ? 'video' : 'audio');
            window.videoStudio?.addClipToTimeline(media);
            const clip = window.videoStudio?.clips?.at(-1);
            if (clip?.mediaId === media.id) {
                window.videoStudio.selectClip(clip);
                window.videoStudio.preview?.seek(clip.start);
            }
            if (media.has_video) {
                const player = document.getElementById('masterVideoPlayer');
                if (player) { player.src = media.url; player.load(); }
            } else {
                window.audioStudio?.loadAudio(media.url, media);
            }
        }
        return media;
    }

    async saveProject() {
        showProcessingOverlay("Saving project", "Writing the project file…");
        try {
            await window.imageStudio?.ready;
            this.collectLiveState();
            const resp = await fetch('/studio/project/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.project)
            });
            const res = await resp.json();
            if (res.status === 'success') {
                this.project.id = res.id;
                this.downloadProjectFile();
                alert("Project saved. The .aviproject file was downloaded and added to your studio library.");
            } else {
                throw new Error(res.error || 'Project save failed');
            }
        } catch (err) {
            alert('The project could not be saved: ' + err.message + '. Try again.');
        }
        hideProcessingOverlay();
    }

    loadProjectFile(file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const data = JSON.parse(e.target.result);
                this.project = this.normalizeProject(data);
                this.renderMediaBin();
                this.switchWorkspace(this.project.workspace);
                this.setMode(this.project.mode);
                window.videoStudio?.loadProjectState(this.project);
                window.audioStudio?.loadProjectState(this.project);
                await window.imageStudio?.loadProjectState?.(this.project);
                this.applyAspectRatio(this.project.aspectRatio);
                this.syncRangeReadouts();
            } catch (err) {
                alert("This file couldn't be opened as a project. Choose a .aviproject file saved from Studio.");
            }
        };
        reader.readAsText(file);
        const input = document.getElementById('inputProjectFile');
        if (input) input.value = '';
    }

    collectLiveState() {
        window.videoStudio?.syncProjectState();
        window.audioStudio?.syncProjectState();
        window.imageStudio?.syncProjectState();
        this.project.updatedAt = new Date().toISOString();
        return this.project;
    }

    downloadProjectFile() {
        const blob = new Blob([JSON.stringify(this.project, null, 2)], {
            type: 'application/x-aviproject+json'
        });
        const link = document.createElement('a');
        const safeName = (this.project.name || 'Untitled Project')
            .replace(/[^a-z0-9_-]+/gi, '_')
            .replace(/^_+|_+$/g, '') || 'Untitled_Project';
        link.href = URL.createObjectURL(blob);
        link.download = safeName + '.aviproject';
        document.body.appendChild(link);
        link.click();
        URL.revokeObjectURL(link.href);
        link.remove();
    }

    normalizeProject(data) {
        if (!data || typeof data !== 'object' || !Array.isArray(data.tracks)) {
            throw new Error('Invalid project structure');
        }

        const trackDefaults = {
            v1: { id: 'v1', type: 'video', name: 'Video V1' },
            t1: { id: 't1', type: 'text', name: 'Text / Subs' },
            a1: { id: 'a1', type: 'audio', name: 'Audio A1' }
        };
        const tracksById = new Map(data.tracks.map(track => [track.id, track]));
        const tracks = Object.values(trackDefaults).map(defaultTrack => {
            const source = tracksById.get(defaultTrack.id)
                || data.tracks.find(track => track.type === defaultTrack.type)
                || {};
            return {
                ...defaultTrack,
                ...source,
                clips: Array.isArray(source.clips) ? source.clips : [],
                volume: Number.isFinite(Number(source.volume)) ? Number(source.volume) : 1,
                pan: Number.isFinite(Number(source.pan)) ? Number(source.pan) : 0,
                muted: Boolean(source.muted),
                locked: Boolean(source.locked),
                solo: Boolean(source.solo)
            };
        });

        return {
            ...this.project,
            ...data,
            version: data.version || '3.0',
            mediaBin: Array.isArray(data.mediaBin) ? data.mediaBin : [],
            tracks,
            layers: Array.isArray(data.layers) ? data.layers : [],
            workspace: ['combo', 'video', 'audio', 'image'].includes(data.workspace) ? data.workspace : 'combo',
            mode: data.mode === 'personal' ? 'personal' : 'pro'
        };
    }

    static escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, character => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        })[character]);
    }

    executeMasterExport() {
        const modal = document.getElementById('exportModal');
        if (modal) modal.classList.remove('active');
        
        const format = document.getElementById('exportFormatSelect').value;
        const quality = document.getElementById('exportQualitySelect').value;
        const aspectRatio = document.getElementById('exportAspectSelect')?.value || this.project.aspectRatio || '16:9';

        showProcessingOverlay("Rendering Master Export...", `Encoding final project as ${format.toUpperCase()} (${quality}, ${aspectRatio})`);

        if (window.videoStudio && (format === 'mp4' || format === 'webm' || format === 'gif')) {
            window.videoStudio.exportVideoTimeline(format, quality, aspectRatio);
        } else if (window.videoStudio && (format === 'mp3' || format === 'wav')) {
            window.videoStudio.exportVideoTimeline(format, quality, aspectRatio);
        } else if (window.imageStudio && format === 'png') {
            window.imageStudio.exportImageSnapshot()
                .catch(() => alert('The image could not be exported.'))
                .finally(() => hideProcessingOverlay());
        } else {
            // Never report a fake success: this format has no renderer for the open workspace.
            hideProcessingOverlay();
            alert(`${String(format).toUpperCase()} export isn't available for this workspace. Choose MP4, WebM, GIF, MP3, WAV, or PNG.`);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.studioCore = new StudioCore();
});
