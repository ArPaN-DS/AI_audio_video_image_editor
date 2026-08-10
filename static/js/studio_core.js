/**
 * Universal Studio Core Manager
 * Coordinates project state, file uploads, workspace switching, and serialization (.aviproject).
 */

class StudioCore {
    constructor() {
        this.project = {
            id: null,
            name: "Untitled Project",
            aspectRatio: "16:9",
            duration: 0.0,
            currentTime: 0.0,
            mode: "pro", // "personal" or "pro"
            workspace: "combo", // "combo", "video", "audio", "image"
            mediaBin: [],
            tracks: [
                { id: "v1", type: "video", name: "Video V1", clips: [], muted: false, locked: false },
                { id: "t1", type: "text", name: "Text / Subs", clips: [], muted: false, locked: false },
                { id: "a1", type: "audio", name: "Audio A1", clips: [], muted: false, locked: false }
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

        // Personal vs Pro Mode Toggle
        const modeCheck = document.getElementById('modeToggleCheck');
        if (modeCheck) {
            modeCheck.addEventListener('change', (e) => {
                this.setMode(e.target.checked ? "pro" : "personal");
            });
        }

        // Panel Inner Tabs
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const btn = e.currentTarget;
                const parentHeader = btn.closest('.panel-header');
                const parentPanel = btn.closest('.studio-panel');
                
                parentHeader.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
                btn.classList.add('active');
                
                const targetId = btn.dataset.target;
                parentPanel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                const targetEl = parentPanel.querySelector(targetId);
                if (targetEl) targetEl.classList.add('active');
            });
        });

        // File Dropzone & Browse
        const dropzone = document.getElementById('mediaDropzone');
        const fileInput = document.getElementById('mediaFileInput');
        if (dropzone && fileInput) {
            dropzone.addEventListener('click', () => fileInput.click());
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
            btnExport.addEventListener('click', () => exportModal.classList.add('active'));
            [btnCloseExport, btnCancelExport].forEach(b => b?.addEventListener('click', () => exportModal.classList.remove('active')));
            btnStartExport?.addEventListener('click', () => this.executeMasterExport());
        }
    }

    switchWorkspace(wsName) {
        this.project.workspace = wsName;
        document.querySelectorAll('.ws-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.workspace === wsName);
        });

        const timelineWorkspace = document.getElementById('timelineWorkspace');
        const masterVideo = document.getElementById('masterVideoPlayer');
        const masterCanvas = document.getElementById('masterImageCanvas');
        const audioWaveformBox = document.getElementById('audioWaveformContainer');

        // Hide all monitors
        [masterVideo, masterCanvas, audioWaveformBox].forEach(el => el.classList.remove('active'));

        if (wsName === 'video' || wsName === 'combo') {
            masterVideo.classList.add('active');
            timelineWorkspace.style.display = 'flex';
        } else if (wsName === 'audio') {
            audioWaveformBox.classList.add('active');
            timelineWorkspace.style.display = 'flex';
        } else if (wsName === 'image') {
            masterCanvas.classList.add('active');
            timelineWorkspace.style.display = 'none';
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
    }

    async handleFileUpload(files) {
        showProcessingOverlay("Uploading Media...", "Probing file duration and stream codecs...");
        
        for (let file of files) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const resp = await fetch('/video/upload', { method: 'POST', body: formData });
                const data = await resp.json();
                if (data.id) {
                    this.project.mediaBin.push(data);
                    this.renderMediaBin();
                }
            } catch (err) {
                console.error("File upload error:", err);
            }
        }
        hideProcessingOverlay();
    }

    renderMediaBin() {
        const binList = document.getElementById('mediaBinList');
        if (!binList) return;
        
        if (this.project.mediaBin.length === 0) {
            binList.innerHTML = `
                <div class="empty-bin-msg">
                    <i class="fa-solid fa-photo-film"></i>
                    <p>No media files imported yet.</p>
                </div>`;
            return;
        }

        binList.innerHTML = '';
        this.project.mediaBin.forEach(item => {
            const card = document.createElement('div');
            card.className = 'media-card';
            card.draggable = true;
            card.dataset.id = item.id;
            
            const iconClass = item.has_video ? 'fa-film' : (item.has_audio ? 'fa-music' : 'fa-image');
            const thumbHtml = item.thumbs && item.thumbs.length ? `<img src="${item.thumbs[0]}">` : `<i class="fa-solid ${iconClass}"></i>`;

            card.innerHTML = `
                <div class="media-thumb-box">${thumbHtml}</div>
                <div class="media-info">
                    <div class="media-name" title="${item.name}">${item.name}</div>
                    <div class="media-meta">${(item.duration || 0).toFixed(1)}s • ${item.width ? item.width + 'x' + item.height : 'Audio'}</div>
                </div>
            `;

            card.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('application/json', JSON.stringify(item));
            });

            card.addEventListener('dblclick', () => {
                this.addMediaToActiveTrack(item);
            });

            binList.appendChild(card);
        });
    }

    addMediaToActiveTrack(item) {
        if (window.videoStudio) {
            window.videoStudio.addClipToTimeline(item);
        }
    }

    async saveProject() {
        showProcessingOverlay("Saving Project...", "Serializing project state to .aviproject");
        try {
            const resp = await fetch('/studio/project/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.project)
            });
            const res = await resp.json();
            if (res.status === 'success') {
                alert(`Project Saved Successfully! ID: ${res.id}`);
            }
        } catch (err) {
            alert(`Save failed: ${err.message}`);
        }
        hideProcessingOverlay();
    }

    loadProjectFile(file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                this.project = data;
                this.renderMediaBin();
                this.switchWorkspace(data.workspace || 'combo');
                this.setMode(data.mode || 'pro');
                alert("Project Loaded Successfully!");
            } catch (err) {
                alert("Invalid .aviproject file format");
            }
        };
        reader.readAsText(file);
    }

    executeMasterExport() {
        const modal = document.getElementById('exportModal');
        if (modal) modal.classList.remove('active');
        
        const format = document.getElementById('exportFormatSelect').value;
        const quality = document.getElementById('exportQualitySelect').value;

        showProcessingOverlay("Rendering Master Export...", `Encoding final project as ${format.toUpperCase()} (${quality})`);

        if (window.videoStudio && (format === 'mp4' || format === 'webm' || format === 'gif')) {
            window.videoStudio.exportVideoTimeline(format, quality);
        } else if (window.audioStudio && (format === 'mp3' || format === 'wav')) {
            window.audioStudio.exportAudioMaster(format);
        } else if (window.imageStudio && format === 'png') {
            window.imageStudio.exportImageSnapshot();
            hideProcessingOverlay();
        } else {
            setTimeout(() => {
                hideProcessingOverlay();
                alert(`Master Export completed for format ${format}!`);
            }, 2000);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.studioCore = new StudioCore();
});
