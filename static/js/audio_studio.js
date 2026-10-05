/**
 * Audio Studio Module
 * WaveSurfer.js v7 player, 10-band parametric EQ, LUFS metering & AI stem demixing.
 */

class AudioStudio {
    constructor() {
        this.wavesurfer = null;
        this.currentAudioMedia = null;
        this.eqBands = { "31": 0, "63": 0, "125": 0, "250": 0, "500": 0, "1000": 0, "2000": 0, "4000": 0, "8000": 0, "16000": 0 };
        this.initWaveSurfer();
        this.initEqSliders();
        this.initAiAudioEvents();
    }

    initWaveSurfer() {
        const container = document.getElementById('wavesurferWaveform');
        if (!container) return;

        this.wavesurfer = WaveSurfer.create({
            container: '#wavesurferWaveform',
            ...AudioStudio.waveformColors(),
            height: 140,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            responsive: true
        });

        this.wavesurfer.on('play', () => this.updatePlayState(true));
        this.wavesurfer.on('pause', () => this.updatePlayState(false));
        // Follow the light / dark theme (theme.js broadcasts `themechange`).
        window.addEventListener('themechange', () => this.wavesurfer?.setOptions?.(AudioStudio.waveformColors()));
    }

    static waveformColors() {
        const styles = getComputedStyle(document.documentElement);
        const token = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
        return {
            waveColor: token('--text-3', '#5A6273'),
            progressColor: token('--track-audio', '#0F9F6E'),
            cursorColor: token('--playhead', '#E11D48')
        };
    }

    loadAudio(url, mediaItem) {
        this.currentAudioMedia = mediaItem;
        if (this.wavesurfer) {
            this.wavesurfer.load(url);
        }
        this.syncProjectState();
    }

    syncProjectState() {
        const project = window.studioCore?.project;
        if (!project) return;
        project.audio = {
            ...(project.audio || {}),
            eqBands: { ...this.eqBands },
            activeMediaId: this.currentAudioMedia?.id || null
        };
    }

    loadProjectState(project) {
        const audioState = project?.audio || {};
        if (audioState.eqBands && typeof audioState.eqBands === 'object') {
            Object.keys(this.eqBands).forEach(frequency => {
                const value = Number(audioState.eqBands[frequency]);
                this.eqBands[frequency] = Number.isFinite(value) ? value : 0;
                const slider = document.querySelector('.eq-slider[data-freq="' + frequency + '"]');
                const label = document.querySelector('.eq-val-' + frequency);
                if (slider) slider.value = this.eqBands[frequency];
                if (label) {
                    label.textContent = (this.eqBands[frequency] > 0 ? '+' : '')
                        + this.eqBands[frequency];
                }
            });
        }
        const media = project?.mediaBin?.find(item => item.id === audioState.activeMediaId);
        if (media?.url) this.loadAudio(media.url, media);
    }

    updatePlayState(isPlaying) {
        const btn = document.getElementById('btnMasterPlayPause');
        if (btn) {
            btn.innerHTML = isPlaying ? '<i class="fa-solid fa-pause" aria-hidden="true"></i>' : '<i class="fa-solid fa-play" aria-hidden="true"></i>';
            btn.setAttribute('aria-label', isPlaying ? 'Pause' : 'Play');
        }
    }

    initEqSliders() {
        const grid = document.getElementById('eqBandsGrid');
        if (!grid) return;

        grid.innerHTML = '';
        const bands = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

        bands.forEach(freq => {
            const freqLabel = freq >= 1000 ? (freq / 1000) + 'k' : String(freq);
            const col = document.createElement('div');
            col.className = 'eq-band-col';
            col.innerHTML = `
                <span class="eq-freq" aria-hidden="true">${freqLabel}</span>
                <input type="range" class="eq-slider" data-freq="${freq}" min="-12" max="12" value="0" step="0.5" aria-label="${freq} Hz gain" aria-valuetext="0 dB">
                <span class="eq-val eq-val-${freq}" aria-hidden="true">0</span>
            `;
            
            const slider = col.querySelector('.eq-slider');
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                this.eqBands[freq.toString()] = val;
                this.syncProjectState();
                col.querySelector(`.eq-val-${freq}`).textContent = (val > 0 ? '+' : '') + val;
                slider.setAttribute('aria-valuetext', (val > 0 ? '+' : '') + val + ' dB');
            });

            grid.appendChild(col);
        });
    }

    initAiAudioEvents() {
        // Measure LUFS
        const btnLufs = document.getElementById('btnCalculateLufs');
        if (btnLufs) {
            btnLufs.addEventListener('click', () => this.measureLufs());
        }

        // Stem Separation Trigger
        const btnDemix = document.getElementById('btnAiStemSeparation');
        if (btnDemix) {
            btnDemix.addEventListener('click', () => this.triggerStemDemixing());
        }

        // Denoise Trigger
        const btnDenoise = document.getElementById('btnAiDenoise');
        if (btnDenoise) {
            btnDenoise.addEventListener('click', () => this.triggerDenoise());
        }
    }

    async measureLufs() {
        if (!this.currentAudioMedia) {
            alert("Add an audio clip to the timeline first, then measure loudness.");
            return;
        }

        showProcessingOverlay("Measuring loudness", "Analyzing integrated loudness and peak level…");
        try {
            const formData = new FormData();
            formData.append('media_id', this.currentAudioMedia.id);
            
            // Fetch audio blob
            const audioBlob = await fetch(this.currentAudioMedia.url).then(r => r.blob());
            const fileData = new FormData();
            fileData.append('file', audioBlob, this.currentAudioMedia.name);

            const resp = await fetch('/audio/lufs', { method: 'POST', body: fileData });
            const data = await resp.json();
            
            if (data.lufs !== undefined) {
                const display = document.getElementById('lufsMeterDisplay');
                if (display) display.textContent = `${data.lufs} LUFS (Peak: ${data.peak_db} dB)`;
            }
        } catch (err) {
            console.error(err);
        }
        hideProcessingOverlay();
    }

    async triggerStemDemixing() {
        if (!this.currentAudioMedia) {
            alert("Add an audio clip to the timeline first, then separate stems.");
            return;
        }

        showProcessingOverlay("Separating stems", "Splitting vocals, drums, bass and other instruments…");
        try {
            const audioBlob = await fetch(this.currentAudioMedia.url).then(r => r.blob());
            const formData = new FormData();
            formData.append('file', audioBlob, this.currentAudioMedia.name);

            const resp = await fetch('/ai/separate-stems', { method: 'POST', body: formData });
            const data = await resp.json();
            
            if (data.status === 'success' && data.stems) {
                alert(`Stem separation finished: ${Object.keys(data.stems).length} stems created.`);
            } else if (data.error) {
                alert(`Stems couldn't be separated: ${data.error}`);
            }
        } catch (err) {
            console.error("Demix error:", err);
        }
        hideProcessingOverlay();
    }

    async triggerDenoise() {
        if (!this.currentAudioMedia) {
            alert("Add an audio clip to the timeline first, then reduce noise.");
            return;
        }

        showProcessingOverlay("Reducing noise", "Cleaning up background noise…");
        try {
            const audioBlob = await fetch(this.currentAudioMedia.url).then(r => r.blob());
            const formData = new FormData();
            formData.append('file', audioBlob, this.currentAudioMedia.name);

            const resp = await fetch('/ai/noise-reduce', { method: 'POST', body: formData });
            const blob = await resp.blob();
            
            const url = URL.createObjectURL(blob);
            this.wavesurfer.load(url);
            alert("Noise reduction finished. The cleaned audio is loaded in the waveform.");
        } catch (err) {
            console.error("Denoise error:", err);
        }
        hideProcessingOverlay();
    }

    exportAudioMaster(format = "wav") {
        if (!this.currentAudioMedia) {
            alert("There is no audio to export. Add an audio clip to the timeline first.");
            return;
        }

        const link = document.createElement('a');
        link.href = this.currentAudioMedia.url;
        link.download = `master_audio.${format}`;
        link.click();
        hideProcessingOverlay();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.audioStudio = new AudioStudio();
});
