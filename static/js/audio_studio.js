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
            waveColor: '#6366f1',
            progressColor: '#ec4899',
            cursorColor: '#06b6d4',
            height: 140,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            responsive: true
        });

        this.wavesurfer.on('play', () => this.updatePlayState(true));
        this.wavesurfer.on('pause', () => this.updatePlayState(false));
    }

    loadAudio(url, mediaItem) {
        this.currentAudioMedia = mediaItem;
        if (this.wavesurfer) {
            this.wavesurfer.load(url);
        }
    }

    updatePlayState(isPlaying) {
        const btn = document.getElementById('btnMasterPlayPause');
        if (btn) {
            btn.innerHTML = isPlaying ? '<i class="fa-solid fa-pause"></i>' : '<i class="fa-solid fa-play"></i>';
        }
    }

    initEqSliders() {
        const grid = document.getElementById('eqBandsGrid');
        if (!grid) return;

        grid.innerHTML = '';
        const bands = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

        bands.forEach(freq => {
            const freqLabel = freq >= 1000 ? (freq / 1000) + 'k' : freq + 'Hz';
            const col = document.createElement('div');
            col.className = 'eq-band-col';
            col.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 4px;';
            col.innerHTML = `
                <span style="font-size: 0.65rem; color: var(--text-muted);">${freqLabel}</span>
                <input type="range" class="eq-slider" data-freq="${freq}" min="-12" max="12" value="0" step="0.5" style="writing-mode: bt-lr; appearance: slider-vertical; width: 18px; height: 90px;">
                <span class="eq-val-${freq}" style="font-size: 0.65rem; color: var(--accent-cyan);">0dB</span>
            `;
            
            const slider = col.querySelector('.eq-slider');
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                this.eqBands[freq.toString()] = val;
                col.querySelector(`.eq-val-${freq}`).textContent = (val > 0 ? '+' : '') + val + 'dB';
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
            alert("Please import or select an audio file first.");
            return;
        }

        showProcessingOverlay("Measuring LUFS Loudness...", "Analyzing frequency spectrum and RMS dynamics...");
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
            alert("Please select an audio file from the Media Bin first.");
            return;
        }

        showProcessingOverlay("Separating 4 Audio Stems...", "Running local neural demixer (Vocals, Drums, Bass, Other)...");
        try {
            const audioBlob = await fetch(this.currentAudioMedia.url).then(r => r.blob());
            const formData = new FormData();
            formData.append('file', audioBlob, this.currentAudioMedia.name);

            const resp = await fetch('/ai/separate-stems', { method: 'POST', body: formData });
            const data = await resp.json();
            
            if (data.status === 'success' && data.stems) {
                alert(`Demixing Complete! Extracted ${Object.keys(data.stems).length} stems.`);
            } else if (data.error) {
                alert(`Demixing info: ${data.error}`);
            }
        } catch (err) {
            console.error("Demix error:", err);
        }
        hideProcessingOverlay();
    }

    async triggerDenoise() {
        if (!this.currentAudioMedia) {
            alert("Please select an audio file first.");
            return;
        }

        showProcessingOverlay("Reducing Noise...", "Applying local noise reduction filter...");
        try {
            const audioBlob = await fetch(this.currentAudioMedia.url).then(r => r.blob());
            const formData = new FormData();
            formData.append('file', audioBlob, this.currentAudioMedia.name);

            const resp = await fetch('/ai/noise-reduce', { method: 'POST', body: formData });
            const blob = await resp.blob();
            
            const url = URL.createObjectURL(blob);
            this.wavesurfer.load(url);
            alert("Noise Reduction Complete!");
        } catch (err) {
            console.error("Denoise error:", err);
        }
        hideProcessingOverlay();
    }

    exportAudioMaster(format = "wav") {
        if (!this.currentAudioMedia) {
            alert("No audio track loaded to export.");
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
