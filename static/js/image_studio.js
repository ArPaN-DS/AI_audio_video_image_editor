/**
 * Image Studio Module
 * Client-side Canvas layer stack manager, adjustment layer filters, background removal & AI upscaler.
 */

class ImageStudio {
    constructor() {
        this.canvas = document.getElementById('masterImageCanvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        
        this.layers = []; // [{ id, name, type, visible, opacity, image, text, x, y, width, height }]
        this.selectedLayer = null;
        this.ready = Promise.resolve();
        this.loadRevision = 0;

        this.initCanvasEvents();
        this.initAiImageButtons();
    }

    initCanvasEvents() {
        if (!this.canvas) return;
        this.canvas.width = 1920;
        this.canvas.height = 1080;

        const btnAddLayer = document.getElementById('btnAddLayerBtn');
        if (btnAddLayer) {
            btnAddLayer.addEventListener('click', () => this.addNewLayer());
        }

        // Adjustments Sliders
        ['propBrightness', 'propContrast', 'propSaturation'].forEach(id => {
            const slider = document.getElementById(id);
            if (slider) {
                slider.addEventListener('input', () => this.renderCanvasLayers());
            }
        });
    }

    initAiImageButtons() {
        const btnRemoveBg = document.getElementById('btnAiRemoveBg');
        if (btnRemoveBg) {
            btnRemoveBg.addEventListener('click', () => this.triggerRemoveBg());
        }

        const btnUpscale = document.getElementById('btnAiUpscale');
        if (btnUpscale) {
            btnUpscale.addEventListener('click', () => this.triggerUpscale());
        }

        const chipRemoveBg = document.getElementById('chipRemoveBg');
        if (chipRemoveBg) {
            chipRemoveBg.addEventListener('click', () => this.triggerRemoveBg());
        }
    }

    addNewLayer(imgElement = null, name = 'New Layer', source = null, mediaId = null) {
        const layer = {
            id: 'layer_' + crypto.randomUUID(),
            name: name,
            type: imgElement ? 'image' : 'vector',
            visible: true,
            opacity: 1.0,
            image: imgElement,
            source: source || (imgElement ? this.imageSource(imgElement) : null),
            mediaId,
            x: 0, y: 0,
            width: imgElement ? imgElement.width : 500,
            height: imgElement ? imgElement.height : 500
        };

        this.layers.push(layer);
        this.selectedLayer = layer;
        this.renderLayerStackUI();
        this.renderCanvasLayers();
        return layer;
    }

    imageSource(image) {
        const raster = document.createElement('canvas');
        raster.width = image.naturalWidth || image.width;
        raster.height = image.naturalHeight || image.height;
        raster.getContext('2d').drawImage(image, 0, 0);
        return raster.toDataURL('image/png');
    }

    loadImage(source) {
        const local = typeof source === 'string' && (
            /^data:image\/(png|jpeg|webp);base64,/.test(source)
            || /^\/(media|processed)\//.test(source)
            || source.startsWith('blob:')
        );
        if (!local) return Promise.reject(new Error('Image source must be local.'));
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error('The saved image could not be opened.'));
            image.src = source;
        });
    }

    importMedia(media) {
        const pending = this.loadImage(media.url).then(image => {
            this.addNewLayer(image, media.name, media.url, media.id);
        });
        this.ready = Promise.all([this.ready.catch(() => {}), pending]);
        return this.ready;
    }

    syncProjectState() {
        const project = window.studioCore?.project;
        if (!project) return;
        project.layers = this.layers.map(layer => ({
            id: layer.id, name: layer.name, type: layer.type,
            visible: layer.visible, opacity: layer.opacity,
            x: layer.x, y: layer.y, width: layer.width, height: layer.height,
            mediaId: layer.mediaId || null,
            source: layer.source || (layer.image ? this.imageSource(layer.image) : null)
        }));
        project.imageCanvas = {
            width: this.canvas.width, height: this.canvas.height,
            adjustments: Object.fromEntries(['propBrightness', 'propContrast', 'propSaturation']
                .map(id => [id, Number(document.getElementById(id)?.value || 0)]))
        };
    }

    loadProjectState(project) {
        const revision = ++this.loadRevision;
        const canvasState = project?.imageCanvas || {};
        this.canvas.width = Math.max(1, Math.min(16384, Number(canvasState.width) || 1920));
        this.canvas.height = Math.max(1, Math.min(16384, Number(canvasState.height) || 1080));
        ['propBrightness', 'propContrast', 'propSaturation'].forEach(id => {
            const slider = document.getElementById(id);
            if (slider) slider.value = Number(canvasState.adjustments?.[id]) || 0;
        });
        this.layers = (project?.layers || []).map(layer => ({
            visible: true,
            opacity: 1,
            x: 0,
            y: 0,
            width: 500,
            height: 500,
            ...layer,
            image: null
        }));
        this.selectedLayer = this.layers[0] || null;
        this.renderLayerStackUI();
        this.renderCanvasLayers();
        const layers = this.layers;
        this.ready = Promise.all(layers.map(async layer => {
            const media = project?.mediaBin?.find(item => item.id === layer.mediaId);
            const source = layer.source || media?.url;
            if (layer.type !== 'image') return;
            if (!source) throw new Error('This layer is missing its saved image.');
            const image = await this.loadImage(source);
            if (revision !== this.loadRevision) return;
            layer.image = image;
            layer.source = source;
        })).then(() => {
            if (revision === this.loadRevision) this.renderCanvasLayers();
        });
        return this.ready;
    }

    renderLayerStackUI() {
        const stackList = document.getElementById('layerStackList');
        if (!stackList) return;

        if (this.layers.length === 0) {
            stackList.innerHTML = `
                <div class="empty-layers-msg">
                    <p class="empty-title">No layers</p>
                    <p>Add an image from Media to start a layer stack. The top layer draws last.</p>
                </div>`;
            return;
        }

        stackList.innerHTML = '';
        this.layers.slice().reverse().forEach(layer => {
            const item = document.createElement('div');
            const safeName = StudioCore.escapeHtml(layer.name || 'Layer');
            item.className = 'layer-item-row';
            item.classList.toggle('is-selected', this.selectedLayer === layer);
            item.classList.toggle('is-hidden', !layer.visible);
            item.innerHTML = `
                <button type="button" class="icon-btn layer-btn btn-vis" aria-pressed="${!layer.visible}"
                    aria-label="${layer.visible ? 'Hide' : 'Show'} ${safeName}" title="${layer.visible ? 'Hide layer' : 'Show layer'}">
                    <i class="fa-solid ${layer.visible ? 'fa-eye' : 'fa-eye-slash'}" aria-hidden="true"></i>
                </button>
                <span class="layer-name" title="${safeName}">${safeName}</span>
                <button type="button" class="icon-btn layer-btn btn-del-layer" aria-label="Delete ${safeName}" title="Delete layer">
                    <i class="fa-solid fa-trash-can" aria-hidden="true"></i>
                </button>
            `;

            item.querySelector('.btn-vis').addEventListener('click', (event) => {
                event.stopPropagation();
                layer.visible = !layer.visible;
                this.renderLayerStackUI();
                this.renderCanvasLayers();
            });

            item.querySelector('.btn-del-layer').addEventListener('click', (event) => {
                event.stopPropagation();
                this.layers = this.layers.filter(l => l.id !== layer.id);
                if (this.selectedLayer === layer) this.selectedLayer = this.layers[0] || null;
                this.renderLayerStackUI();
                this.renderCanvasLayers();
            });

            stackList.appendChild(item);
            item.addEventListener('click', () => {
                this.selectedLayer = layer;
                this.renderLayerStackUI();
            });
        });
    }

    renderCanvasLayers() {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const b = document.getElementById('propBrightness')?.value || 0;
        const c = document.getElementById('propContrast')?.value || 0;
        const s = document.getElementById('propSaturation')?.value || 0;

        this.ctx.filter = `brightness(${100 + parseInt(b)}%) contrast(${100 + parseInt(c)}%) saturate(${100 + parseInt(s)}%)`;

        this.layers.filter(l => l.visible).forEach(layer => {
            this.ctx.globalAlpha = layer.opacity;
            if (layer.image) {
                this.ctx.drawImage(layer.image, layer.x, layer.y, layer.width, layer.height);
            }
        });

        this.ctx.filter = 'none';
        this.ctx.globalAlpha = 1.0;
    }

    async triggerRemoveBg() {
        return this.processCanvasImage('/image/remove-bg', 'Removing background', 'Cutout');
    }

    async triggerUpscale() {
        return this.processCanvasImage('/image/enhance', 'Upscaling 2×',
            'Upscaled image', { scale: 2, model: 'fast' });
    }

    async processCanvasImage(endpoint, title, layerName, options = {}) {
        showProcessingOverlay(title, 'Processing your image locally...');
        let resultUrl = null;
        try {
            await this.ready;
            const blob = await new Promise(resolve => this.canvas.toBlob(resolve, 'image/png'));
            if (!blob) throw new Error('The canvas could not be prepared.');
            const formData = new FormData();
            formData.append('file', blob, 'canvas_image.png');
            Object.entries(options).forEach(([key, value]) => formData.append(key, value));
            const response = await fetch(endpoint, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Image processing failed. Please try again.');
            resultUrl = URL.createObjectURL(await response.blob());
            const image = await this.loadImage(resultUrl);
            this.layers.forEach(layer => { layer.visible = false; });
            this.canvas.width = image.naturalWidth || image.width;
            this.canvas.height = image.naturalHeight || image.height;
            ['propBrightness', 'propContrast', 'propSaturation'].forEach(id => {
                const slider = document.getElementById(id);
                if (slider) slider.value = 0;
            });
            this.addNewLayer(image, layerName);
        } catch (err) {
            alert('The image could not be processed. Make sure an image is on the canvas, then try again.');
        } finally {
            if (resultUrl) URL.revokeObjectURL(resultUrl);
            hideProcessingOverlay();
        }
    }

    async exportImageSnapshot() {
        await this.ready;
        const link = document.createElement('a');
        link.download = 'studio_artwork.png';
        link.href = this.canvas.toDataURL('image/png');
        link.click();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.imageStudio = new ImageStudio();
});
