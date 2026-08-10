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

    addNewLayer(imgElement = null, name = 'New Layer') {
        const layer = {
            id: 'layer_' + Date.now(),
            name: name,
            type: imgElement ? 'image' : 'vector',
            visible: true,
            opacity: 1.0,
            image: imgElement,
            x: 0, y: 0,
            width: imgElement ? imgElement.width : 500,
            height: imgElement ? imgElement.height : 500
        };

        this.layers.push(layer);
        this.selectedLayer = layer;
        this.renderLayerStackUI();
        this.renderCanvasLayers();
    }

    renderLayerStackUI() {
        const stackList = document.getElementById('layerStackList');
        if (!stackList) return;

        if (this.layers.length === 0) {
            stackList.innerHTML = `
                <div class="empty-layers-msg">
                    <i class="fa-solid fa-layer-group"></i>
                    <p>No canvas layers created.</p>
                </div>`;
            return;
        }

        stackList.innerHTML = '';
        this.layers.slice().reverse().forEach(layer => {
            const item = document.createElement('div');
            item.className = 'layer-item-row';
            item.style.cssText = 'display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; background: var(--bg-card); margin-bottom: 4px; border-radius: 4px; font-size: 0.75rem; border: 1px solid var(--border-color);';
            item.innerHTML = `
                <span><i class="fa-solid ${layer.visible ? 'fa-eye' : 'fa-eye-slash'} btn-vis"></i> ${layer.name}</span>
                <span class="btn-del-layer" style="color: #ef4444; cursor: pointer;"><i class="fa-solid fa-trash"></i></span>
            `;

            item.querySelector('.btn-vis').addEventListener('click', () => {
                layer.visible = !layer.visible;
                this.renderLayerStackUI();
                this.renderCanvasLayers();
            });

            item.querySelector('.btn-del-layer').addEventListener('click', () => {
                this.layers = this.layers.filter(l => l.id !== layer.id);
                this.renderLayerStackUI();
                this.renderCanvasLayers();
            });

            stackList.appendChild(item);
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
        showProcessingOverlay("Removing Background...", "Running local U2Net neural background segmentation...");
        try {
            this.canvas.toBlob(async (blob) => {
                if (!blob) return;
                const formData = new FormData();
                formData.append('file', blob, 'canvas_image.png');

                const resp = await fetch('/image/remove-bg', { method: 'POST', body: formData });
                if (resp.ok) {
                    const resultBlob = await resp.blob();
                    const img = new Image();
                    img.onload = () => {
                        this.addNewLayer(img, 'Cutout (No BG)');
                        hideProcessingOverlay();
                    };
                    img.src = URL.createObjectURL(resultBlob);
                } else {
                    hideProcessingOverlay();
                    alert("Background removal failed.");
                }
            }, 'image/png');
        } catch (err) {
            console.error(err);
            hideProcessingOverlay();
        }
    }

    async triggerUpscale() {
        showProcessingOverlay("AI Super-Resolution Upscaling...", "Reconstructing pixels via OpenCV EDSR / FSRCNN...");
        try {
            this.canvas.toBlob(async (blob) => {
                if (!blob) return;
                const formData = new FormData();
                formData.append('file', blob, 'canvas_image.png');
                formData.append('scale', 2);
                formData.append('model', 'fast');

                const resp = await fetch('/image/enhance', { method: 'POST', body: formData });
                if (resp.ok) {
                    const resultBlob = await resp.blob();
                    const img = new Image();
                    img.onload = () => {
                        this.canvas.width = img.width;
                        this.canvas.height = img.height;
                        this.addNewLayer(img, 'AI 2x Upscaled');
                        hideProcessingOverlay();
                    };
                    img.src = URL.createObjectURL(resultBlob);
                } else {
                    hideProcessingOverlay();
                    alert("Upscaling failed.");
                }
            }, 'image/png');
        } catch (err) {
            console.error(err);
            hideProcessingOverlay();
        }
    }

    exportImageSnapshot() {
        const link = document.createElement('a');
        link.download = 'studio_artwork.png';
        link.href = this.canvas.toDataURL('image/png');
        link.click();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.imageStudio = new ImageStudio();
});
