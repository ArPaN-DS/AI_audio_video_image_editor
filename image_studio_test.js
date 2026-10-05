const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const adjustments = Object.fromEntries(['propBrightness', 'propContrast', 'propSaturation']
    .map(id => [id, { value: 0 }]));
const context = { clearRect() {}, drawImage() {}, filter: 'none', globalAlpha: 1 };
const canvas = { width: 20, height: 10, getContext: () => context,
    toDataURL: () => 'data:image/png;base64,cGl4ZWxz' };
const project = { layers: [], mediaBin: [] };
const sandbox = {
    window: { studioCore: { project } },
    crypto: { randomUUID: () => 'test-id' },
    document: {
        addEventListener() {},
        getElementById: id => adjustments[id] || null,
        createElement: () => ({ ...canvas })
    },
    Image: class {
        set src(value) { this.source = value; this.width = 20; this.height = 10;
            queueMicrotask(() => this.onload()); }
    },
    queueMicrotask
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('static/js/image_studio.js', 'utf8') +
    '\nglobalThis.ImageStudioClass = ImageStudio;', sandbox);

(async () => {
    const studio = Object.create(sandbox.ImageStudioClass.prototype);
    Object.assign(studio, { canvas, ctx: context, layers: [], ready: Promise.resolve(), loadRevision: 0 });
    studio.renderLayerStackUI = () => {};
    studio.addNewLayer({ width: 20, height: 10 }, 'Generated artwork');
    studio.layers[0].x = 4;
    studio.layers[0].opacity = 0.6;
    adjustments.propBrightness.value = 12;
    studio.syncProjectState();
    const state = JSON.parse(JSON.stringify(project));
    assert.equal(state.layers[0].source, 'data:image/png;base64,cGl4ZWxz');
    assert.equal(state.layers[0].x, 4);
    assert.equal(state.imageCanvas.adjustments.propBrightness, 12);
    studio.layers = [];
    await studio.loadProjectState(state);
    assert.equal(studio.layers[0].image.width, 20);
    assert.equal(studio.layers[0].opacity, 0.6);
    assert.equal(studio.canvas.width, 20);
    assert.equal(adjustments.propBrightness.value, 12);
    assert.equal(studio.ctx.globalAlpha, 1);
    await assert.rejects(studio.loadImage('https://example.com/private.png'));
    await assert.rejects(studio.loadImage('//example.com/private.png'));
    await studio.importMedia({ id: 'photo.png', name: 'Photo', url: '/media/photo.png' });
    assert.equal(studio.layers.at(-1).mediaId, 'photo.png');
    studio.syncProjectState();
    assert.equal(project.layers.at(-1).source, '/media/photo.png');
    console.log('Image layer persistence regression checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
