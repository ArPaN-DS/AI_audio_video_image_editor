/* ═══════════════════════════════════════
   Image Editor Pro — 100% client-side Canvas
   Nothing is ever uploaded. All editing is
   non-destructive until you hit Download.
   ═══════════════════════════════════════ */

(() => {
    'use strict';

    const $ = (id) => document.getElementById(id);

    // ─── STATE ───
    let base = null;         // offscreen canvas: photo after crop/rotate/flip
    let tool = 'move';
    let adjust = { brightness: 100, contrast: 100, saturate: 100, warmth: 0, blur: 0 };
    let filterName = 'none';
    let texts = [];          // {id, text, x, y, size, color, font, stroke, bold, _bounds}
    let strokes = [];        // {type, color, size, fill, pts|coords}
    let textSeq = 1;
    let selectedText = null;
    let editingText = null;

    let cropRatio = 'free';
    let cropBox = null;      // {x,y,w,h} in canvas-pixel coords (crop tool active)

    const undoStack = [], redoStack = [];
    const MAX_UNDO = 20;

    // ─── DOM ───
    const canvas = $('ieCanvas');
    const ctx = canvas.getContext('2d');
    const wrap = $('ieCanvasWrap');
    const overlay = $('ieOverlay');
    const hero = $('ieHero'), editor = $('ieEditor');

    // ═══════════════════════════════════════
    //  LOAD
    // ═══════════════════════════════════════
    function loadImageFile(file) {
        if (!file || !file.type.startsWith('image/')) { toast('Please choose an image file.', 'warning'); return; }
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            URL.revokeObjectURL(url);
            setBaseFromImage(img);
            hero.classList.add('hidden');
            editor.classList.remove('hidden');
            resetAll(false);
            renderCanvas();
            toast('Image loaded', 'success');
        };
        img.onerror = () => { URL.revokeObjectURL(url); toast('Could not load that image.', 'error'); };
        img.src = url;
    }

    function setBaseFromImage(img) {
        base = document.createElement('canvas');
        base.width = img.naturalWidth;
        base.height = img.naturalHeight;
        base.getContext('2d').drawImage(img, 0, 0);
    }

    function cloneCanvas(src) {
        const c = document.createElement('canvas');
        c.width = src.width; c.height = src.height;
        c.getContext('2d').drawImage(src, 0, 0);
        return c;
    }

    // ═══════════════════════════════════════
    //  RENDER
    // ═══════════════════════════════════════
    function buildFilter() {
        const a = adjust;
        const parts = [
            `brightness(${a.brightness}%)`,
            `contrast(${a.contrast}%)`,
            `saturate(${a.saturate}%)`,
        ];
        if (a.blur > 0) parts.push(`blur(${a.blur}px)`);
        const presets = {
            grayscale: 'grayscale(1)',
            sepia: 'sepia(0.75)',
            vintage: 'sepia(0.4) contrast(1.1) brightness(1.05) saturate(1.25)',
            cool: 'hue-rotate(-12deg) saturate(1.15) brightness(1.03)',
            warm: 'sepia(0.28) saturate(1.35) brightness(1.02)',
            vivid: 'saturate(1.65) contrast(1.15)',
            invert: 'invert(1)',
        };
        if (presets[filterName]) parts.push(presets[filterName]);
        return parts.join(' ');
    }

    function renderCanvas() {
        if (!base) return;
        canvas.width = base.width;
        canvas.height = base.height;

        // 1) photo + adjustments/filters
        ctx.filter = buildFilter();
        ctx.drawImage(base, 0, 0);
        ctx.filter = 'none';

        // 2) warmth as blended overlay
        if (adjust.warmth !== 0) {
            const a = Math.min(0.4, Math.abs(adjust.warmth) / 100 * 0.4);
            ctx.save();
            ctx.globalCompositeOperation = 'soft-light';
            ctx.fillStyle = adjust.warmth > 0
                ? `rgba(255,150,40,${a})` : `rgba(40,130,255,${a})`;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.restore();
        }

        // 3) drawings / shapes
        strokes.forEach(drawStroke);
        if (livePreviewStroke) drawStroke(livePreviewStroke);

        // 4) text overlays
        texts.forEach(drawText);

        updateDims();
    }

    function drawStroke(s) {
        ctx.save();
        ctx.strokeStyle = s.color;
        ctx.fillStyle = s.color;
        ctx.lineWidth = s.size;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        if (s.type === 'path') {
            ctx.beginPath();
            s.pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
            ctx.stroke();
        } else {
            const { x0, y0, x1, y1 } = s.coords;
            if (s.type === 'rect') {
                s.fill ? ctx.fillRect(x0, y0, x1 - x0, y1 - y0)
                       : ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            } else if (s.type === 'ellipse') {
                ctx.beginPath();
                ctx.ellipse((x0 + x1) / 2, (y0 + y1) / 2, Math.abs(x1 - x0) / 2, Math.abs(y1 - y0) / 2, 0, 0, Math.PI * 2);
                s.fill ? ctx.fill() : ctx.stroke();
            } else if (s.type === 'line') {
                ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
            } else if (s.type === 'arrow') {
                drawArrow(x0, y0, x1, y1, s.size);
            }
        }
        ctx.restore();
    }

    function drawArrow(x0, y0, x1, y1, size) {
        const head = Math.max(12, size * 3);
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - head * Math.cos(ang - Math.PI / 6), y1 - head * Math.sin(ang - Math.PI / 6));
        ctx.lineTo(x1 - head * Math.cos(ang + Math.PI / 6), y1 - head * Math.sin(ang + Math.PI / 6));
        ctx.closePath(); ctx.fill();
    }

    function drawText(t) {
        ctx.save();
        ctx.font = `${t.bold ? '700 ' : ''}${t.size}px ${t.font}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.lineJoin = 'round';
        const lines = t.text.split('\n');
        const lh = t.size * 1.15;
        let maxW = 0;
        lines.forEach(l => { maxW = Math.max(maxW, ctx.measureText(l).width); });
        const totalH = lh * lines.length;
        const startY = t.y - totalH / 2 + lh / 2;
        lines.forEach((line, i) => {
            const yy = startY + i * lh;
            if (t.stroke) {
                ctx.lineWidth = Math.max(3, t.size * 0.09);
                ctx.strokeStyle = '#000';
                ctx.strokeText(line, t.x, yy);
            }
            ctx.fillStyle = t.color;
            ctx.fillText(line, t.x, yy);
        });
        ctx.restore();
        t._bounds = { x: t.x - maxW / 2, y: t.y - totalH / 2, w: maxW, h: totalH };

        // selection outline
        if (selectedText === t.id) {
            ctx.save();
            ctx.strokeStyle = '#2563EB';
            ctx.setLineDash([6, 4]);
            ctx.lineWidth = Math.max(2, canvas.width / 400);
            const b = t._bounds, pad = 8;
            ctx.strokeRect(b.x - pad, b.y - pad, b.w + pad * 2, b.h + pad * 2);
            ctx.restore();
        }
    }

    function updateDims() {
        $('ieDims').textContent = base ? `${canvas.width} × ${canvas.height} px` : '—';
    }

    // ═══════════════════════════════════════
    //  UNDO / REDO
    // ═══════════════════════════════════════
    function snapshot() {
        if (!base) return;
        undoStack.push(serialize());
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack.length = 0;
        updateUndoButtons();
    }
    function serialize() {
        return {
            base: cloneCanvas(base),
            adjust: { ...adjust },
            filterName,
            texts: JSON.parse(JSON.stringify(texts)),
            strokes: JSON.parse(JSON.stringify(strokes)),
        };
    }
    function restore(s) {
        base = cloneCanvas(s.base);
        adjust = { ...s.adjust };
        filterName = s.filterName;
        texts = JSON.parse(JSON.stringify(s.texts));
        strokes = JSON.parse(JSON.stringify(s.strokes));
        selectedText = null;
        syncControls();
        renderCanvas();
    }
    function undo() {
        if (!undoStack.length) return;
        redoStack.push(serialize());
        restore(undoStack.pop());
        updateUndoButtons();
    }
    function redo() {
        if (!redoStack.length) return;
        undoStack.push(serialize());
        restore(redoStack.pop());
        updateUndoButtons();
    }
    function updateUndoButtons() {
        $('ieUndoBtn').disabled = !undoStack.length;
        $('ieRedoBtn').disabled = !redoStack.length;
    }

    function syncControls() {
        document.querySelectorAll('.ie-slider[data-adj]').forEach(sl => {
            const key = sl.dataset.adj;
            const input = sl.querySelector('input');
            input.value = adjust[key];
            updateSliderLabel(sl, key);
        });
        document.querySelectorAll('#ieFilterGrid .ie-filter').forEach(b =>
            b.classList.toggle('active', b.dataset.filter === filterName));
    }

    function updateSliderLabel(sl, key) {
        const span = sl.querySelector('.ve-value');
        if (!span) return;
        if (key === 'warmth') span.textContent = adjust.warmth;
        else if (key === 'blur') span.textContent = adjust.blur;
        else span.textContent = (adjust[key] - 100 >= 0 ? '+' : '') + (adjust[key] - 100);
    }

    // ═══════════════════════════════════════
    //  TRANSFORMS (destructive → bake into base)
    // ═══════════════════════════════════════
    function rotate(deg) {
        snapshot();
        const c = document.createElement('canvas');
        const swap = Math.abs(deg) === 90;
        c.width = swap ? base.height : base.width;
        c.height = swap ? base.width : base.height;
        const cc = c.getContext('2d');
        cc.translate(c.width / 2, c.height / 2);
        cc.rotate(deg * Math.PI / 180);
        cc.drawImage(base, -base.width / 2, -base.height / 2);
        transformOverlaysForRotate(deg, base.width, base.height);
        base = c;
        renderCanvas();
    }
    function flip(horizontal) {
        snapshot();
        const c = document.createElement('canvas');
        c.width = base.width; c.height = base.height;
        const cc = c.getContext('2d');
        if (horizontal) { cc.translate(c.width, 0); cc.scale(-1, 1); }
        else { cc.translate(0, c.height); cc.scale(1, -1); }
        cc.drawImage(base, 0, 0);
        base = c;
        // flip overlay coords
        texts.forEach(t => { horizontal ? t.x = base.width - t.x : t.y = base.height - t.y; });
        strokes.forEach(s => flipStroke(s, horizontal));
        renderCanvas();
    }
    function transformOverlaysForRotate(deg, w, h) {
        const rot = (x, y) => deg === 90 ? { x: h - y, y: x } : deg === -90 ? { x: y, y: w - x } : { x, y };
        texts.forEach(t => { const p = rot(t.x, t.y); t.x = p.x; t.y = p.y; });
        strokes.forEach(s => {
            if (s.type === 'path') s.pts = s.pts.map(p => rot(p.x, p.y));
            else { const a = rot(s.coords.x0, s.coords.y0), b = rot(s.coords.x1, s.coords.y1); s.coords = { x0: a.x, y0: a.y, x1: b.x, y1: b.y }; }
        });
    }
    function flipStroke(s, horizontal) {
        const fx = (x) => horizontal ? base.width - x : x;
        const fy = (y) => horizontal ? y : base.height - y;
        if (s.type === 'path') s.pts = s.pts.map(p => ({ x: fx(p.x), y: fy(p.y) }));
        else s.coords = { x0: fx(s.coords.x0), y0: fy(s.coords.y0), x1: fx(s.coords.x1), y1: fy(s.coords.y1) };
    }

    // ═══════════════════════════════════════
    //  CROP
    // ═══════════════════════════════════════
    function startCropTool() {
        cropBox = { x: base.width * 0.1, y: base.height * 0.1, w: base.width * 0.8, h: base.height * 0.8 };
        renderCropOverlay();
    }
    function renderCropOverlay() {
        overlay.innerHTML = '';
        if (!cropBox || tool !== 'crop') return;
        const rect = canvas.getBoundingClientRect();
        const sx = rect.width / canvas.width, sy = rect.height / canvas.height;
        const box = document.createElement('div');
        box.className = 'ie-crop-box';
        box.style.left = (cropBox.x * sx) + 'px';
        box.style.top = (cropBox.y * sy) + 'px';
        box.style.width = (cropBox.w * sx) + 'px';
        box.style.height = (cropBox.h * sy) + 'px';
        ['tl', 'tr', 'bl', 'br'].forEach(h => {
            const hd = document.createElement('div');
            hd.className = 'ie-crop-handle ' + h;
            hd.dataset.handle = h;
            box.appendChild(hd);
        });
        overlay.appendChild(box);
        attachCropDrag(box, sx, sy);
    }
    function attachCropDrag(box, sx, sy) {
        box.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const handle = e.target.dataset.handle;
            const startX = e.clientX, startY = e.clientY;
            const orig = { ...cropBox };
            function move(ev) {
                const dx = (ev.clientX - startX) / sx, dy = (ev.clientY - startY) / sy;
                if (!handle) {
                    cropBox.x = clamp(orig.x + dx, 0, base.width - cropBox.w);
                    cropBox.y = clamp(orig.y + dy, 0, base.height - cropBox.h);
                } else {
                    let { x, y, w, h } = orig;
                    if (handle.includes('l')) { x = orig.x + dx; w = orig.w - dx; }
                    if (handle.includes('r')) { w = orig.w + dx; }
                    if (handle.includes('t')) { y = orig.y + dy; h = orig.h - dy; }
                    if (handle.includes('b')) { h = orig.h + dy; }
                    if (cropRatio !== 'free') {
                        const r = parseFloat(cropRatio);
                        h = w / r;
                        if (handle.includes('t')) y = orig.y + orig.h - h;
                    }
                    if (w > 20 && h > 20 && x >= 0 && y >= 0 && x + w <= base.width && y + h <= base.height) {
                        cropBox = { x, y, w, h };
                    }
                }
                renderCropOverlay();
            }
            function up() { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); }
            document.addEventListener('mousemove', move);
            document.addEventListener('mouseup', up);
        });
    }
    function applyCrop() {
        if (!cropBox) return;
        snapshot();
        const c = document.createElement('canvas');
        c.width = Math.round(cropBox.w); c.height = Math.round(cropBox.h);
        c.getContext('2d').drawImage(base, cropBox.x, cropBox.y, cropBox.w, cropBox.h, 0, 0, c.width, c.height);
        // shift overlay coords
        texts.forEach(t => { t.x -= cropBox.x; t.y -= cropBox.y; });
        strokes.forEach(s => shiftStroke(s, -cropBox.x, -cropBox.y));
        base = c;
        cancelCrop();
        renderCanvas();
        toast('Cropped', 'success');
    }
    function shiftStroke(s, dx, dy) {
        if (s.type === 'path') s.pts.forEach(p => { p.x += dx; p.y += dy; });
        else { s.coords.x0 += dx; s.coords.x1 += dx; s.coords.y0 += dy; s.coords.y1 += dy; }
    }
    function cancelCrop() {
        cropBox = null;
        overlay.innerHTML = '';
        setTool('move');
    }

    // ═══════════════════════════════════════
    //  POINTER → CANVAS COORDS
    // ═══════════════════════════════════════
    function toCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (canvas.width / rect.width),
            y: (e.clientY - rect.top) * (canvas.height / rect.height),
        };
    }

    let livePreviewStroke = null;

    function onCanvasMouseDown(e) {
        if (!base || tool === 'crop') return;
        const p = toCanvasCoords(e);

        if (tool === 'text') {
            const hit = hitText(p);
            if (hit) { selectText(hit); startTextDrag(e, hit); return; }
            addTextAt(p);
            return;
        }
        if (tool === 'move') {
            const hit = hitText(p);
            if (hit) { selectText(hit); startTextDrag(e, hit); }
            else { selectText(null); renderCanvas(); }
            return;
        }
        if (tool === 'draw') { startFreehand(e, p); return; }
        if (tool === 'shape') { startShape(e, p); return; }
    }

    function hitText(p) {
        for (let i = texts.length - 1; i >= 0; i--) {
            const b = texts[i]._bounds;
            if (b && p.x >= b.x - 10 && p.x <= b.x + b.w + 10 && p.y >= b.y - 10 && p.y <= b.y + b.h + 10)
                return texts[i].id;
        }
        return null;
    }
    function selectText(id) {
        selectedText = id;
        const t = texts.find(x => x.id === id);
        if (t) {
            $('ieTextContent').value = t.text;
            $('ieTextSize').value = t.size; $('ieTextSizeVal').textContent = t.size;
            $('ieTextColor').value = t.color;
            $('ieTextFont').value = t.font;
            $('ieTextStroke').classList.toggle('active', t.stroke);
            $('ieTextBold').classList.toggle('active', t.bold);
            editingText = id;
        }
        renderCanvas();
    }

    function addTextAt(p) {
        snapshot();
        const t = {
            id: 't' + (textSeq++),
            text: $('ieTextContent').value || 'Your text',
            x: p.x, y: p.y,
            size: +$('ieTextSize').value,
            color: $('ieTextColor').value,
            font: $('ieTextFont').value,
            stroke: $('ieTextStroke').classList.contains('active'),
            bold: $('ieTextBold').classList.contains('active'),
        };
        texts.push(t);
        selectText(t.id);
        renderCanvas();
    }

    function startTextDrag(e, id) {
        const t = texts.find(x => x.id === id);
        const start = toCanvasCoords(e);
        const ox = t.x, oy = t.y;
        let moved = false, snapped = false;
        function move(ev) {
            const p = toCanvasCoords(ev);
            if (!snapped) { snapshot(); snapped = true; }
            moved = true;
            t.x = ox + (p.x - start.x);
            t.y = oy + (p.y - start.y);
            renderCanvas();
        }
        function up() {
            document.removeEventListener('mousemove', move);
            document.removeEventListener('mouseup', up);
            if (!moved && snapped) undoStack.pop();
        }
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
    }

    function startFreehand(e, p) {
        snapshot();
        const s = { type: 'path', color: $('ieBrushColor').value, size: +$('ieBrushSize').value, pts: [p] };
        function move(ev) { s.pts.push(toCanvasCoords(ev)); livePreviewStroke = s; renderCanvas(); }
        function up() {
            document.removeEventListener('mousemove', move);
            document.removeEventListener('mouseup', up);
            livePreviewStroke = null;
            if (s.pts.length > 1) strokes.push(s); else undoStack.pop();
            renderCanvas();
        }
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
    }

    function startShape(e, p) {
        snapshot();
        const type = document.querySelector('#ieShapeRow .ve-chip.active').dataset.shape;
        const fill = $('ieShapeFill').classList.contains('active') && (type === 'rect' || type === 'ellipse');
        const s = { type, color: $('ieBrushColor').value, size: +$('ieBrushSize').value, fill, coords: { x0: p.x, y0: p.y, x1: p.x, y1: p.y } };
        function move(ev) { const q = toCanvasCoords(ev); s.coords.x1 = q.x; s.coords.y1 = q.y; livePreviewStroke = s; renderCanvas(); }
        function up() {
            document.removeEventListener('mousemove', move);
            document.removeEventListener('mouseup', up);
            livePreviewStroke = null;
            const c = s.coords;
            if (Math.abs(c.x1 - c.x0) > 3 || Math.abs(c.y1 - c.y0) > 3) strokes.push(s); else undoStack.pop();
            renderCanvas();
        }
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
    }

    // ═══════════════════════════════════════
    //  TOOLS
    // ═══════════════════════════════════════
    function setTool(t) {
        tool = t;
        document.querySelectorAll('.ie-tool').forEach(b => b.classList.toggle('active', b.dataset.tool === t));
        wrap.className = 'ie-canvas-wrap tool-' + t;
        $('ieCropSection').classList.toggle('hidden', t !== 'crop');
        $('ieTextSection').classList.toggle('hidden', t !== 'text');
        $('ieDrawSection').classList.toggle('hidden', t !== 'draw' && t !== 'shape');
        if (t === 'crop') startCropTool(); else { cropBox = null; overlay.innerHTML = ''; }
        if (t !== 'move' && t !== 'text') { selectedText = null; renderCanvas(); }
    }

    // ═══════════════════════════════════════
    //  EXPORT
    // ═══════════════════════════════════════
    function download() {
        if (!base) return;
        const sel = selectedText; selectedText = null; renderCanvas(); // hide selection outline
        const fmt = document.querySelector('#ieExportFormat .ve-chip.active').dataset.fmt;
        const q = +$('ieQuality').value / 100;
        const mime = fmt === 'jpeg' ? 'image/jpeg' : fmt === 'webp' ? 'image/webp' : 'image/png';
        canvas.toBlob((blob) => {
            selectedText = sel; renderCanvas();
            if (!blob) { toast('Export failed', 'error'); return; }
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'edited.' + (fmt === 'jpeg' ? 'jpg' : fmt);
            document.body.appendChild(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 4000);
            toast('Downloaded', 'success');
        }, mime, mime === 'image/png' ? undefined : q);
    }

    // ═══════════════════════════════════════
    //  AI ENHANCE / UPSCALE (real super-resolution via local server)
    // ═══════════════════════════════════════
    function currentImageBlob() {
        // Flatten everything (photo + adjustments + text + drawings) so the
        // upscaler enhances exactly what the user sees.
        const sel = selectedText; selectedText = null; renderCanvas();
        return new Promise(resolve => {
            canvas.toBlob(b => { selectedText = sel; renderCanvas(); resolve(b); }, 'image/png');
        });
    }

    let pending = null;  // {afterImg, beforeURL, afterURL, scale, engine, downgraded}

    async function enhance() {
        if (!base) return;
        const scale = +document.querySelector('#ieUpscaleAmount .ve-chip.active').dataset.scale;
        const model = document.querySelector('#ieUpscaleModel .ve-chip.active').dataset.model;
        const btn = $('ieEnhanceBtn');
        btn.disabled = true;

        if (window.ProcessingOverlay) {
            window.ProcessingOverlay.show({
                title: 'AI Super-Resolution Engine',
                stageText: model === 'best' ? 'AI is reconstructing sharp pixel details & edges (EDSR)...' : 'AI is upscaling image resolution (FSRCNN)...',
                category: 'image'
            });
            window.ProcessingOverlay.updateProgress(30, 'Analyzing image structures...');
        }

        try {
            const beforeBlob = await currentImageBlob();
            const beforeURL = URL.createObjectURL(beforeBlob);
            const beforeSize = { w: canvas.width, h: canvas.height };

            const fd = new FormData();
            fd.append('file', beforeBlob, 'image.png');
            fd.append('scale', scale);
            fd.append('model', model);

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(65, 'Enhancing textures & refining high-resolution output...');
            }

            const res = await fetch('/image/enhance', { method: 'POST', body: fd });
            if (!res.ok) {
                URL.revokeObjectURL(beforeURL);
                const e = await res.json().catch(() => ({})); toast(e.error || 'Enhance failed', 'error'); return;
            }

            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.updateProgress(95, 'Preparing before/after comparison...');
            }

            const engine = res.headers.get('X-Enhance-Engine') || '';
            const downgraded = res.headers.get('X-Enhance-Downgraded') === '1';
            const afterURL = URL.createObjectURL(await res.blob());

            const afterImg = new Image();
            afterImg.onload = () => {
                pending = { afterImg, beforeURL, afterURL, scale, engine, downgraded, beforeSize };
                openCompare(beforeURL, afterURL, { beforeSize, afterSize: { w: afterImg.naturalWidth, h: afterImg.naturalHeight }, engine, downgraded });
            };
            afterImg.onerror = () => { URL.revokeObjectURL(beforeURL); URL.revokeObjectURL(afterURL); toast('Could not load enhanced image', 'error'); };
            afterImg.src = afterURL;
        } catch (e) {
            toast('Enhance failed: ' + e.message, 'error');
        } finally {
            if (window.ProcessingOverlay) {
                window.ProcessingOverlay.hide();
            }
            btn.disabled = false;
        }
    }

    // ─── BEFORE / AFTER COMPARE ───
    let compareMeta = null;

    function openCompare(beforeURL, afterURL, meta) {
        compareMeta = meta;
        $('ieCompareBefore').src = beforeURL;
        $('ieCompareAfter').src = afterURL;

        const eng = meta.engine === 'edsr' ? 'Best · EDSR' : meta.engine === 'fsrcnn' ? 'Fast · FSRCNN' : 'Lanczos';
        $('ieCompareInfo').textContent =
            `${meta.beforeSize.w}×${meta.beforeSize.h}  →  ${meta.afterSize.w}×${meta.afterSize.h}  ·  ${eng}` +
            (meta.downgraded ? '  (auto-switched to Fast for size)' : '');

        // Show FIRST so the stage has real dimensions, then size the frame.
        $('ieCompareOverlay').classList.remove('hidden');
        requestAnimationFrame(() => { fitCompareFrame(); setDivider(50); });
    }

    function fitCompareFrame() {
        if (!compareMeta) return;
        const frame = $('ieCompareFrame');
        const stage = frame.parentElement;
        const ar = compareMeta.afterSize.w / compareMeta.afterSize.h;
        const maxW = stage.clientWidth || (window.innerWidth - 36);
        const maxH = stage.clientHeight || (window.innerHeight - 170);
        let w = maxW, h = w / ar;
        if (h > maxH) { h = maxH; w = h * ar; }
        frame.style.width = Math.floor(w) + 'px';
        frame.style.height = Math.floor(h) + 'px';
    }

    function setDivider(pct) {
        pct = clamp(pct, 0, 100);
        $('ieCompareDivider').style.left = pct + '%';
        // BEFORE image shows from left edge up to the divider
        $('ieCompareBefore').style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
        const h = $('ieCompareDivider').querySelector('.ie-cmp-handle');
        if (h) h.style.left = '50%';
    }

    function keepEnhanced() {
        if (!pending) return;
        snapshot();
        setBaseFromImage(pending.afterImg);
        texts = []; strokes = []; selectedText = null;  // baked into pixels now
        renderCanvas();
        const eng = pending.engine === 'edsr' ? 'Best (EDSR)' : pending.engine === 'fsrcnn' ? 'Fast (FSRCNN)' : 'Lanczos';
        toast(`Upscaled ${pending.scale}× · ${eng}`, 'success');
        closeCompare();
    }

    function closeCompare() {
        $('ieCompareOverlay').classList.add('hidden');
        if (pending) { URL.revokeObjectURL(pending.beforeURL); URL.revokeObjectURL(pending.afterURL); }
        pending = null;
    }

    function initCompare() {
        const frame = $('ieCompareFrame');
        function moveFromEvent(clientX) {
            const r = frame.getBoundingClientRect();
            setDivider(((clientX - r.left) / r.width) * 100);
        }
        function down(e) {
            e.preventDefault();
            const cx = e.touches ? e.touches[0].clientX : e.clientX;
            moveFromEvent(cx);
            const move = (ev) => moveFromEvent(ev.touches ? ev.touches[0].clientX : ev.clientX);
            const up = () => {
                document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up);
                document.removeEventListener('touchmove', move); document.removeEventListener('touchend', up);
            };
            document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
            document.addEventListener('touchmove', move, { passive: false }); document.addEventListener('touchend', up);
        }
        frame.addEventListener('mousedown', down);
        frame.addEventListener('touchstart', down, { passive: false });
        window.addEventListener('resize', () => { if (!$('ieCompareOverlay').classList.contains('hidden')) fitCompareFrame(); });
        $('ieCompareKeep').onclick = keepEnhanced;
        $('ieCompareDiscard').onclick = () => { toast('Kept your original', 'info'); closeCompare(); };
        $('ieCompareClose').onclick = closeCompare;
    }

    function showLoading(txt) { $('ieLoadingText').textContent = txt || 'Working...'; $('ieLoadingOverlay').classList.remove('hidden'); }
    function hideLoading() { $('ieLoadingOverlay').classList.add('hidden'); }

    // ═══════════════════════════════════════
    //  RESET
    // ═══════════════════════════════════════
    function resetAll(reRender = true) {
        adjust = { brightness: 100, contrast: 100, saturate: 100, warmth: 0, blur: 0 };
        filterName = 'none';
        texts = []; strokes = [];
        selectedText = null;
        undoStack.length = 0; redoStack.length = 0;
        updateUndoButtons();
        syncControls();
        if (reRender) renderCanvas();
    }

    // ═══════════════════════════════════════
    //  HELPERS
    // ═══════════════════════════════════════
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
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
        setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateX(-50%) translateY(20px)'; setTimeout(() => el.remove(), 400); }, 3000);
    }

    // ═══════════════════════════════════════
    //  WIRING
    // ═══════════════════════════════════════
    function init() {
        // Upload
        $('ieFileInput').addEventListener('change', e => e.target.files[0] && loadImageFile(e.target.files[0]));
        const ua = $('ieUploadArea');
        ['dragover', 'dragenter'].forEach(ev => ua.addEventListener(ev, e => { e.preventDefault(); ua.classList.add('drag-active'); }));
        ['dragleave', 'drop'].forEach(ev => ua.addEventListener(ev, () => ua.classList.remove('drag-active')));
        ua.addEventListener('drop', e => { e.preventDefault(); loadImageFile(e.dataTransfer.files[0]); });

        // Paste from clipboard
        document.addEventListener('paste', e => {
            const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
            if (item) loadImageFile(item.getAsFile());
        });

        // Tools
        document.querySelectorAll('.ie-tool').forEach(b => b.onclick = () => setTool(b.dataset.tool));

        // Canvas interactions
        canvas.addEventListener('mousedown', onCanvasMouseDown);
        window.addEventListener('resize', () => { if (tool === 'crop') renderCropOverlay(); });

        // Transform
        $('ieRotateL').onclick = () => rotate(-90);
        $('ieRotateR').onclick = () => rotate(90);
        $('ieFlipH').onclick = () => flip(true);
        $('ieFlipV').onclick = () => flip(false);

        // Crop
        $('ieCropRatios').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); if (!b) return;
            cropRatio = b.dataset.ratio;
            document.querySelectorAll('#ieCropRatios .ve-chip').forEach(c => c.classList.toggle('active', c === b));
            if (cropRatio !== 'free' && cropBox) { cropBox.h = cropBox.w / parseFloat(cropRatio); renderCropOverlay(); }
        });
        $('ieApplyCrop').onclick = applyCrop;
        $('ieCancelCrop').onclick = cancelCrop;

        // Adjust sliders
        document.querySelectorAll('.ie-slider[data-adj]').forEach(sl => {
            const key = sl.dataset.adj;
            const input = sl.querySelector('input');
            input.addEventListener('input', () => { adjust[key] = +input.value; updateSliderLabel(sl, key); renderCanvas(); });
            input.addEventListener('mousedown', () => snapshot());
        });
        $('ieResetAdjust').onclick = () => { snapshot(); adjust = { brightness: 100, contrast: 100, saturate: 100, warmth: 0, blur: 0 }; syncControls(); renderCanvas(); };

        // Filters
        $('ieFilterGrid').addEventListener('click', e => {
            const b = e.target.closest('.ie-filter'); if (!b) return;
            snapshot();
            filterName = b.dataset.filter;
            document.querySelectorAll('#ieFilterGrid .ie-filter').forEach(c => c.classList.toggle('active', c === b));
            renderCanvas();
        });

        // Text controls (live-edit the selected/active text)
        $('ieTextContent').addEventListener('input', () => { const t = texts.find(x => x.id === editingText); if (t) { t.text = $('ieTextContent').value; renderCanvas(); } });
        $('ieTextSize').addEventListener('input', () => { $('ieTextSizeVal').textContent = $('ieTextSize').value; const t = texts.find(x => x.id === editingText); if (t) { t.size = +$('ieTextSize').value; renderCanvas(); } });
        $('ieTextColor').addEventListener('input', () => { const t = texts.find(x => x.id === editingText); if (t) { t.color = $('ieTextColor').value; renderCanvas(); } });
        $('ieTextFont').addEventListener('change', () => { const t = texts.find(x => x.id === editingText); if (t) { t.font = $('ieTextFont').value; renderCanvas(); } });
        $('ieTextStroke').onclick = () => { $('ieTextStroke').classList.toggle('active'); const t = texts.find(x => x.id === editingText); if (t) { t.stroke = $('ieTextStroke').classList.contains('active'); renderCanvas(); } };
        $('ieTextBold').onclick = () => { $('ieTextBold').classList.toggle('active'); const t = texts.find(x => x.id === editingText); if (t) { t.bold = $('ieTextBold').classList.contains('active'); renderCanvas(); } };

        // Draw / shape controls
        $('ieBrushSize').addEventListener('input', () => $('ieBrushSizeVal').textContent = $('ieBrushSize').value);
        $('ieShapeRow').addEventListener('click', e => { const b = e.target.closest('.ve-chip'); if (!b) return; document.querySelectorAll('#ieShapeRow .ve-chip').forEach(c => c.classList.toggle('active', c === b)); });
        $('ieShapeFill').onclick = () => $('ieShapeFill').classList.toggle('active');

        // AI Enhance / Upscale
        $('ieUpscaleAmount').addEventListener('click', e => { const b = e.target.closest('.ve-chip'); if (!b) return; document.querySelectorAll('#ieUpscaleAmount .ve-chip').forEach(c => c.classList.toggle('active', c === b)); });
        $('ieUpscaleModel').addEventListener('click', e => { const b = e.target.closest('.ve-chip'); if (!b) return; document.querySelectorAll('#ieUpscaleModel .ve-chip').forEach(c => c.classList.toggle('active', c === b)); });
        $('ieEnhanceBtn').onclick = enhance;
        initCompare();

        // Export
        $('ieExportFormat').addEventListener('click', e => {
            const b = e.target.closest('.ve-chip'); if (!b) return;
            document.querySelectorAll('#ieExportFormat .ve-chip').forEach(c => c.classList.toggle('active', c === b));
            $('ieQualityRow').classList.toggle('hidden', b.dataset.fmt === 'png');
        });
        $('ieQuality').addEventListener('input', () => $('ieQualityVal').textContent = $('ieQuality').value);
        $('ieDownloadBtn').onclick = download;

        // Action bar
        $('ieUndoBtn').onclick = undo;
        $('ieRedoBtn').onclick = redo;
        $('ieResetBtn').onclick = () => { if (base) { snapshot(); resetAll(); } };
        $('ieNewBtn').onclick = () => $('ieFileInput').click();

        // Help
        $('ieHelpBtn').onclick = () => $('ieShortcutsOverlay').classList.remove('hidden');
        $('ieCloseShortcuts').onclick = () => $('ieShortcutsOverlay').classList.add('hidden');

        // Keyboard
        document.addEventListener('keydown', onKey);
    }

    function onKey(e) {
        const tag = (e.target.tagName || '').toLowerCase();
        const typing = tag === 'input' || tag === 'textarea';
        if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); undo(); return; }
        if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); redo(); return; }
        if (typing) return;
        if (!base) return;
        if (e.key === 'v' || e.key === 'V') setTool('move');
        else if (e.key === 'c' || e.key === 'C') setTool('crop');
        else if (e.key === 't' || e.key === 'T') setTool('text');
        else if (e.key === 'b' || e.key === 'B') setTool('draw');
        else if (e.key === 'Enter' && tool === 'crop') applyCrop();
        else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedText) {
            snapshot(); texts = texts.filter(x => x.id !== selectedText); selectedText = null; editingText = null; renderCanvas();
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
