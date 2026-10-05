/**
 * Processing overlay — modal progress panel for long-running local jobs.
 * Shows elapsed time, a live activity meter, natural-language stage text,
 * determinate or indeterminate progress, and rotating tips.
 *
 * Public API (unchanged): ProcessingOverlay.show({ title, stageText, stepText, category }),
 * ProcessingOverlay.updateProgress(percent, stageText?, stepText?), ProcessingOverlay.hide().
 */
window.ProcessingOverlay = (function () {
  let timerInterval = null;
  let tipInterval = null;
  let startTime = 0;
  let currentTipIndex = 0;
  let currentCategory = 'general';
  let previousFocus = null;

  const TIPS = {
    audio: [
      { label: 'Tip', text: 'Double-click the waveform to add a cut region at that point.' },
      { label: 'Shortcut', text: 'Press Space to play or pause.' },
      { label: 'Tip', text: 'Add a fade in and fade out to avoid clicks at the start and end of a clip.' },
      { label: 'Shortcut', text: 'Press Ctrl+Z to undo the last cut or edit.' },
      { label: 'Privacy', text: 'Audio is processed on this computer and never leaves it.' },
      { label: 'Tip', text: 'Use Record to capture audio straight from your microphone.' }
    ],
    video: [
      { label: 'Tip', text: 'Drag clips on the timeline to reorder them before exporting.' },
      { label: 'Shortcut', text: 'Press S to split the selected clip at the playhead.' },
      { label: 'Tip', text: 'Text overlays work well for titles and captions; set colour and position in the inspector.' },
      { label: 'Tip', text: 'Need only the soundtrack? Export as MP3 or WAV.' },
      { label: 'Privacy', text: 'Video files stay on this computer. Nothing is uploaded.' },
      { label: 'Tip', text: 'Canvas presets (16:9, 9:16, 1:1) reframe your project for each platform.' }
    ],
    image: [
      { label: 'Shortcut', text: 'Press Ctrl+V to paste an image from the clipboard.' },
      { label: 'Tip', text: 'Increase quality reconstructs detail and edges instead of simply stretching pixels.' },
      { label: 'Shortcut', text: 'Press Ctrl+Z to undo adjustments and drawing.' },
      { label: 'Privacy', text: 'Images are processed on this computer and never leave it.' },
      { label: 'Tip', text: 'Use the Text tool for captions, then drag them into place on the canvas.' }
    ],
    general: [
      { label: 'Privacy', text: 'Your files stay on this computer. Nothing is sent to an online service.' },
      { label: 'Tip', text: 'The theme follows your system setting until you choose one in the header.' },
      { label: 'Tip', text: 'You can drop files anywhere on the workspace to import them.' }
    ]
  };

  // Callers sometimes pass decorative emoji in status strings; keep the copy clean.
  const EMOJI = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B50}\u{2705}\u{274C}\u{FE0F}\u{200D}]/gu;
  function clean(text) {
    return String(text == null ? '' : text).replace(EMOJI, '').replace(/!+(\s|$)/g, '.$1').replace(/\s{2,}/g, ' ').trim();
  }

  function $(id) { return document.getElementById(id); }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  function prefersReducedMotion() {
    try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; }
  }

  function rotateTip() {
    const tipList = TIPS[currentCategory] || TIPS.general;
    if (!tipList.length) return;
    currentTipIndex = (currentTipIndex + 1) % tipList.length;
    const tip = tipList[currentTipIndex];
    const labelEl = $('poTipLabel');
    const textEl = $('poTipText');
    if (!textEl) return;
    const swap = () => {
      if (labelEl) labelEl.textContent = tip.label;
      textEl.textContent = tip.text;
      textEl.classList.remove('fade-out');
    };
    if (prefersReducedMotion()) { swap(); return; }
    textEl.classList.add('fade-out');
    setTimeout(swap, 260);
  }

  function startTimer() {
    stopTimer();
    startTime = Date.now();
    const timerEl = $('poTimerText');
    if (timerEl) timerEl.textContent = '00:00';
    timerInterval = setInterval(() => {
      const el = $('poTimerText');
      if (el) el.textContent = formatTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
  }

  function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  }

  function startTipRotator(category) {
    stopTipRotator();
    currentCategory = TIPS[category] ? category : 'general';
    currentTipIndex = 0;
    const first = (TIPS[currentCategory] || TIPS.general)[0];
    const labelEl = $('poTipLabel');
    const textEl = $('poTipText');
    if (labelEl) labelEl.textContent = first.label;
    if (textEl) textEl.textContent = first.text;
    tipInterval = setInterval(rotateTip, 5000);
  }

  function stopTipRotator() {
    if (tipInterval) { clearInterval(tipInterval); tipInterval = null; }
  }

  function setIndeterminate(on) {
    const track = document.querySelector('#processingOverlay .po-progress-track');
    if (!track) return;
    track.classList.toggle('is-indeterminate', on);
    if (on) track.removeAttribute('aria-valuenow');
  }

  function injectDOM() {
    if ($('processingOverlay') || !document.body) return;
    const div = document.createElement('div');
    div.id = 'processingOverlay';
    div.className = 'hidden';
    div.innerHTML = `
      <div class="po-card" role="dialog" aria-modal="true" aria-labelledby="poTitleText" aria-describedby="poStageText" tabindex="-1">
        <div class="po-header">
          <h2 class="po-title" id="poTitle">
            <span class="po-visualizer" aria-hidden="true">
              <span class="po-wave-bar"></span><span class="po-wave-bar"></span><span class="po-wave-bar"></span><span class="po-wave-bar"></span>
            </span>
            <span class="po-title-text" id="poTitleText">Processing</span>
          </h2>
          <span class="po-timer-pill" title="Elapsed time">
            <i class="fas fa-clock" aria-hidden="true"></i>
            <span class="sr-only">Elapsed time</span>
            <span id="poTimerText">00:00</span>
          </span>
        </div>

        <div class="po-stage-container">
          <p class="po-stage-text" id="poStageText" role="status" aria-live="polite">Starting…</p>
        </div>

        <div class="po-progress-wrapper">
          <div class="po-progress-track is-indeterminate" role="progressbar" aria-label="Progress" aria-valuemin="0" aria-valuemax="100">
            <div class="po-progress-fill" id="poProgressFill"></div>
          </div>
          <div class="po-progress-meta">
            <span id="poProgressStep">Working</span>
            <span id="poProgressPercent">0%</span>
          </div>
        </div>

        <div class="po-tip-box">
          <i class="fas fa-lightbulb po-tip-icon" aria-hidden="true"></i>
          <div class="po-tip-content">
            <div class="po-tip-label" id="poTipLabel">Tip</div>
            <p class="po-tip-text" id="poTipText"></p>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(div);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectDOM);
  } else {
    injectDOM();
  }

  return {
    show: function (options) {
      injectDOM();
      const opts = options || {};
      const titleEl = $('poTitleText');
      const stageEl = $('poStageText');
      const fillEl = $('poProgressFill');
      const percentEl = $('poProgressPercent');
      const stepEl = $('poProgressStep');

      if (titleEl) titleEl.textContent = clean(opts.title || 'Processing') || 'Processing';
      if (stageEl) stageEl.textContent = clean(opts.stageText || opts.message || 'Working on your request…');
      if (fillEl) fillEl.style.width = '0%';
      if (percentEl) percentEl.textContent = '0%';
      if (stepEl) stepEl.textContent = clean(opts.stepText || 'Working');
      setIndeterminate(true);

      startTimer();
      startTipRotator(opts.category || 'general');

      const overlay = $('processingOverlay');
      if (overlay) {
        const wasHidden = overlay.classList.contains('hidden');
        overlay.classList.remove('hidden');
        if (wasHidden) {
          previousFocus = document.activeElement;
          const card = overlay.querySelector('.po-card');
          if (card) { try { card.focus({ preventScroll: true }); } catch (e) { card.focus(); } }
        }
      }
    },

    updateProgress: function (percent, stageText, stepText) {
      const fillEl = $('poProgressFill');
      const percentEl = $('poProgressPercent');
      const stageEl = $('poStageText');
      const stepEl = $('poProgressStep');
      const track = document.querySelector('#processingOverlay .po-progress-track');

      const p = Math.min(100, Math.max(0, Math.round(Number(percent) || 0)));
      setIndeterminate(false);
      if (fillEl) fillEl.style.width = `${p}%`;
      if (percentEl) percentEl.textContent = `${p}%`;
      if (track) track.setAttribute('aria-valuenow', String(p));

      if (stageText && stageEl) {
        const text = clean(stageText);
        if (prefersReducedMotion()) {
          stageEl.textContent = text;
        } else {
          stageEl.style.opacity = '0.4';
          setTimeout(() => { stageEl.textContent = text; stageEl.style.opacity = '1'; }, 140);
        }
      }
      if (stepText && stepEl) stepEl.textContent = clean(stepText);
    },

    hide: function () {
      const overlay = $('processingOverlay');
      const wasOpen = overlay && !overlay.classList.contains('hidden');
      if (overlay) overlay.classList.add('hidden');
      stopTimer();
      stopTipRotator();
      if (wasOpen && previousFocus && typeof previousFocus.focus === 'function' && document.contains(previousFocus)) {
        try { previousFocus.focus({ preventScroll: true }); } catch (e) { /* element gone */ }
      }
      previousFocus = null;
    }
  };
})();
