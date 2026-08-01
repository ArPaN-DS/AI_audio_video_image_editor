/**
 * Interactive Natural-Language Processing Overlay & Engagement System
 * Keeps users engaged with live elapsed time, natural language updates, and pro-tips!
 */

window.ProcessingOverlay = (function () {
  let timerInterval = null;
  let tipInterval = null;
  let startTime = 0;
  let currentTipIndex = 0;
  let currentCategory = 'general';

  // Bank of engaging non-technical Pro-Tips & Shortcuts
  const TIPS = {
    audio: [
      { label: "Pro Tip", text: "Double-click anywhere on the audio waveform to instantly create a new cut region!" },
      { label: "Shortcut", text: "Press Spacebar anytime to play or pause your audio track." },
      { label: "Pro Tip", text: "Use Fade In and Fade Out effects to make smooth transitions at the beginning and end of clips." },
      { label: "Shortcut", text: "Press Ctrl+Z to undo any cut or edit operation." },
      { label: "Privacy First", text: "All audio cutting and AI transcription runs 100% locally on your machine!" },
      { label: "Did You Know?", text: "You can record directly from your microphone using the Record Mic button." }
    ],
    video: [
      { label: "Pro Tip", text: "Drag clips on the timeline to reorder them before rendering your final video." },
      { label: "Shortcut", text: "Press 'S' or click Split to slice a clip at the current playhead position." },
      { label: "Pro Tip", text: "Add text overlays with custom colors and positions to create captions or titles!" },
      { label: "Quick Tools", text: "Need just the soundtrack? Use the 'Extract Audio' tool to save any video as MP3!" },
      { label: "Privacy First", text: "Your video files are processed locally via FFmpeg — no cloud uploads!" },
      { label: "Pro Tip", text: "Use Canvas Presets (16:9, 9:16 Shorts/Reels, 1:1 Square) for instant social media formatting." }
    ],
    image: [
      { label: "Pro Tip", text: "Paste an image straight from your clipboard using Ctrl+V!" },
      { label: "Local AI", text: "The 'Increase Quality' tool uses AI super-resolution to sharpen edges without pixelation." },
      { label: "Shortcut", text: "Press Ctrl+Z to undo adjustments or freehand drawing annotations." },
      { label: "Privacy First", text: "Image editing is 100% browser-based — your pictures never touch the server!" },
      { label: "Meme Generator", text: "Add top and bottom text with classic Impact font to turn any image into a meme!" }
    ],
    general: [
      { label: "Privacy First", text: "All your files stay on your machine — zero cloud tracking or third-party storage." },
      { label: "Tip", text: "Dark mode is automatically matched to your system preference!" },
      { label: "Pro Tip", text: "You can drag & drop files directly anywhere on the studio canvas." }
    ]
  };

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  function rotateTip() {
    const tipList = TIPS[currentCategory] || TIPS.general;
    if (!tipList || tipList.length === 0) return;

    currentTipIndex = (currentTipIndex + 1) % tipList.length;
    const tip = tipList[currentTipIndex];

    const labelEl = document.getElementById('poTipLabel');
    const textEl = document.getElementById('poTipText');

    if (textEl) {
      textEl.classList.add('fade-out');
      setTimeout(() => {
        if (labelEl) labelEl.textContent = tip.label;
        textEl.textContent = tip.text;
        textEl.classList.remove('fade-out');
        textEl.classList.add('fade-in');
        setTimeout(() => textEl.classList.remove('fade-in'), 300);
      }, 250);
    }
  }

  function startTimer() {
    stopTimer();
    startTime = Date.now();
    const timerEl = document.getElementById('poTimerText');
    if (timerEl) timerEl.textContent = '00:00';

    timerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
      if (timerEl) timerEl.textContent = formatTime(elapsedSec);
    }, 1000);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  function startTipRotator(category) {
    stopTipRotator();
    currentCategory = category || 'general';
    currentTipIndex = 0;

    const tipList = TIPS[currentCategory] || TIPS.general;
    const firstTip = tipList[0] || TIPS.general[0];

    const labelEl = document.getElementById('poTipLabel');
    const textEl = document.getElementById('poTipText');
    if (labelEl) labelEl.textContent = firstTip.label;
    if (textEl) textEl.textContent = firstTip.text;

    tipInterval = setInterval(rotateTip, 4500);
  }

  function stopTipRotator() {
    if (tipInterval) {
      clearInterval(tipInterval);
      tipInterval = null;
    }
  }

  function injectDOM() {
    if (document.getElementById('processingOverlay')) return;

    const div = document.createElement('div');
    div.id = 'processingOverlay';
    div.className = 'hidden';
    div.innerHTML = `
      <div class="po-card">
        <div class="po-header">
          <h4 class="po-title" id="poTitle">
            <span>⚡</span> <span id="poTitleText">Processing...</span>
          </h4>
          <div class="po-timer-pill">
            <span>⏱️</span> <span id="poTimerText">00:00</span>
          </div>
        </div>

        <div class="po-visualizer">
          <div class="po-wave-bar"></div>
          <div class="po-wave-bar"></div>
          <div class="po-wave-bar"></div>
          <div class="po-wave-bar"></div>
          <div class="po-wave-bar"></div>
          <div class="po-wave-bar"></div>
        </div>

        <div class="po-stage-container">
          <div class="po-stage-text" id="poStageText">Initializing task...</div>
        </div>

        <div class="po-progress-wrapper">
          <div class="po-progress-track">
            <div class="po-progress-fill" id="poProgressFill" style="width: 0%;"></div>
          </div>
          <div class="po-progress-meta">
            <span id="poProgressStep">Step 1 of 1</span>
            <span id="poProgressPercent">0%</span>
          </div>
        </div>

        <div class="po-tip-box">
          <div class="po-tip-icon">💡</div>
          <div class="po-tip-content">
            <div class="po-tip-label" id="poTipLabel">Pro Tip</div>
            <p class="po-tip-text" id="poTipText">Dark mode is automatically matched to your system preference!</p>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(div);
  }

  // Ensure DOM is ready before injecting
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectDOM);
  } else {
    injectDOM();
  }

  return {
    show: function (options) {
      injectDOM();
      const title = options?.title || 'Processing...';
      const stageText = options?.stageText || 'Working on your request...';
      const category = options?.category || 'general';

      const titleEl = document.getElementById('poTitleText');
      const stageEl = document.getElementById('poStageText');
      const fillEl = document.getElementById('poProgressFill');
      const percentEl = document.getElementById('poProgressPercent');
      const stepEl = document.getElementById('poProgressStep');

      if (titleEl) titleEl.textContent = title;
      if (stageEl) stageEl.textContent = stageText;
      if (fillEl) fillEl.style.width = '0%';
      if (percentEl) percentEl.textContent = '0%';
      if (stepEl) stepEl.textContent = options?.stepText || 'Working...';

      startTimer();
      startTipRotator(category);

      const overlay = document.getElementById('processingOverlay');
      if (overlay) overlay.classList.remove('hidden');
    },

    updateProgress: function (percent, stageText, stepText) {
      const fillEl = document.getElementById('poProgressFill');
      const percentEl = document.getElementById('poProgressPercent');
      const stageEl = document.getElementById('poStageText');
      const stepEl = document.getElementById('poProgressStep');

      const p = Math.min(100, Math.max(0, Math.round(percent)));
      if (fillEl) fillEl.style.width = `${p}%`;
      if (percentEl) percentEl.textContent = `${p}%`;

      if (stageText && stageEl) {
        stageEl.style.opacity = '0.5';
        setTimeout(() => {
          stageEl.textContent = stageText;
          stageEl.style.opacity = '1';
        }, 150);
      }

      if (stepText && stepEl) {
        stepEl.textContent = stepText;
      }
    },

    hide: function () {
      const overlay = document.getElementById('processingOverlay');
      if (overlay) overlay.classList.add('hidden');
      stopTimer();
      stopTipRotator();
    }
  };
})();
