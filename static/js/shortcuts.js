/**
 * Keyboard Shortcuts & Modal Event Dispatcher
 */

document.addEventListener('DOMContentLoaded', () => {
    // Modal Toggles
    const btnShortcuts = document.getElementById('btnShortcutsHelp');
    const modalShortcuts = document.getElementById('shortcutsModal');
    const btnCloseShortcuts = document.getElementById('btnCloseShortcutsModal');

    if (btnShortcuts && modalShortcuts) {
        btnShortcuts.addEventListener('click', () => modalShortcuts.classList.add('active'));
    }
    if (btnCloseShortcuts && modalShortcuts) {
        btnCloseShortcuts.addEventListener('click', () => modalShortcuts.classList.remove('active'));
    }

    // Keybindings listener
    window.addEventListener('keydown', (e) => {
        // Ignore keybindings if user is typing in input or textarea
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select') {
            return;
        }

        // Space -> Play/Pause
        if (e.code === 'Space') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.togglePlayPause();
            else if (window.audioStudio && window.audioStudio.wavesurfer) window.audioStudio.wavesurfer.playPause();
        }

        // S -> Split clip
        if (e.code === 'KeyS' && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.splitSelectedClip();
        }

        // Delete / Backspace -> Delete selected clip
        if (e.code === 'Delete' || e.code === 'Backspace') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.deleteSelectedClip();
        }

        // J -> Shuttle Reverse, K -> Pause, L -> Shuttle Forward
        if (e.code === 'KeyJ') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.nudgeFrame(-5);
        }
        if (e.code === 'KeyK') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.setPlayState(false);
        }
        if (e.code === 'KeyL') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.nudgeFrame(5);
        }

        // Left Arrow / Right Arrow
        if (e.code === 'ArrowLeft') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.nudgeFrame(-1);
        }
        if (e.code === 'ArrowRight') {
            e.preventDefault();
            if (window.videoStudio) window.videoStudio.nudgeFrame(1);
        }

        // ? -> Open Shortcuts Help
        if (e.key === '?') {
            e.preventDefault();
            if (modalShortcuts) modalShortcuts.classList.toggle('active');
        }
    });
});
