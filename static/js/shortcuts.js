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

    // Dialog behaviour for every `.modal-overlay`: focus moves in on open and back on close,
    // Tab stays inside, Esc or a backdrop click closes.
    const dialogs = Array.from(document.querySelectorAll('.modal-overlay'));
    const focusableIn = root => Array.from(root.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )).filter(element => element.offsetParent !== null);
    const openDialog = () => dialogs.find(dialog => dialog.classList.contains('active')) || null;
    dialogs.forEach(dialog => {
        let returnFocus = null;
        if (typeof MutationObserver !== 'undefined') {
            new MutationObserver(() => {
                const active = dialog.classList.contains('active');
                if (active && !dialog.contains(document.activeElement)) {
                    returnFocus = document.activeElement;
                    (dialog.querySelector('.modal-close') || focusableIn(dialog)[0])?.focus();
                } else if (!active && returnFocus) {
                    if (dialog.contains(document.activeElement) || document.activeElement === document.body) returnFocus.focus?.();
                    returnFocus = null;
                }
            }).observe(dialog, { attributes: true, attributeFilter: ['class'] });
        }
        dialog.addEventListener('mousedown', event => {
            if (event.target === dialog) dialog.classList.remove('active');
        });
    });

    const copilotDrawer = document.getElementById('aiAgentDrawer');
    const copilotToggle = document.getElementById('btnToggleAiAgent');

    // Keybindings listener
    window.addEventListener('keydown', (e) => {
        if (e.defaultPrevented) return;

        const dialog = openDialog();
        if (e.key === 'Escape') {
            if (dialog) {
                e.preventDefault();
                dialog.classList.remove('active');
            } else if (copilotDrawer?.classList.contains('open')) {
                e.preventDefault();
                copilotDrawer.classList.remove('open');
                copilotToggle?.focus();
            }
            return;
        }
        if (dialog) {
            // Keep keyboard focus inside the open dialog; suspend editing shortcuts.
            if (e.key === 'Tab') {
                const items = focusableIn(dialog);
                if (!items.length) return;
                const first = items[0];
                const last = items[items.length - 1];
                if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
                else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
                else if (!dialog.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
            }
            return;
        }
        if (e.ctrlKey || e.metaKey || e.altKey) return;

        // Ignore keybindings if user is typing in input or textarea
        const active = document.activeElement;
        const activeTag = active ? active.tagName.toLowerCase() : '';
        if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select' || active?.isContentEditable) {
            return;
        }
        // Let focused buttons, links and tabs keep their native Space / Enter behaviour.
        if ((e.code === 'Space' || e.key === 'Enter')
            && active?.closest?.('button, a[href], summary, [role="tab"], [role="switch"]')) {
            return;
        }
        // Editing shortcuts stay off while focus is inside Copilot.
        if (copilotDrawer?.contains(active)) return;

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
            if (window.videoStudio) window.videoStudio.preview.pause();
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
