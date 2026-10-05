/* ═══════════════════════════════════════
   Theme controller — light / dark
   Shared across Audio, Video and Image editors.
   The no-FOUC snippet in each page's <head> sets the
   initial data-theme before paint; this file wires the
   toggle button, keeps the icon in sync, and broadcasts
   a `themechange` event so canvases/waveforms can recolor.
   ═══════════════════════════════════════ */
(function () {
    'use strict';

    function current() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    }

    function syncButton() {
        var btn = document.getElementById('themeToggle');
        if (!btn) return;
        var dark = current() === 'dark';
        var icon = btn.querySelector('i');
        if (icon) icon.className = dark ? 'fas fa-sun' : 'fas fa-moon';
        // Toggle-button pattern: a constant accessible name + aria-pressed state.
        // The tooltip describes the action the click will perform.
        btn.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
        btn.setAttribute('aria-label', 'Dark theme');
        btn.setAttribute('aria-pressed', dark ? 'true' : 'false');
    }

    // Suppress per-element colour transitions while the palette swaps, so the
    // whole UI changes in one frame instead of fading piecemeal.
    var switchTimer = null;
    function suppressTransitions() {
        var root = document.documentElement;
        if (!root.classList || typeof setTimeout !== 'function') return;
        root.classList.add('theme-switching');
        if (switchTimer) clearTimeout(switchTimer);
        switchTimer = setTimeout(function () { root.classList.remove('theme-switching'); }, 60);
    }

    function apply(theme) {
        suppressTransitions();
        document.documentElement.setAttribute('data-theme', theme);
        try { localStorage.setItem('theme', theme); } catch (e) { /* private mode */ }
        syncButton();
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
    }

    function initMobileNav() {
        var menuBtn = document.getElementById('mobileMenuBtn');
        var drawer = document.getElementById('mobileNavDrawer');
        var backdrop = document.getElementById('mobileNavBackdrop');
        var closeBtn = document.getElementById('closeMobileNav');

        if (!menuBtn || !drawer) return;
        var previousFocus = null;
        var previousOverflow = '';
        var inertElements = [];
        drawer.setAttribute('role', 'dialog');
        drawer.setAttribute('aria-modal', 'true');
        drawer.setAttribute('aria-label', 'Navigation');
        drawer.setAttribute('aria-hidden', 'true');
        menuBtn.setAttribute('aria-controls', drawer.id);
        menuBtn.setAttribute('aria-expanded', 'false');

        function focusableElements() {
            return Array.from(drawer.querySelectorAll('a[href], button, input, select, textarea, [tabindex]'))
                .filter(function (element) { return !element.disabled && element.tabIndex >= 0; });
        }

        function openDrawer() {
            if (!drawer.classList.contains('hidden')) return;
            previousFocus = document.activeElement;
            previousOverflow = document.body.style.overflow;
            inertElements = Array.from(document.body.children)
                .filter(function (element) { return element !== drawer && !element.contains(drawer); })
                .map(function (element) {
                    var state = { element: element, inert: element.inert };
                    element.inert = true;
                    return state;
                });
            drawer.classList.remove('hidden');
            drawer.setAttribute('aria-hidden', 'false');
            document.body.style.overflow = 'hidden';
            if (menuBtn) menuBtn.setAttribute('aria-expanded', 'true');
            (closeBtn || focusableElements()[0])?.focus();
        }

        function closeDrawer() {
            if (drawer.classList.contains('hidden')) return;
            drawer.classList.add('hidden');
            drawer.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = previousOverflow;
            inertElements.forEach(function (state) { state.element.inert = state.inert; });
            inertElements = [];
            if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
            (previousFocus || menuBtn).focus();
        }

        menuBtn.addEventListener('click', openDrawer);
        if (backdrop) backdrop.addEventListener('click', closeDrawer);
        if (closeBtn) closeBtn.addEventListener('click', closeDrawer);
        drawer.addEventListener('click', function (event) {
            if (event.target.closest('a[href]')) closeDrawer();
        });

        document.addEventListener('keydown', function (e) {
            if (drawer.classList.contains('hidden')) return;
            e.stopPropagation();
            if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                closeDrawer();
            } else if (e.key === 'Tab') {
                var elements = focusableElements();
                var first = elements[0];
                var last = elements[elements.length - 1];
                if (e.shiftKey && document.activeElement === first) {
                    e.preventDefault();
                    last?.focus();
                } else if (!e.shiftKey && document.activeElement === last) {
                    e.preventDefault();
                    first?.focus();
                }
            }
        });
        document.addEventListener('focusin', function (event) {
            if (!drawer.classList.contains('hidden') && !drawer.contains(event.target)) {
                (closeBtn || focusableElements()[0])?.focus();
            }
        });
        window.matchMedia('(min-width: 769px)').addEventListener('change', function (event) {
            if (event.matches) closeDrawer();
        });
    }

    function init() {
        syncButton();
        var btn = document.getElementById('themeToggle');
        if (btn) {
            btn.addEventListener('click', function () {
                apply(current() === 'dark' ? 'light' : 'dark');
            });
        }
        initMobileNav();
        window.addEventListener('storage', function (event) {
            if (event.key === 'theme' && (event.newValue === 'light' || event.newValue === 'dark')) {
                suppressTransitions();
                document.documentElement.setAttribute('data-theme', event.newValue);
                syncButton();
                window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: event.newValue } }));
            }
        });
        // Follow the OS preference only while the user hasn't chosen explicitly.
        try {
            var mq = window.matchMedia('(prefers-color-scheme: dark)');
            mq.addEventListener('change', function (e) {
                if (!localStorage.getItem('theme')) {
                    suppressTransitions();
                    document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
                    syncButton();
                    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: current() } }));
                }
            });
        } catch (e) { /* older browsers */ }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
