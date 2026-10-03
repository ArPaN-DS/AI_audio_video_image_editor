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
        var label = dark ? 'Switch to light theme' : 'Switch to dark theme';
        btn.title = label;
        btn.setAttribute('aria-label', label);
        btn.setAttribute('aria-pressed', dark ? 'true' : 'false');
    }

    function apply(theme) {
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

        function openDrawer() {
            drawer.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
            if (menuBtn) menuBtn.setAttribute('aria-expanded', 'true');
        }

        function closeDrawer() {
            drawer.classList.add('hidden');
            document.body.style.overflow = '';
            if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
        }

        menuBtn.addEventListener('click', openDrawer);
        if (backdrop) backdrop.addEventListener('click', closeDrawer);
        if (closeBtn) closeBtn.addEventListener('click', closeDrawer);

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && drawer && !drawer.classList.contains('hidden')) {
                closeDrawer();
            }
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
        // Follow the OS preference only while the user hasn't chosen explicitly.
        try {
            var mq = window.matchMedia('(prefers-color-scheme: dark)');
            mq.addEventListener('change', function (e) {
                if (!localStorage.getItem('theme')) {
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
