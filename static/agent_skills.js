/**
 * Copilot Skills UI — "@" autocomplete for chat composers and the Skills library dialog.
 * Shared by the full Copilot workspace (/agent) and the studio Copilot drawer.
 *
 *   AgentSkills.attachAutocomplete(textarea, { getMediaType, onOpenLibrary })
 *   AgentSkills.openLibrary({ getMediaType, onUse, returnFocus })
 *
 * Pure helpers (filterSkills, findTrigger, applyCompletion) are exported for tests.
 */
(function (root) {
    'use strict';

    const MEDIA_LABELS = { audio: 'Audio', video: 'Video', image: 'Image' };
    let catalogPromise = null;
    let catalogData = null;
    let uid = 0;

    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    // ─── Data ───
    function loadCatalog(force = false) {
        if (catalogPromise && !force) return catalogPromise;
        const fetcher = root.fetch ? root.fetch.bind(root) : null;
        if (!fetcher) return Promise.resolve({ skills: [], categories: [] });
        catalogPromise = fetcher('/api/agent/skills')
            .then(response => response.json())
            .then(data => {
                catalogData = {
                    skills: Array.isArray(data && data.skills) ? data.skills.filter(isValidSkill) : [],
                    categories: Array.isArray(data && data.categories) ? data.categories : []
                };
                return catalogData;
            })
            .catch(() => {
                catalogPromise = null;
                return { skills: [], categories: [] };
            });
        return catalogPromise;
    }

    function isValidSkill(skill) {
        return skill && typeof skill.id === 'string' && /^[a-z][a-z0-9-]*$/.test(skill.id)
            && typeof skill.title === 'string' && typeof skill.description === 'string';
    }

    function setCatalog(data) {
        catalogData = { skills: (data.skills || []).filter(isValidSkill), categories: data.categories || [] };
        catalogPromise = Promise.resolve(catalogData);
    }

    /** Rank skills for a query: id prefix > id contains > title/tags/description words. */
    function filterSkills(skills, query, options = {}) {
        const q = String(query || '').trim().toLowerCase().replace(/^@/, '');
        const media = options.mediaType;
        const category = options.category;
        const scored = [];
        (skills || []).forEach((skill, index) => {
            if (!isValidSkill(skill)) return;
            if (media && Array.isArray(skill.media_types) && !skill.media_types.includes(media)) return;
            if (category && category !== 'all' && skill.category !== category) return;
            let score = 0;
            if (!q) score = 1;
            else if (skill.id.startsWith(q)) score = 100 - skill.id.length / 100;
            else if (skill.id.includes(q)) score = 60;
            else {
                const haystack = [skill.title, skill.description, (skill.tags || []).join(' ')].join(' ').toLowerCase();
                const words = q.split(/[\s-]+/).filter(Boolean);
                const hits = words.filter(word => haystack.includes(word)).length;
                if (hits === words.length) score = 20 + hits;
                else if (hits) score = 5 + hits;
            }
            if (score > 0) scored.push({ skill, score, index });
        });
        scored.sort((a, b) => b.score - a.score || a.index - b.index);
        return scored.map(item => item.skill);
    }

    /** The "@query" being typed just before the caret, or null. */
    function findTrigger(text, caret) {
        const before = String(text || '').slice(0, caret === undefined ? undefined : caret);
        const match = before.match(/(^|[\s(])@([a-z0-9-]{0,40})$/i);
        if (!match) return null;
        return { start: before.length - match[2].length - 1, end: before.length, query: match[2].toLowerCase() };
    }

    /** Replace the "@query" token with "@id " and return the new text and caret. */
    function applyCompletion(text, trigger, skillId) {
        const value = String(text || '');
        const insert = `@${skillId} `;
        if (!trigger) {
            const prefix = value && !/\s$/.test(value) ? value + ' ' : value;
            return { text: prefix + insert, caret: (prefix + insert).length };
        }
        const after = value.slice(trigger.end).replace(/^\S*/, '').replace(/^\s+/, '');
        const next = value.slice(0, trigger.start) + insert + after;
        return { text: next, caret: trigger.start + insert.length };
    }

    function paramsSummary(skill) {
        return (skill.params || []).map(param => {
            let detail = param.name;
            if (Array.isArray(param.enum)) detail += ` (${param.enum.join(' | ')})`;
            else if (param.minimum !== undefined || param.maximum !== undefined) {
                detail += ` (${param.minimum ?? ''}–${param.maximum ?? ''}${param.unit ? ' ' + param.unit : ''})`;
            }
            if (param.required) detail += ', required';
            else if (param.default !== undefined && param.default !== null) detail += `, default ${param.default}`;
            return detail;
        });
    }

    // ─── "@" autocomplete ───
    function attachAutocomplete(textarea, options = {}) {
        if (!textarea || textarea.__agentSkillsAttached) return textarea && textarea.__agentSkillsAttached;
        const doc = root.document;
        const id = `agentSkillsPopup${++uid}`;
        const popup = doc.createElement('div');
        popup.id = id;
        popup.className = 'agent-skills-popup';
        popup.setAttribute('role', 'listbox');
        popup.setAttribute('aria-label', 'Skills');
        popup.hidden = true;
        const host = options.container || textarea.parentNode || doc.body;
        host.appendChild(popup);

        textarea.setAttribute('aria-autocomplete', 'list');
        textarea.setAttribute('aria-controls', id);
        textarea.setAttribute('aria-expanded', 'false');

        const state = { open: false, items: [], active: 0, trigger: null };

        function mediaType() {
            try { return options.getMediaType ? options.getMediaType() || null : null; } catch { return null; }
        }

        function close() {
            state.open = false;
            state.items = [];
            popup.hidden = true;
            popup.innerHTML = '';
            textarea.setAttribute('aria-expanded', 'false');
            textarea.removeAttribute ? textarea.removeAttribute('aria-activedescendant') : textarea.setAttribute('aria-activedescendant', '');
        }

        function render() {
            if (!state.items.length) {
                popup.innerHTML = `<div class="agent-skills-empty" role="presentation">No matching skills${mediaType() ? ` for ${escapeHtml(mediaType())}` : ''}.</div>`
                    + footer();
            } else {
                popup.innerHTML = state.items.map((skill, index) => `
                    <div class="agent-skills-option${index === state.active ? ' is-active' : ''}" role="option"
                         id="${id}_opt${index}" data-skill-id="${escapeHtml(skill.id)}" aria-selected="${index === state.active}">
                        <span class="agent-skills-option-title">${escapeHtml(skill.title)} <span class="agent-skills-id">@${escapeHtml(skill.id)}</span></span>
                        <span class="agent-skills-option-desc">${escapeHtml(skill.description)}</span>
                    </div>`).join('') + footer();
            }
            popup.hidden = false;
            textarea.setAttribute('aria-expanded', 'true');
            if (state.items.length) textarea.setAttribute('aria-activedescendant', `${id}_opt${state.active}`);
        }

        function footer() {
            return `<button type="button" class="agent-skills-browse" data-skills-browse="1">Browse all skills</button>`;
        }

        function update() {
            const caret = textarea.selectionStart ?? String(textarea.value || '').length;
            const trigger = findTrigger(textarea.value, caret);
            if (!trigger) { close(); return; }
            state.trigger = trigger;
            loadCatalog().then(data => {
                const current = findTrigger(textarea.value, textarea.selectionStart ?? String(textarea.value || '').length);
                if (!current || current.query !== trigger.query) return;
                state.items = filterSkills(data.skills, trigger.query, { mediaType: mediaType() }).slice(0, 8);
                state.active = 0;
                state.open = true;
                render();
            });
        }

        function choose(index) {
            const skill = state.items[index];
            if (!skill) return;
            const result = applyCompletion(textarea.value, state.trigger, skill.id);
            textarea.value = result.text;
            if (textarea.setSelectionRange) textarea.setSelectionRange(result.caret, result.caret);
            close();
            if (textarea.focus) textarea.focus();
            if (options.onSelect) options.onSelect(skill);
        }

        function onKeydown(event) {
            if (!state.open) return;
            const key = event.key;
            if (key === 'ArrowDown' || key === 'ArrowUp') {
                if (!state.items.length) return;
                state.active = (state.active + (key === 'ArrowDown' ? 1 : -1) + state.items.length) % state.items.length;
                render();
            } else if ((key === 'Enter' || key === 'Tab') && !event.shiftKey && !event.isComposing) {
                if (!state.items.length) { close(); return; }
                choose(state.active);
            } else if (key === 'Escape') {
                close();
            } else {
                return;
            }
            event.preventDefault();
            if (event.stopImmediatePropagation) event.stopImmediatePropagation();
        }

        textarea.addEventListener('input', update);
        textarea.addEventListener('keydown', onKeydown, true);   // capture: runs before Enter-to-send
        textarea.addEventListener('blur', () => root.setTimeout(close, 150));
        popup.addEventListener('mousedown', event => {
            const target = event.target && event.target.closest ? event.target : null;
            const browse = target && target.closest('[data-skills-browse]');
            const option = target && target.closest('[data-skill-id]');
            if (!browse && !option) return;
            event.preventDefault();
            if (browse) {
                close();
                if (options.onOpenLibrary) options.onOpenLibrary();
                return;
            }
            choose(state.items.findIndex(skill => skill.id === option.getAttribute('data-skill-id')));
        });

        const api = { update, close, choose, onKeydown, state, popup };
        textarea.__agentSkillsAttached = api;
        return api;
    }

    // ─── Skills library dialog ───
    const libraryState = { open: false, query: '', media: 'all', category: 'all', tab: 'all', selected: null, opts: {} };
    let libraryEl = null;
    let libraryReturnFocus = null;

    function libraryResults() {
        const data = catalogData || { skills: [] };
        const skills = libraryState.tab === 'mine' ? data.skills.filter(skill => skill.user) : data.skills;
        return filterSkills(skills, libraryState.query, {
            mediaType: libraryState.media === 'all' ? null : libraryState.media,
            category: libraryState.tab === 'mine' ? null : libraryState.category
        });
    }

    function renderLibrary() {
        if (!libraryEl) return;
        const data = catalogData || { skills: [], categories: [] };
        const results = libraryResults();
        if (!results.some(skill => skill.id === libraryState.selected)) libraryState.selected = results[0] ? results[0].id : null;
        const selected = results.find(skill => skill.id === libraryState.selected) || null;
        const categories = (data.categories || []).filter(category => category.id !== 'mine'
            && data.skills.some(skill => skill.category === category.id));
        const chip = (group, value, label, current) => `<button type="button" class="agent-skills-chip${current === value ? ' is-on' : ''}"
            data-chip-group="${group}" data-chip-value="${escapeHtml(value)}" aria-pressed="${current === value}">${escapeHtml(label)}</button>`;

        const list = results.length ? results.map(skill => `
            <div class="agent-skills-result${skill.id === libraryState.selected ? ' is-active' : ''}" role="option"
                 id="agentSkillsResult_${escapeHtml(skill.id)}" data-skill-id="${escapeHtml(skill.id)}"
                 aria-selected="${skill.id === libraryState.selected}">
                <span class="agent-skills-option-title">${escapeHtml(skill.title)} <span class="agent-skills-id">@${escapeHtml(skill.id)}</span></span>
                <span class="agent-skills-option-desc">${escapeHtml(skill.description)}</span>
            </div>`).join('')
            : `<p class="agent-skills-empty">${libraryState.tab === 'mine'
                ? 'No personal skills yet. After an edit works, say "save this as @my-skill".'
                : 'No skills match your search.'}</p>`;

        const detail = selected ? `
            <h3 class="agent-skills-detail-title">${escapeHtml(selected.title)} <span class="agent-skills-id">@${escapeHtml(selected.id)}</span></h3>
            <p>${escapeHtml(selected.description)}</p>
            <dl class="agent-skills-meta">
                <dt>Works on</dt><dd>${(selected.media_types || []).map(type => escapeHtml(MEDIA_LABELS[type] || type)).join(', ')}</dd>
                <dt>Steps</dt><dd>${(selected.steps || []).map(escapeHtml).join(' → ') || '—'}</dd>
                ${paramsSummary(selected).length ? `<dt>Settings</dt><dd>${paramsSummary(selected).map(escapeHtml).join('<br>')}</dd>` : ''}
                <dt>Example</dt><dd><code>${escapeHtml(selected.example || '@' + selected.id)}</code></dd>
                ${selected.availability && selected.availability.state !== 'ready'
                    ? `<dt>Availability</dt><dd>${escapeHtml(selected.availability.note || '')}</dd>` : ''}
            </dl>
            <div class="agent-skills-actions">
                <button type="button" class="agent-skills-use" data-skill-use="${escapeHtml(selected.id)}">Use @${escapeHtml(selected.id)}</button>
                ${selected.user ? `
                    <button type="button" class="agent-skills-secondary" data-skill-rename="${escapeHtml(selected.id)}">Rename</button>
                    <button type="button" class="agent-skills-danger" data-skill-delete="${escapeHtml(selected.id)}">Delete</button>` : ''}
            </div>` : '<p class="agent-skills-empty">Select a skill to see its details.</p>';

        libraryEl.querySelector('[data-skills-region="filters"]').innerHTML = `
            <div class="agent-skills-tabs" role="tablist" aria-label="Skill lists">
                <button type="button" role="tab" class="agent-skills-tab" data-chip-group="tab" data-chip-value="all" aria-selected="${libraryState.tab === 'all'}">All skills</button>
                <button type="button" role="tab" class="agent-skills-tab" data-chip-group="tab" data-chip-value="mine" aria-selected="${libraryState.tab === 'mine'}">My skills</button>
            </div>
            <div class="agent-skills-chips" role="group" aria-label="Media type">
                ${chip('media', 'all', 'Any media', libraryState.media)}${chip('media', 'audio', 'Audio', libraryState.media)}
                ${chip('media', 'video', 'Video', libraryState.media)}${chip('media', 'image', 'Image', libraryState.media)}
            </div>
            ${libraryState.tab === 'mine' ? '' : `<div class="agent-skills-chips" role="group" aria-label="Category">
                ${chip('category', 'all', 'All categories', libraryState.category)}
                ${categories.map(category => chip('category', category.id, category.label, libraryState.category)).join('')}
            </div>`}`;
        const listEl = libraryEl.querySelector('[data-skills-region="list"]');
        listEl.innerHTML = list;
        if (selected) listEl.setAttribute('aria-activedescendant', `agentSkillsResult_${selected.id}`);
        libraryEl.querySelector('[data-skills-region="detail"]').innerHTML = detail;
        const count = libraryEl.querySelector('[data-skills-region="count"]');
        if (count) count.textContent = `${results.length} skill${results.length === 1 ? '' : 's'}`;
    }

    function buildLibrary() {
        const doc = root.document;
        const backdrop = doc.createElement('div');
        backdrop.className = 'agent-skills-backdrop';
        backdrop.hidden = true;
        backdrop.innerHTML = `
            <div class="agent-skills-dialog" role="dialog" aria-modal="true" aria-labelledby="agentSkillsTitle">
                <header class="agent-skills-header">
                    <h2 id="agentSkillsTitle">Skills</h2>
                    <span class="agent-skills-count" data-skills-region="count" aria-live="polite"></span>
                    <button type="button" class="agent-skills-close" data-skills-close="1" aria-label="Close skills">×</button>
                </header>
                <label class="agent-skills-search-label" for="agentSkillsSearch">Search skills</label>
                <input id="agentSkillsSearch" class="agent-skills-search" type="search" autocomplete="off"
                       placeholder="Search by name, task or keyword" data-skills-search="1">
                <div data-skills-region="filters"></div>
                <div class="agent-skills-body">
                    <div class="agent-skills-list" role="listbox" tabindex="0" aria-label="Skills" data-skills-region="list"></div>
                    <section class="agent-skills-detail" aria-live="polite" data-skills-region="detail"></section>
                </div>
            </div>`;
        doc.body.appendChild(backdrop);

        backdrop.addEventListener('click', event => {
            const target = event.target;
            if (target === backdrop || (target.closest && target.closest('[data-skills-close]'))) { closeLibrary(); return; }
            const chipEl = target.closest && target.closest('[data-chip-group]');
            if (chipEl) {
                const group = chipEl.getAttribute('data-chip-group');
                libraryState[group === 'tab' ? 'tab' : group] = chipEl.getAttribute('data-chip-value');
                renderLibrary();
                return;
            }
            const use = target.closest && target.closest('[data-skill-use]');
            if (use) { useSkill(use.getAttribute('data-skill-use')); return; }
            const remove = target.closest && target.closest('[data-skill-delete]');
            if (remove) { deleteSkill(remove.getAttribute('data-skill-delete')); return; }
            const rename = target.closest && target.closest('[data-skill-rename]');
            if (rename) { renameSkill(rename.getAttribute('data-skill-rename')); return; }
            const option = target.closest && target.closest('[data-skill-id]');
            if (option) { libraryState.selected = option.getAttribute('data-skill-id'); renderLibrary(); }
        });
        backdrop.addEventListener('input', event => {
            if (event.target && event.target.getAttribute && event.target.getAttribute('data-skills-search')) {
                libraryState.query = event.target.value;
                renderLibrary();
            }
        });
        backdrop.addEventListener('keydown', libraryKeydown);
        return backdrop;
    }

    function libraryKeydown(event) {
        if (!libraryState.open) return;
        if (event.key === 'Escape') { event.preventDefault(); closeLibrary(); return; }
        const results = libraryResults();
        const inList = event.target && event.target.getAttribute && event.target.getAttribute('data-skills-region') === 'list';
        const inSearch = event.target && event.target.getAttribute && event.target.getAttribute('data-skills-search');
        if ((inList || inSearch) && (event.key === 'ArrowDown' || event.key === 'ArrowUp') && results.length) {
            const index = Math.max(0, results.findIndex(skill => skill.id === libraryState.selected));
            const next = (index + (event.key === 'ArrowDown' ? 1 : -1) + results.length) % results.length;
            libraryState.selected = results[next].id;
            renderLibrary();
            event.preventDefault();
            return;
        }
        if ((inList || inSearch) && event.key === 'Enter' && libraryState.selected) {
            event.preventDefault();
            useSkill(libraryState.selected);
            return;
        }
        if (event.key === 'Tab') {
            const focusables = Array.from(libraryEl.querySelectorAll('button, input, [tabindex="0"]'))
                .filter(element => !element.disabled && element.offsetParent !== null);
            if (!focusables.length) return;
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            const active = root.document.activeElement;
            if (event.shiftKey && active === first) { last.focus(); event.preventDefault(); }
            else if (!event.shiftKey && active === last) { first.focus(); event.preventDefault(); }
        }
    }

    function openLibrary(opts = {}) {
        libraryState.opts = opts;
        libraryState.open = true;
        libraryReturnFocus = opts.returnFocus || root.document.activeElement;
        let media = null;
        try { media = opts.getMediaType ? opts.getMediaType() : null; } catch { media = null; }
        libraryState.media = MEDIA_LABELS[media] ? media : 'all';
        libraryState.tab = opts.tab || 'all';
        if (!libraryEl) libraryEl = buildLibrary();
        libraryEl.hidden = false;
        if (root.document.body && root.document.body.classList) root.document.body.classList.add('agent-skills-open');
        renderLibrary();
        const search = libraryEl.querySelector('[data-skills-search]');
        if (search) { search.value = libraryState.query; search.focus(); }
        return loadCatalog(true).then(() => { renderLibrary(); return libraryResults(); });
    }

    function closeLibrary() {
        libraryState.open = false;
        if (libraryEl) libraryEl.hidden = true;
        if (root.document.body && root.document.body.classList) root.document.body.classList.remove('agent-skills-open');
        if (libraryReturnFocus && libraryReturnFocus.focus) libraryReturnFocus.focus();
    }

    function useSkill(skillId) {
        const onUse = libraryState.opts.onUse;
        closeLibrary();
        if (onUse) onUse(skillId);
    }

    function deleteSkill(skillId) {
        const doc = root.document;
        const detail = libraryEl.querySelector('[data-skills-region="detail"]');
        detail.insertAdjacentHTML('beforeend', `
            <div class="agent-skills-confirm" role="alertdialog" aria-label="Delete @${escapeHtml(skillId)}">
                <span>Delete @${escapeHtml(skillId)}? This cannot be undone.</span>
                <button type="button" class="agent-skills-secondary" data-confirm="no">Cancel</button>
                <button type="button" class="agent-skills-danger" data-confirm="yes">Delete</button>
            </div>`);
        const box = detail.querySelector('.agent-skills-confirm');
        box.querySelector('[data-confirm="no"]').focus();
        box.addEventListener('click', event => {
            event.stopPropagation();
            const choice = event.target.closest && event.target.closest('[data-confirm]');
            if (!choice) return;
            if (choice.getAttribute('data-confirm') === 'yes') {
                root.fetch(`/api/agent/skills/${encodeURIComponent(skillId)}`, { method: 'DELETE' })
                    .then(() => loadCatalog(true)).then(renderLibrary);
            } else {
                box.remove();
            }
        });
        void doc;
    }

    function renameSkill(skillId) {
        const detail = libraryEl.querySelector('[data-skills-region="detail"]');
        detail.insertAdjacentHTML('beforeend', `
            <form class="agent-skills-rename">
                <label for="agentSkillsRename">New name</label>
                <input id="agentSkillsRename" value="${escapeHtml(skillId)}" pattern="[a-z][a-z0-9-]{1,39}" required>
                <button type="submit" class="agent-skills-use">Save</button>
                <p class="agent-skills-error" role="alert"></p>
            </form>`);
        const form = detail.querySelector('.agent-skills-rename');
        const input = form.querySelector('input');
        input.focus();
        form.addEventListener('submit', event => {
            event.preventDefault();
            root.fetch(`/api/agent/skills/${encodeURIComponent(skillId)}/rename`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_id: input.value.trim().toLowerCase() })
            }).then(response => response.json()).then(data => {
                if (data.error) { form.querySelector('.agent-skills-error').textContent = data.error; return; }
                libraryState.selected = data.skill && data.skill.id;
                return loadCatalog(true).then(renderLibrary);
            });
        });
    }

    /** Insert "@id " into a composer textarea at the caret (used by the library "Use" button). */
    function insertSkill(textarea, skillId) {
        if (!textarea) return;
        const caret = textarea.selectionStart ?? String(textarea.value || '').length;
        const trigger = findTrigger(textarea.value, caret);
        const result = applyCompletion(textarea.value, trigger, skillId);
        textarea.value = result.text;
        if (textarea.setSelectionRange) textarea.setSelectionRange(result.caret, result.caret);
        if (textarea.focus) textarea.focus();
    }

    root.AgentSkills = {
        loadCatalog, setCatalog, filterSkills, findTrigger, applyCompletion, paramsSummary, insertSkill,
        attachAutocomplete, openLibrary, closeLibrary, libraryResults,
        _libraryState: libraryState
    };
})(typeof window !== 'undefined' ? window : globalThis);
