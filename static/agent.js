/**
 * Universal AI Agent Studio — Client Controller
 * Autonomous Multi-Agent Media Orchestrator (Vision, Video, Audio, Inspector)
 */

(function () {
    'use strict';

    // ─── STATE MANAGEMENT ───
    let currentSessionId = null;
    let sessions = []; // [{ id, title, createdAt, messages: [], activeMedia: null }]
    let activeMedia = null; // { filename, original_name, type, size, url }
    let isProcessing = false;
    const pendingUploads = new WeakMap();

    // ─── DOM ELEMENTS ───
    const sidebar = document.getElementById('agentSidebar');
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const sessionsList = document.getElementById('sessionsList');
    const activeMediaBox = document.getElementById('activeMediaBox');
    const activeMediaName = document.getElementById('activeMediaName');
    const activeMediaMeta = document.getElementById('activeMediaMeta');
    const removeActiveMediaBtn = document.getElementById('removeActiveMediaBtn');
    const messagesStream = document.getElementById('messagesStream');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const attachBtn = document.getElementById('attachBtn');
    const filePicker = document.getElementById('filePicker');
    const micBtn = document.getElementById('micBtn');
    const pendingAttachmentCard = document.getElementById('pendingAttachmentCard');
    const pendingFileName = document.getElementById('pendingFileName');
    const pendingFileSize = document.getElementById('pendingFileSize');
    const removePendingBtn = document.getElementById('removePendingBtn');
    const dragOverlay = document.getElementById('dragOverlay');
    const sessionHeaderTitle = document.getElementById('sessionHeaderTitle');
    const clearChatBtn = document.getElementById('clearChatBtn');

    // ─── INITIALIZATION ───
    function init() {
        loadSessionsFromStorage();
        setupEventListeners();
        setupVoiceRecognition();
        setupDragAndDrop();

        if (sessions.length === 0) {
            createNewSession();
        } else {
            switchSession(sessions[0].id);
        }

        // Check if user dropped a file on the landing page
        try {
            const pending = localStorage.getItem('avi_agent_pending_upload');
            if (pending) {
                const mediaObj = JSON.parse(pending);
                localStorage.removeItem('avi_agent_pending_upload');
                setActiveMedia(mediaObj);
                appendSystemNotice(`Imported <strong>${escapeHtml(mediaObj.original_name)}</strong> from the home page. Tell me what to change.`);
            }
        } catch (e) {
            console.error('Pending upload error:', e);
        }
    }

    // ─── SESSION STORAGE ───
    function loadSessionsFromStorage() {
        try {
            const raw = localStorage.getItem('avi_agent_sessions');
            if (raw) {
                const parsed = JSON.parse(raw);
                // Tolerate sessions saved by older or malformed responses instead of breaking the workspace.
                sessions = (Array.isArray(parsed) ? parsed : [])
                    .filter(session => session && typeof session === 'object' && typeof session.id === 'string')
                    .map(session => ({
                        ...session,
                        title: typeof session.title === 'string' ? session.title : 'Conversation',
                        messages: (Array.isArray(session.messages) ? session.messages : [])
                            .filter(message => message && typeof message === 'object')
                    }));
            }
        } catch (e) {
            console.error('Failed to load sessions:', e);
            sessions = [];
        }
    }

    function saveSessionsToStorage() {
        try {
            localStorage.setItem('avi_agent_sessions', JSON.stringify(sessions));
        } catch (e) {
            console.error('Failed to save sessions:', e);
        }
    }

    function getCurrentSession() {
        return sessions.find(s => s.id === currentSessionId);
    }

    function createNewSession() {
        const id = 'sess_' + Date.now();
        const newSession = {
            id: id,
            title: 'New Conversation',
            createdAt: new Date().toISOString(),
            messages: [],
            activeMedia: null
        };
        sessions.unshift(newSession);
        saveSessionsToStorage();
        renderSessionsList();
        switchSession(id);
    }

    function switchSession(id) {
        currentSessionId = id;
        const session = getCurrentSession();
        if (!session) return;

        sessionHeaderTitle.textContent = session.title || 'Conversation';
        activeMedia = session.activeMedia || null;
        updateActiveMediaDisplay();
        updateSubAgentChipStatus(activeMedia ? activeMedia.type : null);
        renderMessages();
        renderSessionsList();
        chatInput.focus();
    }

    function deleteSession(id, e) {
        if (e) e.stopPropagation();
        sessions = sessions.filter(s => s.id !== id);
        saveSessionsToStorage();
        if (sessions.length === 0) {
            createNewSession();
        } else if (currentSessionId === id) {
            switchSession(sessions[0].id);
        } else {
            renderSessionsList();
        }
    }

    function setAttr(element, name, value) {
        if (element && typeof element.setAttribute === 'function') element.setAttribute(name, value);
    }

    function renderSessionsList() {
        sessionsList.innerHTML = '';
        sessions.forEach(sess => {
            const item = document.createElement('div');
            item.className = `session-item ${sess.id === currentSessionId ? 'active' : ''}`;
            item.onclick = () => {
                switchSession(sess.id);
                closeMobileSidebar();
            };

            // The title is a real button (keyboard + screen readers); its click bubbles to the row handler.
            const titleSpan = document.createElement('button');
            titleSpan.type = 'button';
            titleSpan.className = 'session-title session-open-btn';
            if (sess.id === currentSessionId) setAttr(titleSpan, 'aria-current', 'true');
            titleSpan.innerHTML = `<i class="fas fa-message" aria-hidden="true"></i> ${escapeHtml(sess.title || 'Conversation')}`;

            const delBtn = document.createElement('button');
            delBtn.type = 'button';
            delBtn.className = 'session-del-btn';
            delBtn.title = 'Delete chat';
            setAttr(delBtn, 'aria-label', `Delete chat: ${sess.title || 'Conversation'}`);
            delBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
            delBtn.onclick = (e) => deleteSession(sess.id, e);

            item.appendChild(titleSpan);
            item.appendChild(delBtn);
            sessionsList.appendChild(item);
        });
    }

    // ─── ACTIVE MEDIA STATE ───
    function setActiveMedia(mediaObj, session = getCurrentSession()) {
        if (!session || !sessions.includes(session)) return;
        pendingUploads.delete(session);
        session.activeMedia = mediaObj;
        saveSessionsToStorage();
        if (currentSessionId === session.id) {
            activeMedia = mediaObj;
            updateActiveMediaDisplay();
            updateSubAgentChipStatus(mediaObj ? mediaObj.type : null);
        }
    }

    function updateActiveMediaDisplay() {
        if (activeMedia) {
            activeMediaBox.style.display = 'flex';
            activeMediaName.textContent = activeMedia.original_name || activeMedia.filename;
            const sizeStr = activeMedia.size ? formatBytes(activeMedia.size) : '';
            const typeStr = activeMedia.type ? activeMedia.type.toUpperCase() : 'FILE';
            activeMediaMeta.textContent = `${typeStr} • ${sizeStr}`;

            pendingAttachmentCard.style.display = 'flex';
            pendingFileName.textContent = activeMedia.original_name || activeMedia.filename;
            pendingFileSize.textContent = `${typeStr} • ${sizeStr}`;
        } else {
            activeMediaBox.style.display = 'none';
            pendingAttachmentCard.style.display = 'none';
        }
    }

    function updateSubAgentChipStatus(type) {
        document.querySelectorAll('.subagent-chip').forEach(c => c.classList.remove('active'));
        if (type === 'image') {
            const chip = document.getElementById('chipVision');
            if (chip) chip.classList.add('active');
        } else if (type === 'video') {
            const chip = document.getElementById('chipVideo');
            if (chip) chip.classList.add('active');
        } else if (type === 'audio') {
            const chip = document.getElementById('chipAudio');
            if (chip) chip.classList.add('active');
        }
    }

    // ─── FILE UPLOAD PIPELINE ───
    async function uploadMediaFile(file) {
        if (!file) return;
        const session = getCurrentSession();
        if (!session) return;
        const uploadGeneration = {};
        pendingUploads.set(session, uploadGeneration);

        // Show uploading feedback
        pendingAttachmentCard.style.display = 'flex';
        pendingFileName.textContent = `Uploading ${file.name}...`;
        pendingFileSize.textContent = formatBytes(file.size);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/agent/upload', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.error || `Upload failed with status ${resp.status}`);
            }

            const data = await resp.json();
            if (!sessions.includes(session) || pendingUploads.get(session) !== uploadGeneration) return;
            setActiveMedia(data, session);
            if (currentSessionId !== session.id) return;
            showToast(`Uploaded ${data.original_name}`, 'success');

            // Proactively notify in chat stream
            appendSystemNotice(`Uploaded <strong>${escapeHtml(data.original_name)}</strong> (${data.type.toUpperCase()}). Tell me how you want it edited.`);

        } catch (err) {
            if (!sessions.includes(session) || pendingUploads.get(session) !== uploadGeneration) return;
            pendingUploads.delete(session);
            console.error('File upload error:', err);
            if (currentSessionId !== session.id) return;
            updateActiveMediaDisplay();
            showToast(`File upload failed: ${err.message}`, 'error');
        }
    }

    // ─── CHAT SUBMISSION ───
    async function handleSendMessage(predefinedText = null) {
        const text = String(predefinedText || chatInput.value || '').trim();
        if (!text) return;
        if (isProcessing) return;

        const session = getCurrentSession();
        if (!session) return;

        // Auto-title session from first user message
        if (session.messages.length === 0 && text) {
            session.title = text.length > 28 ? text.substring(0, 28) + '...' : text;
            sessionHeaderTitle.textContent = session.title;
            renderSessionsList();
        }

        // 1. Add User Message
        const userMsg = {
            role: 'user',
            content: text,
            attachedMedia: activeMedia ? { ...activeMedia } : null,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        const submittedMedia = session.activeMedia;
        session.messages.push(userMsg);
        saveSessionsToStorage();
        renderMessages();

        // Clear input field & reset height
        if (!predefinedText) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
        }

        // 2. Render Agent Thinking Placeholder
        isProcessing = true;
        sendBtn.disabled = true;
        const thinkingId = 'thinking_' + Date.now();
        renderThinkingBubble(thinkingId);

        try {
            // Prepare conversation context for backend
            const historyPayload = session.messages.slice(0, -1).slice(-8).map(message => ({
                role: message.role === 'user' ? 'user' : 'assistant',
                content: message.role === 'user' ? (message.content || '') : (message.reply || message.content || '')
            }));

            const payload = {
                message: text,
                filename: activeMedia ? activeMedia.filename : '',
                context: activeMedia ? {
                    filename: activeMedia.filename,
                    name: activeMedia.original_name,
                    type: activeMedia.type,
                    size: activeMedia.size
                } : null,
                history: historyPayload,
                session_id: session.id
            };

            const resp = await fetch('/api/agent/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            removeThinkingBubble(thinkingId);

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.error || `Server responded with error ${resp.status}`);
            }

            const rawData = await resp.json();
            const data = rawData && typeof rawData === 'object' ? rawData : {};
            if (!sessions.includes(session)) return;
            // Remember what this edit started from so Split Compare shows before/after, not after/after.
            const inputUrl = submittedMedia && typeof submittedMedia.url === 'string' ? submittedMedia.url : null;

            // If new output file was created, update active media to the newly processed result!
            if (typeof data.output_file === 'string' && typeof data.output_url === 'string' && data.output_file && data.output_url
                && session.activeMedia === submittedMedia && !pendingUploads.has(session)) {
                const outName = data.output_file.split(/[\/\\]/).pop();
                const outExt = outName.split('.').pop().toLowerCase();
                let newType = 'other';
                if (['mp4', 'mov', 'webm', 'mkv', 'gif'].includes(outExt)) newType = 'video';
                else if (['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'opus', 'wma', 'aiff'].includes(outExt)) newType = 'audio';
                else if (['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'].includes(outExt)) newType = 'image';

                setActiveMedia({
                    filename: outName,
                    original_name: outName,
                    type: newType,
                    url: data.output_url
                }, session);
            }

            // 3. Add Agent Message
            const agentMsg = {
                role: 'agent',
                reply: asText(data.reply),
                thought: asText(data.thought),
                delegated_subagent: asText(data.delegated_subagent) || 'Orchestrator',
                clarification_needed: data.clarification_needed === true,
                clarification_options: asTextList(data.clarification_options),
                tools_planned: Array.isArray(data.tools_planned) ? data.tools_planned : [],
                execution_results: Array.isArray(data.execution_results) ? data.execution_results : [],
                output_url: typeof data.output_url === 'string' ? data.output_url : null,
                output_file: typeof data.output_file === 'string' ? data.output_file : null,
                input_url: inputUrl,
                artifacts: Array.isArray(data.artifacts) ? data.artifacts.filter(item => item && typeof item.output_url === 'string') : [],
                suggested_actions: asTextList(data.suggested_actions),
                skills_used: Array.isArray(data.skills_used) ? data.skills_used : [],
                auto_skill: typeof data.auto_skill === 'string' ? data.auto_skill : null,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };

            session.messages.push(agentMsg);
            saveSessionsToStorage();
            if (currentSessionId === session.id) renderMessages();

        } catch (err) {
            console.error('Chat processing error:', err);
            removeThinkingBubble(thinkingId);
            if (!sessions.includes(session)) return;
            const errorMsg = {
                role: 'agent',
                reply: `The request could not be completed (${String(err && err.message || 'unknown error').replace(/[.\s]+$/, '')}). Check that the file is loaded and try again.`,
                is_error: true,
                delegated_subagent: 'Orchestrator',
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };
            session.messages.push(errorMsg);
            saveSessionsToStorage();
            if (currentSessionId === session.id) renderMessages();
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    // ─── RENDERING MESSAGES ───
    function renderMessages() {
        const session = getCurrentSession();
        if (!session || session.messages.length === 0) {
            renderWelcomeHero();
            return;
        }

        messagesStream.innerHTML = '';
        session.messages.forEach(msg => {
            try {
                if (msg.role === 'user') {
                    renderUserMessage(msg);
                } else {
                    renderAgentMessage(msg);
                }
            } catch (err) {
                // One malformed stored message must never make the whole conversation unusable.
                console.error('Skipped an unreadable message:', err);
            }
        });

        // Scroll to bottom smoothly
        messagesStream.scrollTop = messagesStream.scrollHeight;
    }

    function renderWelcomeHero() {
        messagesStream.innerHTML = `
            <div class="welcome-hero">
                <div class="welcome-avatar">
                    <i class="fas fa-brain"></i>
                </div>
                <h1 class="welcome-title">What would you like to edit?</h1>
                <p class="welcome-subtitle">
                    Add a video, audio clip or image, then describe the edit in plain language.
                    You can chain several edits in one message, or type <kbd>@</kbd> to use a skill.
                </p>
                <div class="starter-cards-grid">
                    <button type="button" class="starter-card" data-prompt="Trim video from 2.3 second to 5.3 second">
                        <span class="starter-card-icon tone-indigo" aria-hidden="true"><i class="fas fa-scissors"></i></span>
                        <span class="starter-card-title">Trim a video</span>
                        <span class="starter-card-desc">Keep an exact range, to the millisecond.</span>
                    </button>
                    <button type="button" class="starter-card" data-prompt="Remove background and isolate subject">
                        <span class="starter-card-icon tone-rose" aria-hidden="true"><i class="fas fa-wand-magic-sparkles"></i></span>
                        <span class="starter-card-title">Remove a background</span>
                        <span class="starter-card-desc">Cut out the subject with clean edges.</span>
                    </button>
                    <button type="button" class="starter-card" data-prompt="Transcribe speech to text and clean background noise">
                        <span class="starter-card-icon tone-sky" aria-hidden="true"><i class="fas fa-microphone-lines"></i></span>
                        <span class="starter-card-title">Transcribe speech</span>
                        <span class="starter-card-desc">Text plus SRT and VTT subtitles.</span>
                    </button>
                    <button type="button" class="starter-card" data-prompt="Upscale image to 4x high resolution">
                        <span class="starter-card-icon tone-green" aria-hidden="true"><i class="fas fa-expand-alt"></i></span>
                        <span class="starter-card-title">Upscale an image</span>
                        <span class="starter-card-desc">Enlarge 2x or 4x with restored detail.</span>
                    </button>
                </div>
            </div>
        `;

        document.querySelectorAll('.starter-card').forEach(card => {
            card.onclick = () => {
                const prompt = card.getAttribute('data-prompt');
                handleSendMessage(prompt);
            };
        });
    }

    function renderUserMessage(msg) {
        const row = document.createElement('div');
        row.className = 'message-row user';

        let fileChipHtml = '';
        if (msg.attachedMedia) {
            const icon = msg.attachedMedia.type === 'video' ? 'fa-film' : (msg.attachedMedia.type === 'image' ? 'fa-image' : 'fa-music');
            fileChipHtml = `
                <div class="user-attached-file">
                    <i class="fas ${icon}"></i>
                    <span>${escapeHtml(msg.attachedMedia.original_name || msg.attachedMedia.filename)}</span>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="message-bubble">
                ${fileChipHtml}
                <div class="user-text">${escapeHtml(msg.content)}</div>
            </div>
            <div class="message-avatar" title="You">
                <i class="fas fa-user"></i>
            </div>
        `;
        messagesStream.appendChild(row);
    }

    function assistantName() {
        const brand = window.APP_BRAND;
        return brand && typeof brand.assistant === 'string' && brand.assistant ? brand.assistant : 'Assistant';
    }

    function asText(value) {
        if (typeof value === 'string') return value;
        if (typeof value === 'number' || typeof value === 'boolean') return String(value);
        return '';
    }

    function asTextList(value) {
        const items = Array.isArray(value) ? value : (typeof value === 'string' ? [value] : []);
        return items.map(item => asText(item).trim()).filter(Boolean);
    }

    function safeMediaUrl(value) {
        return typeof value === 'string' && /^\/(?:processed|media)\/[^\s"'<>`\\]+$/.test(value) ? value : null;
    }

    const KNOWN_SUBAGENTS = ['VisionSubAgent', 'VideoSubAgent', 'AudioSubAgent', 'InspectorSubAgent', 'Orchestrator'];
    const PUBLIC_AGENT_LABELS = { VisionSubAgent: 'Image', VideoSubAgent: 'Video', AudioSubAgent: 'Audio',
                                  InspectorSubAgent: 'Inspector', Orchestrator: assistantName() };

    function normalizeAgentMessage(msg) {
        const subagent = asText(msg.delegated_subagent);
        return {
            ...msg,
            reply: asText(msg.reply),
            thought: asText(msg.thought),
            delegated_subagent: KNOWN_SUBAGENTS.includes(subagent) ? subagent : 'Orchestrator',
            clarification_needed: msg.clarification_needed === true,
            clarification_options: asTextList(msg.clarification_options),
            suggested_actions: asTextList(msg.suggested_actions),
            execution_results: (Array.isArray(msg.execution_results) ? msg.execution_results : [])
                .filter(step => step && typeof step === 'object'),
            output_url: safeMediaUrl(msg.output_url),
            input_url: safeMediaUrl(msg.input_url),
            artifacts: (Array.isArray(msg.artifacts) ? msg.artifacts : [])
                .filter(item => item && safeMediaUrl(item.output_url)),
            skills_used: (Array.isArray(msg.skills_used) ? msg.skills_used : [])
                .filter(item => item && typeof item.id === 'string')
                .map(item => ({ id: item.id, title: asText(item.title) || item.id })),
            auto_skill: typeof msg.auto_skill === 'string' ? msg.auto_skill : null
        };
    }

    function renderAgentMessage(rawMsg) {
        const msg = normalizeAgentMessage(rawMsg && typeof rawMsg === 'object' ? rawMsg : {});
        const row = document.createElement('div');
        row.className = 'message-row agent';
        let transcriptText = '';

        const subagent = msg.delegated_subagent || 'Orchestrator';
        const subagentLabel = PUBLIC_AGENT_LABELS[subagent] || assistantName();
        const subagentIcon = subagent === 'VisionSubAgent' ? 'fa-wand-magic-sparkles' :
            (subagent === 'VideoSubAgent' ? 'fa-film' :
                (subagent === 'AudioSubAgent' ? 'fa-wave-square' : 'fa-robot'));
        if (msg.is_error) row.classList.add('is-error');

        // Chain of Thought Accordion
        const skillBadges = msg.skills_used.map(skill => `
            <span class="agent-skill-badge"><i class="fas fa-bolt" aria-hidden="true"></i>
                ${msg.auto_skill === skill.id ? 'Using skill (picked automatically)' : 'Using skill'}: ${escapeHtml(skill.title)}</span>`).join(' ');
        let cotHtml = '';
        if (msg.thought) {
            cotHtml = `
                <details class="cot-accordion">
                    <summary class="cot-summary">
                        <i class="fas fa-chevron-right"></i>
                        <span>Thought Process &amp; Multi-Agent Reasoning</span>
                    </summary>
                    <div class="cot-content">${escapeHtml(msg.thought)}</div>
                </details>
            `;
        }

        // Interactive Media Preview Card
        let mediaHtml = '';
        if (msg.output_url) {
            const url = msg.output_url;
            const ext = url.split('.').pop().toLowerCase();
            let previewEl = '';

            if (['mp4', 'webm', 'mov', 'mkv'].includes(ext)) {
                previewEl = `<video controls playsinline preload="metadata" src="${url}"></video>`;
            } else if (ext === 'gif' || ['png', 'jpg', 'jpeg', 'webp'].includes(ext)) {
                previewEl = `<img src="${url}" alt="Processed Output" loading="lazy" />`;
            } else if (['mp3', 'wav', 'm4a', 'ogg'].includes(ext)) {
                previewEl = `<audio controls src="${url}"></audio>`;
            }

            // Check for speech transcription data
            let transcriptHtml = '';
            const sttStep = (msg.execution_results || []).find(s => s.tool === 'transcribe_audio' && s.data);
            if (sttStep && sttStep.data) {
                const segments = (Array.isArray(sttStep.data.segments) ? sttStep.data.segments : [])
                    .filter(segment => segment && typeof segment === 'object');
                const fullText = asText(sttStep.data.text) || segments.map(segment => asText(segment.text)).join(' ');
                transcriptText = fullText;
                const exports = (Array.isArray(sttStep.data.exports) ? sttStep.data.exports : [])
                    .filter(item => item && /^\/processed\/[^/?#]+\.(txt|srt|vtt)$/.test(item.url));
                const exportLinks = exports.map(item => `<a class="media-action-secondary" href="${escapeHtml(item.url)}" download>${escapeHtml(item.format)}</a>`).join('');
                const segRows = segments.map(s => `
                    <div class="transcript-segment">
                        <span class="timestamp-badge">${Number.isFinite(Number(s.start)) ? Number(s.start).toFixed(1) : '0.0'}s</span>
                        <span>${escapeHtml(asText(s.text))}</span>
                    </div>
                `).join('');

                transcriptHtml = `
                    <div class="transcript-card">
                        <div class="transcript-header">
                            <span><i class="fas fa-file-lines tone-indigo" aria-hidden="true"></i> Speech Transcript (${segments.length} segments)</span>
                            <button class="media-action-secondary transcript-copy-btn" type="button">
                                <i class="fas fa-copy"></i> Copy
                            </button>
                            ${exportLinks}
                        </div>
                        <div class="transcript-body">${segRows || escapeHtml(fullText)}</div>
                    </div>
                `;
            }

            const imageExtensions = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
            const isImageOutput = imageExtensions.includes(ext);
            // Compare against the media this edit started from (stored with the message), never the current selection.
            const inputUrl = msg.input_url;
            const canCompare = isImageOutput && inputUrl && inputUrl !== url
                && imageExtensions.includes(inputUrl.split('.').pop().toLowerCase());
            const compareBtn = canCompare ? `
                <button class="media-action-secondary" onclick="window.agentToggleCompare(this, '${escapeHtml(inputUrl)}', '${escapeHtml(url)}')">
                    <i class="fas fa-columns"></i> Split Compare
                </button>
            ` : '';
            const artifactLinks = msg.artifacts.map(item => `
                <a href="${escapeHtml(item.output_url)}" download class="media-action-secondary">
                    <i class="fas fa-image"></i> ${item.tool === 'extract_frame' ? 'Download thumbnail' : 'Download extra file'}
                </a>`).join('');

            mediaHtml = `
                <div class="media-output-card">
                    <div class="media-output-preview" id="previewBox_${Date.now()}">
                        ${previewEl}
                    </div>
                    <div class="media-output-actions">
                        <a href="${url}" download class="btn-download-media">
                            <i class="fas fa-download"></i> Download Export
                        </a>
                        <div class="media-actions-group">
                            ${compareBtn}
                            ${artifactLinks}
                            <button class="media-action-secondary" onclick="window.open('${escapeHtml(url)}', '_blank')">
                                <i class="fas fa-external-link-alt"></i> Pop Out
                            </button>
                        </div>
                    </div>
                    ${transcriptHtml}
                </div>
            `;
        }

        // Clarification Protocol Interactive Buttons
        let clarifHtml = '';
        if (msg.clarification_needed && msg.clarification_options && msg.clarification_options.length > 0) {
            const btns = msg.clarification_options.map(opt =>
                `<button class="clarification-btn" data-agent-prompt="${escapeHtml(opt)}">${escapeHtml(opt)}</button>`
            ).join('');

            clarifHtml = `
                <div class="clarification-box">
                    <div class="clarification-title">
                        <i class="fas fa-circle-question"></i> Clarification Needed
                    </div>
                    <div class="clarification-options-grid">${btns}</div>
                </div>
            `;
        }

        // Proactive Next-Step Suggestion Pills
        let suggestionsHtml = '';
        if (msg.suggested_actions && msg.suggested_actions.length > 0) {
            const pills = msg.suggested_actions.map(action =>
                `<button class="suggested-action-pill" data-agent-prompt="${escapeHtml(action)}">
                    <i class="fas fa-wand-magic-sparkles" aria-hidden="true"></i> ${escapeHtml(action)}
                </button>`
            ).join('');

            suggestionsHtml = `<div class="suggested-actions-tray">${pills}</div>`;
        }

        row.innerHTML = `
            <div class="message-avatar" title="${subagentLabel}" aria-hidden="true">
                <i class="fas ${subagentIcon}"></i>
            </div>
            <div class="message-bubble">
                <div class="agent-pill ${subagent}">
                    <i class="fas ${subagentIcon}" aria-hidden="true"></i> ${subagentLabel}
                </div>
                ${skillBadges}
                ${cotHtml}
                <div class="agent-text">${formatMarkdown(msg.reply || '')}</div>
                ${mediaHtml}
                ${clarifHtml}
                ${suggestionsHtml}
            </div>
        `;
        messagesStream.appendChild(row);
        row.querySelector('.transcript-copy-btn')?.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(transcriptText);
                showToast('Transcript copied to the clipboard.', 'success');
            } catch {
                showToast('Clipboard is unavailable. Download the TXT transcript instead.', 'error');
            }
        });
        row.querySelectorAll('[data-agent-prompt]').forEach(button => {
            button.addEventListener('click', () => handleSendMessage(button.dataset.agentPrompt));
        });
    }

    const activeThinkingTimers = {};

    const PRO_TIPS_COLLECTION = [
        "Tip: chain edits in one message, for example 'Trim from 1.5s to 4.2s and speed up 1.5x'.",
        "Press Enter to send, or Shift+Enter for a new line.",
        "Your media is processed on this computer.",
        "Try 'Remove the background and upscale 2x'.",
        "Say 'trim the silence' to remove pauses at the start and end.",
        "Finished video edits can be opened in the Studio.",
        "Say 'isolate the vocals' or 'make an instrumental'.",
        "Type @ to pick a ready-made skill, such as @podcast-polish."
    ];

    function renderThinkingBubble(id) {
        const row = document.createElement('div');
        row.className = 'message-row agent';
        row.id = id;
        
        let elapsed = 1;
        let tipIndex = Math.floor(Math.random() * PRO_TIPS_COLLECTION.length);

        row.innerHTML = `
            <div class="message-avatar thinking-avatar" aria-hidden="true">
                <i class="fas fa-wand-magic-sparkles fa-pulse"></i>
            </div>
            <div class="thinking-live-card">
                <div class="thinking-header-row">
                    <div class="thinking-status-wrapper">
                        <div class="thinking-avatar-orb">
                            <i class="fas fa-brain"></i>
                        </div>
                        <span class="thinking-main-title" id="${id}_title">Processing Media Request...</span>
                    </div>
                    <span class="thinking-elapsed-badge">
                        <i class="far fa-clock"></i> <span id="${id}_timer">00:01</span>
                    </span>
                </div>

                <div class="thinking-wave-visualizer">
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                </div>

                <div class="thinking-sub-status" id="${id}_substatus">
                    Ingesting media stream &amp; formulating editing plan...
                </div>

                <div class="thinking-steps-row">
                    <span class="thinking-step-pill active" id="${id}_step1"><i class="fas fa-circle-dot"></i> Ingestion</span>
                    <span class="thinking-step-pill" id="${id}_step2"><i class="fas fa-wand-magic-sparkles"></i> Processing</span>
                    <span class="thinking-step-pill" id="${id}_step3"><i class="fas fa-check-double"></i> Rendering</span>
                </div>

                <div class="thinking-pro-tip-box">
                    <i class="fas fa-lightbulb"></i>
                    <span class="thinking-tip-text" id="${id}_tip">${escapeHtml(PRO_TIPS_COLLECTION[tipIndex])}</span>
                </div>
            </div>
        `;
        messagesStream.appendChild(row);
        messagesStream.scrollTop = messagesStream.scrollHeight;

        // Start Live Timer & Progression Updates
        const timerInterval = setInterval(() => {
            elapsed++;
            const timerEl = document.getElementById(`${id}_timer`);
            const titleEl = document.getElementById(`${id}_title`);
            const substatusEl = document.getElementById(`${id}_substatus`);
            const step1 = document.getElementById(`${id}_step1`);
            const step2 = document.getElementById(`${id}_step2`);
            const step3 = document.getElementById(`${id}_step3`);

            if (timerEl) {
                const m = Math.floor(elapsed / 60).toString().padStart(2, '0');
                const s = (elapsed % 60).toString().padStart(2, '0');
                timerEl.textContent = `${m}:${s}`;
            }

            if (elapsed >= 3 && elapsed < 7) {
                if (titleEl) titleEl.textContent = 'Applying AI Tool Operations...';
                if (substatusEl) substatusEl.textContent = 'Analyzing audio-visual frequencies & executing transforms...';
                if (step1) { step1.className = 'thinking-step-pill done'; step1.innerHTML = '<i class="fas fa-check"></i> Ingestion'; }
                if (step2) { step2.className = 'thinking-step-pill active'; }
            } else if (elapsed >= 7 && elapsed < 12) {
                if (titleEl) titleEl.textContent = 'Rendering Signal Output...';
                if (substatusEl) substatusEl.textContent = 'Encoding media stream with sub-second precision...';
                if (step2) { step2.className = 'thinking-step-pill done'; step2.innerHTML = '<i class="fas fa-check"></i> Processing'; }
                if (step3) { step3.className = 'thinking-step-pill active'; }
            } else if (elapsed >= 12) {
                if (titleEl) titleEl.textContent = 'Finalizing High-Quality Preview...';
                if (substatusEl) substatusEl.textContent = 'Generating media playback elements & packaging reply...';
                if (step3) { step3.className = 'thinking-step-pill done'; step3.innerHTML = '<i class="fas fa-check"></i> Rendering'; }
            }
        }, 1000);

        // Rotate Pro-Tips every 4.2s
        const tipInterval = setInterval(() => {
            tipIndex = (tipIndex + 1) % PRO_TIPS_COLLECTION.length;
            const tipEl = document.getElementById(`${id}_tip`);
            if (tipEl) {
                tipEl.style.opacity = '0';
                setTimeout(() => {
                    tipEl.textContent = PRO_TIPS_COLLECTION[tipIndex];
                    tipEl.style.opacity = '1';
                }, 200);
            }
        }, 4200);

        activeThinkingTimers[id] = { timerInterval, tipInterval };
    }

    function removeThinkingBubble(id) {
        if (activeThinkingTimers[id]) {
            clearInterval(activeThinkingTimers[id].timerInterval);
            clearInterval(activeThinkingTimers[id].tipInterval);
            delete activeThinkingTimers[id];
        }
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendSystemNotice(html) {
        const div = document.createElement('div');
        div.style.textAlign = 'center';
        div.style.fontSize = '0.8rem';
        div.style.color = 'var(--text-hint)';
        div.style.margin = '8px 0';
        div.innerHTML = `<span class="system-notice-chip">${html}</span>`;
        messagesStream.appendChild(div);
        messagesStream.scrollTop = messagesStream.scrollHeight;
    }

    // ─── GLOBAL PROMPT SENDER HELPER ───
    window.agentSendPrompt = function (text) {
        handleSendMessage(text);
    };

    // ─── INTERACTIVE BEFORE / AFTER SLIDER ───
    let activeCompareDrag = null;
    const endCompareDrag = () => { activeCompareDrag = null; };
    const moveCompareDrag = event => { if (activeCompareDrag) activeCompareDrag(event.clientX); };
    window.addEventListener('mouseup', endCompareDrag);
    window.addEventListener('pointerup', endCompareDrag);
    window.addEventListener('mousemove', moveCompareDrag);
    window.addEventListener('pointermove', moveCompareDrag);

    window.agentToggleCompare = function (btn, origUrl, processedUrl) {
        if (!origUrl || !processedUrl) return;
        const card = btn.closest('.media-output-card');
        const preview = card ? card.querySelector('.media-output-preview') : null;
        if (!preview) return;

        if (preview.classList.contains('in-compare-mode')) {
            preview.classList.remove('in-compare-mode');
            preview.innerHTML = `<img src="${processedUrl}" alt="Processed Output" loading="lazy" />`;
            btn.innerHTML = `<i class="fas fa-columns"></i> Split Compare`;
        } else {
            preview.classList.add('in-compare-mode');
            preview.innerHTML = `
                <div class="image-compare-container">
                    <div class="image-compare-wrapper">
                        <img src="${processedUrl}" alt="Processed Result" />
                        <div class="image-compare-before" style="width: 50%;">
                            <img src="${origUrl}" alt="Original Input" />
                        </div>
                        <div class="image-compare-slider" style="left: 50%;">
                            <div class="compare-handle-circle"><i class="fas fa-arrows-left-right"></i></div>
                        </div>
                        <span class="compare-label left">Original</span>
                        <span class="compare-label right">Processed</span>
                    </div>
                </div>
            `;
            btn.innerHTML = `<i class="fas fa-image"></i> Normal View`;

            const container = preview.querySelector('.image-compare-container');
            const beforeBox = preview.querySelector('.image-compare-before');
            const slider = preview.querySelector('.image-compare-slider');

            const updateSlider = (clientX) => {
                const rect = container.getBoundingClientRect();
                let x = clientX - rect.left;
                x = Math.max(0, Math.min(x, rect.width));
                const pct = (x / rect.width) * 100;
                beforeBox.style.width = pct + '%';
                slider.style.left = pct + '%';
            };

            slider.onpointerdown = slider.onmousedown = () => { activeCompareDrag = updateSlider; };
            container.onclick = (e) => updateSlider(e.clientX);
        }
    };

    function closeMobileSidebar() {
        if (sidebar && sidebar.classList.contains('mobile-open')) {
            sidebar.classList.remove('mobile-open');
            if (sidebarToggleBtn) sidebarToggleBtn.setAttribute('aria-expanded', 'false');
        }
    }

    function showInlineConfirm(message, confirmLabel, onConfirm) {
        document.getElementById('agentInlineConfirm')?.remove();
        const box = document.createElement('div');
        box.id = 'agentInlineConfirm';
        box.className = 'agent-inline-confirm';
        box.setAttribute('role', 'alertdialog');
        box.setAttribute('aria-modal', 'false');
        box.setAttribute('aria-label', message);
        box.innerHTML = `
            <span class="agent-inline-confirm-text">${escapeHtml(message)}</span>
            <button type="button" class="agent-inline-confirm-cancel">Cancel</button>
            <button type="button" class="agent-inline-confirm-ok">${escapeHtml(confirmLabel)}</button>`;
        const close = () => { box.remove(); chatInput?.focus(); };
        (document.querySelector('.chat-composer, .composer-dock') || messagesStream.parentNode || document.body).appendChild(box);
        box.querySelector('.agent-inline-confirm-cancel')?.addEventListener('click', close);
        box.querySelector('.agent-inline-confirm-ok')?.addEventListener('click', () => { close(); onConfirm(); });
        box.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
        box.querySelector('.agent-inline-confirm-cancel')?.focus();
    }

    // ─── EVENT LISTENERS ───
    function setupEventListeners() {
        // Toggle Sidebar
        sidebarToggleBtn.onclick = () => {
            if (window.matchMedia('(max-width: 860px)').matches) {
                sidebar.classList.toggle('mobile-open');
                sidebarToggleBtn.setAttribute('aria-expanded', String(sidebar.classList.contains('mobile-open')));
                if (sidebar.classList.contains('mobile-open')) document.getElementById('closeAgentSidebar')?.focus();
            } else {
                sidebar.classList.toggle('collapsed');
                sidebarToggleBtn.setAttribute('aria-expanded', String(!sidebar.classList.contains('collapsed')));
            }
        };
        const closeAgentSidebar = () => {
            sidebar.classList.remove('mobile-open');
            sidebarToggleBtn.setAttribute('aria-expanded', 'false');
            sidebarToggleBtn.focus();
        };
        document.getElementById('closeAgentSidebar')?.addEventListener('click', closeAgentSidebar);
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape' && sidebar.classList.contains('mobile-open')) closeAgentSidebar();
        });

        // New Chat Button
        newChatBtn.onclick = () => {
            createNewSession();
            closeMobileSidebar();
        };

        // Clear Chat
        clearChatBtn.onclick = () => {
            const session = getCurrentSession();
            if (!session || !session.messages.length) return;
            showInlineConfirm('Clear this conversation? This cannot be undone.', 'Clear', () => {
                session.messages = [];
                saveSessionsToStorage();
                renderMessages();
            });
        };

        // Shortcuts Modal
        const shortcutsBtn = document.getElementById('shortcutsBtn');
        const shortcutsModal = document.getElementById('shortcutsModal');
        const closeShortcutsBtn = document.getElementById('closeShortcutsBtn');

        if (shortcutsBtn && shortcutsModal) {
            shortcutsBtn.onclick = () => shortcutsModal.classList.add('active');
            if (closeShortcutsBtn) closeShortcutsBtn.onclick = () => shortcutsModal.classList.remove('active');
            shortcutsModal.onclick = (e) => {
                if (e.target === shortcutsModal) shortcutsModal.classList.remove('active');
            };
        }

        // Global Keyboard Hotkeys
        window.addEventListener('keydown', (e) => {
            // Escape closes modal
            if (e.key === 'Escape' && shortcutsModal && shortcutsModal.classList.contains('active')) {
                shortcutsModal.classList.remove('active');
                return;
            }
            // ? opens shortcuts when not in input
            if (e.key === '?' && document.activeElement !== chatInput && shortcutsModal) {
                shortcutsModal.classList.add('active');
                return;
            }
            // Ctrl + K clears chat
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                clearChatBtn.click();
                return;
            }
            // Ctrl + U triggers file attachment
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'u') {
                e.preventDefault();
                filePicker.click();
                return;
            }
        });

        // Remove Active Media from Drawer
        if (removeActiveMediaBtn) {
            removeActiveMediaBtn.onclick = () => {
                setActiveMedia(null);
            };
        }

        setupComposer();
        setupSkills();
    }

    // ─── SKILLS: "@" autocomplete, library (button, Ctrl+/) ───
    function setupSkills() {
        if (!window.AgentSkills || !chatInput) return;
        const skillsBtn = document.getElementById('skillsBtn');
        const openLibrary = () => window.AgentSkills.openLibrary({
            getMediaType: () => (activeMedia ? activeMedia.type : null),
            returnFocus: skillsBtn || chatInput,
            onUse: skillId => window.AgentSkills.insertSkill(chatInput, skillId)
        });
        window.AgentSkills.attachAutocomplete(chatInput, {
            getMediaType: () => (activeMedia ? activeMedia.type : null),
            onOpenLibrary: openLibrary
        });
        if (skillsBtn) skillsBtn.addEventListener('click', openLibrary);
        window.addEventListener('keydown', event => {
            if ((event.ctrlKey || event.metaKey) && event.key === '/') {
                event.preventDefault();
                openLibrary();
            }
        });
    }

    // ─── COMPOSER: send button, Enter-to-send, attachments ───
    function setupComposer() {
        if (sendBtn) {
            sendBtn.addEventListener('click', event => {
                if (event && event.preventDefault) event.preventDefault();
                handleSendMessage();
            });
        }
        if (chatInput) {
            chatInput.addEventListener('keydown', event => {
                // Enter sends; Shift+Enter inserts a newline; never send mid IME composition.
                if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
                if (event.defaultPrevented) return;          // e.g. an open autocomplete consumed the key
                event.preventDefault();
                handleSendMessage();
            });
            chatInput.addEventListener('input', () => {
                chatInput.style.height = 'auto';
                if (chatInput.scrollHeight) chatInput.style.height = `${Math.min(chatInput.scrollHeight, 200)}px`;
            });
        }
        if (attachBtn && filePicker) {
            attachBtn.addEventListener('click', () => filePicker.click());
        }
        if (filePicker) {
            filePicker.addEventListener('change', () => {
                const file = filePicker.files && filePicker.files[0];
                if (file) uploadMediaFile(file);
                filePicker.value = '';
            });
        }
        if (removePendingBtn) {
            removePendingBtn.addEventListener('click', () => {
                const session = getCurrentSession();
                if (session) pendingUploads.delete(session);
                setActiveMedia(null);
            });
        }
    }

    // ─── TOAST NOTIFICATION SYSTEM ───
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icon = type === 'success' ? 'fa-circle-check' :
            (type === 'error' ? 'fa-circle-exclamation' :
                (type === 'warning' ? 'fa-triangle-exclamation' : 'fa-circle-info'));

        toast.innerHTML = `<i class="fas ${icon}"></i> <span>${escapeHtml(message)}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            if (toast && toast.parentNode) toast.remove();
        }, 3600);
    }
    window.showAgentToast = showToast;

    // ─── VOICE RECOGNITION (WEB SPEECH API) ───
    function setupVoiceRecognition() {
        if (!window.LocalVoiceInput) {
            micBtn.disabled = true;
            micBtn.title = 'Local voice input is unavailable';
            return;
        }
        let dictationSession = null;
        const voice = window.LocalVoiceInput.create({
            onState(state) {
                if (state === 'recording') dictationSession = currentSessionId;
                micBtn.classList.toggle('recording', state === 'recording');
                micBtn.disabled = state === 'transcribing';
                micBtn.title = state === 'recording' ? 'Recording locally — click to transcribe'
                    : state === 'transcribing' ? 'Transcribing locally...' : 'Local voice input';
            },
            onTranscript(transcript) {
                if (currentSessionId !== dictationSession || !transcript) return;
                chatInput.value = chatInput.value ? chatInput.value + ' ' + transcript : transcript;
                chatInput.focus();
                sendBtn.disabled = isProcessing;
            },
            onError(message) { showToast(message, 'error'); }
        });
        micBtn.disabled = !voice.supported;
        micBtn.title = voice.supported ? 'Local voice input' : 'Microphone recording is unavailable in this browser';
        micBtn.onclick = () => voice.toggle();
    }

    // ─── WINDOW DRAG & DROP ───
    function setupDragAndDrop() {
        let dragCounter = 0;

        window.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            dragOverlay.classList.add('active');
        });

        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter <= 0) {
                dragOverlay.classList.remove('active');
                dragCounter = 0;
            }
        });

        window.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;
            dragOverlay.classList.remove('active');

            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                uploadMediaFile(e.dataTransfer.files[0]);
            }
        });
    }

    // ─── HELPERS ───
    function formatBytes(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function formatMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);
        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic: *text*
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Code: `code`
        html = html.replace(/`(.*?)`/g, '<code class="inline-code">$1</code>');
        // Linebreaks
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    // ─── BOOTSTRAP ON LOAD ───
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
