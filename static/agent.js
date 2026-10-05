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
    const uploadProgressContainer = document.getElementById('uploadProgressContainer');
    const uploadProgressFill = document.getElementById('uploadProgressFill');
    const uploadProgressPercent = document.getElementById('uploadProgressPercent');
    const uploadProgressTransferred = document.getElementById('uploadProgressTransferred');
    const uploadProgressSpeed = document.getElementById('uploadProgressSpeed');
    const uploadProgressEta = document.getElementById('uploadProgressEta');
    const activeUploadAbortControllers = new WeakMap();
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

    function humanizeMediaName(filename, originalName) {
        let name = originalName || filename || 'Media';
        if (/^cutout_portrait_[0-9a-f]{16,}/i.test(name)) return 'Portrait Cutout';
        if (/^cutout_[0-9a-f]{16,}/i.test(name)) return 'Subject Cutout';
        if (/^upscale_[0-9a-f]{16,}/i.test(name)) return 'Upscaled Image';
        if (/^clarity_[0-9a-f]{16,}/i.test(name)) return 'Enhanced Image';
        if (/^isolate_voice_[0-9a-f]{16,}/i.test(name)) return 'Isolated Vocals';
        if (/^isolate_[0-9a-f]{16,}/i.test(name)) return 'Stem Isolation';
        if (/^agent_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}/i.test(name)) {
            const ext = (name.split('.').pop() || '').toUpperCase();
            return `Media File (${ext})`;
        }
        if (name.length > 28) {
            const parts = name.split('.');
            const ext = parts.length > 1 ? '.' + parts.pop() : '';
            const base = parts.join('.');
            return base.substring(0, 24) + '...' + ext;
        }
        return name;
    }

    function formatTime(seconds) {
        if (!Number.isFinite(seconds) || seconds <= 0) return '0:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    function updateActiveMediaDisplay() {
        if (activeMedia) {
            activeMediaBox.style.display = 'flex';
            const friendlyName = humanizeMediaName(activeMedia.filename, activeMedia.original_name);
            activeMediaName.textContent = friendlyName;
            activeMediaName.title = activeMedia.original_name || activeMedia.filename || '';

            const sizeStr = activeMedia.size ? formatBytes(activeMedia.size) : '';
            const typeStr = activeMedia.type ? activeMedia.type.toUpperCase() : 'FILE';
            const dimStr = (activeMedia.width && activeMedia.height) ? `${activeMedia.width}×${activeMedia.height} • ` :
                           (activeMedia.duration ? `${formatTime(activeMedia.duration)} • ` : '');
            activeMediaMeta.textContent = `${typeStr} • ${dimStr}${sizeStr}`;

            const activeThumb = document.getElementById('activeMediaThumb');
            if (activeThumb) {
                if (activeMedia.type === 'image' && activeMedia.url) {
                    activeThumb.innerHTML = `<img src="${activeMedia.url}" alt="" loading="lazy" />`;
                } else if (activeMedia.type === 'video' && activeMedia.url) {
                    activeThumb.innerHTML = `<video src="${activeMedia.url}#t=0.1" preload="metadata" muted playsinline></video>`;
                } else if (activeMedia.type === 'audio') {
                    activeThumb.innerHTML = `<i class="fas fa-wave-square"></i>`;
                } else {
                    activeThumb.innerHTML = `<i class="fas fa-file-waveform"></i>`;
                }
            }

            pendingAttachmentCard.style.display = 'flex';
            pendingFileName.textContent = friendlyName;
            pendingFileName.title = activeMedia.original_name || activeMedia.filename || '';
            const pendingTypeBadge = document.getElementById('pendingTypeBadge');
            if (pendingTypeBadge) pendingTypeBadge.textContent = typeStr;
            pendingFileSize.textContent = `${dimStr}${sizeStr}`;

            const pendingFileThumb = document.getElementById('pendingFileThumb');
            if (pendingFileThumb) {
                if (activeMedia.type === 'image' && activeMedia.url) {
                    pendingFileThumb.innerHTML = `<img src="${activeMedia.url}" alt="" loading="lazy" />`;
                } else if (activeMedia.type === 'video' && activeMedia.url) {
                    pendingFileThumb.innerHTML = `<video src="${activeMedia.url}#t=0.1" preload="metadata" muted playsinline></video>`;
                } else if (activeMedia.type === 'audio') {
                    pendingFileThumb.innerHTML = `<i class="fas fa-wave-square"></i>`;
                } else {
                    pendingFileThumb.innerHTML = `<i class="fas fa-paperclip"></i>`;
                }
            }
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

        if (uploadProgressContainer) {
            uploadProgressContainer.style.display = 'flex';
            if (uploadProgressFill) uploadProgressFill.style.width = '0%';
            if (uploadProgressPercent) uploadProgressPercent.textContent = '0%';
            if (uploadProgressTransferred) uploadProgressTransferred.textContent = `0 B / ${formatBytes(file.size)}`;
            if (uploadProgressSpeed) uploadProgressSpeed.textContent = '0 MB/s';
            if (uploadProgressEta) uploadProgressEta.textContent = 'Connecting...';
        }

        const abortController = typeof AbortController !== 'undefined'
            ? new AbortController()
            : { signal: { aborted: false }, abort() { if (this.signal) this.signal.aborted = true; } };
        activeUploadAbortControllers.set(session, abortController);

        const startTime = Date.now();
        let totalUploadedBytes = 0;

        function updateProgress(bytesUploaded) {
            totalUploadedBytes = Math.min(bytesUploaded, file.size);
            const percent = file.size > 0 ? Math.round((totalUploadedBytes / file.size) * 100) : 100;
            const elapsedSeconds = Math.max((Date.now() - startTime) / 1000, 0.1);
            const speedBytesPerSec = totalUploadedBytes / elapsedSeconds;
            const speedMbPerSec = (speedBytesPerSec / (1024 * 1024)).toFixed(1);
            const remainingBytes = Math.max(0, file.size - totalUploadedBytes);
            const etaSeconds = speedBytesPerSec > 0 ? Math.round(remainingBytes / speedBytesPerSec) : 0;

            if (uploadProgressFill) uploadProgressFill.style.width = `${percent}%`;
            if (uploadProgressPercent) uploadProgressPercent.textContent = `${percent}%`;
            if (uploadProgressTransferred) uploadProgressTransferred.textContent = `${formatBytes(totalUploadedBytes)} / ${formatBytes(file.size)}`;
            if (uploadProgressSpeed) uploadProgressSpeed.textContent = `${speedMbPerSec} MB/s`;
            if (uploadProgressEta) uploadProgressEta.textContent = percent >= 100 ? 'Finalizing...' : `${etaSeconds}s left`;
        }

        const CHUNK_THRESHOLD = 4 * 1024 * 1024; // 4MB
        const CHUNK_SIZE = 2.5 * 1024 * 1024; // 2.5MB slices for resilient server streaming

        try {
            let data = null;

            if (file.size <= CHUNK_THRESHOLD) {
                // Direct single-shot upload for small files (100% compatible with test suites and small attachments)
                const formData = new FormData();
                formData.append('file', file);

                updateProgress(Math.round(file.size * 0.4));

                const uploadOpts = { method: 'POST', body: formData };
                if (abortController && abortController.signal) uploadOpts.signal = abortController.signal;

                const resp = await fetch('/api/agent/upload', uploadOpts);

                if (!resp.ok) {
                    const errData = await resp.json().catch(() => ({}));
                    throw new Error(errData.error || `Upload failed with status ${resp.status}`);
                }
                updateProgress(file.size);
                data = await resp.json();
            } else {
                // Resilient chunked upload engine for large/long media with automatic retry
                const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
                const uploadId = 'up_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);

                for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                    if (!sessions.includes(session) || pendingUploads.get(session) !== uploadGeneration) {
                        fetch('/api/agent/upload/abort', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ upload_id: uploadId })
                        }).catch(() => {});
                        return;
                    }

                    const start = chunkIndex * CHUNK_SIZE;
                    const end = Math.min(start + CHUNK_SIZE, file.size);
                    const chunkBlob = file.slice(start, end);

                    // Retry up to 3 times per chunk with backoff on connection drops
                    let chunkSuccess = false;
                    let lastChunkError = null;

                    for (let attempt = 0; attempt < 3; attempt++) {
                        if (abortController && abortController.signal && abortController.signal.aborted) {
                            throw new Error('Upload aborted by user');
                        }

                        try {
                            const chunkForm = new FormData();
                            chunkForm.append('file', chunkBlob, file.name);
                            chunkForm.append('upload_id', uploadId);
                            chunkForm.append('chunk_index', String(chunkIndex));
                            chunkForm.append('total_chunks', String(totalChunks));
                            chunkForm.append('filename', file.name);

                            const chunkOpts = { method: 'POST', body: chunkForm };
                            if (abortController && abortController.signal) chunkOpts.signal = abortController.signal;

                            const chunkResp = await fetch('/api/agent/upload/chunk', chunkOpts);

                            if (!chunkResp.ok) {
                                const errJson = await chunkResp.json().catch(() => ({}));
                                throw new Error(errJson.error || `Chunk ${chunkIndex + 1}/${totalChunks} failed (${chunkResp.status})`);
                            }

                            chunkSuccess = true;
                            updateProgress(end);
                            break;
                        } catch (err) {
                            if (abortController && abortController.signal && abortController.signal.aborted) throw err;
                            lastChunkError = err;
                            await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(2, attempt), 4000)));
                        }
                    }

                    if (!chunkSuccess) {
                        throw new Error(`Upload connection failed at chunk ${chunkIndex + 1}/${totalChunks}: ${lastChunkError ? lastChunkError.message : 'Network error'}`);
                    }
                }

                // All chunks uploaded: Stitch on server
                if (uploadProgressEta) uploadProgressEta.textContent = 'Inspecting media...';

                const completeOpts = {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        upload_id: uploadId,
                        filename: file.name,
                        total_chunks: totalChunks
                    })
                };
                if (abortController && abortController.signal) completeOpts.signal = abortController.signal;

                const completeResp = await fetch('/api/agent/upload/complete', completeOpts);

                if (!completeResp.ok) {
                    const completeErr = await completeResp.json().catch(() => ({}));
                    throw new Error(completeErr.error || 'Failed to finalize uploaded media on server');
                }

                data = await completeResp.json();
            }

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
        } finally {
            if (uploadProgressContainer) uploadProgressContainer.style.display = 'none';
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
            const mediaType = (msg.attachedMedia.type || 'FILE').toUpperCase();
            fileChipHtml = `
                <div class="user-attached-file">
                    <span class="user-file-icon-wrap"><i class="fas ${icon}"></i></span>
                    <span class="user-file-name">${escapeHtml(msg.attachedMedia.original_name || msg.attachedMedia.filename)}</span>
                    <span class="user-file-type-pill">${escapeHtml(mediaType)}</span>
                </div>
            `;
        }

        const timeStr = msg.timestamp ? `<div class="user-message-time">${escapeHtml(msg.timestamp)}</div>` : '';

        row.innerHTML = `
            <div class="message-bubble">
                ${fileChipHtml}
                <div class="user-text">${escapeHtml(msg.content)}</div>
                ${timeStr}
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
    const PUBLIC_AGENT_LABELS = {
        VisionSubAgent: 'Image Specialist',
        VideoSubAgent: 'Video Studio',
        AudioSubAgent: 'Audio Studio',
        InspectorSubAgent: 'Media Inspector',
        Orchestrator: assistantName()
    };

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
                (subagent === 'AudioSubAgent' ? 'fa-wave-square' :
                    (subagent === 'InspectorSubAgent' ? 'fa-sliders' : 'fa-sparkles')));
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
                        <i class="fas fa-chevron-right" aria-hidden="true"></i>
                        <i class="fas fa-sparkles cot-spark-icon" aria-hidden="true"></i>
                        <span>Reasoning process</span>
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
            const imageExtensions = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
            const videoExtensions = ['mp4', 'webm', 'mov', 'mkv'];
            const audioExtensions = ['mp3', 'wav', 'm4a', 'ogg', 'flac'];
            const isImageOutput = imageExtensions.includes(ext);
            const isVideoOutput = videoExtensions.includes(ext);
            const isAudioOutput = audioExtensions.includes(ext);

            let previewEl = '';
            if (isVideoOutput) {
                previewEl = `<video controls playsinline preload="metadata" src="${url}"></video>`;
            } else if (isImageOutput) {
                previewEl = `<img src="${url}" alt="Processed Output" loading="lazy" />`;
            } else if (isAudioOutput) {
                previewEl = `<div class="audio-embed-wrapper" data-url="${escapeHtml(url)}" data-input="${escapeHtml(msg.input_url || '')}" data-name="${escapeHtml(msg.output_file || '')}"></div>`;
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

            // Compare against the media this edit started from (stored with the message)
            const inputUrl = msg.input_url;
            const canCompareImage = isImageOutput && inputUrl && inputUrl !== url
                && imageExtensions.includes(inputUrl.split('.').pop().toLowerCase());
            const canCompareVideo = isVideoOutput && inputUrl && inputUrl !== url
                && videoExtensions.includes(inputUrl.split('.').pop().toLowerCase());

            const compareBtn = canCompareImage ? `
                <button class="media-action-secondary" onclick="window.agentToggleCompare(this, '${escapeHtml(inputUrl)}', '${escapeHtml(url)}')">
                    <i class="fas fa-columns"></i> Split Compare
                </button>
            ` : (canCompareVideo ? `
                <button class="media-action-secondary" onclick="window.agentToggleVideoCompare(this, '${escapeHtml(inputUrl)}', '${escapeHtml(url)}')">
                    <i class="fas fa-columns"></i> Split Compare
                </button>
            ` : '');

            const copyBtn = isImageOutput ? `
                <button class="media-action-secondary" onclick="window.copyImageToClipboard('${escapeHtml(url)}')">
                    <i class="fas fa-copy"></i> Copy
                </button>
            ` : '';

            const snapshotBtn = isVideoOutput ? `
                <button class="media-action-secondary" onclick="window.agentSnapshotVideo(this, '${escapeHtml(url)}')">
                    <i class="fas fa-camera"></i> Snapshot Frame
                </button>
            ` : '';

            const inspectBtn = `
                <button class="media-action-secondary" onclick="window.openMediaInspector({ url: '${escapeHtml(url)}', filename: '${escapeHtml(msg.output_file || '')}', original_name: '${escapeHtml(msg.output_file || '')}', type: '${isVideoOutput ? 'video' : (isAudioOutput ? 'audio' : 'image')}', input_url: '${escapeHtml(inputUrl || '')}' })">
                    <i class="fas fa-expand"></i> Inspect
                </button>
            `;
            const stageBar = isImageOutput ? `
                <div class="preview-stage-bar">
                    <button type="button" class="preview-stage-btn active" data-stage="checker" title="Checkerboard (Transparency)"><i class="fas fa-border-all"></i></button>
                    <button type="button" class="preview-stage-btn" data-stage="dark" title="Dark Slate Studio"><i class="fas fa-circle" style="color:#0f172a"></i></button>
                    <button type="button" class="preview-stage-btn" data-stage="light" title="Crisp White"><i class="fas fa-circle" style="color:#ffffff"></i></button>
                    <button type="button" class="preview-stage-btn" data-stage="accent" title="High Contrast"><i class="fas fa-circle" style="color:#0ea5e9"></i></button>
                </div>
            ` : '';
            const artifactLinks = msg.artifacts.map(item => `
                <a href="${escapeHtml(item.output_url)}" download class="media-action-secondary">
                    <i class="fas fa-image"></i> ${item.tool === 'extract_frame' ? 'Download thumbnail' : 'Download extra file'}
                </a>`).join('');

            mediaHtml = `
                <div class="media-output-card">
                    <div class="media-output-preview" id="previewBox_${Date.now()}">
                        ${stageBar}
                        ${previewEl}
                    </div>
                    <div class="media-output-actions">
                        <a href="${url}" download class="btn-download-media">
                            <i class="fas fa-download"></i> Download Export
                        </a>
                        <div class="media-actions-group">
                            ${compareBtn}
                            ${copyBtn}
                            ${snapshotBtn}
                            ${inspectBtn}
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

        const timeMetaHtml = msg.timestamp ? `<span class="message-time-meta" title="${escapeHtml(msg.timestamp)}">${escapeHtml(msg.timestamp)}</span>` : '';
        const skillsHeaderHtml = skillBadges ? `<div class="message-header-meta">${skillBadges}</div>` : '';

        row.innerHTML = `
            <div class="message-avatar" title="${subagentLabel}" aria-hidden="true">
                <i class="fas ${subagentIcon}"></i>
            </div>
            <div class="message-bubble">
                ${skillsHeaderHtml}
                ${cotHtml}
                <div class="agent-text">${formatMarkdown(msg.reply || '')}</div>
                ${mediaHtml}
                ${clarifHtml}
                ${suggestionsHtml}
                <div class="message-actions-toolbar">
                    <button type="button" class="bubble-action-btn copy-msg-btn" onclick="window.copyAgentReply(this)" title="Copy response text" aria-label="Copy response">
                        <i class="fas fa-copy"></i> <span>Copy text</span>
                    </button>
                    ${timeMetaHtml}
                </div>
            </div>
        `;
        messagesStream.appendChild(row);
        if (row.querySelectorAll) {
            row.querySelectorAll('.audio-embed-wrapper').forEach(w => {
                if (window.renderStudioAudioPlayer) {
                    window.renderStudioAudioPlayer(w, {
                        url: w.dataset.url,
                        inputUrl: w.dataset.input,
                        filename: w.dataset.name,
                        originalName: w.dataset.name
                    });
                }
            });
        }
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
        row.querySelectorAll('.preview-stage-btn').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                const preview = btn.closest('.media-output-preview');
                if (!preview) return;
                preview.querySelectorAll('.preview-stage-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                preview.classList.remove('stage-checker', 'stage-dark', 'stage-light', 'stage-accent');
                const stage = btn.dataset.stage;
                if (stage && stage !== 'checker') preview.classList.add(`stage-${stage}`);
            };
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

        const hasAttachment = Boolean(activeMedia && activeMedia.filename);
        const initialTitle = hasAttachment
            ? `Analyzing ${activeMedia.type || 'media'}...`
            : 'Thinking...';
        const initialSubstatus = hasAttachment
            ? 'Inspecting format, tracks, and formulating editing steps...'
            : 'Analyzing request and preparing response...';

        let elapsed = 1;

        row.innerHTML = `
            <div class="message-avatar thinking-avatar" aria-hidden="true">
                <i class="fas fa-wand-magic-sparkles"></i>
            </div>
            <div class="thinking-live-card">
                <div class="thinking-header-row">
                    <div class="thinking-status-wrapper">
                        <span class="thinking-glow-dot"></span>
                        <span class="thinking-main-title" id="${id}_title">${escapeHtml(initialTitle)}</span>
                    </div>
                    <span class="thinking-elapsed-badge">
                        <i class="far fa-clock"></i> <span id="${id}_timer">1s</span>
                    </span>
                </div>

                <div class="thinking-shimmer-track">
                    <div class="thinking-shimmer-bar"></div>
                </div>

                <div class="thinking-sub-status" id="${id}_substatus">
                    ${escapeHtml(initialSubstatus)}
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

            if (timerEl) {
                timerEl.textContent = `${elapsed}s`;
            }

            if (elapsed >= 3 && elapsed < 7) {
                if (titleEl) titleEl.textContent = hasAttachment ? 'Applying Tool Operations...' : 'Formulating Response...';
                if (substatusEl) substatusEl.textContent = hasAttachment ? 'Executing media tools & processing transforms...' : 'Synthesizing capabilities & drafting clear reply...';
            } else if (elapsed >= 7 && elapsed < 12) {
                if (titleEl) titleEl.textContent = hasAttachment ? 'Rendering Output...' : 'Refining Answer...';
                if (substatusEl) substatusEl.textContent = hasAttachment ? 'Encoding media and generating preview...' : 'Validating formatting and finalizing...';
            } else if (elapsed >= 12) {
                if (titleEl) titleEl.textContent = hasAttachment ? 'Finalizing Preview...' : 'Completing Response...';
                if (substatusEl) substatusEl.textContent = 'Packaging results...';
            }
        }, 1000);

        activeThinkingTimers[id] = { timerInterval };
    }

    function removeThinkingBubble(id) {
        if (activeThinkingTimers[id]) {
            if (activeThinkingTimers[id].timerInterval) clearInterval(activeThinkingTimers[id].timerInterval);
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

    // ─── INTERACTIVE BEFORE / AFTER SLIDER (PRECISION CLIP-PATH) ───
    function setupCompareInteractions(container) {
        if (!container) return;
        const line = container.querySelector('.compare-divider-line');
        const handle = container.querySelector('.compare-divider-handle');
        const tooltip = container.querySelector('.compare-percent-tooltip');
        let isDragging = false;

        const setSplitPct = (pct) => {
            const clamped = Math.max(0, Math.min(100, pct));
            container.style.setProperty('--split-pos', clamped + '%');
            if (line) line.style.left = clamped + '%';
            if (handle) handle.setAttribute('aria-valuenow', Math.round(clamped));
            if (tooltip) tooltip.textContent = Math.round(clamped) + '%';
        };

        const onPointerDown = (e) => {
            if (e.target.closest('.compare-mode-toolbar') || e.target.closest('.preview-stage-bar')) return;
            isDragging = true;
            line?.classList.add('dragging');
            container.setPointerCapture?.(e.pointerId);
            onPointerMove(e);
        };

        const onPointerMove = (e) => {
            if (!isDragging && e.type !== 'click') return;
            const rect = container.getBoundingClientRect();
            if (!rect.width) return;
            const x = e.clientX - rect.left;
            const pct = (x / rect.width) * 100;
            setSplitPct(pct);
        };

        const onPointerUp = (e) => {
            if (isDragging) {
                isDragging = false;
                line?.classList.remove('dragging');
                try { container.releasePointerCapture?.(e.pointerId); } catch (_) {}
            }
        };

        container.addEventListener('pointerdown', onPointerDown);
        container.addEventListener('pointermove', onPointerMove);
        container.addEventListener('pointerup', onPointerUp);
        container.addEventListener('pointercancel', onPointerUp);

        // Keyboard arrow controls for slider
        handle?.addEventListener('keydown', (e) => {
            const currentPct = parseFloat(container.style.getPropertyValue('--split-pos') || '50');
            if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                e.preventDefault();
                setSplitPct(currentPct - (e.shiftKey ? 10 : 2));
            } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                e.preventDefault();
                setSplitPct(currentPct + (e.shiftKey ? 10 : 2));
            } else if (e.key === 'Home') {
                e.preventDefault();
                setSplitPct(0);
            } else if (e.key === 'End') {
                e.preventDefault();
                setSplitPct(100);
            }
        });

        // Stage buttons
        container.querySelectorAll('.preview-stage-btn').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                container.querySelectorAll('.preview-stage-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                container.classList.remove('stage-checker', 'stage-dark', 'stage-light', 'stage-accent');
                const stage = btn.dataset.stage;
                if (stage && stage !== 'checker') container.classList.add(`stage-${stage}`);
            };
        });

        // Mode buttons
        container.querySelectorAll('.compare-mode-btn').forEach(btn => {
            if (btn.classList.contains('compare-hold-btn')) {
                const holdStart = (e) => { e.preventDefault(); container.classList.add('mode-hold-original'); };
                const holdEnd = (e) => { e.preventDefault(); container.classList.remove('mode-hold-original'); };
                btn.onpointerdown = holdStart;
                btn.onpointerup = holdEnd;
                btn.onpointercancel = holdEnd;
                btn.onmouseleave = holdEnd;
            } else {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    container.querySelectorAll('.compare-mode-btn:not(.compare-hold-btn)').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    container.classList.remove('mode-split', 'mode-side-by-side', 'mode-difference');
                    const mode = btn.dataset.mode;
                    if (mode && mode !== 'split') container.classList.add(`mode-${mode}`);
                };
            }
        });
    }

    window.agentToggleCompare = function (btn, origUrl, processedUrl) {
        if (!origUrl || !processedUrl) return;
        const card = btn.closest('.media-output-card');
        const preview = card ? card.querySelector('.media-output-preview') : null;
        if (!preview) return;

        if (preview.classList.contains('in-compare-mode')) {
            preview.classList.remove('in-compare-mode');
            preview.innerHTML = `
                <div class="preview-stage-bar">
                    <button type="button" class="preview-stage-btn active" data-stage="checker" title="Checkerboard"><i class="fas fa-border-all"></i></button>
                    <button type="button" class="preview-stage-btn" data-stage="dark" title="Dark Slate Studio"><i class="fas fa-circle" style="color:#0f172a"></i></button>
                    <button type="button" class="preview-stage-btn" data-stage="light" title="Crisp White"><i class="fas fa-circle" style="color:#ffffff"></i></button>
                    <button type="button" class="preview-stage-btn" data-stage="accent" title="High Contrast"><i class="fas fa-circle" style="color:#0ea5e9"></i></button>
                </div>
                <img src="${processedUrl}" alt="Processed Output" loading="lazy" />
            `;
            btn.innerHTML = `<i class="fas fa-columns"></i> Split Compare`;
            preview.querySelectorAll('.preview-stage-btn').forEach(b => {
                b.onclick = (e) => {
                    e.stopPropagation();
                    preview.querySelectorAll('.preview-stage-btn').forEach(x => x.classList.remove('active'));
                    b.classList.add('active');
                    preview.classList.remove('stage-checker', 'stage-dark', 'stage-light', 'stage-accent');
                    const stg = b.dataset.stage;
                    if (stg && stg !== 'checker') preview.classList.add(`stage-${stg}`);
                };
            });
        } else {
            preview.classList.add('in-compare-mode');
            preview.innerHTML = `
                <div class="image-compare-container" style="--split-pos: 50%;">
                    <div class="preview-stage-bar">
                        <button type="button" class="preview-stage-btn active" data-stage="checker" title="Checkerboard"><i class="fas fa-border-all"></i></button>
                        <button type="button" class="preview-stage-btn" data-stage="dark" title="Dark Slate"><i class="fas fa-circle" style="color:#0f172a"></i></button>
                        <button type="button" class="preview-stage-btn" data-stage="light" title="Crisp White"><i class="fas fa-circle" style="color:#ffffff"></i></button>
                        <button type="button" class="preview-stage-btn" data-stage="accent" title="High Contrast"><i class="fas fa-circle" style="color:#0ea5e9"></i></button>
                    </div>
                    <div class="compare-mode-toolbar">
                        <button type="button" class="compare-mode-btn active" data-mode="split" title="Split Wipe"><i class="fas fa-columns"></i> Split</button>
                        <button type="button" class="compare-mode-btn" data-mode="side-by-side" title="Side by Side"><i class="fas fa-table-columns"></i> Side</button>
                        <button type="button" class="compare-mode-btn" data-mode="difference" title="X-Ray Difference"><i class="fas fa-circle-half-stroke"></i> Diff</button>
                        <button type="button" class="compare-mode-btn compare-hold-btn" title="Hold to view original (or hold Space)"><i class="fas fa-eye"></i> Hold</button>
                    </div>
                    <div class="image-compare-wrapper">
                        <img class="compare-img compare-after" src="${processedUrl}" alt="Processed Output" />
                        <div class="compare-before-layer">
                            <img class="compare-img compare-before" src="${origUrl}" alt="Original Input" />
                        </div>
                        <div class="compare-divider-line" style="left: 50%;">
                            <div class="compare-divider-handle" tabindex="0" role="slider" aria-label="Comparison divider position" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
                                <i class="fas fa-arrows-left-right"></i>
                                <span class="compare-percent-tooltip">50%</span>
                            </div>
                        </div>
                        <span class="compare-pill compare-pill-before">Original</span>
                        <span class="compare-pill compare-pill-after">Processed</span>
                    </div>
                </div>
            `;
            btn.innerHTML = `<i class="fas fa-image"></i> Normal View`;
            const container = preview.querySelector('.image-compare-container');
            setupCompareInteractions(container);
        }
    };

    // ─── SYNCHRONIZED VIDEO COMPARISON ───
    window.agentToggleVideoCompare = function (btn, origUrl, processedUrl) {
        if (!origUrl || !processedUrl) return;
        const card = btn.closest ? btn.closest('.media-output-card') : null;
        const preview = card ? card.querySelector('.media-output-preview') : null;
        if (!preview) return;

        if (preview.classList.contains('in-compare-mode')) {
            preview.classList.remove('in-compare-mode');
            preview.innerHTML = `
                <video controls playsinline preload="metadata" src="${processedUrl}"></video>
            `;
            btn.innerHTML = `<i class="fas fa-columns"></i> Split Compare`;
        } else {
            preview.classList.add('in-compare-mode');
            preview.innerHTML = `
                <div class="video-compare-container" style="--split-pos: 50%;">
                    <div class="compare-mode-toolbar">
                        <button type="button" class="compare-mode-btn active" data-mode="split" title="Split Wipe"><i class="fas fa-columns"></i> Split</button>
                        <button type="button" class="compare-mode-btn" data-mode="side-by-side" title="Side by Side"><i class="fas fa-table-columns"></i> Side</button>
                        <button type="button" class="compare-mode-btn compare-hold-btn" title="Hold to view original (or hold Space)"><i class="fas fa-eye"></i> Hold</button>
                    </div>
                    <div class="video-compare-wrapper">
                        <video class="video-after" playsinline preload="auto" src="${processedUrl}"></video>
                        <div class="video-before-layer">
                            <video class="video-before" playsinline preload="auto" src="${origUrl}"></video>
                        </div>
                        <div class="compare-divider-line" style="left: 50%;">
                            <div class="compare-divider-handle" tabindex="0" role="slider" aria-label="Comparison divider position" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
                                <i class="fas fa-arrows-left-right"></i>
                                <span class="compare-percent-tooltip">50%</span>
                            </div>
                        </div>
                        <span class="compare-pill compare-pill-before">Original</span>
                        <span class="compare-pill compare-pill-after">Processed</span>
                    </div>
                    <div class="video-studio-controls">
                        <div class="video-scrub-row">
                            <input type="range" class="video-scrub-slider" min="0" max="100" step="0.1" value="0" aria-label="Scrub video timeline" />
                            <span class="video-time-label">0:00 / 0:00</span>
                        </div>
                        <div class="video-controls-left">
                            <button type="button" class="video-ctrl-btn video-play-btn"><i class="fas fa-play"></i> Play</button>
                            <button type="button" class="video-ctrl-btn video-step-back-btn" title="Step backward 1 frame (-1f)"><i class="fas fa-backward-step"></i> -1f</button>
                            <button type="button" class="video-ctrl-btn video-step-fwd-btn" title="Step forward 1 frame (+1f)"><i class="fas fa-forward-step"></i> +1f</button>
                            <button type="button" class="video-ctrl-btn video-loop-btn active" title="Toggle Loop"><i class="fas fa-repeat"></i> Loop</button>
                        </div>
                        <div class="video-controls-right">
                            <button type="button" class="video-ctrl-btn video-speed-btn" data-speed="1">1.0×</button>
                            <button type="button" class="video-ctrl-btn video-snap-btn" title="Capture high-res frame PNG"><i class="fas fa-camera"></i> Snapshot</button>
                        </div>
                    </div>
                </div>
            `;
            btn.innerHTML = `<i class="fas fa-video"></i> Normal View`;
            const container = preview.querySelector('.video-compare-container');
            setupVideoCompareInteractions(container);
        }
    };

    function setupVideoCompareInteractions(container) {
        if (!container) return;
        const vAfter = container.querySelector ? container.querySelector('.video-after') : null;
        const vBefore = container.querySelector ? container.querySelector('.video-before') : null;
        const playBtn = container.querySelector ? container.querySelector('.video-play-btn') : null;
        const scrubSlider = container.querySelector ? container.querySelector('.video-scrub-slider') : null;
        const timeLabel = container.querySelector ? container.querySelector('.video-time-label') : null;
        const stepBackBtn = container.querySelector ? container.querySelector('.video-step-back-btn') : null;
        const stepFwdBtn = container.querySelector ? container.querySelector('.video-step-fwd-btn') : null;
        const loopBtn = container.querySelector ? container.querySelector('.video-loop-btn') : null;
        const speedBtn = container.querySelector ? container.querySelector('.video-speed-btn') : null;
        const snapBtn = container.querySelector ? container.querySelector('.video-snap-btn') : null;
        const divider = container.querySelector ? container.querySelector('.compare-divider-line') : null;
        const tooltip = container.querySelector ? container.querySelector('.compare-percent-tooltip') : null;

        let isPlaying = false;
        let isLooping = true;
        let playbackRate = 1.0;

        function updatePlayState(playing) {
            isPlaying = playing;
            if (playBtn) playBtn.innerHTML = isPlaying ? `<i class="fas fa-pause"></i> Pause` : `<i class="fas fa-play"></i> Play`;
            if (isPlaying) {
                if (vAfter && vAfter.play) vAfter.play().catch(() => {});
                if (vBefore && vBefore.play) vBefore.play().catch(() => {});
            } else {
                if (vAfter && vAfter.pause) vAfter.pause();
                if (vBefore && vBefore.pause) vBefore.pause();
            }
        }

        if (playBtn) playBtn.onclick = () => updatePlayState(!isPlaying);

        if (vAfter && vAfter.addEventListener) {
            vAfter.addEventListener('timeupdate', () => {
                const cur = vAfter.currentTime || 0;
                const dur = vAfter.duration || 1;
                if (scrubSlider && !scrubSlider.matches?.(':active')) {
                    scrubSlider.value = (cur / dur) * 100;
                }
                if (timeLabel) timeLabel.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
                if (vBefore && Math.abs((vBefore.currentTime || 0) - cur) > 0.06) {
                    vBefore.currentTime = cur;
                }
            });

            vAfter.addEventListener('ended', () => {
                if (isLooping) {
                    vAfter.currentTime = 0;
                    if (vBefore) vBefore.currentTime = 0;
                    if (vAfter.play) vAfter.play().catch(() => {});
                    if (vBefore && vBefore.play) vBefore.play().catch(() => {});
                } else {
                    updatePlayState(false);
                }
            });

            vAfter.addEventListener('loadedmetadata', () => {
                if (timeLabel) timeLabel.textContent = `0:00 / ${formatTime(vAfter.duration || 0)}`;
            });
        }

        if (scrubSlider) {
            scrubSlider.oninput = () => {
                const dur = vAfter?.duration || 1;
                const target = (scrubSlider.value / 100) * dur;
                if (vAfter) vAfter.currentTime = target;
                if (vBefore) vBefore.currentTime = target;
                if (timeLabel) timeLabel.textContent = `${formatTime(target)} / ${formatTime(dur)}`;
            };
        }

        const frameTime = 1 / 30;
        if (stepBackBtn) {
            stepBackBtn.onclick = () => {
                updatePlayState(false);
                const t = Math.max(0, (vAfter?.currentTime || 0) - frameTime);
                if (vAfter) vAfter.currentTime = t;
                if (vBefore) vBefore.currentTime = t;
            };
        }
        if (stepFwdBtn) {
            stepFwdBtn.onclick = () => {
                updatePlayState(false);
                const dur = vAfter?.duration || 1000;
                const t = Math.min(dur, (vAfter?.currentTime || 0) + frameTime);
                if (vAfter) vAfter.currentTime = t;
                if (vBefore) vBefore.currentTime = t;
            };
        }

        if (loopBtn) {
            loopBtn.onclick = () => {
                isLooping = !isLooping;
                loopBtn.classList.toggle('active', isLooping);
            };
        }

        if (speedBtn) {
            const speeds = [0.25, 0.5, 1.0, 1.5, 2.0];
            speedBtn.onclick = () => {
                const idx = speeds.indexOf(playbackRate);
                playbackRate = speeds[(idx + 1) % speeds.length];
                speedBtn.textContent = `${playbackRate}×`;
                if (vAfter) vAfter.playbackRate = playbackRate;
                if (vBefore) vBefore.playbackRate = playbackRate;
            };
        }

        if (snapBtn) {
            snapBtn.onclick = () => {
                if (!vAfter) return;
                window.captureVideoFrame(vAfter);
            };
        }

        if (divider && container.getBoundingClientRect) {
            let isDragging = false;
            const setSplitPos = (clientX) => {
                const rect = container.getBoundingClientRect();
                const pos = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
                container.style.setProperty('--split-pos', `${pos}%`);
                divider.style.left = `${pos}%`;
                if (tooltip) tooltip.textContent = `${Math.round(pos)}%`;
            };

            divider.addEventListener?.('pointerdown', (e) => {
                isDragging = true;
                divider.classList.add('dragging');
                if (divider.setPointerCapture) divider.setPointerCapture(e.pointerId);
                e.preventDefault();
            });
            window.addEventListener('pointermove', (e) => {
                if (isDragging) setSplitPos(e.clientX);
            });
            window.addEventListener('pointerup', () => {
                if (isDragging) {
                    isDragging = false;
                    divider.classList.remove('dragging');
                }
            });
        }

        if (container.querySelectorAll) {
            container.querySelectorAll('.compare-mode-btn').forEach(b => {
                b.onclick = () => {
                    if (b.classList.contains('compare-hold-btn')) return;
                    container.querySelectorAll('.compare-mode-btn:not(.compare-hold-btn)').forEach(x => x.classList.remove('active'));
                    b.classList.add('active');
                    container.classList.remove('mode-split', 'mode-side-by-side');
                    const m = b.dataset.mode;
                    if (m === 'side-by-side') container.classList.add('mode-side-by-side');
                    else container.classList.add('mode-split');
                };
            });
        }

        const holdBtn = container.querySelector ? container.querySelector('.compare-hold-btn') : null;
        if (holdBtn && holdBtn.addEventListener) {
            const startHold = () => container.classList.add('mode-hold-original');
            const endHold = () => container.classList.remove('mode-hold-original');
            holdBtn.addEventListener('mousedown', startHold);
            holdBtn.addEventListener('touchstart', startHold);
            window.addEventListener('mouseup', endHold);
            window.addEventListener('touchend', endHold);
        }
    }

    // ─── 1-CLICK HIGH-RES VIDEO SNAPSHOT ───
    window.captureVideoFrame = async function (videoEl) {
        if (!videoEl || !videoEl.videoWidth) {
            showToast('Video is loading. Try again in a moment.', 'warning');
            return;
        }
        try {
            const canvas = document.createElement('canvas');
            canvas.width = videoEl.videoWidth;
            canvas.height = videoEl.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

            const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
            if (!blob) throw new Error('Canvas conversion failed');

            try {
                if (navigator.clipboard?.write) {
                    await navigator.clipboard.write([
                        new ClipboardItem({ 'image/png': blob })
                    ]);
                    showToast(`Captured snapshot (${canvas.width}×${canvas.height}) & copied to clipboard!`, 'success');
                } else {
                    showToast(`Captured frame snapshot (${canvas.width}×${canvas.height})`, 'success');
                }
            } catch (clipErr) {
                showToast(`Captured frame snapshot (${canvas.width}×${canvas.height})`, 'success');
            }

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `snapshot_${Math.round((videoEl.currentTime || 0) * 100) / 100}s.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 4000);
        } catch (e) {
            showToast('Could not capture frame: ' + e.message, 'error');
        }
    };

    window.agentSnapshotVideo = function (btn, videoUrl) {
        const card = btn.closest ? btn.closest('.media-output-card') : null;
        const video = card ? card.querySelector('video') : null;
        if (video && video.videoWidth) {
            window.captureVideoFrame(video);
        } else {
            const tempVid = document.createElement('video');
            tempVid.crossOrigin = 'anonymous';
            tempVid.src = videoUrl;
            tempVid.onloadedmetadata = () => { tempVid.currentTime = 0; };
            tempVid.onseeked = () => { window.captureVideoFrame(tempVid); };
        }
    };

    // ─── STUDIO AUDIO PLAYER & A/B MASTERING ENGINE ───
    window.renderStudioAudioPlayer = function (container, options) {
        if (!container || !options || !options.url) return;
        const url = options.url;
        const inputUrl = options.inputUrl && options.inputUrl !== url ? options.inputUrl : null;
        const filename = options.filename || 'Audio Track';
        const title = humanizeMediaName(filename, options.originalName);

        const abControls = inputUrl ? `
            <div class="audio-ab-switcher" role="group" aria-label="A/B Audio Comparison">
                <button type="button" class="audio-ab-btn" data-track="a" title="Audition Original Audio (A)">[A] Original</button>
                <button type="button" class="audio-ab-btn active" data-track="b" title="Audition Enhanced Audio (B)">[B] Enhanced</button>
            </div>
        ` : '';

        container.innerHTML = `
            <div class="audio-studio-player">
                <div class="audio-player-header">
                    <div class="audio-player-meta">
                        <div class="audio-player-glyph"><i class="fas fa-wave-square"></i></div>
                        <div class="audio-player-title-col">
                            <span class="audio-player-title">${escapeHtml(title)}</span>
                            <span class="audio-player-sub">${inputUrl ? 'Mastering Comparison Active' : 'Master Track'}</span>
                        </div>
                    </div>
                    ${abControls}
                </div>
                <div class="audio-waveform-stage" tabindex="0" role="slider" aria-label="Scrub audio waveform">
                    <canvas class="audio-waveform-canvas"></canvas>
                    <div class="audio-playhead-line" style="left: 0%;"></div>
                    <span class="audio-hover-tooltip">0:00</span>
                </div>
                <div class="audio-meter-band">
                    <canvas class="audio-spectrum-canvas"></canvas>
                    <div class="audio-vu-meters" title="Stereo Output Level">
                        <div class="audio-vu-bar"><div class="audio-vu-fill audio-vu-l"></div></div>
                        <div class="audio-vu-bar"><div class="audio-vu-fill audio-vu-r"></div></div>
                    </div>
                </div>
                <div class="audio-transport-bar">
                    <div class="audio-transport-left">
                        <button type="button" class="audio-play-trigger" title="Play / Pause"><i class="fas fa-play"></i></button>
                        <button type="button" class="audio-skip-btn audio-skip-back" title="Rewind 5s"><i class="fas fa-rotate-left"></i></button>
                        <button type="button" class="audio-skip-btn audio-skip-fwd" title="Forward 5s"><i class="fas fa-rotate-right"></i></button>
                        <span class="audio-time-counter">0:00 / 0:00</span>
                    </div>
                    <div class="audio-transport-right">
                        <button type="button" class="audio-speed-btn" data-speed="1">1.0×</button>
                        <button type="button" class="audio-loop-btn active" title="Toggle Loop"><i class="fas fa-repeat"></i></button>
                    </div>
                </div>
                <audio class="audio-active-element" preload="auto" src="${url}"></audio>
                ${inputUrl ? `<audio class="audio-input-element" preload="auto" src="${inputUrl}"></audio>` : ''}
            </div>
        `;

        setupAudioStudioPlayer(container, options);
    };

    function setupAudioStudioPlayer(container, options) {
        if (!container || !container.querySelector) return;
        const player = container.querySelector('.audio-studio-player');
        if (!player) return;

        const enhancedAudio = player.querySelector('.audio-active-element');
        const originalAudio = player.querySelector('.audio-input-element');
        const playBtn = player.querySelector('.audio-play-trigger');
        const skipBackBtn = player.querySelector('.audio-skip-back');
        const skipFwdBtn = player.querySelector('.audio-skip-fwd');
        const timeCounter = player.querySelector('.audio-time-counter');
        const speedBtn = player.querySelector('.audio-speed-btn');
        const loopBtn = player.querySelector('.audio-loop-btn');
        const stage = player.querySelector('.audio-waveform-stage');
        const waveCanvas = player.querySelector('.audio-waveform-canvas');
        const playhead = player.querySelector('.audio-playhead-line');
        const tooltip = player.querySelector('.audio-hover-tooltip');
        const specCanvas = player.querySelector('.audio-spectrum-canvas');
        const vuL = player.querySelector('.audio-vu-l');
        const vuR = player.querySelector('.audio-vu-r');
        const abButtons = player.querySelectorAll ? player.querySelectorAll('.audio-ab-btn') : [];

        let activeAudio = enhancedAudio;
        let isPlaying = false;
        let isLooping = true;
        let playbackRate = 1.0;
        let animId = null;

        const barCount = 72;
        const peaks = [];
        let hash = 0;
        const srcStr = options.url || '';
        for (let i = 0; i < srcStr.length; i++) {
            hash = ((hash << 5) - hash) + srcStr.charCodeAt(i);
            hash |= 0;
        }
        for (let i = 0; i < barCount; i++) {
            const seed = Math.abs(Math.sin(hash + i * 0.45));
            peaks.push(0.18 + seed * 0.78);
        }

        function drawWaveform(progressPct) {
            if (!waveCanvas || !waveCanvas.getContext) return;
            const w = waveCanvas.width = waveCanvas.clientWidth || 320;
            const h = waveCanvas.height = waveCanvas.clientHeight || 72;
            const ctx = waveCanvas.getContext('2d');
            ctx.clearRect(0, 0, w, h);

            const barWidth = Math.max(2, (w / barCount) - 2);
            const mid = h / 2;

            for (let i = 0; i < barCount; i++) {
                const x = i * (w / barCount) + 1;
                const barH = peaks[i] * (h * 0.75);
                const isPlayed = (i / barCount) <= progressPct;

                if (isPlayed) {
                    const grad = ctx.createLinearGradient(0, mid - barH / 2, 0, mid + barH / 2);
                    grad.addColorStop(0, '#818cf8');
                    grad.addColorStop(1, '#ec4899');
                    ctx.fillStyle = grad;
                } else {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
                }

                if (ctx.roundRect) {
                    ctx.beginPath();
                    ctx.roundRect(x, mid - barH / 2, barWidth, barH, 2);
                    ctx.fill();
                } else {
                    ctx.fillRect(x, mid - barH / 2, barWidth, barH);
                }
            }
        }

        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(() => drawWaveform(0));
        }

        function updateMeters() {
            if (isPlaying) {
                const baseEnergy = 0.55 + Math.random() * 0.35;
                const leftEnergy = Math.min(100, Math.max(10, (baseEnergy + (Math.random() * 0.15 - 0.07)) * 100));
                const rightEnergy = Math.min(100, Math.max(10, (baseEnergy + (Math.random() * 0.15 - 0.07)) * 100));
                if (vuL && vuL.style) vuL.style.width = `${leftEnergy}%`;
                if (vuR && vuR.style) vuR.style.width = `${rightEnergy}%`;

                if (specCanvas && specCanvas.getContext) {
                    const sw = specCanvas.width = specCanvas.clientWidth || 200;
                    const sh = specCanvas.height = specCanvas.clientHeight || 16;
                    const sctx = specCanvas.getContext('2d');
                    sctx.clearRect(0, 0, sw, sh);
                    const bands = 24;
                    const bWidth = (sw / bands) - 1.5;
                    for (let b = 0; b < bands; b++) {
                        const hFactor = Math.abs(Math.sin(Date.now() * 0.006 + b * 0.6)) * (0.3 + Math.random() * 0.7);
                        const barHeight = Math.max(2, hFactor * sh);
                        sctx.fillStyle = `hsl(${220 + b * 4}, 90%, 65%)`;
                        sctx.fillRect(b * (bWidth + 1.5), sh - barHeight, bWidth, barHeight);
                    }
                }

                if (typeof requestAnimationFrame === 'function') {
                    animId = requestAnimationFrame(updateMeters);
                }
            } else {
                if (vuL && vuL.style) vuL.style.width = '0%';
                if (vuR && vuR.style) vuR.style.width = '0%';
                if (specCanvas && specCanvas.getContext) {
                    const sctx = specCanvas.getContext('2d');
                    sctx.clearRect(0, 0, specCanvas.width, specCanvas.height);
                }
            }
        }

        function togglePlay(shouldPlay) {
            isPlaying = typeof shouldPlay === 'boolean' ? shouldPlay : !isPlaying;
            if (playBtn) {
                playBtn.innerHTML = isPlaying ? `<i class="fas fa-pause"></i>` : `<i class="fas fa-play"></i>`;
                playBtn.classList.toggle('playing', isPlaying);
            }
            if (isPlaying) {
                if (activeAudio && activeAudio.play) activeAudio.play().catch(() => {});
                if (typeof requestAnimationFrame === 'function') {
                    animId = requestAnimationFrame(updateMeters);
                }
            } else {
                if (activeAudio && activeAudio.pause) activeAudio.pause();
                if (animId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(animId);
                updateMeters();
            }
        }

        if (playBtn) playBtn.onclick = () => togglePlay();

        function onTimeUpdate() {
            const cur = activeAudio?.currentTime || 0;
            const dur = activeAudio?.duration || 1;
            const pct = Math.min(1, Math.max(0, cur / dur));
            if (playhead && playhead.style) playhead.style.left = `${pct * 100}%`;
            drawWaveform(pct);
            if (timeCounter) timeCounter.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
        }

        if (enhancedAudio && enhancedAudio.addEventListener) enhancedAudio.addEventListener('timeupdate', onTimeUpdate);
        if (originalAudio && originalAudio.addEventListener) originalAudio.addEventListener('timeupdate', onTimeUpdate);

        const onEnded = () => {
            if (isLooping) {
                if (activeAudio) activeAudio.currentTime = 0;
                if (activeAudio && activeAudio.play) activeAudio.play().catch(() => {});
            } else {
                togglePlay(false);
            }
        };
        if (enhancedAudio && enhancedAudio.addEventListener) enhancedAudio.addEventListener('ended', onEnded);
        if (originalAudio && originalAudio.addEventListener) originalAudio.addEventListener('ended', onEnded);

        if (stage && stage.addEventListener && stage.getBoundingClientRect) {
            let isScrubbing = false;
            const scrubTo = (clientX) => {
                const rect = stage.getBoundingClientRect();
                const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
                const dur = activeAudio?.duration || 1;
                if (activeAudio) activeAudio.currentTime = pct * dur;
                onTimeUpdate();
            };

            stage.addEventListener('pointerdown', (e) => {
                isScrubbing = true;
                if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId);
                scrubTo(e.clientX);
            });
            stage.addEventListener('pointermove', (e) => {
                const rect = stage.getBoundingClientRect();
                const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const dur = activeAudio?.duration || 0;
                if (tooltip && tooltip.style) {
                    tooltip.style.left = `${pct * 100}%`;
                    tooltip.textContent = formatTime(pct * dur);
                }
                if (isScrubbing) scrubTo(e.clientX);
            });
            stage.addEventListener('pointerup', () => { isScrubbing = false; });
        }

        if (skipBackBtn) {
            skipBackBtn.onclick = () => {
                if (activeAudio) activeAudio.currentTime = Math.max(0, (activeAudio.currentTime || 0) - 5);
                onTimeUpdate();
            };
        }
        if (skipFwdBtn) {
            skipFwdBtn.onclick = () => {
                if (activeAudio) activeAudio.currentTime = Math.min(activeAudio.duration || 1000, (activeAudio.currentTime || 0) + 5);
                onTimeUpdate();
            };
        }

        if (speedBtn) {
            const speeds = [0.75, 1.0, 1.25, 1.5, 2.0];
            speedBtn.onclick = () => {
                const idx = speeds.indexOf(playbackRate);
                playbackRate = speeds[(idx + 1) % speeds.length];
                speedBtn.textContent = `${playbackRate}×`;
                if (enhancedAudio) enhancedAudio.playbackRate = playbackRate;
                if (originalAudio) originalAudio.playbackRate = playbackRate;
            };
        }

        if (loopBtn) {
            loopBtn.onclick = () => {
                isLooping = !isLooping;
                loopBtn.classList.toggle('active', isLooping);
            };
        }

        function switchTrack(track) {
            const prev = activeAudio;
            const target = track === 'a' ? originalAudio : enhancedAudio;
            if (!target || target === prev) return;

            const t = prev ? (prev.currentTime || 0) : 0;
            if (prev && prev.pause) prev.pause();
            target.currentTime = t;
            activeAudio = target;
            activeAudio.playbackRate = playbackRate;

            if (isPlaying && target.play) {
                target.play().catch(() => {});
            }

            abButtons.forEach(b => b.classList.toggle('active', b.dataset.track === track));
            onTimeUpdate();
        }

        abButtons.forEach(btn => {
            btn.onclick = () => switchTrack(btn.dataset.track);
        });
    }

    // ─── CLIPBOARD IMAGE COPY ───
    window.copyImageToClipboard = async function (imageUrl) {
        if (!imageUrl) return;
        try {
            const resp = await fetch(imageUrl);
            const blob = await resp.blob();
            let pngBlob = blob;
            if (blob.type !== 'image/png') {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.src = URL.createObjectURL(blob);
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = reject;
                });
                const canvas = document.createElement('canvas');
                canvas.width = img.naturalWidth || img.width;
                canvas.height = img.naturalHeight || img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                pngBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
            }
            await navigator.clipboard.write([
                new ClipboardItem({ 'image/png': pngBlob })
            ]);
            showToast('Image copied to clipboard!', 'success');
        } catch (err) {
            console.error('Clipboard copy failed:', err);
            showToast('Could not copy image directly. Use Download instead.', 'error');
        }
    };

    // ─── CINEMA-GRADE MEDIA INSPECTOR LIGHTBOX ───
    let inspectorZoom = 1;
    let inspectorPanX = 0;
    let inspectorPanY = 0;
    let isInspectorPanning = false;
    let startPanX = 0;
    let startPanY = 0;

    function applyInspectorTransform() {
        const wrap = document.getElementById('inspectorCanvasWrap');
        const zoomDisplay = document.getElementById('inspectorZoomLevel');
        if (wrap) wrap.style.transform = `translate(${inspectorPanX}px, ${inspectorPanY}px) scale(${inspectorZoom})`;
        if (zoomDisplay) zoomDisplay.textContent = Math.round(inspectorZoom * 100) + '%';
    }

    window.openMediaInspector = function (media) {
        if (!media || !media.url) return;
        const modal = document.getElementById('mediaInspectorModal');
        if (!modal) return;

        const titleEl = document.getElementById('inspectorTitle');
        const badgeEl = document.getElementById('inspectorTypeBadge');
        const metaEl = document.getElementById('inspectorMetaBadge');
        const canvasWrap = document.getElementById('inspectorCanvasWrap');
        const downloadBtn = document.getElementById('inspectorDownloadBtn');
        const copyBtn = document.getElementById('inspectorCopyBtn');
        const quickActions = document.getElementById('inspectorQuickActions');
        const compareHint = document.getElementById('inspectorCompareHint');

        inspectorZoom = 1;
        inspectorPanX = 0;
        inspectorPanY = 0;
        applyInspectorTransform();

        const ext = (media.filename || media.url || '').split('.').pop().toLowerCase();
        const type = media.type || (['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'].includes(ext) ? 'image' :
                     (['mp4', 'mov', 'webm', 'mkv'].includes(ext) ? 'video' :
                     (['mp3', 'wav', 'flac', 'ogg', 'm4a'].includes(ext) ? 'audio' : 'file')));

        if (badgeEl) badgeEl.textContent = type.toUpperCase();
        if (titleEl) {
            titleEl.textContent = humanizeMediaName(media.filename, media.original_name);
            titleEl.title = media.original_name || media.filename || '';
        }
        const sizeStr = media.size ? formatBytes(media.size) : '';
        const dimStr = (media.width && media.height) ? `${media.width}×${media.height} • ` :
                       (media.duration ? `${formatTime(media.duration)} • ` : '');
        if (metaEl) metaEl.textContent = `${dimStr}${sizeStr} • ${ext.toUpperCase()}`;

        if (downloadBtn) {
            downloadBtn.href = media.url;
            downloadBtn.download = media.original_name || media.filename || `export.${ext}`;
        }

        if (copyBtn) {
            copyBtn.style.display = type === 'image' ? 'inline-flex' : 'none';
            copyBtn.onclick = () => window.copyImageToClipboard(media.url);
        }

        if (compareHint) {
            compareHint.style.display = (media.input_url && media.input_url !== media.url) ? 'inline' : 'none';
        }

        if (canvasWrap) {
            if (type === 'image') {
                if (media.input_url && media.input_url !== media.url) {
                    canvasWrap.innerHTML = `
                        <div class="image-compare-container" style="--split-pos: 50%;">
                            <div class="compare-mode-toolbar">
                                <button type="button" class="compare-mode-btn active" data-mode="split" title="Split Wipe"><i class="fas fa-columns"></i> Split</button>
                                <button type="button" class="compare-mode-btn" data-mode="side-by-side" title="Side by Side"><i class="fas fa-table-columns"></i> Side</button>
                                <button type="button" class="compare-mode-btn" data-mode="difference" title="X-Ray Difference"><i class="fas fa-circle-half-stroke"></i> Diff</button>
                                <button type="button" class="compare-mode-btn compare-hold-btn" title="Hold to view original (or hold Space)"><i class="fas fa-eye"></i> Hold</button>
                            </div>
                            <div class="image-compare-wrapper">
                                <img class="compare-img compare-after" src="${media.url}" alt="Processed Output" />
                                <div class="compare-before-layer">
                                    <img class="compare-img compare-before" src="${media.input_url}" alt="Original Input" />
                                </div>
                                <div class="compare-divider-line" style="left: 50%;">
                                    <div class="compare-divider-handle" tabindex="0" role="slider" aria-label="Comparison divider position" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
                                        <i class="fas fa-arrows-left-right"></i>
                                        <span class="compare-percent-tooltip">50%</span>
                                    </div>
                                </div>
                                <span class="compare-pill compare-pill-before">Original</span>
                                <span class="compare-pill compare-pill-after">Processed</span>
                            </div>
                        </div>
                    `;
                    setupCompareInteractions(canvasWrap.querySelector('.image-compare-container'));
                } else {
                    canvasWrap.innerHTML = `<img src="${media.url}" alt="Inspector Preview" />`;
                }
            } else if (type === 'video') {
                if (media.input_url && media.input_url !== media.url) {
                    canvasWrap.innerHTML = `
                        <div class="video-compare-container" style="--split-pos: 50%; max-width: 900px; margin: 0 auto;">
                            <div class="compare-mode-toolbar">
                                <button type="button" class="compare-mode-btn active" data-mode="split" title="Split Wipe"><i class="fas fa-columns"></i> Split</button>
                                <button type="button" class="compare-mode-btn" data-mode="side-by-side" title="Side by Side"><i class="fas fa-table-columns"></i> Side</button>
                                <button type="button" class="compare-mode-btn compare-hold-btn" title="Hold to view original (or hold Space)"><i class="fas fa-eye"></i> Hold</button>
                            </div>
                            <div class="video-compare-wrapper">
                                <video class="video-after" playsinline preload="auto" src="${media.url}"></video>
                                <div class="video-before-layer">
                                    <video class="video-before" playsinline preload="auto" src="${media.input_url}"></video>
                                </div>
                                <div class="compare-divider-line" style="left: 50%;">
                                    <div class="compare-divider-handle" tabindex="0" role="slider" aria-label="Comparison divider position" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
                                        <i class="fas fa-arrows-left-right"></i>
                                        <span class="compare-percent-tooltip">50%</span>
                                    </div>
                                </div>
                                <span class="compare-pill compare-pill-before">Original</span>
                                <span class="compare-pill compare-pill-after">Processed</span>
                            </div>
                            <div class="video-studio-controls">
                                <div class="video-scrub-row">
                                    <input type="range" class="video-scrub-slider" min="0" max="100" step="0.1" value="0" aria-label="Scrub video timeline" />
                                    <span class="video-time-label">0:00 / 0:00</span>
                                </div>
                                <div class="video-controls-left">
                                    <button type="button" class="video-ctrl-btn video-play-btn"><i class="fas fa-play"></i> Play</button>
                                    <button type="button" class="video-ctrl-btn video-step-back-btn" title="Step backward 1 frame (-1f)"><i class="fas fa-backward-step"></i> -1f</button>
                                    <button type="button" class="video-ctrl-btn video-step-fwd-btn" title="Step forward 1 frame (+1f)"><i class="fas fa-forward-step"></i> +1f</button>
                                    <button type="button" class="video-ctrl-btn video-loop-btn active" title="Toggle Loop"><i class="fas fa-repeat"></i> Loop</button>
                                </div>
                                <div class="video-controls-right">
                                    <button type="button" class="video-ctrl-btn video-speed-btn" data-speed="1">1.0×</button>
                                    <button type="button" class="video-ctrl-btn video-snap-btn" title="Capture high-res frame PNG"><i class="fas fa-camera"></i> Snapshot</button>
                                </div>
                            </div>
                        </div>
                    `;
                    setupVideoCompareInteractions(canvasWrap.querySelector('.video-compare-container'));
                } else {
                    canvasWrap.innerHTML = `
                        <div class="video-compare-container" style="max-width: 900px; margin: 0 auto;">
                            <div class="video-compare-wrapper">
                                <video class="video-after" playsinline preload="auto" src="${media.url}"></video>
                            </div>
                            <div class="video-studio-controls">
                                <div class="video-scrub-row">
                                    <input type="range" class="video-scrub-slider" min="0" max="100" step="0.1" value="0" aria-label="Scrub video timeline" />
                                    <span class="video-time-label">0:00 / 0:00</span>
                                </div>
                                <div class="video-controls-left">
                                    <button type="button" class="video-ctrl-btn video-play-btn"><i class="fas fa-play"></i> Play</button>
                                    <button type="button" class="video-ctrl-btn video-step-back-btn" title="Step backward 1 frame (-1f)"><i class="fas fa-backward-step"></i> -1f</button>
                                    <button type="button" class="video-ctrl-btn video-step-fwd-btn" title="Step forward 1 frame (+1f)"><i class="fas fa-forward-step"></i> +1f</button>
                                    <button type="button" class="video-ctrl-btn video-loop-btn active" title="Toggle Loop"><i class="fas fa-repeat"></i> Loop</button>
                                </div>
                                <div class="video-controls-right">
                                    <button type="button" class="video-ctrl-btn video-speed-btn" data-speed="1">1.0×</button>
                                    <button type="button" class="video-ctrl-btn video-snap-btn" title="Capture high-res frame PNG"><i class="fas fa-camera"></i> Snapshot</button>
                                </div>
                            </div>
                        </div>
                    `;
                    setupVideoCompareInteractions(canvasWrap.querySelector('.video-compare-container'));
                }
            } else if (type === 'audio') {
                canvasWrap.innerHTML = `<div class="audio-inspector-holder" style="max-width:580px; margin: 0 auto;"></div>`;
                renderStudioAudioPlayer(canvasWrap.querySelector('.audio-inspector-holder'), {
                    url: media.url,
                    inputUrl: media.input_url,
                    filename: media.filename,
                    originalName: media.original_name
                });
            }
        }

        if (quickActions) {
            quickActions.innerHTML = '';
            if (type === 'video' || type === 'audio') {
                const studioBtn = document.createElement('a');
                studioBtn.className = 'inspector-action-chip';
                studioBtn.href = `/studio?load=${encodeURIComponent(media.filename || '')}`;
                studioBtn.innerHTML = `<i class="fas fa-layer-group"></i> <span>Open in Studio</span>`;
                quickActions.appendChild(studioBtn);
            }
            if (type === 'image') {
                const imgEditBtn = document.createElement('a');
                imgEditBtn.className = 'inspector-action-chip';
                imgEditBtn.href = `/image?load=${encodeURIComponent(media.filename || '')}`;
                imgEditBtn.innerHTML = `<i class="fas fa-paintbrush"></i> <span>Open in Image Editor</span>`;
                quickActions.appendChild(imgEditBtn);

                const studioBtn = document.createElement('a');
                studioBtn.className = 'inspector-action-chip';
                studioBtn.href = `/studio?load=${encodeURIComponent(media.filename || '')}`;
                studioBtn.innerHTML = `<i class="fas fa-layer-group"></i> <span>Send to Studio</span>`;
                quickActions.appendChild(studioBtn);
            }
        }

        modal.classList.remove('hidden');
    };

    window.closeMediaInspector = function () {
        const modal = document.getElementById('mediaInspectorModal');
        if (modal) modal.classList.add('hidden');
        const video = modal?.querySelector('video');
        if (video) video.pause();
        const audio = modal?.querySelector('audio');
        if (audio) audio.pause();
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

        // Inspect active media / pending media
        document.getElementById('inspectActiveMediaBtn')?.addEventListener('click', () => {
            if (activeMedia) window.openMediaInspector(activeMedia);
        });
        document.getElementById('previewPendingBtn')?.addEventListener('click', () => {
            if (activeMedia) window.openMediaInspector(activeMedia);
        });

        // Inspector modal controls
        document.getElementById('inspectorCloseBtn')?.addEventListener('click', window.closeMediaInspector);
        document.getElementById('inspectorBackdrop')?.addEventListener('click', window.closeMediaInspector);

        document.getElementById('inspectorZoomIn')?.addEventListener('click', () => {
            inspectorZoom = Math.min(8, inspectorZoom * 1.25);
            applyInspectorTransform();
        });
        document.getElementById('inspectorZoomOut')?.addEventListener('click', () => {
            inspectorZoom = Math.max(0.25, inspectorZoom / 1.25);
            applyInspectorTransform();
        });
        document.getElementById('inspectorZoomReset')?.addEventListener('click', () => {
            inspectorZoom = 1;
            inspectorPanX = 0;
            inspectorPanY = 0;
            applyInspectorTransform();
        });

        // Inspector stage background selector
        document.querySelectorAll('#inspectorStageSelector .stage-btn').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                document.querySelectorAll('#inspectorStageSelector .stage-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const viewport = document.getElementById('inspectorViewport');
                if (!viewport) return;
                viewport.classList.remove('stage-checker', 'stage-dark', 'stage-light', 'stage-accent');
                const stage = btn.dataset.stage;
                if (stage && stage !== 'checker') viewport.classList.add(`stage-${stage}`);
            };
        });

        // Inspector pan & zoom interaction
        const inspectorViewport = document.getElementById('inspectorViewport');
        const inspectorWrap = document.getElementById('inspectorCanvasWrap');
        if (inspectorViewport && inspectorWrap) {
            inspectorViewport.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.88 : 1.14;
                inspectorZoom = Math.max(0.25, Math.min(8, inspectorZoom * delta));
                applyInspectorTransform();
            }, { passive: false });

            inspectorWrap.addEventListener('pointerdown', (e) => {
                if (e.target.closest('.compare-divider-line') || e.target.closest('.compare-mode-toolbar') || e.target.closest('audio') || e.target.closest('video')) return;
                isInspectorPanning = true;
                startPanX = e.clientX - inspectorPanX;
                startPanY = e.clientY - inspectorPanY;
                inspectorWrap.classList.add('panning');
                inspectorWrap.setPointerCapture?.(e.pointerId);
            });
            inspectorWrap.addEventListener('pointermove', (e) => {
                if (!isInspectorPanning) return;
                inspectorPanX = e.clientX - startPanX;
                inspectorPanY = e.clientY - startPanY;
                applyInspectorTransform();
            });
            const endPan = (e) => {
                if (isInspectorPanning) {
                    isInspectorPanning = false;
                    inspectorWrap.classList.remove('panning');
                    try { inspectorWrap.releasePointerCapture?.(e.pointerId); } catch (_) {}
                }
            };
            inspectorWrap.addEventListener('pointerup', endPan);
            inspectorWrap.addEventListener('pointercancel', endPan);
        }

        // Global Clipboard Paste: Upload image files pasted anywhere on the page
        window.addEventListener('paste', async (e) => {
            if (!e.clipboardData || !e.clipboardData.items) return;
            const items = e.clipboardData.items;
            for (let i = 0; i < items.length; i++) {
                if (items[i].kind === 'file') {
                    const file = items[i].getAsFile();
                    if (file) {
                        e.preventDefault();
                        showToast(`Pasting ${file.name || 'image'} from clipboard...`, 'info');
                        await uploadMediaFile(file);
                        break;
                    }
                }
            }
        });

        // Global Keyboard Hotkeys
        window.addEventListener('keydown', (e) => {
            const inspectorModal = document.getElementById('mediaInspectorModal');
            // Escape closes inspector first, then shortcuts modal
            if (e.key === 'Escape') {
                if (inspectorModal && !inspectorModal.classList.contains('hidden')) {
                    window.closeMediaInspector();
                    return;
                }
                if (shortcutsModal && shortcutsModal.classList.contains('active')) {
                    shortcutsModal.classList.remove('active');
                    return;
                }
            }
            // Space hold to compare (when compare container exists)
            if (e.code === 'Space' && document.activeElement !== chatInput) {
                const activeCompare = document.querySelector('.image-compare-container');
                if (activeCompare) {
                    activeCompare.classList.add('mode-hold-original');
                }
            }
            // Zoom hotkeys in inspector
            if (inspectorModal && !inspectorModal.classList.contains('hidden')) {
                if (e.key === '+' || e.key === '=') {
                    e.preventDefault();
                    inspectorZoom = Math.min(8, inspectorZoom * 1.25);
                    applyInspectorTransform();
                    return;
                }
                if (e.key === '-' || e.key === '_') {
                    e.preventDefault();
                    inspectorZoom = Math.max(0.25, inspectorZoom / 1.25);
                    applyInspectorTransform();
                    return;
                }
                if (e.key === '0') {
                    e.preventDefault();
                    inspectorZoom = 1;
                    inspectorPanX = 0;
                    inspectorPanY = 0;
                    applyInspectorTransform();
                    return;
                }
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

        window.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                document.querySelectorAll('.image-compare-container.mode-hold-original').forEach(el => {
                    el.classList.remove('mode-hold-original');
                });
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
                if (session) {
                    const ctrl = activeUploadAbortControllers.get(session);
                    if (ctrl) ctrl.abort();
                    pendingUploads.delete(session);
                }
                if (uploadProgressContainer) uploadProgressContainer.style.display = 'none';
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

        // 1. Extract and protect multiline code blocks (```lang ... ```)
        const codeBlocks = [];
        let src = String(text).replace(/```([a-zA-Z0-9_\-\.\+]*)\r?\n([\s\S]*?)```/g, (match, lang, code) => {
            const index = codeBlocks.length;
            const cleanLang = (lang || 'code').trim().toUpperCase();
            const escapedCode = escapeHtml(code.trim());
            codeBlocks.push(
                `<div class="code-block-wrapper">` +
                    `<div class="code-block-header">` +
                        `<span class="code-lang-tag"><i class="fas fa-terminal" aria-hidden="true"></i> ${escapeHtml(cleanLang)}</span>` +
                        `<button type="button" class="copy-code-btn" onclick="window.copyCodeSnippet(this)" title="Copy snippet" aria-label="Copy snippet">` +
                            `<i class="fas fa-copy" aria-hidden="true"></i> <span>Copy</span>` +
                        `</button>` +
                    `</div>` +
                    `<pre class="code-block-pre"><code>${escapedCode}</code></pre>` +
                `</div>`
            );
            return `__CODE_BLOCK_${index}__`;
        });

        // 2. Escape HTML for safe rendering
        let html = escapeHtml(src);

        // 3. Headings (###, ##, #)
        html = html.replace(/^(?:&gt;|\>)?\s*###\s+(.+)$/gm, '<h4 class="agent-md-h4">$1</h4>');
        html = html.replace(/^(?:&gt;|\>)?\s*##\s+(.+)$/gm, '<h3 class="agent-md-h3">$1</h3>');
        html = html.replace(/^(?:&gt;|\>)?\s*#\s+(.+)$/gm, '<h2 class="agent-md-h2">$1</h2>');

        // 4. Blockquotes (> text)
        html = html.replace(/^\&gt;\s+(.+)$/gm, '<blockquote class="agent-md-quote"><i class="fas fa-quote-left quote-glyph" aria-hidden="true"></i> $1</blockquote>');

        // 5. Unordered lists (- item or * item)
        html = html.replace(/^(?:[-*]|\&bull;)\s+(.+)$/gm, '<li class="agent-md-li">$1</li>');
        html = html.replace(/(<li class="agent-md-li">[\s\S]*?<\/li>)+/g, '<ul class="agent-md-ul">$&</ul>');

        // 6. Ordered lists (1. item)
        html = html.replace(/^\d+\.\s+(.+)$/gm, '<li class="agent-md-oli">$1</li>');
        html = html.replace(/(<li class="agent-md-oli">[\s\S]*?<\/li>)+/g, '<ol class="agent-md-ol">$&</ol>');

        // 7. Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // 8. Italic: *text*
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // 9. Strikethrough: ~~text~~
        html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
        // 10. Keycaps: [[Key]]
        html = html.replace(/\[\[(.*?)\]\]/g, '<kbd class="agent-md-kbd">$1</kbd>');
        // 11. Inline Code: `code`
        html = html.replace(/`(.*?)`/g, '<code class="inline-code">$1</code>');
        // 12. Safe Links: [title](url)
        html = html.replace(/\[(.*?)\]\(((?:https?:\/\/|\/|#)[^\s<>"']+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="agent-md-link">$1 <i class="fas fa-arrow-up-right-from-square" aria-hidden="true"></i></a>');

        // 13. Linebreaks (clean up around block elements)
        html = html.replace(/\n/g, '<br>');
        html = html.replace(/<br>\s*<(ul|ol|li|h2|h3|h4|blockquote|\/ul|\/ol|\/blockquote)/g, '<$1');
        html = html.replace(/<\/(ul|ol|li|h2|h3|h4|blockquote)><br>/g, '</$1>');

        // 14. Restore protected code blocks
        html = html.replace(/__CODE_BLOCK_(\d+)__/g, (m, idx) => codeBlocks[Number(idx)] || '');

        return html;
    }

    // ─── GLOBAL UTILITY ACTIONS ───
    window.copyCodeSnippet = async function (btn) {
        try {
            const wrapper = btn && typeof btn.closest === 'function' ? btn.closest('.code-block-wrapper') : null;
            if (!wrapper) return;
            const codeEl = wrapper.querySelector('code');
            const text = codeEl ? (codeEl.textContent || codeEl.innerText || '') : '';
            if (navigator && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                await navigator.clipboard.writeText(text);
            }
            const span = wrapper.querySelector('.copy-code-btn span');
            const icon = wrapper.querySelector('.copy-code-btn i');
            if (span) span.textContent = 'Copied!';
            if (icon) icon.className = 'fas fa-check';
            btn.classList.add('copied');
            setTimeout(() => {
                if (span) span.textContent = 'Copy';
                if (icon) icon.className = 'fas fa-copy';
                btn.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Failed to copy code snippet:', err);
        }
    };

    window.copyAgentReply = async function (btn) {
        try {
            const bubble = btn && typeof btn.closest === 'function' ? btn.closest('.message-bubble') : null;
            if (!bubble) return;
            const textEl = bubble.querySelector('.agent-text');
            const text = textEl ? (textEl.innerText || textEl.textContent || '') : '';
            if (navigator && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                await navigator.clipboard.writeText(text);
            }
            const span = btn.querySelector('span');
            const icon = btn.querySelector('i');
            if (span) span.textContent = 'Copied!';
            if (icon) icon.className = 'fas fa-check';
            btn.classList.add('copied');
            setTimeout(() => {
                if (span) span.textContent = 'Copy text';
                if (icon) icon.className = 'fas fa-copy';
                btn.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Failed to copy reply text:', err);
        }
    };

    // ─── BOOTSTRAP ON LOAD ───
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
