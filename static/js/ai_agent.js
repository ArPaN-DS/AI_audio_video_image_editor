/**
 * Studio assistant drawer — chat, "@" skills and edit results for the selected studio media.
 */

document.addEventListener('DOMContentLoaded', () => {
    const btnToggleAiAgent = document.getElementById('btnToggleAiAgent');
    const aiAgentDrawer = document.getElementById('aiAgentDrawer');
    const btnCloseAgentDrawer = document.getElementById('btnCloseAgentDrawer');
    const agentChatHistory = document.getElementById('agentChatHistory');
    const agentInputText = document.getElementById('agentInputText');
    const btnSendAgent = document.getElementById('btnSendAgent');
    const btnMicAgent = document.getElementById('btnMicAgent');
    const agentQuickChips = document.getElementById('agentQuickChips');
    const agentActiveTarget = document.getElementById('agentActiveTarget');

    if (!aiAgentDrawer) return;

    let isSending = false;
    const conversationHistory = [];
    const memorySessionId = 'studio_' + Date.now().toString(36);
    if (btnMicAgent && window.LocalVoiceInput) {
        const voice = window.LocalVoiceInput.create({
            onState(state) {
                btnMicAgent.classList.toggle('recording', state === 'recording');
                btnMicAgent.disabled = state === 'transcribing';
                btnMicAgent.title = state === 'recording' ? 'Recording locally — click to transcribe'
                    : state === 'transcribing' ? 'Transcribing locally...' : 'Local voice input';
            },
            onTranscript(transcript) {
                agentInputText.value = [agentInputText.value, transcript].filter(Boolean).join(' ');
            },
            onError(message) { appendBotMessage(message); }
        });
        btnMicAgent.disabled = !voice.supported;
        btnMicAgent.addEventListener('click', () => voice.toggle());
    } else if (btnMicAgent) {
        btnMicAgent.disabled = true;
    }

    // Dynamic Context Updater — Tracks Active Target File in Studio
    function updateActiveFileDisplay() {
        if (!agentActiveTarget) return;

        let activeName = null;
        const activeMedia = window.studioCore?.getActiveMedia();
        if (activeMedia) {
            activeName = activeMedia.name || activeMedia.filename || activeMedia.id;
        }

        if (activeName) {
            agentActiveTarget.textContent = activeName;
            agentActiveTarget.classList.add('has-file');
        } else {
            agentActiveTarget.textContent = 'No media selected';
            agentActiveTarget.classList.remove('has-file');
        }
    }

    // Monitor studio state updates
    setInterval(updateActiveFileDisplay, 1000);

    // Toggle Drawer Open / Close
    if (btnToggleAiAgent) {
        btnToggleAiAgent.setAttribute('aria-expanded', String(aiAgentDrawer.classList.contains('open')));
        btnToggleAiAgent.addEventListener('click', () => {
            aiAgentDrawer.classList.toggle('open');
            btnToggleAiAgent.setAttribute('aria-expanded', String(aiAgentDrawer.classList.contains('open')));
            if (aiAgentDrawer.classList.contains('open')) {
                agentInputText.focus();
                updateActiveFileDisplay();
            }
        });
    }

    if (btnCloseAgentDrawer) {
        btnCloseAgentDrawer.addEventListener('click', () => {
            aiAgentDrawer.classList.remove('open');
            btnToggleAiAgent?.setAttribute('aria-expanded', 'false');
        });
    }

    // Quick chips click handler
    if (agentQuickChips) {
        agentQuickChips.addEventListener('click', (e) => {
            const btn = e.target.closest('.chip-btn');
            if (btn) {
                const prompt = btn.getAttribute('data-prompt');
                if (prompt) {
                    agentInputText.value = prompt;
                    sendAgentMessage();
                }
            }
        });
    }

    // Send button & enter key
    if (btnSendAgent) {
        btnSendAgent.addEventListener('click', sendAgentMessage);
    }

    if (agentInputText) {
        agentInputText.addEventListener('keydown', (e) => {
            // The skills popup handles Enter first (capture phase) and marks the event as handled.
            if (e.key === 'Enter' && !e.defaultPrevented && !e.isComposing) {
                e.preventDefault();
                sendAgentMessage();
            }
        });
    }

    // ─── Skills: "@" autocomplete + library ───
    const studioMediaType = () => {
        const media = window.studioCore?.getActiveMedia?.();
        if (!media) return null;
        return media.type || (media.has_video ? 'video' : media.has_audio ? 'audio' : 'image');
    };
    const btnAgentSkills = document.getElementById('btnAgentSkills');
    const openSkillsLibrary = () => window.AgentSkills?.openLibrary({
        getMediaType: studioMediaType,
        returnFocus: btnAgentSkills || agentInputText,
        onUse: skillId => window.AgentSkills.insertSkill(agentInputText, skillId)
    });
    if (window.AgentSkills && agentInputText) {
        agentInputText.parentNode?.classList?.add('agent-skills-host');
        window.AgentSkills.attachAutocomplete(agentInputText, {
            getMediaType: studioMediaType,
            onOpenLibrary: openSkillsLibrary
        });
    }
    btnAgentSkills?.addEventListener('click', openSkillsLibrary);

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'agent-message user-msg';
        msgDiv.innerHTML = `
            <div class="msg-bubble"><p>${escapeHtml(text)}</p></div>
            <div class="agent-avatar"><i class="fa-solid fa-user"></i></div>
        `;
        agentChatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendBotMessage(replyText, thoughtText = '', executionSteps = [], outputUrl = null, artifacts = []) {
        executionSteps = Array.isArray(executionSteps) ? executionSteps.filter(step => step && typeof step === 'object') : [];
        artifacts = Array.isArray(artifacts)
            ? artifacts.filter(item => item && typeof item.output_url === 'string' && /^\/(?:processed|media)\//.test(item.output_url))
            : [];
        replyText = typeof replyText === 'string' ? replyText : '';
        thoughtText = typeof thoughtText === 'string' ? thoughtText : '';
        outputUrl = typeof outputUrl === 'string' && /^\/(?:processed|media)\//.test(outputUrl) ? outputUrl : null;
        const msgDiv = document.createElement('div');
        msgDiv.className = 'agent-message bot-msg';
        const transcript = executionSteps?.find(step => step.tool === 'transcribe_audio' && step.status === 'success')?.data;
        const transcriptText = transcript?.text || transcript?.segments?.map(segment => segment.text).join(' ') || '';
        const transcriptExports = (transcript?.exports || []).filter(item => /^\/processed\/[^/?#]+\.(txt|srt|vtt)$/.test(item.url));
        const transcriptHtml = transcript ? `<div class="agent-transcript">
            <strong>Speech Transcript</strong><p>${escapeHtml(transcriptText || 'No speech detected.')}</p>
            <button type="button" class="btn btn-secondary btn-sm transcript-copy-btn">Copy transcript</button>
            ${transcriptExports.map(item => `<a class="btn btn-secondary btn-sm" href="${escapeHtml(item.url)}" download>${escapeHtml(item.format)}</a>`).join('')}
        </div>` : '';

        let stepsHtml = '';
        if (executionSteps && executionSteps.length > 0) {
            stepsHtml = '<div class="agent-steps-container">';
            executionSteps.forEach(step => {
                const icon = step.status === 'success' ? 'fa-circle-check text-success' : 'fa-circle-exclamation text-danger';
                stepsHtml += `
                    <div class="agent-step-item">
                        <i class="fa-solid ${icon}"></i>
                        <span><strong>${escapeHtml(step.tool)}:</strong> ${escapeHtml(step.message)}</span>
                    </div>
                `;
            });
            stepsHtml += '</div>';
        }

        // Accordion for AI Reasoning (Clean & Mature UI)
        let thoughtHtml = '';
        if (thoughtText) {
            thoughtHtml = `
                <details class="agent-reasoning-accordion">
                    <summary><i class="fa-solid fa-brain"></i> AI Reasoning & Analysis</summary>
                    <div class="reasoning-content">${escapeHtml(thoughtText)}</div>
                </details>
            `;
        }

        let previewHtml = '';
        if (outputUrl) {
            previewHtml = `
                <div class="agent-output-preview">
                    <button class="btn btn-secondary btn-sm btn-preview-media" data-url="${escapeHtml(outputUrl)}">
                        <i class="fa-solid fa-play"></i> Preview
                    </button>
                    <a href="${escapeHtml(outputUrl)}" download class="btn btn-primary btn-sm">
                        <i class="fa-solid fa-download"></i> Download
                    </a>
                    ${artifacts.map(item => `<a href="${escapeHtml(item.output_url)}" download class="btn btn-secondary btn-sm">
                        <i class="fa-solid fa-image"></i> ${item.tool === 'extract_frame' ? 'Thumbnail' : 'Extra file'}
                    </a>`).join('')}
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="agent-avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="msg-bubble">
                ${thoughtHtml}
                <p class="reply-text">${escapeHtml(replyText)}</p>
                ${stepsHtml}
                ${transcriptHtml}
                ${previewHtml}
            </div>
        `;

        agentChatHistory.appendChild(msgDiv);
        msgDiv.querySelector('.transcript-copy-btn')?.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(transcriptText);
            } catch {
                appendBotMessage('Clipboard is unavailable. Download the TXT transcript instead.');
            }
        });

        // Bind preview button click
        const previewBtn = msgDiv.querySelector('.btn-preview-media');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => {
                const url = previewBtn.getAttribute('data-url');
                if (window.studioCore) {
                    window.studioCore.loadProcessedMedia(url).catch(() => appendBotMessage('The edited file is ready, but its studio preview could not be opened.'));
                } else {
                    window.open(url, '_blank');
                }
            });
        }

        scrollToBottom();
    }

    let copilotLoadingTimer = null;
    let copilotElapsedSec = 0;

    function appendLoadingIndicator() {
        copilotElapsedSec = 1;
        const msgDiv = document.createElement('div');
        msgDiv.className = 'agent-message bot-msg loading-msg';
        msgDiv.id = 'agentLoadingIndicator';
        msgDiv.innerHTML = `
            <div class="agent-avatar copilot-loading-avatar" aria-hidden="true"><i class="fa-solid fa-wand-magic-sparkles fa-pulse"></i></div>
            <div class="msg-bubble copilot-loading-bubble" role="status" aria-live="polite">
                <div class="copilot-loading-head">
                    <span class="copilot-loading-title" id="copilotLiveTitle">Working on your request</span>
                    <span class="copilot-loading-timer" id="copilotLiveTimer">00:01</span>
                </div>
                <div class="copilot-loading-bars" aria-hidden="true">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
                <p class="loading-text copilot-loading-text" id="copilotLiveText">Preparing the edit plan.</p>
            </div>
        `;
        agentChatHistory.appendChild(msgDiv);
        scrollToBottom();

        if (copilotLoadingTimer) clearInterval(copilotLoadingTimer);
        copilotLoadingTimer = setInterval(() => {
            copilotElapsedSec++;
            const tEl = document.getElementById('copilotLiveTimer');
            const txtEl = document.getElementById('copilotLiveText');
            if (tEl) {
                const m = Math.floor(copilotElapsedSec / 60).toString().padStart(2, '0');
                const s = (copilotElapsedSec % 60).toString().padStart(2, '0');
                tEl.textContent = `${m}:${s}`;
            }
            if (txtEl) {
                if (copilotElapsedSec >= 3 && copilotElapsedSec < 7) {
                    txtEl.textContent = "Applying the edits.";
                } else if (copilotElapsedSec >= 7) {
                    txtEl.textContent = "Finishing the result.";
                }
            }
        }, 1000);
    }

    function removeLoadingIndicator() {
        if (copilotLoadingTimer) {
            clearInterval(copilotLoadingTimer);
            copilotLoadingTimer = null;
        }
        const loading = document.getElementById('agentLoadingIndicator');
        if (loading) loading.remove();
    }

    function scrollToBottom() {
        agentChatHistory.scrollTop = agentChatHistory.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    async function sendAgentMessage() {
        const text = agentInputText.value.trim();
        if (!text || isSending) return;
        isSending = true;
        if (btnSendAgent) btnSendAgent.disabled = true;

        agentInputText.value = '';
        appendUserMessage(text);
        appendLoadingIndicator();

        // Get active file name if available from studio state
        const core = window.studioCore;
        const project = core?.project;
        const activeMedia = core?.getActiveMedia();
        const activeFilename = activeMedia?.filename || activeMedia?.id || null;

        try {
            const response = await fetch('/api/agent/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    filename: activeFilename,
                    history: conversationHistory.slice(-20),
                    session_id: memorySessionId,
                    context: {
                        active_workspace: project?.workspace || 'combo',
                        type: activeMedia?.type || (activeMedia?.has_video ? 'video' : activeMedia?.has_audio ? 'audio' : activeMedia ? 'image' : null),
                        duration: activeMedia?.duration
                    }
                })
            });

            const data = await response.json();
            removeLoadingIndicator();

            if (['success', 'partial', 'failed'].includes(data.status)) {
                conversationHistory.push({ role: 'user', content: text }, { role: 'assistant', content: data.reply || '' });
                appendBotMessage(
                    data.reply,
                    data.thought,
                    data.execution_results,
                    data.output_url,
                    data.artifacts
                );

                // Auto-refresh waveform or video/image preview if an edit produced a file
                const edited = Array.isArray(data.execution_results) && data.execution_results.some(step => step && step.status === 'success'
                    && !['inspect_media', 'transcribe_audio'].includes(step.tool));
                if (edited && data.output_url && core && core.project === project) {
                    try {
                        await core.loadProcessedMedia(data.output_url, core.getActiveMedia()?.id === activeMedia?.id);
                    } catch {
                        appendBotMessage('The edited file is ready to download, but its studio preview could not be imported.');
                    }
                }
            } else {
                appendBotMessage(`Error: ${data.error || 'Failed to process request.'}`);
            }

        } catch (err) {
            removeLoadingIndicator();
            appendBotMessage(`${(window.APP_BRAND && window.APP_BRAND.assistant) || 'The assistant'} could not be reached (${String(err && err.message || 'network error').replace(/[.\s]+$/, '')}). Check that the studio is running and try again.`);
        } finally {
            isSending = false;
            if (btnSendAgent) btnSendAgent.disabled = false;
        }
    }
});
