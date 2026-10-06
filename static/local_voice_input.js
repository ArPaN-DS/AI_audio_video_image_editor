(function () {
    'use strict';

    const formats = [
        'audio/webm;codecs=opus', 'audio/webm',
        'audio/ogg;codecs=opus', 'audio/ogg', 'audio/mp4'
    ];

    function create({ onTranscript, onState, onError, onAudioLevel } = {}) {
        let mimeType = null;
        try {
            mimeType = formats.find(format => window.MediaRecorder?.isTypeSupported(format)) || null;
        } catch (_) {}

        const supported = Boolean(mimeType && window.navigator?.mediaDevices?.getUserMedia
            && window.Blob && window.FormData && window.fetch && window.AbortController);
        let state = 'idle';
        let disposed = false;
        let requestingPermission = false;
        let generation = 0;
        let stream = null;
        let recorder = null;
        let recordingTimer = null;
        let requestController = null;
        let audioCtx = null;
        let analyser = null;
        let animFrame = null;

        function notify(callback, value) {
            if (disposed || typeof callback !== 'function') return;
            try { callback(value); } catch (_) {}
        }

        function setState(value) {
            if (state === value) return;
            state = value;
            notify(onState, value);
        }

        function isCurrent(token) {
            return !disposed && token === generation;
        }

        function releaseTracks(mediaStream) {
            mediaStream.getTracks().forEach(track => {
                try { track.stop(); } catch (_) {}
            });
        }

        function releaseStream() {
            if (!stream) return;
            releaseTracks(stream);
            stream = null;
        }

        function clearRecordingTimer() {
            if (recordingTimer === null) return;
            window.clearTimeout(recordingTimer);
            recordingTimer = null;
        }

        function cleanup() {
            clearRecordingTimer();
            if (animFrame) {
                window.cancelAnimationFrame(animFrame);
                animFrame = null;
            }
            if (audioCtx) {
                try { audioCtx.close(); } catch (_) {}
                audioCtx = null;
                analyser = null;
            }
            if (recorder) {
                recorder.ondataavailable = null;
                recorder.onstop = null;
                recorder.onerror = null;
                try {
                    if (recorder.state !== 'inactive') recorder.stop();
                } catch (_) {}
                recorder = null;
            }
            releaseStream();
            if (requestController) {
                requestController.abort();
                requestController = null;
            }
        }

        function fail(message, token) {
            if (!isCurrent(token)) return;
            generation++;
            cleanup();
            setState('idle');
            notify(onError, message);
        }

        async function transcribe(chunks, recordingMime, token) {
            try {
                const audio = new window.Blob(chunks, { type: recordingMime });
                if (!audio.size) {
                    fail('No audio was recorded. Please try again.', token);
                    return;
                }
                const extension = {
                    'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a'
                }[recordingMime.split(';')[0].toLowerCase()];
                if (!extension) {
                    fail('This recording format is not supported for local dictation.', token);
                    return;
                }
                const formData = new window.FormData();
                formData.append('file', audio, `dictation.${extension}`);
                requestController = new window.AbortController();
                const response = await window.fetch('/ai/transcribe', {
                    method: 'POST', body: formData, signal: requestController.signal,
                    mode: 'same-origin', credentials: 'same-origin', redirect: 'error'
                });
                if (!isCurrent(token)) return;
                if (!response.ok) throw new Error();
                const data = await response.json();
                if (!isCurrent(token)) return;
                if (!data || data.available === false || data.error) {
                    fail('Local transcription is unavailable. Please try again later.', token);
                    return;
                }
                const segmentText = Array.isArray(data.segments) ? data.segments
                    .map(segment => typeof segment?.text === 'string' ? segment.text.trim() : '')
                    .filter(Boolean).join(' ') : '';
                const transcript = [data.full_text, data.text, segmentText]
                    .find(text => typeof text === 'string' && text.trim())?.trim() || '';
                if (!transcript) {
                    fail('No speech was detected. Please try again.', token);
                    return;
                }
                requestController = null;
                setState('idle');
                notify(onTranscript, transcript);
            } catch (_) {
                fail('Local transcription failed. Please try again.', token);
            }
        }

        function stopRecording() {
            if (disposed || state !== 'recording' || !recorder) return;
            const token = generation;
            const activeRecorder = recorder;
            clearRecordingTimer();
            setState('transcribing');
            if (!isCurrent(token)) return;
            try {
                activeRecorder.stop();
                releaseStream();
            } catch (_) {
                fail('Could not finish the audio recording. Please try again.', token);
            }
        }

        async function toggle() {
            if (disposed) return;
            if (!supported) {
                notify(onError, 'Local dictation is not supported in this browser. Please type your command.');
                return;
            }
            if (state === 'recording') {
                stopRecording();
                return;
            }
            if (requestingPermission || state === 'transcribing') return;

            requestingPermission = true;
            const token = ++generation;
            try {
                const mediaStream = await window.navigator.mediaDevices.getUserMedia({ audio: true });
                if (!isCurrent(token)) {
                    releaseTracks(mediaStream);
                    return;
                }
                stream = mediaStream;
                recorder = new window.MediaRecorder(stream, { mimeType });
                const activeRecorder = recorder;
                const chunks = [];
                activeRecorder.ondataavailable = event => {
                    if (isCurrent(token) && event.data?.size) chunks.push(event.data);
                };
                activeRecorder.onerror = () => fail('Audio recording failed. Please try again.', token);
                activeRecorder.onstop = () => {
                    if (!isCurrent(token)) return;
                    clearRecordingTimer();
                    releaseStream();
                    activeRecorder.ondataavailable = null;
                    activeRecorder.onstop = null;
                    activeRecorder.onerror = null;
                    recorder = null;
                    setState('transcribing');
                    if (isCurrent(token)) transcribe(chunks, activeRecorder.mimeType || mimeType, token);
                };
                activeRecorder.start();
                if (!isCurrent(token)) return;

                try {
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                    if (AudioContextClass) {
                        audioCtx = new AudioContextClass();
                        const sourceNode = audioCtx.createMediaStreamSource(stream);
                        analyser = audioCtx.createAnalyser();
                        analyser.fftSize = 64;
                        sourceNode.connect(analyser);
                        const freqData = new Uint8Array(analyser.frequencyBinCount);
                        const pumpLevel = () => {
                            if (!isCurrent(token) || state !== 'recording') return;
                            analyser.getByteFrequencyData(freqData);
                            let total = 0;
                            for (let i = 0; i < freqData.length; i++) total += freqData[i];
                            const avg = total / (freqData.length * 255);
                            notify(onAudioLevel, avg);
                            animFrame = window.requestAnimationFrame(pumpLevel);
                        };
                        pumpLevel();
                    }
                } catch (_) {}

                setState('recording');
                if (isCurrent(token)) recordingTimer = window.setTimeout(stopRecording, 60000);
            } catch (error) {
                const message = ['NotAllowedError', 'PermissionDeniedError', 'SecurityError'].includes(error?.name)
                    ? 'Microphone access was denied. Allow access or type your command.'
                    : ['NotFoundError', 'DevicesNotFoundError'].includes(error?.name)
                        ? 'No microphone was found. Connect one or type your command.'
                        : 'Could not start audio recording. Please try again.';
                fail(message, token);
            } finally {
                requestingPermission = false;
            }
        }

        function dispose() {
            if (disposed) return;
            disposed = true;
            generation++;
            window.removeEventListener?.('pagehide', dispose);
            cleanup();
        }

        window.addEventListener?.('pagehide', dispose);
        notify(onState, state);
        return { toggle, dispose, supported };
    }

    window.LocalVoiceInput = { create };
})();
