const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((accept, decline) => { resolve = accept; reject = decline; });
    return { promise, resolve, reject };
}

function createHarness(options = {}) {
    const states = [];
    const transcripts = [];
    const errors = [];
    const recorders = [];
    const timers = new Map();
    const requests = [];
    const windowListeners = new Map();
    let permissionCalls = 0;
    let timerId = 0;
    const track = { stops: 0, stop() { this.stops++; } };
    const stream = { getTracks: () => [track] };
    const pendingPermission = deferred();
    class Recorder {
        static isTypeSupported(type) { return type === (options.mimeType || 'audio/webm;codecs=opus'); }
        constructor(mediaStream, configuration) {
            if (options.constructorError) throw new Error('SECRET recorder internals');
            assert.equal(mediaStream, stream);
            this.mimeType = configuration.mimeType;
            this.state = 'inactive';
            this.stops = 0;
            recorders.push(this);
        }
        start() {
            if (options.startError) throw new Error('SECRET start internals');
            this.state = 'recording';
        }
        stop() {
            this.stops++;
            this.state = 'inactive';
            if (options.stopError) throw new Error('SECRET stop internals');
            if (!options.delayedStop) this.finishStop();
        }
        finishStop() {
            if (!options.emptyAudio) this.ondataavailable?.({ data: new Blob(['voice'], { type: this.mimeType }) });
            this.onstop?.();
        }
    }
    class UploadForm {
        constructor() { this.entries = []; }
        append(...args) { this.entries.push(args); }
    }
    const window = {
        addEventListener(name, callback) {
            if (!windowListeners.has(name)) windowListeners.set(name, new Set());
            windowListeners.get(name).add(callback);
        },
        removeEventListener(name, callback) { windowListeners.get(name)?.delete(callback); },
        navigator: { mediaDevices: { getUserMedia(constraints) {
            assert.equal(constraints.audio, true);
            assert.equal(Object.keys(constraints).length, 1);
            permissionCalls++;
            return pendingPermission.promise;
        } } },
        MediaRecorder: options.unsupported ? undefined : Recorder,
        Blob, FormData: UploadForm, AbortController,
        setTimeout(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; },
        clearTimeout(id) { timers.delete(id); },
        fetch(url, configuration) {
            const response = deferred();
            requests.push({ url, configuration, response });
            return response.promise;
        }
    };
    const sandbox = { window };
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync('static/local_voice_input.js', 'utf8'), sandbox);
    const controller = window.LocalVoiceInput.create({
        onState: state => states.push(state),
        onTranscript: text => transcripts.push(text),
        onError: message => errors.push(message)
    });
    return {
        window, controller, states, transcripts, errors, timers, requests, recorders, track, windowListeners,
        pagehide() {
            [...(windowListeners.get('pagehide') || [])].forEach(callback => callback());
        },
        get permissionCalls() { return permissionCalls; },
        async start() {
            const pending = controller.toggle();
            pendingPermission.resolve(stream);
            await pending;
        },
        grant() { pendingPermission.resolve(stream); },
        deny(name) { pendingPermission.reject(Object.assign(new Error('SECRET permission internals'), { name })); },
        respond(data, ok = true) { requests.at(-1).response.resolve({ ok, json: async () => data }); }
    };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

async function run() {
    const success = createHarness();
    assert.equal(success.controller.supported, true);
    await success.start();
    assert.equal(success.timers.size, 1);
    assert.equal([...success.timers.values()][0].delay, 60000);
    await success.controller.toggle();
    assert.equal(success.track.stops, 1);
    assert.equal(success.timers.size, 0);
    assert.deepEqual(success.states, ['idle', 'recording', 'transcribing']);
    const request = success.requests[0];
    assert.equal(request.url, '/ai/transcribe');
    assert.equal(request.configuration.method, 'POST');
    assert.equal(request.configuration.mode, 'same-origin');
    assert.equal(request.configuration.credentials, 'same-origin');
    assert.equal(request.configuration.redirect, 'error');
    assert.equal(request.configuration.body.entries[0][0], 'file');
    assert.equal(request.configuration.body.entries[0][2], 'dictation.webm');
    assert.equal(request.configuration.body.entries[0][1].type, 'audio/webm;codecs=opus');
    assert.equal(request.configuration.headers, undefined);
    await success.controller.toggle();
    assert.equal(success.permissionCalls, 1);
    success.respond({ available: true, full_text: '  Trim this clip  ' });
    await settle();
    assert.deepEqual(success.transcripts, ['Trim this clip']);
    assert.deepEqual(success.states, ['idle', 'recording', 'transcribing', 'idle']);
    assert.equal(success.errors.length, 0);
    assert.equal(success.recorders[0].onstop, null);
    await success.start();
    success.controller.dispose();
    assert.equal(success.track.stops, 2);
    assert.equal(success.windowListeners.get('pagehide').size, 0);

    for (const [data, expected] of [
        [{ full_text: ' Preferred ', text: 'Secondary', segments: [{ text: 'Last' }] }, 'Preferred'],
        [{ text: ' Text response ' }, 'Text response'],
        [{ full_text: ' ', text: ' Text fallback ' }, 'Text fallback'],
        [{ segments: [{ text: ' Segment one ' }, { text: 'Segment two' }] }, 'Segment one Segment two'],
        [{ full_text: null, text: 42, segments: [null, {}, { text: ' ' }, { text: 42 }, { text: ' Valid segment ' }] }, 'Valid segment']
    ]) {
        const shape = createHarness();
        await shape.start();
        await shape.controller.toggle();
        shape.respond({ available: true, ...data });
        await settle();
        assert.deepEqual(shape.transcripts, [expected]);
        assert.equal(shape.errors.length, 0);
        shape.controller.dispose();
    }

    const navigation = createHarness();
    await navigation.start();
    navigation.pagehide();
    assert.equal(navigation.track.stops, 1);
    assert.equal(navigation.recorders[0].stops, 1);
    assert.equal(navigation.timers.size, 0);
    assert.equal(navigation.requests.length, 0);
    assert.equal(navigation.windowListeners.get('pagehide').size, 0);
    await navigation.controller.toggle();
    assert.equal(navigation.permissionCalls, 1);

    const navigationPermission = createHarness();
    const navigationPending = navigationPermission.controller.toggle();
    navigationPermission.pagehide();
    navigationPermission.grant();
    await navigationPending;
    assert.equal(navigationPermission.track.stops, 1);
    assert.equal(navigationPermission.recorders.length, 0);
    assert.deepEqual(navigationPermission.states, ['idle']);

    const navigationResponse = createHarness();
    await navigationResponse.start();
    await navigationResponse.controller.toggle();
    navigationResponse.pagehide();
    assert.equal(navigationResponse.requests[0].configuration.signal.aborted, true);
    navigationResponse.respond({ available: true, text: 'Ignored after navigation' });
    await settle();
    assert.equal(navigationResponse.transcripts.length, 0);
    assert.equal(navigationResponse.errors.length, 0);

    for (const [mimeType, extension] of [['audio/webm', 'webm'], ['audio/ogg;codecs=opus', 'ogg'], ['audio/ogg', 'ogg'], ['audio/mp4', 'm4a']]) {
        const format = createHarness({ mimeType });
        await format.start();
        await format.controller.toggle();
        assert.equal(format.requests[0].configuration.body.entries[0][2], `dictation.${extension}`);
        format.respond({ available: true, full_text: 'Format works' });
        await settle();
        assert.deepEqual(format.transcripts, ['Format works']);
        format.controller.dispose();
    }

    for (const options of [{ unsupported: true }, { mimeType: 'audio/unsupported' }]) {
        const unsupported = createHarness(options);
        assert.equal(unsupported.controller.supported, false);
        await unsupported.controller.toggle();
        assert.equal(unsupported.permissionCalls, 0);
        assert.match(unsupported.errors[0], /not supported/);
    }

    for (const name of ['NotAllowedError', 'NotFoundError', 'NotReadableError']) {
        const permission = createHarness();
        const pending = permission.controller.toggle();
        await permission.controller.toggle();
        assert.equal(permission.permissionCalls, 1);
        permission.deny(name);
        await pending;
        assert.deepEqual(permission.states, ['idle']);
        assert.equal(permission.errors.length, 1);
        assert.doesNotMatch(permission.errors[0], /SECRET/);
    }

    const latePermission = createHarness();
    const pending = latePermission.controller.toggle();
    latePermission.controller.dispose();
    latePermission.controller.dispose();
    latePermission.grant();
    await pending;
    assert.equal(latePermission.track.stops, 1);
    assert.equal(latePermission.recorders.length, 0);
    assert.equal(latePermission.requests.length, 0);
    assert.deepEqual(latePermission.states, ['idle']);
    assert.equal(latePermission.errors.length, 0);

    const deniedAfterDispose = createHarness();
    const deniedPending = deniedAfterDispose.controller.toggle();
    deniedAfterDispose.controller.dispose();
    deniedAfterDispose.deny('NotAllowedError');
    await deniedPending;
    assert.equal(deniedAfterDispose.errors.length, 0);
    assert.deepEqual(deniedAfterDispose.states, ['idle']);

    const disposedRecording = createHarness();
    await disposedRecording.start();
    disposedRecording.controller.dispose();
    await disposedRecording.controller.toggle();
    assert.equal(disposedRecording.track.stops, 1);
    assert.equal(disposedRecording.recorders[0].stops, 1);
    assert.equal(disposedRecording.timers.size, 0);
    assert.equal(disposedRecording.requests.length, 0);
    assert.deepEqual(disposedRecording.states, ['idle', 'recording']);

    const delayedStop = createHarness({ delayedStop: true });
    await delayedStop.start();
    await delayedStop.controller.toggle();
    await delayedStop.controller.toggle();
    assert.equal(delayedStop.track.stops, 1);
    assert.equal(delayedStop.requests.length, 0);
    assert.equal(delayedStop.recorders[0].stops, 1);
    delayedStop.recorders[0].finishStop();
    delayedStop.respond({ available: true, full_text: 'Final audio chunk' });
    await settle();
    assert.deepEqual(delayedStop.transcripts, ['Final audio chunk']);
    assert.equal(delayedStop.track.stops, 1);

    const disposedStop = createHarness({ delayedStop: true });
    await disposedStop.start();
    await disposedStop.controller.toggle();
    disposedStop.controller.dispose();
    disposedStop.recorders[0].finishStop();
    assert.equal(disposedStop.requests.length, 0);
    assert.equal(disposedStop.track.stops, 1);

    const timed = createHarness();
    await timed.start();
    [...timed.timers.values()][0].callback();
    assert.equal(timed.recorders[0].stops, 1);
    assert.equal(timed.track.stops, 1);
    assert.equal(timed.timers.size, 0);
    timed.respond({ available: true, full_text: 'Automatically stopped' });
    await settle();
    assert.deepEqual(timed.transcripts, ['Automatically stopped']);

    const disposedResponse = createHarness();
    await disposedResponse.start();
    await disposedResponse.controller.toggle();
    disposedResponse.controller.dispose();
    assert.equal(disposedResponse.requests[0].configuration.signal.aborted, true);
    disposedResponse.respond({ available: true, full_text: 'Must not be delivered' });
    await settle();
    assert.equal(disposedResponse.transcripts.length, 0);
    assert.equal(disposedResponse.errors.length, 0);
    assert.deepEqual(disposedResponse.states, ['idle', 'recording', 'transcribing']);

    const rejectedAfterDispose = createHarness();
    await rejectedAfterDispose.start();
    await rejectedAfterDispose.controller.toggle();
    rejectedAfterDispose.controller.dispose();
    rejectedAfterDispose.requests[0].response.reject(new Error('SECRET late network error'));
    await settle();
    assert.equal(rejectedAfterDispose.errors.length, 0);
    assert.equal(rejectedAfterDispose.transcripts.length, 0);

    const disposedJson = createHarness();
    await disposedJson.start();
    await disposedJson.controller.toggle();
    const pendingJson = deferred();
    disposedJson.requests[0].response.resolve({ ok: true, json: () => pendingJson.promise });
    await settle();
    disposedJson.controller.dispose();
    pendingJson.resolve({ available: true, full_text: 'Must also be ignored' });
    await settle();
    assert.equal(disposedJson.transcripts.length, 0);
    assert.equal(disposedJson.errors.length, 0);

    const emptyAudio = createHarness({ emptyAudio: true });
    await emptyAudio.start();
    await emptyAudio.controller.toggle();
    assert.equal(emptyAudio.requests.length, 0);
    assert.match(emptyAudio.errors[0], /No audio/);
    assert.equal(emptyAudio.states.at(-1), 'idle');

    for (const [data, ok] of [
        [{ available: false, error: 'SECRET model details' }, true],
        [{ available: true, error: 'SECRET traceback', full_text: 'Ignore' }, true],
        [{ error: 'SECRET server details' }, false],
        [{ available: true, full_text: '  ' }, true],
        [null, true]
    ]) {
        const failed = createHarness();
        await failed.start();
        await failed.controller.toggle();
        failed.respond(data, ok);
        await settle();
        assert.equal(failed.errors.length, 1);
        assert.doesNotMatch(failed.errors[0], /SECRET/);
        assert.equal(failed.transcripts.length, 0);
        assert.equal(failed.states.at(-1), 'idle');
        assert.equal(failed.track.stops, 1);
        assert.equal(failed.timers.size, 0);
    }

    for (const failure of ['network', 'json']) {
        const failed = createHarness();
        await failed.start();
        await failed.controller.toggle();
        if (failure === 'network') failed.requests[0].response.reject(new Error('SECRET network details'));
        else failed.requests[0].response.resolve({ ok: true, json: async () => { throw new Error('SECRET JSON'); } });
        await settle();
        assert.equal(failed.states.at(-1), 'idle');
        assert.equal(failed.errors.length, 1);
        assert.doesNotMatch(failed.errors[0], /SECRET/);
    }

    for (const options of [{ constructorError: true }, { startError: true }, { stopError: true }]) {
        const failed = createHarness(options);
        await failed.start();
        if (options.stopError) await failed.controller.toggle();
        assert.equal(failed.track.stops, 1);
        assert.equal(failed.timers.size, 0);
        assert.equal(failed.states.at(-1), 'idle');
        assert.equal(failed.errors.length, 1);
        assert.doesNotMatch(failed.errors[0], /SECRET/);
    }

    const recorderError = createHarness();
    await recorderError.start();
    recorderError.recorders[0].onerror({ error: new Error('SECRET recording failure') });
    assert.equal(recorderError.track.stops, 1);
    assert.equal(recorderError.requests.length, 0);
    assert.equal(recorderError.states.at(-1), 'idle');
    assert.equal(recorderError.timers.size, 0);
    assert.doesNotMatch(recorderError.errors[0], /SECRET/);

    const source = fs.readFileSync('static/local_voice_input.js', 'utf8');
    assert.doesNotMatch(source, /SpeechRecognition|https?:\/\//);
    console.log('Local voice input recording, formats, privacy, and cancellation regressions passed.');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
