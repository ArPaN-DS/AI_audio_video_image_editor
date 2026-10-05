# AI Assistant Guide

Open `/agent` for a dedicated conversation, or open the AI assistant from `/studio` to edit selected studio media.
Media processing and voice transcription use the local application; no subscription is required.

## Editing

- Upload media or select an existing studio clip before requesting an edit.
- Use precise instructions, such as “Trim from 0.2 to 0.8 seconds” or “Adjust speed to 0.75x”.
- Separate ordered actions with “then”: “Extract audio as WAV then trim from 0.2 to 0.8 seconds”.
- Audio speed changes preserve pitch and support 0.25x–4x.
- Soundtrack normalization, fades, noise cleaning, and voice enhancement preserve the video picture.
- If an action fails, dependent edits stop; the last completed result remains downloadable.

## Speech and Subtitles

Ask “Transcribe speech” or “Generate subtitles” after uploading audio or video.
The reply includes the transcript and TXT, SRT, and VTT download links when timed speech is available.
Untimed transcription provides TXT only. No detected speech produces no invented subtitle cues.

For voice commands, click the microphone, grant recording permission, speak, then click again.
Recording stops automatically after 60 seconds. The local application transcribes the recording into
the command input; review it before sending. Recording requires a supported browser and microphone
access, and transcription requires the local speech capability to be available.
