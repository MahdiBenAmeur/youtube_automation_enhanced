# Enhanced YouTube Automation

Starter structure for a YouTube Shorts automation project.

## Target flow

1. Pick a Shorts concept.
2. Generate or assemble a short script.
3. Build scenes, captions, voice, music, and overlays.
4. Render a vertical video.
5. Save the final file and upload it as a YouTube Short.

## Recommended structure

- `src/youtube_automation/`
  - Main package.
- `src/youtube_automation/shorts/`
  - Shorts-specific workflow code.
- `src/youtube_automation/shorts/builders/`
  - Script builders that create video definitions.
- `src/youtube_automation/shorts/renderers/`
  - Video assembly logic for subtitles, voice, clips, and final render.
- `src/youtube_automation/shorts/uploaders/`
  - Upload and publish logic for YouTube Shorts.
- `src/youtube_automation/services/`
  - External integrations such as ffmpeg, TTS, LLMs, image generation, and YouTube API.
- `src/youtube_automation/domain/`
  - Shared models for scripts, scenes, assets, renders, and upload jobs.
- `configs/channels/`
  - Per-channel defaults such as title style, hashtags, upload settings.
- `configs/shorts/`
  - Shorts formats, durations, template settings, caption styles.
- `assets/templates/`
  - Reusable render templates, overlays, transitions.
- `assets/audio/`
  - Music, sound effects, voice presets.
- `assets/visuals/`
  - Background loops, stickers, icons, logos.
- `workspaces/projects/`
  - Generated Shorts project data before render.
- `workspaces/renders/`
  - Final rendered videos.
- `workspaces/uploads/`
  - Upload queue, published metadata, logs.
- `scripts/`
  - Entry scripts such as `build_short.py` and `upload_short.py`.
- `tests/`
  - Unit and workflow tests.

## Practical code split

- `builders`
  - Create a structured video plan from a topic or prompt.
- `renderers`
  - Convert the plan into scenes and a final vertical video.
- `uploaders`
  - Push finished videos to YouTube with title, description, tags, and scheduling.

This structure is optimized for code that builds a Short, saves it, and uploads it.

## Uploading a short

Call the uploader function with a rendered video path and the public title:

```python
from youtube_automation.shorts.uploaders.upload_short import upload_short

result = upload_short(
    video_path="workspaces/renders/fight_001.mp4",
    title="Epic Arena Clash",
)
```

Behavior:

- Uploads with `privacyStatus="public"`.
- Sets the description to `fighting simulation {video_number}`.
- Only increments `video_number` after YouTube returns a successful upload response.

CLI usage:

```bash
python scripts/upload_short.py workspaces/renders/fight_001.mp4 "Epic Arena Clash"
```
