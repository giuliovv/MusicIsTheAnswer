# StereoMusic

Spatial audio demos that pan sound based on detected objects in the camera feed.

## Requirements

- Python 3.10+
- Virtualenv with project dependencies installed
- A webcam and audio output device

## Quick start

```bash
source .venv_spatial/bin/activate
python -m stereomusic.spatial_tracker
```

## Usage

Track a specific object type (e.g. only track your phone):

```bash
python -m stereomusic.spatial_tracker_hrtf -t "cup" -f

```

`-f` is for fast mode, will downsize and use only the object

Run the test suite:

```bash
python test_spatial_tracking.py
```

Audio-only test (no tracking):

```bash
python -m stereomusic.spatial_audio_hrtf
```

HRTF version (stronger spatial cues):

```bash
python -m stereomusic.spatial_tracker_hrtf stereomusic/test_tone.wav person
```

Run with debug window:

```bash
python -m stereomusic.spatial_tracker_hrtf -t "cell phone" -d
```

## How it feels

The audio pans left/right based on object position in the frame, and gets louder as the object gets closer.
