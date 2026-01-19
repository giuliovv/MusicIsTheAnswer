# StereoMusic

Spatial audio demos that pan sound based on detected objects in the camera feed.

## Requirements

- Python 3.11 (recommended, matches main project)
- Virtualenv with project dependencies installed
- A webcam and audio output device

## Quick start

```bash
cd MusicIsTheAnswer
python -m venv .venv_spatial
source .venv_spatial/bin/activate

# Install stereomusic-specific requirements (on top of root ones if needed)
pip install -r stereomusic/requirements.txt

# Run basic stereo tracker (no HRTF)
python -m stereomusic.spatial_tracker
```

## Usage

Track a specific object type (e.g. only track a cup):

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
python -m stereomusic.spatial_tracker_hrtf stereomusic/chiptune.wav person
```

Run with debug window:

```bash
python -m stereomusic.spatial_tracker_hrtf -t "cell phone" -d
```

### Common commands

| Goal                                   | Command example                                                   |
|----------------------------------------|-------------------------------------------------------------------|
| Basic stereo tracking (no HRTF)        | `python -m stereomusic.spatial_tracker`                           |
| Track only cups (HRTF, fast mode)     | `python -m stereomusic.spatial_tracker_hrtf -t cup -f`         |
| Track cups with chiptune (HRTF, HAILO)     | `python -m stereomusic.spatial_tracker_hrtf stereomusic/chiptune.wav cup --hailo` |
| HRTF tracking with debug window       | `python -m stereomusic.spatial_tracker_hrtf -t cup --hailo -d`  |

## Modules overview

- `spatial_audio.py` – simple stereo panning with distance cues.
- `spatial_audio_hrtf.py` – HRTF-based 3D positioning using pyroomacoustics
	and MIT KEMAR SOFA data.
- `object_detector.py` – YOLOv8-based object detector (Ultralytics).
- `spatial_tracker.py` – basic tracker: camera → YOLO → stereo audio.
- `spatial_tracker_hrtf.py` – enhanced tracker with HRTF audio, optional
	Hailo acceleration and debug visualization.

## Audio assets and credits

- `chiptune.wav` is a chiptune demo file derived from audio by
	orginaljun on Freesound: https://freesound.org/people/orginaljun/sounds/396173/
	(see Freesound for licensing terms).

## How it feels

The audio pans left/right based on object position in the frame, and gets louder as the object gets closer.
