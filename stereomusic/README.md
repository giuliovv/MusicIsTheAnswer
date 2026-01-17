  Full interactive demo (audio follows objects):
  source .venv_spatial/bin/activate
  python -m stereomusic.spatial_tracker

  Track a specific object type (e.g., only track your phone):
  python -m stereomusic.spatial_tracker stereomusic/test_tone.wav "cell phone"

  Run test suite:
  python test_spatial_tracking.py

  The audio will play from the direction where objects are detected - move an object left/right and you'll hear the sound pan. Closer objects will be louder.

If you want to test just the audio without tracking first:

  python -m stereomusic.spatial_audio_hrtf


Test HRTF version (with better spatial clues)
python -m stereomusic.spatial_tracker_hrtf stereomusic/test_tone.wav person