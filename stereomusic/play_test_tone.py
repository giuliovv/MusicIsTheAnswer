"""Play a test tone through the default ALSA output.

This module is intended for quick audio testing on the Raspberry Pi
without touching the ElevenLabs / main assistant code. It reuses the
existing generate_test_tone utility and calls `aplay` to play the file.
"""

import subprocess
from pathlib import Path

from .generate_test_tone import generate_tone


def main() -> None:
    """Generate (if needed) and play a test tone WAV file."""
    wav_path = Path(__file__).with_name("test_tone.wav")

    # Only regenerate if the file doesn't exist yet
    if not wav_path.exists():
        generate_tone(str(wav_path))

    print(f"[stereomusic] Playing {wav_path} via aplay...")

    try:
        subprocess.run(["aplay", str(wav_path)], check=True)
    except FileNotFoundError:
        print("[stereomusic] `aplay` not found. Install `alsa-utils` or use another player.")
    except subprocess.CalledProcessError as exc:
        print(f"[stereomusic] `aplay` failed with exit code {exc.returncode}.")


if __name__ == "__main__":
    main()
