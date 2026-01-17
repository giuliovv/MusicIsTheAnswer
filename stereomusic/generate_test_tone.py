"""Generate a test tone WAV file for testing spatial audio."""

import numpy as np
import soundfile as sf

def generate_tone(filename: str = "test_tone.wav", duration: float = 5.0, frequency: float = 440.0):
    """Generate a simple sine wave tone."""
    samplerate = 44100
    t = np.linspace(0, duration, int(samplerate * duration), dtype=np.float32)

    # Create a pleasant tone with some harmonics
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    tone += 0.25 * np.sin(2 * np.pi * frequency * 2 * t)  # octave
    tone += 0.125 * np.sin(2 * np.pi * frequency * 3 * t)  # fifth

    # Normalize
    tone = tone / np.max(np.abs(tone)) * 0.7

    sf.write(filename, tone, samplerate)
    print(f"Generated: {filename} ({duration}s at {frequency}Hz)")

if __name__ == "__main__":
    generate_tone()
