"""
Spatial Audio Player - Plays audio from a position in stereo space.
Position ranges from -1.0 (full left) to 1.0 (full right).

Distance cues:
- Volume attenuation (inverse distance)
- Low-pass filter (high frequencies attenuate with distance)
- Reverb/wet mix (more reverb = farther)
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time


class SimpleReverb:
    """Simple reverb using delay lines - vectorized for performance."""

    def __init__(self, samplerate: int):
        self.samplerate = samplerate
        # Single delay line for simplicity and speed
        self.delay_samples = int(samplerate * 0.05)  # 50ms delay
        self.buffer = np.zeros((self.delay_samples, 2), dtype=np.float32)
        self.write_pos = 0
        self.decay = 0.3

    def process(self, chunk: np.ndarray, wet_mix: float) -> np.ndarray:
        """Apply reverb with wet/dry mix."""
        if wet_mix < 0.01:
            return chunk

        n = len(chunk)
        wet = np.zeros_like(chunk)

        # Read from circular buffer
        read_pos = self.write_pos
        for i in range(n):
            wet[i] = self.buffer[read_pos]
            self.buffer[read_pos] = chunk[i] * 0.5 + self.buffer[read_pos] * self.decay
            read_pos = (read_pos + 1) % self.delay_samples

        self.write_pos = read_pos
        return chunk * (1 - wet_mix) + wet * wet_mix


class LowPassFilter:
    """Simple one-pole low-pass filter."""

    def __init__(self):
        self.prev = np.zeros(2, dtype=np.float32)

    def process(self, chunk: np.ndarray, alpha: float) -> np.ndarray:
        """Apply low-pass filter. alpha: 0.0 (muffled) to 1.0 (no filter)."""
        if alpha >= 0.99:
            return chunk

        result = np.empty_like(chunk)
        beta = 1.0 - alpha
        prev = self.prev.copy()

        for i in range(len(chunk)):
            result[i] = alpha * chunk[i] + beta * prev
            prev = result[i]

        self.prev = prev
        return result


class SpatialAudioPlayer:
    def __init__(self, audio_file: str):
        """Load an audio file for spatial playback."""
        self.data, self.samplerate = sf.read(audio_file, dtype='float32')

        # Convert mono to stereo if needed
        if len(self.data.shape) == 1:
            self.data = np.column_stack([self.data, self.data])

        # Convert to mono first for clean spatial positioning
        if self.data.shape[1] >= 2:
            mono = (self.data[:, 0] + self.data[:, 1]) / 2
            self.data = np.column_stack([mono, mono])

        self.position = 0.0  # -1.0 (left) to 1.0 (right)
        self.distance = 1.0  # 1.0 = close, larger = farther
        self.playing = False
        self.loop = False
        self._current_frame = 0
        self._stream = None
        self._lock = threading.Lock()

        # Distance processing effects
        self._reverb = SimpleReverb(self.samplerate)
        self._lowpass = LowPassFilter()

    def set_position(self, x: float):
        """Set horizontal position: -1.0 (left) to 1.0 (right)."""
        with self._lock:
            self.position = max(-1.0, min(1.0, x))

    def set_distance(self, d: float):
        """Set distance: 1.0 = full volume, larger values = quieter."""
        with self._lock:
            self.distance = max(0.5, d)

    def _apply_spatial(self, chunk: np.ndarray) -> np.ndarray:
        """Apply stereo panning and distance effects."""
        with self._lock:
            pos = self.position
            dist = self.distance

        # Constant power panning
        angle = (pos + 1) * np.pi / 4
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        # === DISTANCE CUES ===

        # 1. Volume attenuation
        volume = 1.0 / dist

        # 2. Low-pass filter (farther = more muffled)
        cutoff = min(1.0, 2.0 / dist)
        chunk = self._lowpass.process(chunk, cutoff)

        # 3. Reverb (farther = more reverb)
        wet_mix = min(0.5, (dist - 0.5) * 0.2)
        chunk = self._reverb.process(chunk, max(0, wet_mix))

        # Apply panning and volume
        result = chunk.copy()
        result[:, 0] *= left_gain * volume
        result[:, 1] *= right_gain * volume

        return result

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice stream."""
        remaining = len(self.data) - self._current_frame

        if remaining <= 0:
            if self.loop:
                self._current_frame = 0
                remaining = len(self.data)
            else:
                outdata.fill(0)
                self.playing = False
                raise sd.CallbackStop()

        chunk_size = min(frames, remaining)
        chunk = self.data[self._current_frame:self._current_frame + chunk_size].copy()

        # Apply spatial audio
        spatial_chunk = self._apply_spatial(chunk)

        outdata[:chunk_size] = spatial_chunk
        outdata[chunk_size:] = 0

        self._current_frame += chunk_size

    def play(self, loop: bool = False):
        """Start playing the audio."""
        self.loop = loop
        self._current_frame = 0
        self.playing = True

        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            callback=self._audio_callback,
            dtype='float32',
            blocksize=2048,  # Larger buffer to prevent underruns
            latency='high'
        )
        self._stream.start()

    def stop(self):
        """Stop playback."""
        self.playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def wait(self):
        """Wait for playback to finish."""
        while self.playing:
            time.sleep(0.1)


def demo():
    """Demo: move sound from left to right."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python spatial_audio.py <audio_file.wav>")
        print("\nDemo will move the sound from left to right.")
        return

    audio_file = sys.argv[1]
    print(f"Loading: {audio_file}")

    player = SpatialAudioPlayer(audio_file)

    print("Playing audio, moving from left to right...")
    print("Position: -1.0 (left) -> 0.0 (center) -> 1.0 (right)")

    player.play(loop=True)

    try:
        duration = 10.0
        steps = 100

        for i in range(steps):
            pos = -1.0 + (2.0 * i / (steps - 1))
            player.set_position(pos)
            print(f"\rPosition: {pos:+.2f} {'<' if pos < -0.3 else '>' if pos > 0.3 else '|':^5}", end="", flush=True)
            time.sleep(duration / steps)

        print("\n\nDone! Press Ctrl+C to exit or let it loop...")
        player.wait()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        player.stop()


if __name__ == "__main__":
    demo()
