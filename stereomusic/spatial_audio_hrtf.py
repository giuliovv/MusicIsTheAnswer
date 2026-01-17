"""
Enhanced Spatial Audio with HRTF-like processing and elevation cues.

Provides more accurate 3D positioning than simple panning:
- Interaural Time Difference (ITD): sound arrives at one ear first
- Interaural Level Difference (ILD): sound is louder in closer ear
- Elevation cues: frequency filtering to indicate up/down
- Room acoustics: pyroomacoustics for realistic reverb
"""

from __future__ import annotations
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
from typing import Optional

# Speed of sound and head model
SPEED_OF_SOUND = 343.0  # m/s
HEAD_RADIUS = 0.0875    # meters (average human head radius)


class HRTFSpatialPlayer:
    """
    Spatial audio player with HRTF-like processing.

    Position is specified as:
    - azimuth: -1.0 (full left) to 1.0 (full right)
    - elevation: -1.0 (below) to 1.0 (above)
    - distance: 1.0 (close) to 5.0 (far)
    """

    def __init__(self, audio_file: str):
        """Load an audio file for spatial playback."""
        self.data, self.samplerate = sf.read(audio_file, dtype='float32')

        # Convert to mono for clean spatial positioning
        if len(self.data.shape) == 1:
            self.mono = self.data
        else:
            self.mono = np.mean(self.data, axis=1)

        # Spatial parameters
        self.azimuth = 0.0      # -1.0 (left) to 1.0 (right)
        self.elevation = 0.0   # -1.0 (below) to 1.0 (above)
        self.distance = 1.0    # 1.0 (close) to 5.0+ (far)

        self.playing = False
        self.loop = False
        self._current_frame = 0
        self._stream = None
        self._lock = threading.Lock()

        # Processing state
        self._itd_buffer_left = np.zeros(100, dtype=np.float32)
        self._itd_buffer_right = np.zeros(100, dtype=np.float32)
        self._itd_write_pos = 0

        # Filters for elevation (simple IIR state)
        self._lpf_state_l = 0.0
        self._lpf_state_r = 0.0
        self._hpf_state_l = 0.0
        self._hpf_state_r = 0.0

        # Reverb delay lines
        self._reverb_buffer = np.zeros((int(self.samplerate * 0.1), 2), dtype=np.float32)
        self._reverb_pos = 0

    def set_position(self, azimuth: float, elevation: float = 0.0):
        """
        Set spatial position.

        Args:
            azimuth: -1.0 (full left) to 1.0 (full right)
            elevation: -1.0 (below) to 1.0 (above)
        """
        with self._lock:
            self.azimuth = max(-1.0, min(1.0, azimuth))
            self.elevation = max(-1.0, min(1.0, elevation))

    def set_distance(self, d: float):
        """Set distance: 1.0 = close, larger = farther."""
        with self._lock:
            self.distance = max(0.5, d)

    def _calculate_itd_samples(self, azimuth: float) -> tuple[int, int]:
        """
        Calculate Interaural Time Difference in samples.

        Sound reaches the closer ear first. Max ITD is ~0.7ms for sounds
        directly to the side.
        """
        # Azimuth angle in radians (-pi/2 to pi/2)
        angle = azimuth * (np.pi / 2)

        # Woodworth formula for ITD
        itd_seconds = (HEAD_RADIUS / SPEED_OF_SOUND) * (np.sin(angle) + angle)

        itd_samples = int(abs(itd_seconds) * self.samplerate)
        itd_samples = min(itd_samples, 50)  # Cap at ~1ms

        if azimuth > 0:  # Sound from right
            return (itd_samples, 0)  # Left ear delayed
        else:  # Sound from left
            return (0, itd_samples)  # Right ear delayed

    def _calculate_ild(self, azimuth: float) -> tuple[float, float]:
        """
        Calculate Interaural Level Difference.

        The ear facing the sound is louder due to head shadow.
        Effect is frequency-dependent but we use a simple approximation.
        """
        # ILD can be up to 20dB for high frequencies
        # Use ~6dB max for a reasonable effect
        angle = azimuth * (np.pi / 2)

        # Head shadow approximation
        shadow = 0.3 * np.sin(angle)  # Max ~6dB difference

        left_gain = 1.0 - max(0, shadow)
        right_gain = 1.0 + min(0, shadow)

        return (left_gain, right_gain)

    def _apply_elevation_filter(self, left: np.ndarray, right: np.ndarray,
                                 elevation: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply strong elevation cues through frequency filtering and pitch perception.

        - Higher elevation: boost high frequencies significantly + slight volume boost
        - Lower elevation: heavy low-pass filter (very muffled)

        Uses aggressive filtering to make the difference clearly audible.
        """
        if abs(elevation) < 0.05:
            return left, right

        result_left = left.copy()
        result_right = right.copy()

        if elevation > 0:
            # HIGH sounds: Brighten significantly
            # Apply strong high-shelf boost by emphasizing differences
            strength = elevation  # 0 to 1

            # Strong high-pass emphasis (makes sound "airy" and "above")
            alpha = 0.3 + 0.5 * strength  # 0.3 to 0.8
            prev_l, prev_r = self._hpf_state_l, self._hpf_state_r

            for i in range(len(left)):
                # High-pass: output difference from previous sample
                hp_l = left[i] - prev_l
                hp_r = right[i] - prev_r
                # Mix high-passed signal with original
                result_left[i] = left[i] + hp_l * alpha * 2.0
                result_right[i] = right[i] + hp_r * alpha * 2.0
                prev_l, prev_r = left[i], right[i]

            self._hpf_state_l, self._hpf_state_r = prev_l, prev_r

            # Slight volume boost for "up" (things above feel more present)
            result_left *= (1.0 + 0.2 * strength)
            result_right *= (1.0 + 0.2 * strength)

        else:
            # LOW sounds: Muffle heavily
            strength = -elevation  # 0 to 1

            # Very aggressive low-pass (makes sound "below" / "underground")
            # Lower alpha = more muffled
            alpha = 0.3 - 0.2 * strength  # 0.3 down to 0.1
            alpha = max(0.1, alpha)

            prev_l, prev_r = self._lpf_state_l, self._lpf_state_r

            for i in range(len(left)):
                result_left[i] = alpha * left[i] + (1 - alpha) * prev_l
                result_right[i] = alpha * right[i] + (1 - alpha) * prev_r
                prev_l, prev_r = result_left[i], result_right[i]

            self._lpf_state_l, self._lpf_state_r = prev_l, prev_r

            # Volume reduction for "down" (things below feel more distant)
            result_left *= (1.0 - 0.3 * strength)
            result_right *= (1.0 - 0.3 * strength)

        return result_left, result_right

    def _apply_distance(self, left: np.ndarray, right: np.ndarray,
                        distance: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply distance cues:
        - Volume attenuation (inverse square approximation)
        - Low-pass filter (air absorption)
        - Reverb (more reverb = farther)
        """
        # Volume attenuation
        volume = 1.0 / (distance ** 0.7)  # Gentler than inverse square

        left = left * volume
        right = right * volume

        # Air absorption (simple low-pass for distance > 2)
        if distance > 2:
            alpha = max(0.3, 1.0 - (distance - 2) * 0.15)
            for i in range(1, len(left)):
                left[i] = alpha * left[i] + (1 - alpha) * left[i-1]
                right[i] = alpha * right[i] + (1 - alpha) * right[i-1]

        # Simple reverb for distance
        if distance > 1.5:
            wet = min(0.4, (distance - 1.5) * 0.15)
            reverb_len = len(self._reverb_buffer)

            for i in range(len(left)):
                # Read from delay buffer
                read_pos = (self._reverb_pos + i) % reverb_len
                reverb_l = self._reverb_buffer[read_pos, 0]
                reverb_r = self._reverb_buffer[read_pos, 1]

                # Mix
                left[i] = left[i] * (1 - wet) + reverb_l * wet
                right[i] = right[i] * (1 - wet) + reverb_r * wet

                # Write to delay buffer with feedback
                write_pos = (self._reverb_pos + i + int(0.05 * self.samplerate)) % reverb_len
                self._reverb_buffer[write_pos, 0] = left[i] * 0.3
                self._reverb_buffer[write_pos, 1] = right[i] * 0.3

            self._reverb_pos = (self._reverb_pos + len(left)) % reverb_len

        return left, right

    def _apply_spatial(self, chunk: np.ndarray) -> np.ndarray:
        """Apply full 3D spatial processing."""
        with self._lock:
            azimuth = self.azimuth
            elevation = self.elevation
            distance = self.distance

        n = len(chunk)

        # 1. Calculate ITD (time difference)
        delay_left, delay_right = self._calculate_itd_samples(azimuth)

        # 2. Calculate ILD (level difference)
        gain_left, gain_right = self._calculate_ild(azimuth)

        # 3. Apply constant-power panning on top of ILD
        angle = (azimuth + 1) * np.pi / 4
        pan_left = np.cos(angle)
        pan_right = np.sin(angle)

        # Create stereo with ITD
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)

        # Apply delays using circular buffer
        for i in range(n):
            # Write to buffer
            self._itd_buffer_left[self._itd_write_pos] = chunk[i]
            self._itd_buffer_right[self._itd_write_pos] = chunk[i]

            # Read with delay
            read_left = (self._itd_write_pos - delay_left) % len(self._itd_buffer_left)
            read_right = (self._itd_write_pos - delay_right) % len(self._itd_buffer_right)

            left[i] = self._itd_buffer_left[read_left]
            right[i] = self._itd_buffer_right[read_right]

            self._itd_write_pos = (self._itd_write_pos + 1) % len(self._itd_buffer_left)

        # Apply ILD and panning
        left = left * gain_left * pan_left
        right = right * gain_right * pan_right

        # 4. Apply elevation filtering
        left, right = self._apply_elevation_filter(left, right, elevation)

        # 5. Apply distance cues
        left, right = self._apply_distance(left, right, distance)

        # Combine to stereo
        result = np.column_stack([left, right])
        return result

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice stream."""
        remaining = len(self.mono) - self._current_frame

        if remaining <= 0:
            if self.loop:
                self._current_frame = 0
                remaining = len(self.mono)
            else:
                outdata.fill(0)
                self.playing = False
                raise sd.CallbackStop()

        chunk_size = min(frames, remaining)
        chunk = self.mono[self._current_frame:self._current_frame + chunk_size].copy()

        # Apply spatial processing
        spatial_chunk = self._apply_spatial(chunk)

        outdata[:chunk_size] = spatial_chunk
        outdata[chunk_size:] = 0

        self._current_frame += chunk_size

    def play(self, loop: bool = False):
        """Start playing with spatial audio."""
        self.loop = loop
        self._current_frame = 0
        self.playing = True

        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            callback=self._audio_callback,
            dtype='float32',
            blocksize=2048,
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
    """Demo moving sound in 3D space."""
    import sys
    import os

    if len(sys.argv) < 2:
        audio_file = os.path.join(os.path.dirname(__file__), "test_tone.wav")
    else:
        audio_file = sys.argv[1]

    print(f"Loading: {audio_file}")
    player = HRTFSpatialPlayer(audio_file)

    print("\n=== HRTF Spatial Audio Demo ===")
    print("Sound will move in 3D space:\n")

    player.play(loop=True)

    try:
        # Move left to right
        print("1. Moving LEFT to RIGHT...")
        for i in range(50):
            az = -1.0 + (2.0 * i / 49)
            player.set_position(az, 0.0)
            time.sleep(0.06)

        # Move up and down
        print("2. Moving DOWN to UP...")
        player.set_position(0.0, -1.0)
        for i in range(50):
            el = -1.0 + (2.0 * i / 49)
            player.set_position(0.0, el)
            time.sleep(0.06)

        # Circle around
        print("3. Circling around (azimuth + elevation)...")
        for i in range(100):
            angle = 2 * np.pi * i / 100
            az = np.sin(angle)
            el = np.cos(angle) * 0.5
            player.set_position(az, el)
            time.sleep(0.05)

        # Distance demo
        print("4. Moving CLOSE to FAR...")
        player.set_position(0.0, 0.0)
        for i in range(50):
            d = 1.0 + (4.0 * i / 49)
            player.set_distance(d)
            time.sleep(0.08)

        print("\nDone! Press Ctrl+C to exit.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        player.stop()


if __name__ == "__main__":
    demo()
