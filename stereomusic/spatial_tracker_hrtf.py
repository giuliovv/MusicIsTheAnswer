"""
Enhanced Spatial Object Tracker with HRTF audio.

Uses the HRTFSpatialPlayer for more accurate 3D positioning including:
- Precise left/right via ITD (interaural time difference)
- Elevation cues (up/down) via frequency filtering
- Better distance cues via reverb and filtering
"""

from __future__ import annotations
import os
import sys
import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass

# Suppress ALSA warnings before importing audio libraries
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def _suppress_alsa_errors():
    """Suppress ALSA error messages on Linux."""
    try:
        from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int
        # Define the error handler function type
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

        def py_error_handler(filename, line, function, err, fmt):
            pass  # Suppress all ALSA errors

        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

        asound = CDLL('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)

        # Keep reference to prevent garbage collection
        _suppress_alsa_errors._handler = c_error_handler
    except Exception:
        pass  # Not on Linux or ALSA not available

_suppress_alsa_errors()

from .object_detector import ObjectDetector, Detection, get_detector
from .spatial_audio_hrtf import HRTFSpatialPlayer


@dataclass
class TrackedObject:
    """An object being tracked with its current spatial position."""
    detection: Detection
    last_seen: float


class SpatialObjectTrackerHRTF:
    """
    Enhanced tracker using HRTF-based spatial audio.

    Provides more accurate 3D positioning than the basic tracker:
    - Horizontal: ITD + ILD for precise left/right
    - Vertical: Frequency cues for up/down
    - Depth: Reverb + filtering for distance
    """

    def __init__(
        self,
        audio_file: str,
        camera=None,
        target_class: Optional[str] = None,
        detector: Optional[ObjectDetector] = None,
    ):
        self.audio_file = audio_file
        self.target_class = target_class
        self.detector = detector or get_detector()

        # Camera setup
        if camera is None:
            from camera.usb_camera import USBCamera
            self.camera = USBCamera()
            self._own_camera = True
        else:
            self.camera = camera
            self._own_camera = False

        # Audio player (HRTF version)
        self.player: Optional[HRTFSpatialPlayer] = None

        # Tracking state
        self.tracked: Optional[TrackedObject] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_detection: Optional[Callable[[Detection], None]] = None
        self.on_lost: Optional[Callable[[], None]] = None

        # Settings
        self.update_interval = 0.1
        self.lost_timeout = 0.5

        # Smoothing for position (reduces jitter)
        self._smooth_x = 0.0
        self._smooth_el = 0.0
        self._smooth_dist = 3.0
        self._smoothing = 0.3  # 0 = no smoothing, 1 = very smooth

    def start(self, loop_audio: bool = True):
        """Start tracking with HRTF spatial audio."""
        if self._running:
            return

        if not self.camera.is_available():
            raise RuntimeError("Camera not available")

        self._loop_audio = loop_audio
        self._audio_started = False

        # Pre-load audio file (but don't play yet)
        print("Loading audio file...")
        self.player = HRTFSpatialPlayer(self.audio_file)
        self.player.set_position(0.0, 0.0)
        self.player.set_distance(5.0)  # Start quiet

        # Pre-load YOLO model with loading indicator
        print("Loading YOLO model (first time may download)...")
        sys.stdout.flush()
        self.detector._ensure_initialized()
        print("YOLO ready!")

        self._running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

        print(f"\nTracking: {self.target_class or 'any object'}")
        print("(Audio starts when object detected)\n")

    def stop(self):
        """Stop tracking and audio."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.player:
            self.player.stop()
            self.player = None

        if self._own_camera:
            self.camera.release()

        print("HRTF Spatial tracking stopped")

    def _tracking_loop(self):
        """Main tracking loop."""
        frame_count = 0
        while self._running:
            try:
                image_bytes = self.camera.capture()
                if image_bytes is None:
                    time.sleep(self.update_interval)
                    continue

                frame_count += 1
                detections = self.detector.detect_from_bytes(image_bytes)

                # Filter by target class
                if self.target_class:
                    detections = [
                        d for d in detections
                        if d.class_name.lower() == self.target_class.lower()
                    ]

                now = time.time()

                if detections:
                    best = detections[0]
                    self.tracked = TrackedObject(detection=best, last_seen=now)

                    # Start audio on first detection
                    if not self._audio_started and self.player:
                        self.player.play(loop=self._loop_audio)
                        self._audio_started = True

                    # Get raw positions
                    raw_x = best.spatial_x
                    raw_el = best.spatial_elevation
                    raw_dist = best.estimated_distance

                    # Smooth the values
                    self._smooth_x = self._smooth_x * self._smoothing + raw_x * (1 - self._smoothing)
                    self._smooth_el = self._smooth_el * self._smoothing + raw_el * (1 - self._smoothing)
                    self._smooth_dist = self._smooth_dist * self._smoothing + raw_dist * (1 - self._smoothing)

                    # Update HRTF audio with smoothed values
                    if self.player:
                        self.player.set_position(self._smooth_x, self._smooth_el)
                        self.player.set_distance(self._smooth_dist)

                    if self.on_detection:
                        self.on_detection(best)

                elif self.tracked:
                    if now - self.tracked.last_seen > self.lost_timeout:
                        self.tracked = None
                        if self.player:
                            self.player.set_distance(5.0)  # Fade out
                        if self.on_lost:
                            self.on_lost()

            except Exception as e:
                print(f"Tracking error: {e}")

            time.sleep(self.update_interval)

    def get_current_detection(self) -> Optional[Detection]:
        """Get the currently tracked detection."""
        if self.tracked:
            return self.tracked.detection
        return None


def run_demo(audio_file: str, target_class: Optional[str] = None):
    """Run demo with HRTF spatial audio."""
    print("\n" + "=" * 45)
    print("  HRTF Spatial Object Tracker")
    print("=" * 45)
    print("\nYou'll hear 3D audio cues:")
    print("  LEFT/RIGHT - sound pans to object position")
    print("  UP/DOWN    - bright=high, muffled=low")
    print("  DISTANCE   - louder=close, reverb=far")
    print("\nPress Ctrl+C to stop.\n")

    tracker = SpatialObjectTrackerHRTF(
        audio_file=audio_file,
        target_class=target_class,
    )

    def on_detect(det: Detection):
        # More detailed position info
        if det.spatial_x < -0.5:
            lr = "FAR LEFT"
        elif det.spatial_x < -0.2:
            lr = "LEFT"
        elif det.spatial_x > 0.5:
            lr = "FAR RIGHT"
        elif det.spatial_x > 0.2:
            lr = "RIGHT"
        else:
            lr = "CENTER"

        if det.spatial_elevation > 0.3:
            ud = "HIGH"
        elif det.spatial_elevation < -0.3:
            ud = "LOW"
        else:
            ud = "MID"

        if det.estimated_distance < 1.5:
            dist = "very close"
        elif det.estimated_distance < 2.5:
            dist = "close"
        elif det.estimated_distance < 3.5:
            dist = "medium"
        else:
            dist = "far"

        print(f"\r[{det.class_name}] {lr} {ud} ({dist})          ", end="", flush=True)

    def on_lost():
        print("\r[searching...]                    ", end="", flush=True)

    tracker.on_detection = on_detect
    tracker.on_lost = on_lost

    try:
        tracker.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        tracker.stop()


if __name__ == "__main__":
    default_audio = os.path.join(os.path.dirname(__file__), "test_tone.wav")

    audio_file = sys.argv[1] if len(sys.argv) >= 2 else default_audio
    target = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)

    run_demo(audio_file, target)
