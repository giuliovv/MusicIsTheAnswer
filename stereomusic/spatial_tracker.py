"""
Spatial Object Tracker - Connects camera, object detection, and spatial audio.

This is the main integration module that:
1. Captures video frames from camera
2. Runs object detection to find objects
3. Updates spatial audio position based on detected objects
4. Provides audio feedback so a blind user knows where objects are
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass

from .object_detector import ObjectDetector, Detection, get_detector
from .spatial_audio import SpatialAudioPlayer


@dataclass
class TrackedObject:
    """An object being tracked with its current spatial position."""
    detection: Detection
    last_seen: float  # timestamp


class SpatialObjectTracker:
    """
    Main class that tracks objects and provides spatial audio feedback.

    Usage:
        tracker = SpatialObjectTracker(audio_file="music.wav")
        tracker.start()  # Starts tracking loop
        # ... user can hear objects in space ...
        tracker.stop()
    """

    def __init__(
        self,
        audio_file: str,
        camera=None,
        target_class: Optional[str] = None,
        detector: Optional[ObjectDetector] = None,
    ):
        """
        Initialize the tracker.

        Args:
            audio_file: Path to audio file to play spatially
            camera: Camera instance (from camera module) or None to create USB camera
            target_class: If set, only track this object class (e.g., "person", "bottle")
            detector: ObjectDetector instance or None to use default
        """
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

        # Audio player
        self.player: Optional[SpatialAudioPlayer] = None

        # Tracking state
        self.tracked: Optional[TrackedObject] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_detection: Optional[Callable[[Detection], None]] = None
        self.on_lost: Optional[Callable[[], None]] = None

        # Settings
        self.update_interval = 0.1  # seconds between detection updates
        self.lost_timeout = 0.5     # seconds before object considered lost

    def start(self, loop_audio: bool = True):
        """Start tracking and playing spatial audio."""
        if self._running:
            return

        # Check camera
        if not self.camera.is_available():
            raise RuntimeError("Camera not available")

        # Start audio
        self.player = SpatialAudioPlayer(self.audio_file)
        self.player.set_position(0.0)  # Start centered
        self.player.set_distance(3.0)  # Start at medium distance (quieter)
        self.player.play(loop=loop_audio)

        # Start tracking thread
        self._running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

        print("Spatial tracking started")
        print(f"Tracking: {self.target_class or 'any object'}")

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

        print("Spatial tracking stopped")

    def _tracking_loop(self):
        """Main tracking loop running in background thread."""
        frame_count = 0
        while self._running:
            try:
                # Capture frame
                image_bytes = self.camera.capture()
                if image_bytes is None:
                    print("[DEBUG] Camera capture returned None")
                    time.sleep(self.update_interval)
                    continue

                frame_count += 1
                if frame_count == 1:
                    print(f"[DEBUG] First frame captured: {len(image_bytes)} bytes")

                # Detect objects
                detections = self.detector.detect_from_bytes(image_bytes)

                if frame_count <= 3:
                    print(f"[DEBUG] Frame {frame_count}: {len(detections)} detections")

                # Filter by target class if specified
                if self.target_class:
                    all_classes = [d.class_name for d in detections]
                    detections = [
                        d for d in detections
                        if d.class_name.lower() == self.target_class.lower()
                    ]
                    if frame_count <= 3:
                        print(f"[DEBUG] Classes found: {all_classes}, after filter: {len(detections)}")

                # Update tracking
                now = time.time()

                if detections:
                    # Track the most confident detection
                    best = detections[0]
                    self.tracked = TrackedObject(detection=best, last_seen=now)

                    # Update audio position
                    if self.player:
                        self.player.set_position(best.spatial_x)
                        self.player.set_distance(best.estimated_distance)

                    # Callback
                    if self.on_detection:
                        self.on_detection(best)

                elif self.tracked:
                    # Check if object is lost
                    if now - self.tracked.last_seen > self.lost_timeout:
                        self.tracked = None
                        # Move to center and quiet when lost
                        if self.player:
                            self.player.set_distance(5.0)  # Quiet
                        if self.on_lost:
                            self.on_lost()

            except Exception as e:
                print(f"Tracking error: {e}")

            time.sleep(self.update_interval)

    def get_current_detection(self) -> Optional[Detection]:
        """Get the currently tracked detection, if any."""
        if self.tracked:
            return self.tracked.detection
        return None


def run_demo(audio_file: str, target_class: Optional[str] = None):
    """
    Run a demo of spatial object tracking.

    Args:
        audio_file: Path to audio file to play
        target_class: Optional class to track (e.g., "person", "cell phone")
    """
    print("\n=== Spatial Object Tracker Demo ===\n")
    print("This will play audio from the direction of detected objects.")
    print("Point your camera at objects to hear them!")
    print("Press Ctrl+C to stop.\n")

    tracker = SpatialObjectTracker(
        audio_file=audio_file,
        target_class=target_class,
    )

    # Set up callbacks for feedback
    def on_detect(det: Detection):
        pos = "LEFT" if det.spatial_x < -0.3 else "RIGHT" if det.spatial_x > 0.3 else "CENTER"
        dist = "close" if det.estimated_distance < 2 else "far" if det.estimated_distance > 3 else "medium"
        print(f"\r[{det.class_name}] {pos} ({dist})     ", end="", flush=True)

    def on_lost():
        print("\r[searching...]              ", end="", flush=True)

    tracker.on_detection = on_detect
    tracker.on_lost = on_lost

    try:
        tracker.start()

        # Keep running until interrupted
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        tracker.stop()


if __name__ == "__main__":
    import sys
    import os

    # Default to test tone in same directory
    default_audio = os.path.join(os.path.dirname(__file__), "test_tone.wav")

    if len(sys.argv) >= 2:
        audio_file = sys.argv[1]
    else:
        audio_file = default_audio

    target = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print(f"\nUsage: python spatial_tracker.py [audio_file] [target_class]")
        print(f"Example: python spatial_tracker.py music.wav person")
        sys.exit(1)

    run_demo(audio_file, target)
