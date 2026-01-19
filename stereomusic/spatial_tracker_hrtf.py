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
import signal
import threading
from typing import Optional, Callable, Union, Tuple, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables (e.g. CAMERA_TYPE)
load_dotenv()

from dotenv import load_dotenv

# Suppress ALSA warnings before importing audio libraries
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Load .env so CAMERA_TYPE/USB_CAMERA_INDEX are respected when running this
# module directly (e.g., `python -m stereomusic.spatial_tracker_hrtf`).
load_dotenv()

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

from .object_detector import ObjectDetector, Detection, get_detector, create_fast_detector
from .spatial_audio_hrtf import HRTFSpatialPlayer
try:
    from .hailo_detector import HailoObjectDetector
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


@dataclass
class TrackedObject:
    """An object being tracked with its current spatial position."""
    detection: Detection
    last_seen: float


class DebugVisualizer:
    """Shows camera feed with detection overlay (Main Thread Only)."""

    def __init__(self, target_class: Optional[str] = None):
        self._target_class = target_class
        self._window_created = False
        self._window_name = 'Spatial Tracker Debug'

    def init_window(self):
        """Create the window (must be called from main thread)."""
        if not self._window_created:
            import cv2
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, 800, 600)
            self._window_created = True

    def close(self):
        """Close the window."""
        if self._window_created:
            import cv2
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
            self._window_created = False

    def update_and_show(self, frame_data, detections: list, tracked_detection=None) -> bool:
        """
        Update the display and process UI events.
        Target class is highlighted.
        frame_data can be bytes (JPEG) or numpy array (BGR).
        Returns False if user requested quit (q/ESC), True otherwise.
        """
        import cv2
        import numpy as np

        if not self._window_created:
            self.init_window()

        if frame_data is None:
            # Just pump events
            key = cv2.waitKey(10) & 0xFF
            return not (key == ord('q') or key == 27)

        # Handle both numpy array and bytes input
        if isinstance(frame_data, np.ndarray):
            frame = frame_data.copy()
            # Debug: log first frame info
            if not hasattr(self, '_first_display_logged'):
                self._first_display_logged = True
                print(f"Visualizer received numpy array: shape={frame.shape}, "
                      f"dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
        else:
            # Decode from JPEG bytes
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            key = cv2.waitKey(10) & 0xFF
            return not (key == ord('q') or key == 27)

        h, w = frame.shape[:2]

        # Draw crosshair at center
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (100, 100, 100), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (100, 100, 100), 1)

        # Draw all detections
        for det in detections:
            # Calculate box coordinates
            x1 = int((det.x_center - det.width/2) * w)
            y1 = int((det.y_center - det.height/2) * h)
            x2 = int((det.x_center + det.width/2) * w)
            y2 = int((det.y_center + det.height/2) * h)

            # Color: green if tracked, gray if not target class
            is_target = (self._target_class is None or
                        det.class_name.lower() == self._target_class.lower())
            is_tracked = tracked_detection and det.class_name == tracked_detection.class_name

            if is_tracked:
                color = (0, 255, 0)  # Green for tracked
                thickness = 2
            elif is_target:
                color = (0, 255, 255)  # Yellow for target class
                thickness = 1
            else:
                color = (128, 128, 128)  # Gray for other
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label: class, confidence, and estimated distance
            label = f"{det.class_name} {det.confidence:.0%} d={det.estimated_distance:.1f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Position info for tracked object
            if is_tracked:
                # Spatial position
                lr = "L" if det.spatial_x < -0.2 else "R" if det.spatial_x > 0.2 else "C"
                ud = "UP" if det.spatial_elevation > 0.2 else "DN" if det.spatial_elevation < -0.2 else "MID"
                pos_text = f"[{lr}] [{ud}] dist:{det.estimated_distance:.1f}"
                cv2.putText(frame, pos_text, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show tracking status
        status = f"Tracking: {self._target_class or 'any'}"
        cv2.putText(frame, status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Debug: log before imshow
        if not hasattr(self, '_imshow_logged'):
            self._imshow_logged = True
            print(f"About to imshow: frame shape={frame.shape}, dtype={frame.dtype}")

        cv2.imshow(self._window_name, frame)

        # Check for quit (q or ESC)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            return False

        return True


class SpatialObjectTrackerHRTF:
    """
    Enhanced tracker using HRTF-based spatial audio.
    """

    def __init__(
        self,
        audio_file: str,
        camera=None,
        target_class: Optional[str] = None,
        detector: Optional[ObjectDetector] = None,
        debug: bool = False,
        fast_mode: bool = False,
        use_hailo: bool = False,
    ):
        self.audio_file = audio_file
        self.target_class = target_class
        self.debug = debug
        self.fast_mode = fast_mode
        self.use_hailo = use_hailo

        if use_hailo and not HAILO_AVAILABLE:
            print("Warning: Hailo requested but not available. Falling back to CPU.")
            self.use_hailo = False

        if self.use_hailo:
            print("Initializing Hailo AI acceleration...")
            self.hailo_detector = HailoObjectDetector()
            self.camera = None
            self._own_camera = False
            self.detector = None
        else:
            self.hailo_detector = None
            if detector:
                self.detector = detector
            elif fast_mode:
                self.detector = create_fast_detector(target_class)
            else:
                self.detector = get_detector()

            # Camera setup
            if camera is None:
                camera_type = os.getenv("CAMERA_TYPE")
                if not camera_type:
                    try:
                        from camera.pi_camera import PiCamera
                        if PiCamera().is_available():
                            camera_type = "pi"
                    except Exception:
                        pass
                
                camera_type = (camera_type or "usb").lower()
                if camera_type == "pi":
                    from camera.pi_camera import PiCamera
                    self.camera = PiCamera()
                else:
                    from camera.usb_camera import USBCamera
                    camera_index = int(os.getenv("USB_CAMERA_INDEX", 0))
                    self.camera = USBCamera(camera_index)

                self._own_camera = True
            else:
                self.camera = camera
                self._own_camera = False

        self.player: Optional[HRTFSpatialPlayer] = None
        self.tracked: Optional[TrackedObject] = None
        
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.on_detection: Optional[Callable[[Detection], None]] = None
        self.on_lost: Optional[Callable[[], None]] = None

        self.update_interval = 0.05 if fast_mode else 0.1
        self.lost_timeout = 0.5

        self._smooth_x = 0.0
        self._smooth_el = 0.0
        self._smooth_dist = 3.0
        self._smoothing = 0.5 if fast_mode else 0.3
        self._dist_smoothing = 0.7 if fast_mode else 0.5

        # Thread-safe storage for debug visualizer (frame can be bytes or numpy array)
        self._debug_lock = threading.Lock()
        self._latest_debug_data: Optional[Tuple[Union[bytes, "np.ndarray"], List[Detection], Optional[Detection]]] = None

    def start(self, loop_audio: bool = True):
        """Start tracking."""
        if self._running:
            return

        if not self.use_hailo and not self.camera.is_available():
            raise RuntimeError("Camera not available")

        self._loop_audio = loop_audio
        self._audio_started = False

        print("Loading audio file...")
        self.player = HRTFSpatialPlayer(self.audio_file)
        self.player.set_position(0.0, 0.0)
        self.player.set_distance(5.0)

        if self.use_hailo:
            print("Starting Hailo inference pipeline...")
            self.hailo_detector.start()
            time.sleep(2.0)
        else:
            mode_str = "FAST mode (320px)" if self.fast_mode else "normal mode (640px)"
            print(f"Loading YOLO model ({mode_str})...")
            sys.stdout.flush()
            self.detector._ensure_initialized()
            print("YOLO ready!")

        self._running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

        print(f"\nTracking: {self.target_class or 'any object'}")
        print("(Audio starts when object detected)\n")

    def stop(self):
        """Stop tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.player:
            self.player.stop()
            self.player = None

        if self.use_hailo and self.hailo_detector:
            self.hailo_detector.stop()

        if self._own_camera and self.camera:
            self.camera.release()
            
        print("HRTF Spatial tracking stopped")

    def get_debug_data(self):
        """Get latest data for debug display (called from main thread).
        Returns tuple of (frame_data, detections, tracked_detection) or None.
        frame_data can be bytes (JPEG) or numpy array (BGR).
        """
        with self._debug_lock:
            return self._latest_debug_data

    def _tracking_loop(self):
        """Main tracking loop (runs in background thread)."""
        while self._running:
            try:
                if self.use_hailo:
                    frame_array, all_detections = self.hailo_detector.get_latest()
                    if frame_array is None:
                        time.sleep(self.update_interval)
                        continue
                    # Pass numpy array directly - visualizer handles both types
                    frame_data = frame_array.copy()
                    # Debug: log first frame in tracking loop
                    if not hasattr(self, '_first_track_frame_logged'):
                        self._first_track_frame_logged = True
                        print(f"Tracking loop got frame: shape={frame_data.shape}, "
                              f"min={frame_data.min()}, max={frame_data.max()}")
                else:
                    frame_data = self.camera.capture()
                    if frame_data is None:
                        time.sleep(self.update_interval)
                        continue
                    all_detections = self.detector.detect_from_bytes(frame_data)

                # Filter by target class
                if self.target_class:
                    detections = [
                        d for d in all_detections
                        if d.class_name.lower() == self.target_class.lower()
                    ]
                else:
                    detections = all_detections

                now = time.time()
                
                # Update tracked object
                tracked_det = None
                if detections:
                    best = detections[0]
                    self.tracked = TrackedObject(detection=best, last_seen=now)
                    tracked_det = best

                    if self.player:
                        if not self._audio_started:
                            self.player.play(loop=self._loop_audio)
                            self._audio_started = True
                        elif not self.player.playing:
                            # Resume if paused
                            self.player.resume()

                    # Spatial Audio Logic
                    raw_x = best.spatial_x
                    raw_el = best.spatial_elevation
                    raw_dist = best.estimated_distance

                    self._smooth_x = self._smooth_x * self._smoothing + raw_x * (1 - self._smoothing)
                    self._smooth_el = self._smooth_el * self._smoothing + raw_el * (1 - self._smoothing)
                    self._smooth_dist = self._smooth_dist * self._dist_smoothing + raw_dist * (1 - self._dist_smoothing)

                    if self.player:
                        self.player.set_position(self._smooth_x, self._smooth_el)
                        self.player.set_distance(self._smooth_dist)

                    if self.on_detection:
                        self.on_detection(best)

                elif self.tracked:
                    # Lost logic
                    if now - self.tracked.last_seen > self.lost_timeout:
                        self.tracked = None
                        if self.player:
                            self.player.pause()
                        if self.on_lost:
                            self.on_lost()
                
                # Store data for debug visualizer
                if self.debug and frame_data is not None:
                    with self._debug_lock:
                        self._latest_debug_data = (frame_data, all_detections, tracked_det)

            except Exception as e:
                print(f"Tracking error: {e}")

            time.sleep(self.update_interval)

    def get_current_detection(self) -> Optional[Detection]:
        if self.tracked:
            return self.tracked.detection
        return None


def run_demo(
    audio_file: str,
    target_class: Optional[str] = None,
    debug: bool = False,
    fast: bool = False,
    use_hailo: bool = False,
):
    """Run demo with HRTF spatial audio."""
    print("\n" + "=" * 45)
    print("  HRTF Spatial Object Tracker")
    print("=" * 45)
    print("\nYou'll hear 3D audio cues:")
    print("  LEFT/RIGHT - sound pans to object position")
    print("  UP/DOWN    - bright=high, muffled=low")
    print("  DISTANCE   - louder=close, reverb=far")
    if fast:
        print("\nFAST mode: optimized for Raspberry Pi / smooth tracking")
    if use_hailo:
        print("\nHAILO mode: Hardware acceleration enabled")
    if debug:
        print("DEBUG mode: window will show camera + detections")
    print("\nPress Ctrl+C or 'q' to stop.\n")

    # Create visualizer BEFORE tracker to avoid Qt/GLib threading conflicts
    visualizer = None
    if debug:
        visualizer = DebugVisualizer(target_class)
        visualizer.init_window()  # Create window before GLib main loop starts

    tracker = SpatialObjectTrackerHRTF(
        audio_file=audio_file,
        target_class=target_class,
        debug=debug,
        fast_mode=fast,
        use_hailo=use_hailo,
    )

    # Flag for clean shutdown
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nShutdown requested...")
        shutdown_requested.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def on_detect(det: Detection):
        pass  # Let visualizer handle UI

    def on_lost():
        pass

    tracker.on_detection = on_detect
    tracker.on_lost = on_lost

    try:
        tracker.start()

        # Main thread loop
        _main_loop_debug_logged = False
        while not shutdown_requested.is_set():
            if debug and visualizer:
                # Get latest data from tracker
                data = tracker.get_debug_data()
                if data:
                    frame_data, dets, tracked = data
                    if not _main_loop_debug_logged:
                        _main_loop_debug_logged = True
                        print(f"Main loop got data: frame_data type={type(frame_data)}, "
                              f"shape={frame_data.shape if hasattr(frame_data, 'shape') else 'N/A'}")
                    keep_running = visualizer.update_and_show(frame_data, dets, tracked)
                else:
                    keep_running = visualizer.update_and_show(None, [], None)

                if not keep_running:
                    break
            else:
                # No GUI, just sleep but check shutdown flag
                shutdown_requested.wait(timeout=0.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        tracker.stop()
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    import argparse

    default_audio = os.path.join(os.path.dirname(__file__), "test_tone.wav")

    parser = argparse.ArgumentParser(
        description="Spatial Object Tracker with HRTF 3D audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stereomusic.spatial_tracker_hrtf                       # Track any object
  python -m stereomusic.spatial_tracker_hrtf -t person             # Track only people
  python -m stereomusic.spatial_tracker_hrtf -t person -f          # Fast mode (for Pi)
  python -m stereomusic.spatial_tracker_hrtf -t "cell phone" -d    # Debug view
  python -m stereomusic.spatial_tracker_hrtf -t person -f -d       # Fast + debug

Performance tips:
  - Use -f (fast mode) for Raspberry Pi or smoother tracking
  - Use -t to specify target class (faster than detecting all)
  - Fast mode uses 320px input (vs 640px normal) = ~4x faster
        """
    )
    parser.add_argument("audio", nargs="?", default=default_audio,
                        help="Audio file to play (default: test_tone.wav)")
    parser.add_argument("-t", "--target", default=None,
                        help="Target object class to track (e.g., 'person', 'cell phone')")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Show debug window with camera feed and detections")
    parser.add_argument("-f", "--fast", action="store_true",
                        help="Fast mode: 320px input + class filtering (recommended for Pi if no Hailo)")
    parser.add_argument("--hailo", action="store_true", help="Use Hailo AI Hat for acceleration")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        sys.exit(1)

    run_demo(args.audio, args.target, args.debug, args.fast, args.hailo)
