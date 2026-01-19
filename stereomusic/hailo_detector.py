
import sys
import os
import time
import threading
import json
import queue

# Add system site-packages to path for access to gi (PyGObject) and hailo
# These are system packages that can't be easily pip-installed in a venv
_system_site_packages = [
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.11/dist-packages',
    '/usr/lib/python3.12/dist-packages',
    '/usr/lib/python3.13/dist-packages',
]
for _path in _system_site_packages:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo
from .object_detector import Detection

class HailoObjectDetector:
    """
    Object detector using Hailo AI Hat (RPi 5) via GStreamer.
    Replaces both the camera source and the inference engine.
    """
    def __init__(self, hef_path=None, minimal=False):
        # Auto-detect Hailo device type and select appropriate HEF
        self.device_type = self._detect_hailo_device()
        
        if hef_path is None:
            # Auto-select HEF based on device
            if self.device_type == "hailo8":
                self.hef_path = "/usr/share/hailo-models/yolov8s_h8.hef"
            else:
                self.hef_path = "/usr/share/hailo-models/yolov8s_h8l.hef" if os.path.exists("/usr/share/hailo-models/yolov8s_h8l.hef") else hef_path
        else:
            self.hef_path = hef_path
            
        self.minimal = minimal
        self.running = False
        self.pipeline = None
        self._bus = None
        self.thread = None
        
        self.latest_frame = None
        self.latest_detections = []
        self.lock = threading.Lock()
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Post-process library path
        self.post_process_so = self._find_post_process_so()
        if not self.post_process_so:
            print("WARNING: Could not find libyolo_post.so. Inference might fail.")
            self.post_process_so = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_post.so"
        
        # Config file for post-processing (COCO labels)
        self.config_json = self._find_config_json()
        
        print(f"Hailo device: {self.device_type}")
        print(f"Using HEF: {self.hef_path}")
        print(f"Post-process: {self.post_process_so}")
    
    def _detect_hailo_device(self):
        """Detect whether we have Hailo8 or Hailo8L."""
        try:
            import subprocess
            result = subprocess.run(['hailortcli', 'fw-control', 'identify'], 
                                    capture_output=True, text=True, timeout=5)
            output = result.stdout.lower()
            if 'hailo-8l' in output or 'h8l' in output:
                return "hailo8l"
            elif 'hailo-8' in output or 'h8' in output:
                return "hailo8"
        except Exception:
            pass
        # Default to hailo8 (full version)
        return "hailo8"
    
    def _find_config_json(self):
        """Find the YOLO config JSON file."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "yolov8_coco.json"),
            "/usr/share/hailo-models/yolov8.json",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _find_post_process_so(self):
        # Use libyolo_hailortpp_post.so for YOLOv8 (same as hailo-apps-infra)
        candidates = [
            "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
            "/usr/lib/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _create_pipeline_string(self):
        # Configuration
        # We use libcamerasrc (RPi Camera)
        # 640x640 input for YOLOv8 (standard)
        # Use aspect ratio preserving scaling with videobox for letterboxing

        # Pipeline components:
        # 1. Source: libcamerasrc -> videoscale (preserve aspect) -> videobox (letterbox) -> tee
        # 2. Branch A (Inference): -> hailonet -> hailofilter -> identity (callback) -> fakesink
        # 3. Branch B (Display/Python): -> videoconvert -> appsink

        input_width = 640
        input_height = 640

        # Camera outputs 1536x864 (16:9). To fit in 640x640:
        # Scale to 640x360 (preserving 16:9), then add 140px padding top and bottom
        # Or we can just scale to fit and accept some distortion for simplicity

        # Source (RPi Cam) - capture at native resolution, then scale
        # Using videoscale with add-borders=true for letterboxing
        source = (
            f"libcamerasrc name=src ! "
            f"video/x-raw, format=RGB, width=1536, height=864 ! "
            f"queue name=src_q max-size-buffers=3 leaky=downstream ! "
            f"videoscale add-borders=true n-threads=2 ! "
            f"video/x-raw, format=RGB, width={input_width}, height={input_height}, pixel-aspect-ratio=1/1 ! "
            f"videoconvert n-threads=2 ! "
            f"tee name=t"
        )

        # Inference Branch - use filter_letterbox since we're now letterboxing
        batch_size = 1

        inference = (
            f"t. ! queue name=inf_q max-size-buffers=3 leaky=no ! "
            f"videoconvert n-threads=2 ! "
            f"hailonet name=hailonet hef-path={self.hef_path} batch-size={batch_size} force-writable=true ! "
            f"queue name=filter_q max-size-buffers=3 ! "
            f"hailofilter so-path={self.post_process_so} function-name=filter_letterbox qos=false ! "
            f"queue name=identity_q ! identity name=identity_callback ! fakesink"
        )

        # Output Branch (to Python) - get frames for visualization
        output = (
            f"t. ! queue name=out_q max-size-buffers=2 leaky=downstream ! "
            f"videoconvert n-threads=2 ! video/x-raw, format=BGR ! "
            f"appsink name=app_sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        return f"{source} {inference} {output}"

    def start(self):
        if self.running:
            return

        pipeline_str = self._create_pipeline_string()
        print(f"Starting Hailo pipeline...")
        # print(f"Pipeline: {pipeline_str}")
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            print(f"Failed to create pipeline: {e}")
            raise

        # Connect signals
        appsink = self.pipeline.get_by_name("app_sink")
        if appsink:
            appsink.connect("new-sample", self._on_new_sample)
        else:
            print("Error: Could not find app_sink")

        identity = self.pipeline.get_by_name("identity_callback")
        if identity:
            identity_pad = identity.get_static_pad("sink")
            identity_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_probe)
        else:
            print("Error: Could not find identity_callback")

        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        # Monitor bus for errors using polling instead of GLib main loop
        # This avoids conflicts with OpenCV's Qt backend
        bus = self.pipeline.get_bus()
        self._bus = bus

        # Start a simple bus polling thread instead of GLib main loop
        self.thread = threading.Thread(target=self._bus_poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False

        # Wait for polling thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

        self._bus = None
        self.thread = None

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Get actual dimensions from caps
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        if width is None or height is None:
            # Fallback to expected dimensions
            width, height = 640, 640

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            # Create array from buffer with actual dimensions (BGR = 3 bytes per pixel)
            expected_size = width * height * 3
            actual_size = len(map_info.data)

            if actual_size >= expected_size:
                arr = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                # Make a copy before unmapping (required!)
                frame_copy = arr.copy()
                with self.lock:
                    self.latest_frame = frame_copy
                    # Debug: print once when first frame arrives
                    if not hasattr(self, '_first_frame_logged'):
                        self._first_frame_logged = True
                        print(f"First frame received: {width}x{height}, shape={frame_copy.shape}, "
                              f"min={frame_copy.min()}, max={frame_copy.max()}")
            else:
                print(f"Buffer size mismatch: expected {expected_size}, got {actual_size}")
        except Exception as e:
            print(f"Frame conversion error: {e}")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_probe(self, pad, info):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        # Debug: log that probe is being called
        if not hasattr(self, '_probe_called_logged'):
            self._probe_called_logged = True
            print("Probe callback is being called")

        # Get detections from Hailo meta
        roi = hailo.get_roi_from_buffer(buffer)
        detections = []

        # Debug: check all objects in ROI
        all_objects = list(roi.get_objects())
        if all_objects and not hasattr(self, '_roi_objects_logged'):
            self._roi_objects_logged = True
            print(f"ROI has {len(all_objects)} objects")
            for i, obj in enumerate(all_objects[:5]):  # Show first 5
                print(f"  Object {i}: type={type(obj).__name__}")

        for obj in roi.get_objects_typed(hailo.HAILO_DETECTION):
            # Parse to our Detection format
            # Hailo gives bbox in normalized coordinates relative to ROI (which is full frame here)
            bbox = obj.get_bbox()

            # Confidence
            conf = obj.get_confidence()
            label = obj.get_label()

            # Convert to Detection object
            det = Detection(
                class_name=label,
                confidence=conf,
                x_center=(bbox.xmin() + bbox.xmax()) / 2,
                y_center=(bbox.ymin() + bbox.ymax()) / 2,
                width=bbox.width(),
                height=bbox.height()
            )
            detections.append(det)

        # Debug: log first detection
        if detections and not hasattr(self, '_first_detection_logged'):
            self._first_detection_logged = True
            print(f"First detection: {detections[0].class_name} ({detections[0].confidence:.2f}) "
                  f"at ({detections[0].x_center:.2f}, {detections[0].y_center:.2f})")

        with self.lock:
            self.latest_detections = detections

        return Gst.PadProbeReturn.OK

    def _bus_poll_loop(self):
        """Poll GStreamer bus for messages instead of using GLib main loop."""
        while self.running:
            if self._bus:
                msg = self._bus.timed_pop(100 * Gst.MSECOND)  # 100ms timeout
                if msg:
                    self._handle_bus_message(msg)
            else:
                time.sleep(0.1)

    def _handle_bus_message(self, message):
        """Handle a GStreamer bus message."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer Error: {err}, {debug}")
            self.running = False
        elif t == Gst.MessageType.EOS:
            print("End of stream")
            self.running = False

    def get_latest(self):
        """Return (frame, detections)"""
        with self.lock:
            return self.latest_frame, self.latest_detections

