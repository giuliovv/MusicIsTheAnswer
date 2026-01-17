
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
        self.loop = None
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
        # Hailo-8L requires specific hef
        
        # Pipeline components:
        # 1. Source: libcamerasrc -> videoscale -> video/x-raw,640x640 -> split (tee)
        # 2. Branch A (Inference): -> hailonet -> hailofilter -> identity (callback) -> fakesink
        # 3. Branch B (Display/Python): -> videoconvert -> appsink
        
        input_width = 640
        input_height = 640
        
        # Source (RPi Cam) with proper format negotiation
        source = (
            f"libcamerasrc name=src ! "
            f"video/x-raw, format=RGB, width=1536, height=864 ! "
            f"queue name=src_q max-size-buffers=3 leaky=downstream ! "
            f"videoscale n-threads=2 ! "
            f"videoconvert n-threads=2 ! "
            f"video/x-raw, format=RGB, width={input_width}, height={input_height} ! "
            f"tee name=t"
        )
        
        # Inference Branch
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
        
        # Start GLib loop in thread (needed for GStreamer messages usually, though often not strictly required for just flow)
        # but better to have it for bus handling
        self.loop = GLib.MainLoop()
        self.thread = threading.Thread(target=self.loop.run)
        self.thread.start()
        
        # Monitor bus for errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def stop(self):
        if not self.running:
            return
            
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.loop:
            self.loop.quit()
        
        if self.thread:
            self.thread.join()

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
            
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Convert buffer to numpy array
        # This is a bit complex in pure python-gst without specific helpers, 
        # but basic approach:
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
            
        try:
            # We know it's 640x640 BGR (3 bytes per pixel)
            arr = np.ndarray(
                shape=(640, 640, 3),
                dtype=np.uint8,
                buffer=map_info.data
            )
            
            with self.lock:
                self.latest_frame = arr.copy()
        finally:
            buf.unmap(map_info)
            
        return Gst.FlowReturn.OK

    def _on_probe(self, pad, info):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK
            
        # Get detections from Hailo meta
        roi = hailo.get_roi_from_buffer(buffer)
        detections = []
        
        for obj in roi.get_objects_typed(hailo.HAILO_DETECTION):
            # Parse to our Detection format
            # Hailo gives bbox in normalized coordinates relative to ROI (which is full frame here)
            bbox = obj.get_bbox()
            
            # Confidence
            conf = obj.get_confidence()
            label = obj.get_label()
            
            # Convert to our Detection class format (from object_detector.py) if we imported it
            # But let's just store dicts or similar for now or duplicate the class
            # We will use a simple dict struct to avoid circular deps or complex imports inside thread
            
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
            
        with self.lock:
            self.latest_detections = detections
            
        return Gst.PadProbeReturn.OK

    def _on_bus_message(self, bus, message):
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

