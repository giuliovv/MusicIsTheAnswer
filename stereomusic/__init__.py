"""
Spatial Music - Audio that follows objects in space.

Modules:
    spatial_audio: Basic spatial audio player (simple panning)
    spatial_audio_hrtf: Enhanced HRTF-based 3D audio with elevation
    object_detector: YOLO-based object detection
    spatial_tracker: Basic integration (camera + detection + audio)
    spatial_tracker_hrtf: Enhanced integration with HRTF audio
"""

from .spatial_audio import SpatialAudioPlayer
from .spatial_audio_hrtf import HRTFSpatialPlayer
from .object_detector import ObjectDetector, Detection, get_detector, create_fast_detector
from .spatial_tracker import SpatialObjectTracker
from .spatial_tracker_hrtf import SpatialObjectTrackerHRTF
