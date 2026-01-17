"""
Spatial Music - Audio that follows objects in space.

Modules:
    spatial_audio: Core spatial audio player
    object_detector: YOLO-based object detection
    spatial_tracker: Integration of camera + detection + audio
"""

from .spatial_audio import SpatialAudioPlayer
from .object_detector import ObjectDetector, Detection, get_detector
from .spatial_tracker import SpatialObjectTracker
