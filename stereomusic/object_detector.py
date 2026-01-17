"""
Object Detection module using YOLO.
Returns bounding boxes with class labels and coordinates.
"""

from __future__ import annotations  # Python 3.8 compatibility
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class Detection:
    """A detected object with its bounding box and metadata."""
    class_name: str
    confidence: float
    x_center: float  # 0.0 (left edge) to 1.0 (right edge) - normalized
    y_center: float  # 0.0 (top) to 1.0 (bottom) - normalized
    width: float     # Normalized width (0.0 to 1.0)
    height: float    # Normalized height (0.0 to 1.0)

    @property
    def area(self) -> float:
        """Relative area of the bounding box (0.0 to 1.0)."""
        return self.width * self.height

    @property
    def spatial_x(self) -> float:
        """Convert to spatial audio x position: -1.0 (left) to 1.0 (right)."""
        return (self.x_center - 0.5) * 2.0

    @property
    def spatial_elevation(self) -> float:
        """
        Convert y position to elevation: -1.0 (below) to 1.0 (above).

        Note: In image coordinates, y=0 is top, y=1 is bottom.
        We invert this so objects at top of frame are "above" (positive elevation).
        """
        return (0.5 - self.y_center) * 2.0

    @property
    def estimated_distance(self) -> float:
        """
        Estimate distance based on bounding box size.
        Larger objects are assumed to be closer.
        Returns: 1.0 (very close) to 5.0 (far away)
        """
        # Use area as proxy for distance, but map it through
        # a clamped, roughly linear curve so small jitters in
        # the YOLO box size don't flip "closer" to "farther".

        # Clamp area into a reasonable range for typical objects
        # in the frame. Very tiny boxes (<0.5% of image) are
        # treated as "far", very large boxes (>25%) as "very close".
        area = max(0.0005, float(self.area))
        min_area = 0.005   # ~0.5% of image -> far
        max_area = 0.25    # 25% of image   -> very close

        if area <= min_area:
            closeness = 0.0
        elif area >= max_area:
            closeness = 1.0
        else:
            closeness = (area - min_area) / (max_area - min_area)

        # Map closeness (0 = far, 1 = close) to distance 5..1
        distance = 5.0 - 4.0 * closeness
        return distance


class ObjectDetector:
    """YOLO-based object detector with optimization options."""

    # Common class IDs for filtering (COCO dataset)
    COMMON_CLASSES = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7,
        'cat': 15,
        'dog': 16,
        'backpack': 24,
        'umbrella': 25,
        'handbag': 26,
        'bottle': 39,
        'cup': 41,
        'fork': 42,
        'knife': 43,
        'spoon': 44,
        'bowl': 45,
        'banana': 46,
        'apple': 47,
        'chair': 56,
        'couch': 57,
        'bed': 59,
        'laptop': 63,
        'mouse': 64,
        'remote': 65,
        'keyboard': 66,
        'cell phone': 67,
        'book': 73,
        'clock': 74,
        'scissors': 76,
        'toothbrush': 79,
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        input_size: int = 640,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the detector.

        Args:
            model_name: YOLO model to use (yolov8n.pt is smallest/fastest)
            confidence_threshold: Minimum confidence to report detections
            input_size: Input image size (smaller = faster). Options: 640, 480, 320, 256
            target_classes: List of class names to detect (None = all). Filtering at
                           inference is faster than filtering after.
        """
        self.model = None
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.target_classes = target_classes
        self._class_ids = None  # Will be set on init
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if not self._initialized:
            try:
                from ultralytics import YOLO
                print(f"Loading YOLO model: {self.model_name} (input size: {self.input_size})")
                self.model = YOLO(self.model_name)

                # Set up class filtering if target classes specified
                if self.target_classes:
                    self._class_ids = []
                    for cls_name in self.target_classes:
                        cls_lower = cls_name.lower()
                        if cls_lower in self.COMMON_CLASSES:
                            self._class_ids.append(self.COMMON_CLASSES[cls_lower])
                        else:
                            # Try to find in model's class names
                            for idx, name in self.model.names.items():
                                if name.lower() == cls_lower:
                                    self._class_ids.append(idx)
                                    break
                    if self._class_ids:
                        print(f"Filtering to classes: {self.target_classes} (IDs: {self._class_ids})")

                self._initialized = True
                print("YOLO model loaded successfully")
            except ImportError:
                raise ImportError(
                    "ultralytics not installed. Run: pip install ultralytics"
                )

    def detect_from_bytes(self, image_bytes: bytes) -> list[Detection]:
        """
        Detect objects in a JPEG image.

        Args:
            image_bytes: JPEG image data

        Returns:
            List of Detection objects
        """
        import cv2

        # Decode JPEG bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return []

        return self.detect_from_frame(frame)

    def detect_from_frame(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect objects in a numpy frame (BGR format from OpenCV).

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        self._ensure_initialized()

        height, width = frame.shape[:2]

        # Build inference kwargs
        kwargs = {
            'verbose': False,
            'conf': self.confidence_threshold,
            'imgsz': self.input_size,
        }

        # Add class filtering if set (faster than filtering after)
        if self._class_ids:
            kwargs['classes'] = self._class_ids

        # Run inference
        results = self.model(frame, **kwargs)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                # Normalize coordinates to 0-1 range
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    x_center=x_center,
                    y_center=y_center,
                    width=box_width,
                    height=box_height
                ))

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_from_file(self, filepath: str) -> list[Detection]:
        """Detect objects in an image file."""
        import cv2
        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not read image: {filepath}")
        return self.detect_from_frame(frame)


# Singleton instance for reuse
_detector: Optional[ObjectDetector] = None


def get_detector(
    model_name: str = "yolov8n.pt",
    input_size: int = 640,
    target_classes: Optional[List[str]] = None,
) -> ObjectDetector:
    """Get or create the singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = ObjectDetector(
            model_name=model_name,
            input_size=input_size,
            target_classes=target_classes,
        )
    return _detector


def create_fast_detector(target_class: Optional[str] = None) -> ObjectDetector:
    """
    Create a fast detector optimized for real-time tracking.

    Uses:
    - Smaller input size (320 instead of 640)
    - Class filtering at inference time
    - Lower confidence threshold for smoother tracking
    """
    target_classes = [target_class] if target_class else None
    return ObjectDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.4,
        input_size=320,  # Much faster than 640
        target_classes=target_classes,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python object_detector.py <image_file>")
        print("\nDetects objects and shows their spatial positions.")
        sys.exit(1)

    detector = get_detector()
    detections = detector.detect_from_file(sys.argv[1])

    print(f"\nFound {len(detections)} objects:\n")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det.class_name} ({det.confidence:.1%})")
        print(f"   Position: x={det.spatial_x:+.2f} (left/right)")
        print(f"   Distance: {det.estimated_distance:.1f} (1=close, 5=far)")
        print()
