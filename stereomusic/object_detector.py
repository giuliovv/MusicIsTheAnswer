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
        # Use area as proxy for distance
        # Large area (>0.3) = close, small area (<0.02) = far
        area = max(0.001, self.area)
        # Inverse relationship: bigger = closer
        distance = 1.0 / (area * 10 + 0.2)
        return max(1.0, min(5.0, distance))


class ObjectDetector:
    """YOLO-based object detector."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the detector.

        Args:
            model_name: YOLO model to use (yolov8n.pt is smallest/fastest)
            confidence_threshold: Minimum confidence to report detections
        """
        self.model = None
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if not self._initialized:
            try:
                from ultralytics import YOLO
                print(f"Loading YOLO model: {self.model_name}")
                self.model = YOLO(self.model_name)
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

        # Run inference
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

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


def get_detector(model_name: str = "yolov8n.pt") -> ObjectDetector:
    """Get or create the singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = ObjectDetector(model_name)
    return _detector


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
