"""
USB Camera implementation using OpenCV.
Works with any standard USB webcam.
"""

from __future__ import annotations  # Python 3.8 compatibility
import cv2
from typing import Optional, List
from .base import CameraCapture


class USBCamera(CameraCapture):
    """USB camera capture using OpenCV."""
    
    def __init__(self, device_index: int = 0):
        """
        Initialize USB camera.
        
        Args:
            device_index: Camera device index (usually 0 for first camera)
        """
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
    
    def _get_capture(self) -> Optional[cv2.VideoCapture]:
        """Get or create video capture object."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device_index)
            # Give camera time to initialize
            if self._cap.isOpened():
                # Set resolution (adjust as needed)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self._cap
    
    def is_available(self) -> bool:
        """Check if USB camera is available."""
        cap = self._get_capture()
        return cap is not None and cap.isOpened()
    
    def capture(self) -> Optional[bytes]:
        """
        Capture an image from the USB camera.
        
        Returns:
            bytes: JPEG image data, or None if capture failed
        """
        cap = self._get_capture()
        if cap is None or not cap.isOpened():
            print("Error: Could not open USB camera")
            return None
        
        # Flush the buffer by reading several frames
        # This ensures we get a fresh frame, not a stale buffered one
        for _ in range(5):
            cap.grab()
        
        # Capture fresh frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            return None
        
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            print("Error: Could not encode frame as JPEG")
            return None
        
        return buffer.tobytes()
    
    def release(self):
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()


def list_available_cameras(max_index: int = 10) -> list[int]:
    """
    List available camera device indices.
    
    Args:
        max_index: Maximum index to check
        
    Returns:
        list: Available camera indices
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


if __name__ == "__main__":
    # Quick test
    print("Checking for available cameras...")
    cameras = list_available_cameras()
    print(f"Found cameras at indices: {cameras}")
    
    if cameras:
        print(f"\nTesting camera at index {cameras[0]}...")
        cam = USBCamera(cameras[0])
        if cam.is_available():
            print("Camera is available!")
            if cam.save_capture("test_capture.jpg"):
                print("Saved test image to test_capture.jpg")
            else:
                print("Failed to save test image")
        cam.release()
    else:
        print("No cameras found!")
