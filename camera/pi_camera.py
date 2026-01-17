"""
Raspberry Pi Camera 3 implementation using picamera2.

This uses a subprocess call to system Python because:
- picamera2/libcamera only works with system Python (apt install)
- Our venv uses Python 3.11 for tflite-runtime compatibility
- This hybrid approach lets both work together

To use:
1. Install: sudo apt install -y python3-picamera2
2. Change CAMERA_TYPE=pi in .env
"""

import subprocess
import sys
import base64
from typing import Optional
from .base import CameraCapture


# Python script that runs with system Python to capture images
CAPTURE_SCRIPT = '''
import sys
import base64
import io

try:
    from picamera2 import Picamera2
    from PIL import Image
    import numpy as np
    
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    
    # Small delay to let camera warm up
    import time
    time.sleep(0.5)
    
    # Capture
    array = cam.capture_array()

    # Rotate 180 degrees so the image is upright if the camera
    # is mounted upside down (common on some Pi setups).
    array = np.rot90(array, 2)
    
    # Convert to JPEG
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    
    # Output as base64
    print(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    
    cam.stop()
    cam.close()
    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
'''

# Script to check if camera is available
CHECK_SCRIPT = '''
import sys
try:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.close()
    print("OK")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
'''


class PiCamera(CameraCapture):
    """
    Raspberry Pi Camera 3 capture using picamera2 via system Python.
    
    Uses subprocess to call system Python (which has picamera2 installed via apt)
    while allowing the main application to run in a venv with different Python version.
    """
    
    def __init__(self):
        """Initialize Pi Camera."""
        self._available: Optional[bool] = None
    
    def _get_system_python(self) -> str:
        """Get the path to system Python 3."""
        # Try common locations for system Python
        candidates = [
            "/usr/bin/python3",
            "/usr/bin/python",
        ]
        for python in candidates:
            try:
                result = subprocess.run(
                    [python, "-c", "import picamera2; print('ok')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return python
            except Exception:
                continue
        return "/usr/bin/python3"  # fallback
    
    def is_available(self) -> bool:
        """Check if Pi Camera is available via system Python."""
        if self._available is not None:
            return self._available
        
        try:
            python = self._get_system_python()
            result = subprocess.run(
                [python, "-c", CHECK_SCRIPT],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._available = result.returncode == 0 and "OK" in result.stdout
            if not self._available:
                print(f"Pi Camera check failed: {result.stderr}")
            return self._available
        except subprocess.TimeoutExpired:
            print("Pi Camera check timed out")
            self._available = False
            return False
        except Exception as e:
            print(f"Pi Camera check error: {e}")
            self._available = False
            return False
    
    def capture(self) -> Optional[bytes]:
        """
        Capture an image from the Pi Camera using system Python.
        
        Returns:
            bytes: JPEG image data, or None if capture failed
        """
        try:
            python = self._get_system_python()
            result = subprocess.run(
                [python, "-c", CAPTURE_SCRIPT],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Pi Camera capture failed: {result.stderr}")
                return None
            
            # Decode base64 output
            image_b64 = result.stdout.strip()
            if not image_b64:
                print("Pi Camera returned empty image")
                return None
            
            return base64.b64decode(image_b64)
            
        except subprocess.TimeoutExpired:
            print("Pi Camera capture timed out")
            return None
        except Exception as e:
            print(f"Pi Camera capture error: {e}")
            return None
    
    def release(self):
        """Release camera resources (no-op for subprocess approach)."""
        pass


if __name__ == "__main__":
    print("Testing Pi Camera (via system Python subprocess)...")
    cam = PiCamera()
    
    print("Checking availability...")
    if cam.is_available():
        print("✅ Camera is available!")
        print("Capturing image...")
        if cam.save_capture("test_pi_capture.jpg"):
            print("✅ Saved test image to test_pi_capture.jpg")
        else:
            print("❌ Failed to capture image")
    else:
        print("❌ Camera not available")
        print("   Make sure picamera2 is installed: sudo apt install python3-picamera2")
    
    cam.release()
