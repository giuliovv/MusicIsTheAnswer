
import cv2
import sys
import os
import subprocess
import time

def check_usb_cameras():
    print("\n--- Checking USB Cameras (OpenCV) ---")
    available_indices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  [SUCCESS] Frame captured from index {i}: {frame.shape}")
                available_indices.append(i)
            else:
                print(f"  [WARNING] Camera open, but failed to read frame from index {i}")
            cap.release()
        # else:
            # print(f"  No camera at index {i}")
    
    if not available_indices:
        print("No working USB cameras found via OpenCV.")
    else:
        print(f"Working USB camera indices: {available_indices}")

    return available_indices

def check_pi_camera():
    print("\n--- Checking Raspberry Pi Camera (picamera2) ---")
    # Using the same logic as camera/pi_camera.py
    
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
    
    # Try system python
    python_paths = ["/usr/bin/python3", "/usr/bin/python", sys.executable]
    
    found = False
    for py in python_paths:
        print(f"Testing with python: {py}")
        try:
            result = subprocess.run(
                [py, "-c", CHECK_SCRIPT],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print(f"  [SUCCESS] Pi Camera detected using {py}")
                found = True
                break
            else:
                print(f"  [FAILED] {result.stderr.strip() if result.stderr else 'Unknown error'}")
        except Exception as e:
            print(f"  [ERROR] Failed to run subprocess: {e}")
            
    if not found:
        print("Pi Camera not detected via picamera2.")
    return found

def main():
    print(f"Python executable: {sys.executable}")
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Error: OpenCV (cv2) module not found in this environment!")
        return

    usb_indices = check_usb_cameras()
    pi_cam = check_pi_camera()

    print("\n--- Summary ---")
    if pi_cam:
        print("Recommendation: Use CAMERA_TYPE=pi")
    elif usb_indices:
        print(f"Recommendation: Use CAMERA_TYPE=usb (Index: {usb_indices[0]})")
    else:
        print("No functional cameras found. Check connections or permissions.")

if __name__ == "__main__":
    main()
