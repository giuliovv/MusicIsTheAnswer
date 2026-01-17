#!/usr/bin/env python3
"""
Test script for spatial object tracking.

This script tests the complete pipeline:
1. Camera capture (USB webcam)
2. Object detection (YOLO)
3. Spatial audio output (stereo positioning)

Run this on your laptop to verify everything works before deploying to Pi.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_camera():
    """Test camera is working."""
    print("\n=== Testing Camera ===")
    from camera.usb_camera import USBCamera, list_available_cameras

    cameras = list_available_cameras()
    if not cameras:
        print("ERROR: No cameras found!")
        return False

    print(f"Found cameras at indices: {cameras}")

    cam = USBCamera(cameras[0])
    if not cam.is_available():
        print("ERROR: Camera not available!")
        return False

    print("Camera is available")

    # Try to capture
    image_bytes = cam.capture()
    if image_bytes is None:
        print("ERROR: Could not capture image!")
        cam.release()
        return False

    print(f"Captured image: {len(image_bytes)} bytes")
    cam.release()
    return True


def test_detection():
    """Test object detection is working."""
    print("\n=== Testing Object Detection ===")
    from camera.usb_camera import USBCamera, list_available_cameras
    from stereomusic.object_detector import get_detector

    cameras = list_available_cameras()
    if not cameras:
        print("Skipping - no camera")
        return False

    cam = USBCamera(cameras[0])
    image_bytes = cam.capture()
    cam.release()

    if image_bytes is None:
        print("Skipping - could not capture")
        return False

    print("Initializing YOLO detector (first run downloads model)...")
    detector = get_detector()

    print("Running detection...")
    detections = detector.detect_from_bytes(image_bytes)

    print(f"Found {len(detections)} objects:")
    for det in detections[:5]:  # Show top 5
        pos = "LEFT" if det.spatial_x < -0.3 else "RIGHT" if det.spatial_x > 0.3 else "CENTER"
        print(f"  - {det.class_name} ({det.confidence:.0%}) at {pos}")

    return True


def test_spatial_audio():
    """Test spatial audio is working."""
    print("\n=== Testing Spatial Audio ===")
    import time
    from stereomusic.spatial_audio import SpatialAudioPlayer

    audio_file = os.path.join(os.path.dirname(__file__), "stereomusic", "test_tone.wav")

    if not os.path.exists(audio_file):
        print("Generating test tone...")
        from stereomusic.generate_test_tone import generate_tone
        generate_tone(audio_file)

    print(f"Loading audio: {audio_file}")
    player = SpatialAudioPlayer(audio_file)

    print("Playing test: sound moves LEFT -> CENTER -> RIGHT")
    player.play(loop=True)

    positions = [(-1.0, "LEFT"), (0.0, "CENTER"), (1.0, "RIGHT")]
    for pos, name in positions:
        player.set_position(pos)
        print(f"  Position: {name}")
        time.sleep(1.0)

    player.stop()
    print("Audio test complete")
    return True


def test_full_tracking():
    """Test the full tracking pipeline."""
    print("\n=== Testing Full Spatial Tracking ===")
    print("This will track objects and play audio from their direction.")
    print("Point your camera at objects (try a person, phone, or cup).")
    print("Press Ctrl+C after a few seconds to stop.\n")

    import time
    from stereomusic.spatial_tracker import SpatialObjectTracker

    audio_file = os.path.join(os.path.dirname(__file__), "stereomusic", "test_tone.wav")

    tracker = SpatialObjectTracker(audio_file=audio_file)

    # Callbacks
    def on_detect(det):
        pos = "LEFT" if det.spatial_x < -0.3 else "RIGHT" if det.spatial_x > 0.3 else "CENTER"
        dist = "close" if det.estimated_distance < 2 else "far" if det.estimated_distance > 3 else "medium"
        print(f"  [{det.class_name}] {pos}, {dist}")

    tracker.on_detection = on_detect

    try:
        tracker.start()
        print("Tracking started! Move objects around...\n")

        # Run for 10 seconds
        for i in range(10):
            time.sleep(1)
            print(f"  ... {10-i} seconds remaining")

        print("\nTest complete!")
        return True

    except KeyboardInterrupt:
        print("\nStopped by user")
        return True
    finally:
        tracker.stop()


def main():
    print("=" * 50)
    print("Spatial Object Tracking - Test Suite")
    print("=" * 50)

    # Run tests in order
    tests = [
        ("Camera", test_camera),
        ("Object Detection", test_detection),
        ("Spatial Audio", test_spatial_audio),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nAll basic tests passed!")
        response = input("\nRun full tracking test? (y/n): ")
        if response.lower() == 'y':
            test_full_tracking()
    else:
        print("\nSome tests failed. Fix issues before running full tracking.")


if __name__ == "__main__":
    main()
