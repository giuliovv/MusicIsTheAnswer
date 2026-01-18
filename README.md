# 🎧 Spatial Audio Object Detection for the Blind

**Winner of the EF x Raspberry Pi x Asteroid Accessibility Hackathon (17 January 2025)**

A real-time spatial awareness system that helps blind and visually impaired users understand their surroundings through **3D audio positioning**. Objects are detected via on-device AI and represented as stereo sounds that originate from their actual location in space.

## The Problem

Existing solutions for blind users have critical limitations:

**Traditional LLM approach:**
- Camera → LLM → "There's a cup on your left, a phone to your right, and keys in the center"
- **Mental overload**: User must parse verbal descriptions and mentally map positions
- **Slow**: Waiting for full sentence descriptions
- **Overlapping voices**: Multiple objects create confusing audio clutter
- **Not instinctive**: Requires conscious processing of language

**Our solution:**
- Camera → YOLO → Positioned audio beeps
- **Intuitive**: Brain processes spatial audio instantly, no thinking required
- **Fast**: Immediate perception of surroundings
- **Parallel**: Multiple objects heard simultaneously without confusion
- **Natural**: Like hearing sounds in the real world

The difference is like reading turn-by-turn directions versus using GPS with map visualization—one requires mental processing, the other is immediate understanding.

## Our Solution

We use **spatial audio instead of verbal descriptions** to represent object positions:

1. **Camera captures the scene** → YOLOv8n detects objects and positions on-device
2. **Position mapped to stereo field** → Left/right pan based on horizontal position
3. **Sound plays from object's location** → User hears where each object actually is
4. **Natural spatial understanding** → Brain processes 3D audio instinctively

**Key innovation**: You can still have a conversation while receiving spatial information. The audio cues provide continuous spatial awareness without interrupting dialogue or requiring you to parse complex verbal descriptions. Less mental load, more natural interaction.

## Why Spatial Audio > Voice Descriptions

### Traditional LLM Camera Approach
```
User: "What do you see?"
Assistant: "I see a coffee mug on your left, about 30 degrees, 
           a laptop directly in front of you, 
           and your phone on the right side, near the edge of the table,
           and keys slightly to the left of center..."

Problems:
❌ Must wait for full description
❌ Mental processing required to map positions
❌ Can't have conversation while describing
❌ Multiple objects = long, confusing descriptions
❌ High cognitive load
```

### Our Spatial Audio Approach
```
[Beep from left] [Beep from center] [Beep from right] [Beep from center-left]

User: "Hey, where did I put my keys?"
[While talking, user hears beep from center-left]
User: *reaches toward the sound*

Advantages:
✅ Instant perception
✅ No mental translation needed
✅ Can talk while perceiving space
✅ Multiple objects heard simultaneously without confusion
✅ Zero cognitive load - just like hearing in real life
```

### The Key Insight
**Music doesn't interrupt conversation.** You can have a normal dialogue while the spatial audio provides continuous environmental awareness—like background music that carries information. With voice descriptions, you have to stop and listen to each object enumeration.

## How It Works

### Real-Time Detection
Point the camera at your environment and the system:
- Detects objects using YOLOv8n running on Raspberry Pi AI HAT
- Calculates bounding box positions in the camera frame
- Maps objects to normalized stereo field
- Generates positioned audio for each detected object

### Spatial Audio Engine
The system:
- Detects objects using YOLOv8n on Raspberry Pi AI HAT
- Calculates normalized positions from bounding boxes (x: -1 to +1)
- Generates stereo audio with position-based panning
- Plays distinct sounds for each object
- Provides real-time spatial feedback

### Example Experience
```
[Camera detects objects on table]
[Beep sound from left side] - Coffee mug detected on left
[Different beep from right side] - Phone detected on right
[Center beep] - Laptop detected straight ahead
```

The user instinctively knows the spatial layout from audio positioning alone.

## Technical Implementation

### Architecture
```
┌─────────────────┐
│  Pi Camera 3    │ → Captures scene at 30fps
└────────┬────────┘
         │
┌────────▼────────┐
│  YOLOv8n on     │ → Detects objects + bounding boxes
│  Raspberry Pi   │   (runs on AI HAT accelerator)
│  AI HAT         │
└────────┬────────┘
         │
┌────────▼────────┐
│   Spatial       │ → Maps bbox position → stereo field
│   Audio         │   Generates positioned sound
│ (pyroomacoustics)│
└────────┬────────┘
         │
┌────────▼────────┐
│   Headphones    │ → User hears 3D soundscape
└─────────────────┘
```

### Key Components
- **Object detection**: YOLOv8n optimized for Hailo AI accelerator
- **Hardware acceleration**: Hailo-8L AI accelerator (13 TOPS) on Raspberry Pi AI HAT
- **Vision**: Raspberry Pi Camera Module 3
- **Spatial audio**: pyroomacoustics for acoustic simulation and stereo positioning
- **Audio output**: PyAudio for low-latency stereo playback
- **Hardware**: Raspberry Pi 5 + AI HAT + Camera Module 3 + USB headphones

### Stereo Panning Algorithm
```python
def bbox_to_stereo(bbox, frame_width):
    """
    Maps object bounding box to stereo field
    bbox: (x_min, y_min, x_max, y_max) from YOLO
    frame_width: camera frame width in pixels
    """
    # Calculate center of bounding box
    x_center = (bbox[0] + bbox[2]) / 2
    
    # Normalize to -1 (left) to +1 (right)
    x_normalized = (x_center / frame_width) * 2 - 1
    
    # Pan calculation for stereo field
    left_gain = (1 - x_normalized) / 2
    right_gain = (1 + x_normalized) / 2
    
    return left_gain, right_gain
```

## Hardware Requirements

| Component | Purpose | Notes |
|-----------|---------|-------|
| **Raspberry Pi 5** | Main compute | Required for AI HAT compatibility |
| **Raspberry Pi AI HAT** | AI acceleration | Hailo-8L accelerator (13 TOPS) |
| **Camera Module 3** | Scene capture | USB webcam also supported |
| **USB Headphones** | Stereo audio output | Must have separate L/R channels |

**Note**: The AI HAT is essential for running YOLOv8n at usable framerates (30fps). Without hardware acceleration, performance would be too slow for real-time feedback.

## Setup

See the main [README.md](../README.md) for complete Raspberry Pi setup instructions.

### Quick Start for Spatial Audio
```bash
cd ~/MusicIsTheAnswer
source .venv/bin/activate

# Test spatial positioning
python stereomusic/test_spatial_tracking.py

# Run full assistant with spatial audio
python main.py
```

## Usage

### Running the Demo

Create a venv and install the requirements then:

```bash
python -m stereomusic.spatial_tracker_hrtf -t "cup" -f
```

The system will:
1. Initialize the camera and AI HAT
2. Start real-time object detection
3. Play positioned audio for each detected object
4. Continuously update as you move objects or camera

### What You'll Hear
- **Left side objects**: Sound predominantly in left ear
- **Right side objects**: Sound predominantly in right ear
- **Center objects**: Balanced sound in both ears
- **Multiple objects**: Simultaneous sounds from different positions

Press `Ctrl+C` to stop.

7. **Working prototype**: Fully functional demo on actual hardware

## Impact & Future Development

### Potential Applications
- **Home navigation**: "Where did I leave my keys?"
- **Kitchen assistance**: "Guide me to the milk in the fridge"
- **Office productivity**: "Find the document on my desk"
- **Social situations**: "Who's in the room and where are they?"
- **Shopping**: "Is this the product I'm looking for?"

## Technical Highlights

### Edge AI Processing
- **YOLOv8n**: Lightweight YOLO variant optimized for edge devices
- **Hailo-8L acceleration**: 13 TOPS performance enabling 30fps detection
- **On-device inference**: No cloud dependency, works offline
- **80 object classes**: COCO dataset trained model

### Audio Engineering
- **pyroomacoustics**: Room acoustics simulation library for realistic spatial positioning
- **Bounding box mapping**: Direct conversion from pixel coordinates to stereo field
- **Distinct sound signatures**: Different tones per object category
- **Low-latency playback**: PyAudio for real-time audio output
- **Acoustic realism**: Simulates natural sound propagation in 3D space

### System Optimization
- **Efficient pipeline**: Camera → YOLO → Audio in <50ms total latency
- **Concurrent processing**: Parallel detection and audio generation
- **Resource management**: Optimized for Pi 5's limited resources

### Technology Stack
- [Raspberry Pi AI HAT](https://www.raspberrypi.com/products/ai-hat/) with Hailo-8L accelerator
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [Hailo](https://hailo.ai/) - AI hardware acceleration (13 TOPS)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) - Spatial audio simulation

## Links

- **Main Project**: [MusicIsTheAnswer](https://github.com/giuliovv/MusicIsTheAnswer)
- **Hackathon**: [EF x Raspberry Pi x Asteroid](https://luma.com/jp9aruxj)

---

*Built with ❤️ for accessibility. Spatial audio brings sight to sound.*
