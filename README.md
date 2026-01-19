# 🎧 Spatial Audio Object Detection for the Blind

**Winner of the EF x Raspberry Pi x Asteroid Accessibility Hackathon (17 January 2025)**

A real-time spatial awareness system that helps blind and visually impaired users understand their surroundings through **3D audio positioning**. Objects are detected via on-device AI and represented as stereo sounds that originate from the object's actual location in space.

## The Problem

Existing solutions for blind users have critical limitations:

**Traditional Audio approach:**
- Camera → LLM → "There's a cup on your left, a phone to your right, and keys in the center"
- **Mental overload**: User must parse verbal descriptions and mentally map positions
- **Slow**: Waiting for full sentence descriptions
- **Overlapping voices**: Multiple objects create confusing audio clutter
- **Not instinctive**: Requires conscious processing of language

**Our solution:**
- Camera → YOLO → Positioned music
- **Intuitive**: Brain processes spatial audio instantly, no thinking required
- **Fast**: Immediate perception of surroundings
- **Parallel**: Multiple objects heard simultaneously without confusion
- **Natural**: Like hearing sounds in the real world
- **Fun**: Music makes the experience enjoyable

The difference is like reading turn-by-turn directions versus using GPS with map visualization—one requires mental processing, the other is immediate understanding.

## Our Solution

We use **spatial audio instead of verbal descriptions** to represent object positions:

1. **Camera captures the scene** → YOLOv8n detects objects and positions on-device
2. **Position mapped to stereo field** → Left/right pan based on horizontal position
3. **Sound plays from object's location** → User hears where each object actually is
4. **Natural spatial understanding** → Brain processes 3D audio instinctively

**Key innovation**: You can still have a conversation while receiving spatial information. The audio cues provide continuous spatial awareness without interrupting dialogue or requiring you to parse complex verbal descriptions. Less mental load, more natural interaction.

## Why Spatial Audio > Voice Descriptions

### Traditional Audio Description Approach
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

User: [Whithout prompting, the user hears sounds from center-left]
User: *reaches toward the sound*

Advantages:
✅ Instant perception
✅ No mental translation needed
✅ Can talk while perceiving space
✅ Multiple objects heard simultaneously without confusion
✅ Zero cognitive load - just like hearing in real life
```

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

## Repository layout

At a high level this repo is split into two parts:

- `main.py` – cash-recognition voice assistant (wake word + ElevenLabs + Gemini).
- `stereomusic/` – spatial audio and object-tracking demos, main hackathon folder (YOLO → 3D audio). 
- `camera/` – camera abstraction for USB and Pi Camera.
- `tools/` – Gemini-powered tools, e.g. cash recognition, packaging reader.
- `test_spatial_tracking.py` – laptop-friendly test for the spatial tracker.
- `test_full_flow.py` – end-to-end test of camera → Gemini → ElevenLabs.

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
│(pyroomacoustics)│
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

### Python version

This project is developed and tested on **Python 3.11**. The full cash-recognition
assistant (wake word + ElevenLabs + Gemini) currently requires Python 3.11 (and
not 3.13) because of `tflite-runtime`.

The `stereomusic` spatial-audio demos are pure Python and may also work on
other 3.x versions, but 3.11 is the recommended and supported version across
the repo.

## Complete Setup Guide (Raspberry Pi 5)

### Step 1: Flash Raspberry Pi OS

1. **Download** [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your computer.
2. Insert your **Micro SD card** (you may need a Micro SD → SD adapter).
3. Open Raspberry Pi Imager and:
   - **Choose Device**: Raspberry Pi 5
   - **Choose OS**: Raspberry Pi OS (64-bit)
   - **Choose Storage**: Your SD card
4. Click the **⚙️ gear icon** (or "Edit Settings") and configure:
   - ✅ Set hostname: Choose a name (e.g., `raspberrypi` or `mypi`)
   - ✅ Set username and password: **Write these down!**
   - ✅ Configure WiFi: Enter your network name and password
   - ✅ Enable SSH: Use password authentication
5. Click **Save**, then **Write**.
6. Wait for write + verification to complete.

### Step 2: First Boot

1. **Insert** the SD card into your Raspberry Pi.
2. **Connect** power (USB-C).
3. **Wait** 2–3 minutes for first boot to complete.
4. The green LED should blink occasionally (activity).

### Step 3: Connect via SSH

On your computer (Mac/Windows/Linux), open a terminal:

```bash
ssh youruser@yourhostname.local
```

Replace `youruser` and `yourhostname` with the values you set. Enter your password when prompted.

> **Troubleshooting**: If connection fails, wait another minute. If still failing, connect an HDMI monitor to see what's happening.

### Step 4: Copy the Project Files

On your **computer** (not the Pi), clone or download this repo, then copy it to the Pi:

```bash
# Clone the repo (on your computer)
git clone https://github.com/giuliovv/MusicIsTheAnswer.git

# Copy to Pi
scp -r MusicIsTheAnswer youruser@yourhostname.local:~/
```

Or if you already have the folder locally:

```bash
scp -r /path/to/MusicIsTheAnswer youruser@yourhostname.local:~/
```

### Step 5: Install Python 3.11 with pyenv

> ⚠️ **Important:** Recent Raspberry Pi OS releases ship with Python 3.13 by default,
> but the wake word library (`tflite-runtime`) requires Python 3.11. Install
> Python 3.11 using `pyenv`.

SSH into your Pi and run:

```bash
# Install pyenv dependencies
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to your shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc

# Install Python 3.11 (takes 5–10 minutes on Pi 5 - be patient!)
pyenv install 3.11.9
```

> **Note**: When installing Python, it can appear to hang on "Patching..." – this is
> normal; compiling Python on a Pi takes time. You can check progress with `top`
> in another terminal.

### Step 6: Run the Pi Setup Script

```bash
cd ~/MusicIsTheAnswer
chmod +x setup_pi.sh
./setup_pi.sh
```

This installs all system dependencies and Python packages for the assistant and
spatial audio demos.

### Step 7: Configure Audio Device

Before running the assistant, configure your USB headphones as the default audio device.

First, check your audio devices:

```bash
aplay -l   # List playback devices
arecord -l # List recording devices
```

You'll see output like:

```text
card 3: Audio [AB13X USB Audio], device 0: USB Audio [USB Audio]
```

Note the card number for your USB headphones (e.g., `3`).

Create the ALSA configuration:

```bash
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "hw:3,0"
    }
    capture.pcm {
        type plug
        slave.pcm "hw:3,0"
    }
}

ctl.!default {
    type hw
    card 3
}
EOF
```

> **Important**: Replace `3` with your actual card number if different.

Test audio output:

```bash
speaker-test -c 2 -t wav -D hw:3,0
```

You should hear "Front Left, Front Right" in your headphones. Press `Ctrl+C` to stop.

### Quick Start for Spatial Audio
```bash
cd ~/MusicIsTheAnswer
python -m venv .venv
source .venv/bin/activate

# Install main assistant dependencies
pip install -r requirements.txt

# (Optional) install additional deps for stereomusic demos
pip install -r stereomusic/requirements.txt

# Test spatial positioning only
python test_spatial_tracking.py

# Run full assistant with spatial audio and voice interface
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

### Audio assets and credits

- The `stereomusic/chiptune.wav` demo file is based on a chiptune
    downloaded from [originaljun on Freesound](https://freesound.org/people/orginaljun/sounds/396173/)
    (see Freesound for licensing details).

## Known limitations

- Requires (USB) headphones with clear left/right channels for the spatial cues to
    make sense.
- First run of YOLOv8 downloads model weights, which can take time and needs
    an internet connection once.
- Real-time performance and HRTF audio are tuned for Raspberry Pi 5 + AI HAT;
    slower hardware may get lower frame rates.

## Links

- **Main Project**: [MusicIsTheAnswer](https://github.com/giuliovv/MusicIsTheAnswer)
- **Hackathon**: [EF x Raspberry Pi x Asteroid](https://luma.com/jp9aruxj)

---

*Built with ❤️ for accessibility. Spatial audio brings sight to sound.*
