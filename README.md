# Cash Recognition Voice Assistant

A voice-activated assistant for Raspberry Pi that helps blind and visually impaired users identify banknotes.

**Say "Hey Pi" → Ask about your money → Get spoken feedback**

## Features

- 🎤 **Wake word activation** - Say "Hey Pi" to start
- 💬 **Natural conversation** - Chat with the assistant powered by ElevenLabs
- 📷 **Camera recognition** - Identifies banknotes using Google Gemini AI
- 🔊 **Voice output** - Clear spoken responses via headphones
- 💷 **Multi-currency** - Works with British Pounds, Euros, US Dollars, and more

## Hardware Required

| Item | Notes |
|------|-------|
| **Raspberry Pi 5** | Pi 4 also works, but Pi 5 recommended |
| **Micro SD Card** | 32GB+ recommended |
| **USB-C Power Supply** | 27W for Pi 5 (official PSU recommended) |
| **Camera** | USB webcam (e.g., Logitech C920) OR **Raspberry Pi Camera Module 3** (recommended) |
| **USB Headphones with Mic** | USB-C or USB-A headphones with built-in microphone |
| **USB-C to USB-A Adapter** | If using USB-A peripherals with Pi 5 |

> **Pi Camera Note**: If using Pi Camera Module 3 with Pi 5, you'll need the 22-pin adapter cable (often included in the box) since Pi 5 uses a smaller camera connector than Pi 4.

### Optional
- HDMI cable + monitor (for initial troubleshooting only)
- Raspberry Pi AI HAT+ (for future local inference)

## Complete Setup Guide

### Step 1: Flash Raspberry Pi OS

1. **Download** [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your computer
2. Insert your **Micro SD card** (you may need a Micro SD → SD adapter for your computer)
3. Open Raspberry Pi Imager and:
   - **Choose Device**: Raspberry Pi 5
   - **Choose OS**: Raspberry Pi OS (64-bit)
   - **Choose Storage**: Your SD card
4. Click the **⚙️ gear icon** (or "Edit Settings") and configure:
   - ✅ Set hostname: Choose a name (e.g., `raspberrypi` or `mypi`)
   - ✅ Set username and password: **Write these down!**
   - ✅ Configure WiFi: Enter your network name and password
   - ✅ Enable SSH: Use password authentication
5. Click **Save**, then **Write**
6. Wait for write + verification to complete

### Step 2: First Boot

1. **Insert** the SD card into your Raspberry Pi
2. **Connect** power (USB-C)
3. **Wait** 2-3 minutes for first boot to complete
4. The green LED should blink occasionally (activity)

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
git clone https://github.com/yourusername/AccessibilityHackathon.git

# Copy to Pi
scp -r AccessibilityHackathon youruser@yourhostname.local:~/
```

Or if you already have the folder:
```bash
scp -r /path/to/AccessibilityHackathon youruser@yourhostname.local:~/
```

### Step 5: Install Python 3.11

> ⚠️ **Important:** Raspberry Pi OS (Debian Trixie) comes with Python 3.13, but the wake word library (`tflite-runtime`) requires Python 3.11. You MUST install Python 3.11 using pyenv.

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

# Install Python 3.11 (takes 5-10 minutes on Pi 5 - be patient!)
pyenv install 3.11.9
```

> **Note**: When installing Python, it will appear to hang on "Patching..." - this is normal. Compiling Python takes time. You can check progress with `top` in another terminal.

### Step 6: Run Setup Script

```bash
cd ~/AccessibilityHackathon
chmod +x setup_pi.sh
./setup_pi.sh
```

This installs all system dependencies and Python packages.

### Step 7: Configure Audio Device

Before running the assistant, you need to configure your USB headphones as the default audio device.

First, check your audio devices:
```bash
aplay -l   # List playback devices
arecord -l # List recording devices
```

You'll see output like:
```
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
        slave.pcm "hw:2,0"
    }
    capture.pcm {
        type plug
        slave.pcm "hw:2,0"
    }
}

ctl.!default {
    type hw
    card 2
}
EOF
```

> **Important**: Replace `3` with your actual card number if different!

Test audio output:
```bash
speaker-test -c 2 -t wav -D hw:2,0
```

You should hear "Front Left, Front Right" in your headphones. Press Ctrl+C to stop.

### Step 8: Configure API Keys

```bash
# Create .env from template
cp env.template .env

# Edit the file
nano .env
```

Add your API keys:
- **ELEVENLABS_API_KEY** - From [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys)
- **ELEVENLABS_AGENT_ID** - You'll get this after Step 10
- **GOOGLE_API_KEY** - From [Google AI Studio](https://aistudio.google.com/app/apikey)

Save with Ctrl+O, Enter, Ctrl+X.

### Step 9: Train Wake Word

The assistant uses a custom wake word to activate. By default, we use **"Hey Pi"**, but **you can choose any phrase you like** (e.g., "Hey Assistant", "Hello Pi", "Cash Helper").

```bash
source .venv/bin/activate
python train_wakeword.py
```

**Option A - Record your own samples:**
- Record 3+ samples saying your chosen wake phrase
- Save as WAV files in `hotword_training_audio/`
- The wake word will be trained from your recordings

**Option B - Use text-to-speech:**
- The script can generate samples using Google TTS

After training, you'll see: `✅ Wake word trained successfully!`

> **Tip**: Choose a phrase that's easy to say and distinctive. Two-syllable phrases like "Hey Pi" work well because they're short but unique enough to avoid false triggers.

### Step 10: Create ElevenLabs Agent

Follow the detailed guide in [SETUP_AGENT.md](SETUP_AGENT.md).

Quick summary:
1. Go to [ElevenLabs Agents](https://elevenlabs.io/app/conversational-ai/agents)
2. Create a new agent with "Blank Template"
3. Add the system prompt (see SETUP_AGENT.md)
4. Add the `identify_cash` **Client Tool**
5. Copy the Agent ID to your `.env` file

### Step 11: Test Components

```bash
source .venv/bin/activate

# Test camera + Gemini vision
python test_camera_gemini.py

# Test full flow (camera → Gemini → speech)
python test_full_flow.py
```

### Step 12: Run the Assistant!

```bash
python main.py
```

Say **"Hey Pi"** and then ask: "What banknote am I holding?"

## Project Structure

```
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── env.template            # Environment variables template
├── setup_pi.sh             # Raspberry Pi setup script
├── SETUP_AGENT.md          # ElevenLabs agent setup guide
├── train_wakeword.py       # Wake word training script
├── test_camera_gemini.py   # Camera + Gemini test
├── test_full_flow.py       # Full flow test
├── camera/
│   ├── base.py             # Camera interface
│   ├── usb_camera.py       # USB camera (OpenCV)
│   └── pi_camera.py        # Pi Camera 3 (picamera2)
├── tools/
│   └── cash_recognition.py # Gemini banknote analysis
├── hotword_training_audio/ # Wake word training samples
└── hotword_refs/           # Trained wake word model
```

## Usage Examples

**Basic usage:**
> "Hey Pi"  
> "What money do I have?"  
> *"I can see a twenty pound note. That's twenty pounds."*

**Checking a single note:**
> "Hey Pi"  
> "What's this note?"  
> *"That's a fifty pound note, the polymer version showing Alan Turing."*

## Troubleshooting

### Python Version Issues
**Problem**: `ERROR: No matching distribution found for tflite-runtime`

**Solution**: You're using Python 3.13. Install Python 3.11 via pyenv (see Step 5).

---

### pyenv Install Stuck on "Patching..."
**Problem**: `pyenv install 3.11.9` appears frozen

**Solution**: This is normal! Compiling Python on a Pi takes 5-10 minutes. Check it's working with:
```bash
ps aux | grep pyenv
```

---

### Audio: "Invalid sample rate" Error
**Problem**: `OSError: [Errno -9997] Invalid sample rate`

**Solution**: Your USB audio device isn't set as default. Create `~/.asoundrc` (see Step 7).

---

### Wake Word: "no wav file found"
**Problem**: Training fails with "only wav files are supported"

**Solution**: Convert your audio files to WAV:
```bash
cd ~/AccessibilityHackathon/hotword_training_audio
for f in *.mp3 *.m4a; do
  ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.*}.wav"
done
rm *.mp3 *.m4a  # Remove originals
```

---

### USB Camera Not Found
**Problem**: "Camera not available" error with USB camera

**Solution**:
1. Check camera is connected: `ls /dev/video*`
2. Try different USB port (use USB 3.0 blue ports)
3. Test camera: `ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg`

---

### Pi Camera: "Camera frontend has timed out"
**Problem**: Camera is detected but capture fails with timeout error

**Solution**: This is a **hardware connection issue**:
1. Power off the Pi completely
2. Re-seat the ribbon cable on both ends (camera and Pi)
3. Check orientation: gold contacts face toward the lens (camera) and USB ports (Pi 5)
4. Make sure clips are **fully closed**
5. Power on and test: `rpicam-hello -t 5000`

---

### Pi Camera: "No module named 'libcamera'"
**Problem**: picamera2 import fails in your venv

**Solution**: This is expected! The code uses system Python for camera operations:
1. Make sure system Python has picamera2: `/usr/bin/python3 -c "from picamera2 import Picamera2; print('OK')"`
2. If that fails: `sudo apt install python3-picamera2`
3. Our `pi_camera.py` handles this automatically via subprocess

---

### Agent Doesn't Use the Tool
**Problem**: Agent says "I don't have access to tools"

**Solution**: 
1. Ensure the tool is added as a **Client Tool** (not webhook)
2. Tool name must be exactly `identify_cash`
3. Update the System Prompt to mention the tool
4. Click Save in the ElevenLabs dashboard

---

### WebSocket Timeout
**Problem**: `TimeoutError: timed out while waiting for handshake response`

**Solution**: 
1. Check internet: `curl -I https://api.elevenlabs.io`
2. Verify Agent ID is correct
3. Try again (sometimes temporary)

---

### SSH Connection Refused
**Problem**: Can't connect to Pi via SSH

**Solution**:
1. Wait 2-3 minutes after first boot
2. Ensure Pi is on same network: `ping yourhostname.local`
3. Check SSH was enabled in Raspberry Pi Imager settings
4. Connect HDMI monitor to see Pi's IP address

## Using Pi Camera 3 (Instead of USB Camera)

The Raspberry Pi Camera Module 3 provides better image quality than USB cameras. Here's how to set it up:

### Hardware Connection (Pi 5)

1. **Power off** your Raspberry Pi completely
2. **Pi 5 requires an adapter cable** - the camera comes with a standard 15-pin cable, but Pi 5 uses a smaller 22-pin connector:
   - Disconnect the standard cable from the camera board (lift the black clip, slide cable out)
   - Connect the 22-pin adapter cable to the camera (gold contacts face the lens)
   - Connect the other end to the Pi 5's **CAM/DISP 0** port (gold contacts face USB ports)
3. Make sure both clips are **firmly closed**
4. Power on the Pi

### Software Setup

```bash
# Install picamera2 for system Python (required - don't skip this!)
sudo apt install -y python3-picamera2

# Install libcap-dev (needed for pip install in venv)
sudo apt install -y libcap-dev

# Verify camera is detected
rpicam-hello --list-cameras
# Should show: "0 : imx708 [4608x2592 ...]"

# Update your .env file
nano ~/AccessibilityHackathon/.env
# Change: CAMERA_TYPE=pi

# Test the camera
cd ~/AccessibilityHackathon
source .venv/bin/activate
python test_camera_capture.py
```

> **Note**: On Pi 5 with modern Raspberry Pi OS, the camera is enabled by default - no need to use `raspi-config`.

> **Technical Note**: picamera2 requires system Python because it depends on `libcamera` (a C++ library). Our code uses a subprocess approach to call system Python for camera operations while the main app runs in the Python 3.11 venv for wake word compatibility.

## Credits

- [ElevenLabs Conversational AI](https://elevenlabs.io/docs/conversational-ai) - Voice agent platform
- [ElevenLabs Raspberry Pi Example](https://github.com/elevenlabs/elevenlabs-examples/tree/main/examples/conversational-ai/raspberry-pi) - Reference implementation
- [Google Gemini](https://ai.google.dev/) - Multimodal AI for image analysis
- [EfficientWord-Net](https://github.com/Ant-Brain/EfficientWord-Net) - Wake word detection

## License

MIT License
