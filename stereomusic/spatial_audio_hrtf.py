"""
Enhanced Spatial Audio using Pyroomacoustics and MIT KEMAR HRTF.

Uses Measured HRTF data (SOFA format) for accurate 3D positioning:
- Azimuth: Precise ITD and ILD from KEMAR mannequin.
- Elevation: Spectral cues (pinna notches) from measured data.
"""

from __future__ import annotations
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
from typing import Optional, Tuple
import scipy.signal
from scipy.spatial import KDTree
import pyroomacoustics as pra
import warnings

# Suppress pyroomacoustics warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SOFALoader:
    """Singleton to load and cache the SOFA HRTF data."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SOFALoader, cls).__new__(cls)
                cls._instance._load_data()
            return cls._instance

    def _load_data(self):
        print("Loading MIT KEMAR HRTF data...")
        try:
            # Download/Get path
            pra.datasets.download_sofa_files()
            db = pra.datasets.SOFADatabase()
            path = db['mit_kemar_normal_pinna']
            
            # Load file
            # open_sofa_file returns a tuple: (data, fs, coords, ... ) or similar depending on version
            # Based on inspection: data, fs, coords, receivers, _, _
            sofa_file = pra.sofa.open_sofa_file(path.path)
            self.data, self.fs, self.coords, self.receivers, _, _ = sofa_file
            
            # coords is (3, N) -> (Az, El, Dist)
            # Transpose for KDTree
            self.tree = KDTree(self.coords.T)
            print(f"HRTF Loaded. Samples: {self.data.shape[2]}, FS: {self.fs}Hz, Positions: {self.coords.shape[1]}")
            
        except Exception as e:
            print(f"Error loading HRTF: {e}")
            raise

    def get_hrir(self, az_rad: float, el_rad: float, target_fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Left/Right HRIR for given azimuth/elevation (radians).
        Resamples if target_fs differs from HRTF fs.
        """
        # MIT KEMAR Coords (Available range based on inspection: Az 0..6.2, El 0..2.3, Dist 1.4)
        # We assume input Azimuth is 0..2pi (Standard Spherical)
        # We assume input Elevation is 0..pi (Colatitude)
        
        # 1.4 is the fixed distance in the dataset
        dist = 1.4
        
        # Query nearest
        d, idx = self.tree.query([az_rad, el_rad, dist])
        
        # Get Impulse Responses: Shape (N_meas, 2, N_samples)
        ir_left = self.data[idx, 0, :]
        ir_right = self.data[idx, 1, :]
        
        # Resample if needed
        if target_fs != self.fs:
            num_samples = int(len(ir_left) * target_fs / self.fs)
            ir_left = scipy.signal.resample(ir_left, num_samples)
            ir_right = scipy.signal.resample(ir_right, num_samples)
            
        return ir_left, ir_right


class HRTFSpatialPlayer:
    def __init__(self, audio_file: str):
        """Load an audio file for spatial playback."""
        # Load Audio
        self.data, self.samplerate = sf.read(audio_file, dtype='float32')

        # Convert to mono
        if len(self.data.shape) > 1:
            self.mono = np.mean(self.data, axis=1)
        else:
            self.mono = self.data

        # Load HRTF Data
        self.hrtf_loader = SOFALoader()
        
        # State
        self.azimuth = 0.0      # -1.0 (left) to 1.0 (right)
        self.elevation = 0.0   # -1.0 (below) to 1.0 (above)
        self.distance = 1.0    # meters
        
        self.playing = False
        self.loop = False
        self._current_frame = 0
        self._stream = None
        self._lock = threading.Lock()
        
        # Convolution State (Overlap-Add)
        # We process in chunks. To do continuous convolution, we need to handle the "tail"
        # Since HRIR changes, we'll do:
        # 1. Convolve current chunk with current HRIR
        # 2. Add overlap from previous chunk
        # 3. Save new overlap
        # This is strictly "Overlap-Add" but with time-varying filter.
        # Ideally we cross-fade filters, but switching per chunk (20ms) is usually okayish if changes are smooth.
        
        self.overlap_left = np.zeros(2048, dtype=np.float32) # Enough for resampled HRIR
        self.overlap_right = np.zeros(2048, dtype=np.float32)

    def set_position(self, azimuth: float, elevation: float = 0.0):
        """
        Set spatial position.
        azimuth: -1.0 (left) to 1.0 (right)
        elevation: -1.0 (down) to 1.0 (up)
        """
        with self._lock:
            self.azimuth = max(-1.0, min(1.0, azimuth))
            self.elevation = max(-1.0, min(1.0, elevation))

    def set_distance(self, d: float):
        with self._lock:
            self.distance = max(0.2, d)

    def _map_coordinates(self, az: float, el: float) -> Tuple[float, float]:
        """
        Map logical coords (-1..1) to SOFA Spherical Coords (radians).
        
        Logic:
        Input Az: 0=Center, -1=Left, 1=Right
        SOFA Az: 0=Front, pi/2=Left, 3pi/2=Right (Counter-Clockwise)
        Wait, usually Azimuth is CCW.
        Front (0 deg) -> 0 rad
        Left (90 deg) -> PI/2 rad (Positive Azimuth)
        Right (-90 deg) -> 3PI/2 rad (or -PI/2)
        
        Our Input:
        0 -> 0
        -1 (Left) -> PI/2
        1 (Right) -> 3PI/2 (270 deg)
        
        Input El: 0=Horizon, 1=Up, -1=Down
        SOFA El (Colatitude): 0=Top, pi/2=Horizon, pi=Bottom
        
        Mapping El:
        1 (Up) -> 0
        0 (Horizon) -> PI/2
        -1 (Down) -> PI (or max available 2.3)
        """
        # Azimuth Mapping
        if az < 0:
            # -1 (Left) -> pi/2
            target_az = abs(az) * (np.pi / 2)
        else:
            # 1 (Right) -> 3pi/2 (which is -pi/2 effectively)
            # Linear interpolation 0 -> 0, 1 -> 3pi/2?
            # No, standard is 0->Front, Right is usually Negative angle or >180.
            # Let's use 2pi - (az * pi/2)
            target_az = (2 * np.pi) - (az * np.pi / 2)
            if target_az >= 2 * np.pi:
                target_az = 0.0

        # Elevation Mapping (Colatitude)
        # 1.0 (Up) -> 0.0
        # 0.0 (Horizon) -> 1.57 (pi/2)
        # -1.0 (Down) -> 3.14 (pi)
        
        # Linear map: y = mx + c
        # 1 -> 0
        # 0 -> 1.57
        # -1 -> 3.14
        # Slope = -1.57
        # y = -1.57 * el + 1.57
        target_el = (-1.5707 * el) + 1.5707
        
        return target_az, target_el

    def _apply_spatial(self, chunk: np.ndarray) -> np.ndarray:
        with self._lock:
            az = self.azimuth
            el = self.elevation
            dist = self.distance
        
        # 1. Get HRIRs
        target_az, target_el = self._map_coordinates(az, el)
        ir_l, ir_r = self.hrtf_loader.get_hrir(target_az, target_el, self.samplerate)
        
        # 2. Convolve
        # Simple Overlap-Add
        out_l = scipy.signal.convolve(chunk, ir_l, mode='full')
        out_r = scipy.signal.convolve(chunk, ir_r, mode='full')
        
        # Add overlap
        n_overlap = len(self.overlap_left)
        out_len = len(out_l)
        
        # Extend with overlap if needed (usually out_l is chunk + N - 1)
        # Ensure overlap buffer is same size as signal tail
        
        # Add previous overlap
        add_len = min(len(out_l), n_overlap)
        out_l[:add_len] += self.overlap_left[:add_len]
        out_r[:add_len] += self.overlap_right[:add_len]
        
        # Save new overlap for next chunk
        chunk_size = len(chunk)
        new_overlap_l = out_l[chunk_size:]
        new_overlap_r = out_r[chunk_size:]
        
        # Update buffer (resize if necessary)
        self.overlap_left = np.zeros(len(new_overlap_l), dtype=np.float32)
        self.overlap_left[:] = new_overlap_l
        
        self.overlap_right = np.zeros(len(new_overlap_r), dtype=np.float32)
        self.overlap_right[:] = new_overlap_r
        
        # Truncate to chunk size for output
        out_l = out_l[:chunk_size]
        out_r = out_r[:chunk_size]
        
        # 3. Apply Distance Cues (Volume + Air Absorption)
        # Make gain change more gradually with distance, but with
        # a slightly higher base level so the sound feels present.
        base_gain = 1.8
        exponent = 0.9
        gain = base_gain / (max(0.5, dist) ** exponent)

        # Keep far sounds quieter, but don't let them disappear.
        min_gain = 0.2
        if gain < min_gain:
            gain = min_gain

        # Limit max gain to avoid clipping when very close
        gain = min(gain, 3.0)
        
        out_l *= gain
        out_r *= gain
        
        # Simple Low-Pass for large distances (Air absorption > 20m, but we simulate effect earlier)
        # For simplicity, we just use gain for now as HRTF already modifies spectrum heavily.
        
        return np.column_stack([out_l, out_r])

    def _audio_callback(self, outdata, frames, time_info, status):
        remaining = len(self.mono) - self._current_frame
        
        if remaining <= 0:
            if self.loop:
                self._current_frame = 0
                remaining = len(self.mono)
            else:
                outdata.fill(0)
                self.playing = False
                raise sd.CallbackStop()

        chunk_size = min(frames, remaining)
        chunk = self.mono[self._current_frame:self._current_frame + chunk_size]
        
        # Process
        spatial_data = self._apply_spatial(chunk)
        
        outdata[:chunk_size] = spatial_data
        outdata[chunk_size:] = 0
        
        self._current_frame += chunk_size

    def play(self, loop: bool = False):
        self.loop = loop
        self._current_frame = 0
        self.playing = True
        
        # Reset overlap
        self.overlap_left.fill(0)
        self.overlap_right.fill(0)
        
        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            callback=self._audio_callback,
            dtype='float32',
            blocksize=2048
        )
        self._stream.start()

    def stop(self):
        self.playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def wait(self):
        while self.playing:
            time.sleep(0.1)

def demo():
    import sys
    import os
    
    if len(sys.argv) < 2:
        # Check for default test file
        default_file = os.path.join(os.path.dirname(__file__), "test_tone.wav")
        if os.path.exists(default_file):
            audio_file = default_file
        else:
            print("Usage: python spatial_audio_hrtf.py <audio_file>")
            return
    else:
        audio_file = sys.argv[1]

    print(f"Loading: {audio_file}")
    player = HRTFSpatialPlayer(audio_file)
    
    print("\n=== HRTF Spatial Audio Demo (Pyroomacoustics) ===")
    print("Listen for precise 3D positioning using MIT KEMAR data.\n")
    
    player.play(loop=True)
    
    try:
        # Move Azimuth
        print("1. Azimuth: Left -> Center -> Right")
        for i in range(50):
            az = -1.0 + (2.0 * i / 49)
            player.set_position(az, 0.0)
            print(f"\rAz: {az:.2f}   ", end="")
            time.sleep(0.1)
        print()
            
        # Move Elevation
        print("2. Elevation: Up -> Horizon -> Down")
        player.set_position(0.0, 1.0)
        for i in range(50):
            el = 1.0 - (2.0 * i / 49)
            player.set_position(0.0, el)
            print(f"\rEl: {el:.2f}   ", end="")
            time.sleep(0.1)
        print()
        
        print("Done. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        player.stop()

if __name__ == "__main__":
    demo()
