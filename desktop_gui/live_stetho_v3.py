"""Live Stethoscope AI Diagnostic Tool (V3 Deep Learning CNN)
==========================================================
Supports:
  1. Live PC / USB Microphone streaming:
     python live_stetho_v3.py --mic

  2. Live ESP32 + INMP441 I2S streaming over USB Serial:
     python live_stetho_v3.py --esp32 COM5

  3. Audio File Diagnosis:
     python live_stetho_v3.py --file audio.wav
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from collections import deque

import numpy as np
import scipy.signal
import librosa

# Determine base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stetho_cnn_fused.npz"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(r"c:\Users\Dell\Ai Box\test_vercel_stetho\models\stetho_cnn_fused.npz")

SAMPLE_RATE = 16000
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 32,000 samples
N_MELS = 64
N_FRAMES = 128
LABELS = ["normal", "crackle", "wheeze", "both"]

# 4th-order Butterworth Bandpass filter (100 Hz - 2000 Hz)
_SOS = scipy.signal.butter(4, [100.0 / 8000.0, 2000.0 / 8000.0], btype="band", output="sos")

def bandpass_filter(y: np.ndarray) -> np.ndarray:
    if len(y) < 30:
        return y
    return scipy.signal.sosfiltfilt(_SOS, y).astype(np.float32)

def extract_logmel(y: np.ndarray) -> np.ndarray:
    """Extract 64-band Log-Mel Spectrogram (64 x 128) matching V3 CNN."""
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    else:
        y = y[:WINDOW_SAMPLES]
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS,
        fmin=100, fmax=2000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < N_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - mel_db.shape[1])), mode="edge")
    else:
        mel_db = mel_db[:, :N_FRAMES]
    return ((mel_db + 80.0) / 80.0).clip(0.0, 1.0).astype(np.float32)

def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def conv2d_np(x, w, b):
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    pad_h, pad_w = kH // 2, kW // 2
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    shape = (B, H, W, C_in, kH, kW)
    strides = (x_pad.strides[0], x_pad.strides[2], x_pad.strides[3], x_pad.strides[1], x_pad.strides[2], x_pad.strides[3])
    cols = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
    cols_flat = cols.reshape(B, H, W, -1)
    w_flat = w.reshape(C_out, -1)
    out = np.tensordot(cols_flat, w_flat, axes=([-1], [-1]))
    return np.transpose(out, (0, 3, 1, 2)) + b.reshape(1, -1, 1, 1)

def maxpool2d_np(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

class V3Classifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"V3 Fused weights file not found: {model_path}")
        print(f"Loading V3 StethoDeepCNN weights from: {model_path.name}...")
        self.w = dict(np.load(str(model_path)))
        print("V3 Model loaded successfully (18 fused tensor layers, pure NumPy engine)!\n")

    def predict(self, audio_chunk: np.ndarray):
        """Audio chunk (length >= 32000 float32) -> (predicted_label, confidence, prob_dict)"""
        filtered = bandpass_filter(audio_chunk)
        spec = extract_logmel(filtered)[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 128)
        
        # 4-stage ConvNet forward pass
        x = gelu_np(conv2d_np(spec, self.w['w1'], self.w['b1']))
        x = gelu_np(conv2d_np(x, self.w['w2'], self.w['b2']))
        x = maxpool2d_np(x)
        x = gelu_np(conv2d_np(x, self.w['w3'], self.w['b3']))
        x = gelu_np(conv2d_np(x, self.w['w4'], self.w['b4']))
        x = maxpool2d_np(x)
        x = gelu_np(conv2d_np(x, self.w['w5'], self.w['b5']))
        x = gelu_np(conv2d_np(x, self.w['w6'], self.w['b6']))
        x = maxpool2d_np(x)
        x = gelu_np(conv2d_np(x, self.w['w7'], self.w['b7']))
        x = x.mean(axis=(2, 3))
        x = gelu_np(x @ self.w['fc1_w'].T + self.w['fc1_b'])
        logits = x @ self.w['fc2_w'].T + self.w['fc2_b']
        probs = softmax(logits)[0]
        
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        conf = float(probs[idx])
        prob_dict = {LABELS[i]: float(probs[i]) for i in range(4)}
        return label, conf, prob_dict

def render_ui(label: str, conf: float, probs: dict, rms: float, source_name: str):
    """Render a clean clinical diagnostic UI in terminal."""
    # Volume bar using standard ASCII
    bars = int(min(20, max(0, rms * 150)))
    vol_str = "#" * bars + "-" * (20 - bars)

    if label == "normal":
        status_tag = "[ NORMAL VESICULAR SOUND ]"
        action = "Normal respiratory acoustics. Clear vesicular airflow. No immediate referral."
    elif label == "crackle":
        status_tag = "[ !!! ABNORMAL: CRACKLES DETECTED !!! ]"
        action = "Clinical rales/crackles (Pneumonia/Edema/Fibrosis). Pulmonologist referral recommended."
    elif label == "wheeze":
        status_tag = "[ !!! ABNORMAL: WHEEZE DETECTED !!! ]"
        action = "Continuous musical wheeze (Asthma/COPD airway constriction). Bronchodilator triage."
    else:
        status_tag = "[ !!! ABNORMAL: MIXED CRACKLE + WHEEZE !!! ]"
        action = "Combined airway adventitious sounds. Urgent clinical auscultation assessment."

    prob_line = " | ".join([
        f"{lbl.upper()}: {probs[lbl]*100:4.1f}%" for lbl in LABELS
    ])

    print("-" * 75)
    print(f" Source: {source_name} | Signal Level: [{vol_str}] | Conf: {conf*100:.1f}%")
    print(f" Diagnosis: {status_tag}")
    print(f" Probabilities: {prob_line}")
    print(f" Recommendation: {action}")
    print("-" * 75)

# ----------------- Microphone Streaming Worker -----------------
class MicrophoneStream:
    def __init__(self, sample_rate=SAMPLE_RATE, maxlen=WINDOW_SAMPLES * 2):
        import sounddevice as sd
        self.sd = sd
        self.sample_rate = sample_rate
        self.maxlen = maxlen
        self.buffer = np.zeros(maxlen, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].astype(np.float32)
        with self.lock:
            self.buffer = np.roll(self.buffer, -len(mono))
            self.buffer[-len(mono):] = mono

    def start(self, device_idx=None):
        self.stream = self.sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1024,
            callback=self._audio_callback
        )
        self.stream.start()

    def get_latest_window(self, n=WINDOW_SAMPLES):
        with self.lock:
            return self.buffer[-n:].copy()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# ----------------- ESP32 Serial Streaming Worker -----------------
class ESP32SerialStream:
    def __init__(self, port: str, baud=921600, maxlen=WINDOW_SAMPLES * 2):
        import serial
        self.ser = serial.Serial(port, baud, timeout=1)
        self.sync_seq = b"\xA5\x5A\xA5\x5A"
        self.maxlen = maxlen
        self.buffer = np.zeros(maxlen, dtype=np.float32)
        self.raw_bytes = bytearray()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                chunk = self.ser.read(4096)
                if not chunk:
                    continue
                self.raw_bytes += chunk
                while len(self.raw_bytes) >= 6:
                    idx = self.raw_bytes.find(self.sync_seq)
                    if idx < 0:
                        self.raw_bytes[:] = self.raw_bytes[-3:]
                        break
                    self.raw_bytes[:] = self.raw_bytes[idx + 4:]
                    if len(self.raw_bytes) < 2:
                        break
                    cnt = int.from_bytes(self.raw_bytes[:2], "little")
                    self.raw_bytes[:] = self.raw_bytes[2:]
                    if cnt == 0 or cnt > 4096:
                        continue
                    need = cnt * 2
                    if len(self.raw_bytes) < need:
                        break
                    samples_i16 = np.frombuffer(bytes(self.raw_bytes[:need]), dtype="<i2")
                    self.raw_bytes[:] = self.raw_bytes[need:]
                    samples_f32 = (samples_i16 / 32768.0).astype(np.float32)
                    with self.lock:
                        n = len(samples_f32)
                        self.buffer = np.roll(self.buffer, -n)
                        self.buffer[-n:] = samples_f32
            except Exception:
                time.sleep(0.01)

    def get_latest_window(self, n=WINDOW_SAMPLES):
        with self.lock:
            return self.buffer[-n:].copy()

    def stop(self):
        self.running = False
        self.ser.close()

# ----------------- Main CLI Execution -----------------
def main():
    parser = argparse.ArgumentParser(description="Live Digital Stethoscope Auscultation Analyzer (V3 Deep CNN)")
    parser.add_argument("--mic", action="store_true", help="Capture live respiratory sound from PC / USB microphone")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (default: system mic)")
    parser.add_argument("--esp32", type=str, default=None, help="ESP32 Serial COM port (e.g. COM3, COM5)")
    parser.add_argument("--file", type=str, default=None, help="Classify pre-recorded WAV or MP3 audio file")
    parser.add_argument("--interval", type=float, default=1.0, help="Diagnostic classification interval in seconds (default: 1.0s)")
    args = parser.parse_args()

    v3 = V3Classifier()

    # 1. Single File Analysis Mode
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return
        print(f"Analyzing respiratory audio file: {file_path.name}...")
        y, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        duration = len(y) / SAMPLE_RATE
        print(f"Duration: {duration:.2f}s | Sample Rate: {sr} Hz")
        
        # Analyze in 2.0s segments with 1.0s hop
        hop = int(SAMPLE_RATE * 1.0)
        win = WINDOW_SAMPLES
        seg_probs = []
        for pos in range(0, max(1, len(y) - win + 1), hop):
            chunk = y[pos:pos + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            _, _, p = v3.predict(chunk)
            seg_probs.append(list(p.values()))
        
        avg_probs = np.mean(seg_probs, axis=0) if seg_probs else [0.25, 0.25, 0.25, 0.25]
        prob_dict = {LABELS[i]: float(avg_probs[i]) for i in range(4)}
        overall_idx = int(np.argmax(avg_probs))
        overall_label = LABELS[overall_idx]
        overall_conf = float(avg_probs[overall_idx])
        rms = float(np.sqrt(np.mean(y**2)))
        
        render_ui(overall_label, overall_conf, prob_dict, rms, file_path.name)
        return

    # 2. Live ESP32 Stethoscope Mode
    elif args.esp32:
        print(f"\n=======================================================")
        print(f" Connecting to ESP32 Digital Stethoscope on {args.esp32}...")
        print(f" INMP441 I2S Mic -> ESP32 -> Serial (921600 baud)")
        print(f" Press Ctrl+C to stop auscultation monitoring.")
        print(f"=======================================================\n")
        esp = ESP32SerialStream(args.esp32)
        time.sleep(1.5)  # Warm up serial buffer
        try:
            while True:
                chunk = esp.get_latest_window(WINDOW_SAMPLES)
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms < 0.001:
                    print(f"Waiting for audio signal from ESP32 on {args.esp32}... (Microphone quiet)", end="\r")
                else:
                    label, conf, probs = v3.predict(chunk)
                    render_ui(label, conf, probs, rms, f"ESP32 ({args.esp32})")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            esp.stop()
            print("\nESP32 Auscultation stream stopped.")
        return

    # 3. Live PC / USB Microphone Mode
    elif args.mic or len(sys.argv) == 1:
        print(f"\n=======================================================")
        print(f" Starting Live Auscultation via Microphone...")
        print(f" Sampling: 16 kHz | Window: 2.0 seconds")
        print(f" Point microphone/chestpiece to chest and breathe normally.")
        print(f" Press Ctrl+C to stop auscultation.")
        print(f"=======================================================\n")
        mic = MicrophoneStream(SAMPLE_RATE)
        mic.start(args.device)
        time.sleep(1.5)  # Fill rolling window
        try:
            while True:
                chunk = mic.get_latest_window(WINDOW_SAMPLES)
                rms = float(np.sqrt(np.mean(chunk**2)))
                label, conf, probs = v3.predict(chunk)
                render_ui(label, conf, probs, rms, "Live Microphone")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            mic.stop()
            print("\nMicrophone Auscultation stopped.")
        return

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
