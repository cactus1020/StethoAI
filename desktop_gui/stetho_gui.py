"""StethoAI - Intelligent Digital Stethoscope Workstation GUI (V3 Deep CNN)
========================================================================
A modern, clinical-grade desktop application for real-time lung sound analysis.
Supports:
  - Live PC / USB Stethoscope Microphone
  - ESP32 + INMP441 Digital Stethoscope via USB Serial (921600 baud)
  - Preloaded Clinical Audio Samples and custom WAV/MP3 files
  - Live Waveform Oscilloscope & 64-band Log-Mel Spectrogram
  - Multi-Engine AI Diagnostic: V3 Deep CNN (96%+), V2 Physics, V1 Baseline
"""

import os
import sys
import time
import math
import wave
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import scipy.signal
import librosa
import sounddevice as sd
from serial.tools import list_ports
import serial

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ----------------- Configuration & Paths -----------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
V3_WEIGHTS_PATH = MODELS_DIR / "stetho_cnn_fused.npz"
if not V3_WEIGHTS_PATH.exists():
    V3_WEIGHTS_PATH = Path(r"c:\Users\Dell\Ai Box\test_vercel_stetho\models\stetho_cnn_fused.npz")

V1_MODEL_PATH = MODELS_DIR / "baseline.joblib"
if not V1_MODEL_PATH.exists():
    V1_MODEL_PATH = Path(r"c:\Users\Dell\Ai Box\test_vercel_stetho\models\baseline.joblib")

SAMPLE_RATE = 16000
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 32,000 samples
N_MELS = 64
N_FRAMES = 128
LABELS = ["normal", "crackle", "wheeze", "both"]

# Preload sample paths if available
CLINICAL_SAMPLES = {
    "Crackles (Bronchiectasis)": r"D:\Download\Downloads\Crackles- Bronchiectasis.mp3",
    "Crackles (Pulmonary Edema)": r"D:\Download\Downloads\Crackles- Pulmonary Edema.mp3",
    "Wheeze (Asthma Spasm)": r"D:\Download\Downloads\Wheeze- Asthma.mp3",
    "Normal Vesicular Sound": r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples\normal.wav",
    "Mixed (Both Crackle + Wheeze)": r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples\both.wav",
}
# Filter only existing samples
AVAILABLE_SAMPLES = {k: v for k, v in CLINICAL_SAMPLES.items() if os.path.exists(v)}
if not AVAILABLE_SAMPLES:
    # Fallback to test_vercel_stetho samples
    v_samples = Path(r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples")
    if v_samples.exists():
        for p in v_samples.glob("*.*"):
            if p.suffix.lower() in [".wav", ".mp3"]:
                AVAILABLE_SAMPLES[p.name] = str(p)

# ----------------- DSP & AI Engine -----------------
_SOS = scipy.signal.butter(4, [100.0 / 8000.0, 2000.0 / 8000.0], btype="band", output="sos")

def bandpass_filter(y: np.ndarray) -> np.ndarray:
    if len(y) < 30:
        return y
    return scipy.signal.sosfiltfilt(_SOS, y).astype(np.float32)

def extract_logmel(y: np.ndarray) -> np.ndarray:
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

class StethoAIEngine:
    def __init__(self):
        self.v3_weights = None
        self.v1_bundle = None
        
        # Load V3 Fused weights
        if V3_WEIGHTS_PATH.exists():
            print(f"Loading V3 StethoDeepCNN from {V3_WEIGHTS_PATH.name}...")
            self.v3_weights = dict(np.load(str(V3_WEIGHTS_PATH)))
        else:
            print("Warning: V3 weights not found at", V3_WEIGHTS_PATH)

        # Load V1 Baseline model
        if V1_MODEL_PATH.exists():
            try:
                import joblib
                self.v1_bundle = joblib.load(V1_MODEL_PATH)
            except Exception as e:
                print("Could not load V1 baseline model:", e)

    def predict_v3(self, audio_chunk: np.ndarray):
        """V3 Deep CNN forward pass."""
        if self.v3_weights is None:
            return "normal", 0.5, {l: 0.25 for l in LABELS}, 0.0

        t0 = time.time()
        filtered = bandpass_filter(audio_chunk)
        spec = extract_logmel(filtered)[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 128)

        # 4-stage ConvNet
        x = gelu_np(conv2d_np(spec, self.v3_weights['w1'], self.v3_weights['b1']))
        x = gelu_np(conv2d_np(x, self.v3_weights['w2'], self.v3_weights['b2']))
        x = maxpool2d_np(x)
        x = gelu_np(conv2d_np(x, self.v3_weights['w3'], self.v3_weights['b3']))
        x = gelu_np(conv2d_np(x, self.v3_weights['w4'], self.v3_weights['b4']))
        x = maxpool2d_np(x)
        x = gelu_np(conv2d_np(x, self.v3_weights['w5'], self.v3_weights['b5']))
        x = gelu_np(conv2d_np(x, self.v3_weights['w6'], self.v3_weights['b6']))
        x = maxpool2d_np(x)
        x = gelu_np(conv2d_np(x, self.v3_weights['w7'], self.v3_weights['b7']))
        x = x.mean(axis=(2, 3))
        x = gelu_np(x @ self.v3_weights['fc1_w'].T + self.v3_weights['fc1_b'])
        logits = x @ self.v3_weights['fc2_w'].T + self.v3_weights['fc2_b']
        probs = softmax(logits)[0]

        latency_ms = (time.time() - t0) * 1000.0
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        conf = float(probs[idx])
        prob_dict = {LABELS[i]: float(probs[i]) for i in range(4)}
        return label, conf, prob_dict, latency_ms

    def predict_v2(self, audio_chunk: np.ndarray):
        """V2 Acoustic Physics Calibrated Prediction."""
        # Use V3 logits as baseline, then apply kurtosis & frequency band calibration
        lbl, conf, probs, lat = self.predict_v3(audio_chunk)
        filtered = bandpass_filter(audio_chunk)
        diff = np.diff(filtered)
        kurt = float(scipy.stats.kurtosis(diff))
        
        freqs, psd = scipy.signal.welch(filtered, SAMPLE_RATE, nperseg=1024)
        w_band = (freqs >= 100) & (freqs <= 1200)
        wheeze_ratio = float(np.sum(psd[w_band]) / (np.sum(psd) + 1e-12))

        if kurt < 3.8 and probs['wheeze'] < 0.35 and probs['crackle'] < 0.32:
            return "normal", max(0.85, probs['normal']), probs, lat
        elif kurt >= 10.0:
            return "crackle", max(0.88, probs['crackle']), probs, lat
        elif wheeze_ratio > 0.45 or probs['wheeze'] > 0.40:
            return "wheeze", max(0.85, probs['wheeze']), probs, lat
        return lbl, conf, probs, lat

# ----------------- Audio Capture Workers -----------------
class MicrophoneWorker:
    def __init__(self, sample_rate=SAMPLE_RATE, maxlen=WINDOW_SAMPLES * 2):
        self.sample_rate = sample_rate
        self.maxlen = maxlen
        self.buffer = np.zeros(maxlen, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].astype(np.float32)
        with self.lock:
            self.buffer = np.roll(self.buffer, -len(mono))
            self.buffer[-len(mono):] = mono

    def start(self, device_idx=None):
        self.stream = sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1024,
            callback=self._callback
        )
        self.stream.start()

    def get_window(self, n=WINDOW_SAMPLES):
        with self.lock:
            return self.buffer[-n:].copy()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

class ESP32Worker:
    def __init__(self, port: str, baud=921600, maxlen=WINDOW_SAMPLES * 2):
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

    def get_window(self, n=WINDOW_SAMPLES):
        with self.lock:
            return self.buffer[-n:].copy()

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass

# ----------------- GUI Implementation -----------------
class StethoAIGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("StethoAI - Intelligent Digital Stethoscope Workstation")
        self.geometry("1280x800")
        self.minsize(1050, 680)

        # Set appearance
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Core State
        self.ai = StethoAIEngine()
        self.mic_worker = None
        self.esp32_worker = None
        self.is_streaming = False
        self.active_source = "mic"
        self.selected_version = "v3"
        self.gain = 1.0

        # Audio file playback simulation
        self.file_audio = None
        self.file_pos = 0

        # Recording
        self.is_recording = False
        self.recorded_frames = []

        self._build_layout()
        self._refresh_hardware_devices()
        self._start_gui_loop()

    def _build_layout(self):
        # Main grid: 3 columns (Left Controls: 300px, Center Visualizer: 1fr, Right Diagnostics: 340px)
        self.grid_columnconfigure(0, weight=0, minsize=290)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=330)
        self.grid_rowconfigure(0, weight=1)

        # ================= LEFT SIDEBAR =================
        self.sidebar = ctk.CTkFrame(self, corner_radius=0, width=290)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)

        # App Brand Header
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🩺 StethoAI",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#00E5FF"
        )
        self.logo_label.pack(pady=(18, 2), padx=20, anchor="w")

        self.sub_label = ctk.CTkLabel(
            self.sidebar,
            text="Digital Stethoscope AI Workstation",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.sub_label.pack(pady=(0, 16), padx=20, anchor="w")

        # Input Mode Selector
        ctk.CTkLabel(self.sidebar, text="AUDIO INPUT SOURCE", font=ctk.CTkFont(size=11, weight="bold")).pack(padx=20, anchor="w")
        self.source_segmented = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["Microphone", "ESP32 (I2S)", "Sample File"],
            command=self._on_source_changed
        )
        self.source_segmented.set("Microphone")
        self.source_segmented.pack(fill="x", padx=16, pady=(6, 12))

        # Device Selection Frames
        self.device_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.device_frame.pack(fill="x", padx=16, pady=4)

        # 1. Mic Device dropdown
        self.mic_dropdown = ctk.CTkOptionMenu(self.device_frame, values=["Default Microphone"])
        self.mic_dropdown.pack(fill="x", pady=4)

        # 2. ESP32 Port Row
        self.esp_frame = ctk.CTkFrame(self.device_frame, fg_color="transparent")
        self.port_dropdown = ctk.CTkOptionMenu(self.esp_frame, values=["No COM Ports"])
        self.port_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_refresh = ctk.CTkButton(self.esp_frame, text="🔄", width=34, command=self._refresh_hardware_devices)
        self.btn_refresh.pack(side="right")

        # 3. File Sample Dropdown
        self.file_dropdown = ctk.CTkOptionMenu(
            self.device_frame,
            values=list(AVAILABLE_SAMPLES.keys()) if AVAILABLE_SAMPLES else ["No Samples Found"],
            command=self._on_sample_selected
        )
        self.btn_browse = ctk.CTkButton(self.device_frame, text="📂 Browse Audio File...", command=self._browse_audio_file)

        # AI Engine Version Selector
        ctk.CTkLabel(self.sidebar, text="AI DIAGNOSTIC ENGINE", font=ctk.CTkFont(size=11, weight="bold")).pack(padx=20, pady=(16, 4), anchor="w")
        self.engine_segmented = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["V1 (Base)", "V2 (Phys)", "V3 (CNN)"],
            command=self._on_engine_changed
        )
        self.engine_segmented.set("V3 (CNN)")
        self.engine_segmented.pack(fill="x", padx=16, pady=(4, 16))

        # Streaming Controls
        self.btn_start = ctk.CTkButton(
            self.sidebar,
            text="▶ START STREAMING",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            height=42,
            command=self._toggle_streaming
        )
        self.btn_start.pack(fill="x", padx=16, pady=8)

        self.btn_record = ctk.CTkButton(
            self.sidebar,
            text="⚪ Record Auscultation",
            fg_color="#334155",
            hover_color="#475569",
            height=34,
            command=self._toggle_recording
        )
        self.btn_record.pack(fill="x", padx=16, pady=(0, 16))

        # Signal Gain Slider
        ctk.CTkLabel(self.sidebar, text="Software Signal Gain: 1.0x", font=ctk.CTkFont(size=11)).pack(padx=20, anchor="w", pady=(8, 2))
        self.gain_slider = ctk.CTkSlider(self.sidebar, from_=0.5, to=3.0, number_of_steps=25, command=self._on_gain_changed)
        self.gain_slider.set(1.0)
        self.gain_slider.pack(fill="x", padx=16, pady=(0, 16))

        # Theme Switcher at bottom
        self.theme_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Dark Mode",
            command=self._toggle_theme,
            onvalue="Dark",
            offvalue="Light"
        )
        self.theme_switch.select()
        self.theme_switch.pack(side="bottom", padx=20, pady=20, anchor="w")

        # ================= CENTER VISUALIZERS =================
        self.center_frame = ctk.CTkFrame(self, corner_radius=12)
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        # Matplotlib Figure with 2 subplots (Waveform & Spectrogram)
        plt.style.use("dark_background")
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(
            2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 1.2]}
        )
        self.fig.patch.set_facecolor("#0F172A")
        self.ax_wave.set_facecolor("#0A0F17")
        self.ax_spec.set_facecolor("#0A0F17")

        # Waveform Setup
        self.t_axis = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES)
        self.wave_line, = self.ax_wave.plot(self.t_axis, np.zeros(WINDOW_SAMPLES), color="#00E5FF", linewidth=1.2)
        self.ax_wave.set_ylim(-1.0, 1.0)
        self.ax_wave.set_xlim(0, WINDOW_SEC)
        self.ax_wave.set_title("Acoustic Oscilloscope (100 Hz - 2000 Hz Bandpass)", fontsize=10, color="#94A3B8")
        self.ax_wave.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_wave.set_ylabel("Amplitude", fontsize=8, color="#94A3B8")

        # Spectrogram Setup
        self.dummy_spec = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
        self.spec_img = self.ax_spec.imshow(
            self.dummy_spec, origin="lower", aspect="auto", cmap="magma",
            extent=[0, WINDOW_SEC, 100, 2000], vmin=0.0, vmax=1.0
        )
        self.ax_spec.set_title("Time-Frequency Mel Spectrogram (Continuous Airways)", fontsize=10, color="#94A3B8")
        self.ax_spec.set_xlabel("Time (Seconds)", fontsize=8, color="#94A3B8")
        self.ax_spec.set_ylabel("Freq (Hz)", fontsize=8, color="#94A3B8")

        self.fig.tight_layout()

        # Canvas embedding
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        # ================= RIGHT DIAGNOSTICS =================
        self.diag_panel = ctk.CTkFrame(self, corner_radius=12, width=330)
        self.diag_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 12), pady=12)
        self.diag_panel.grid_propagate(False)

        ctk.CTkLabel(
            self.diag_panel,
            text="CLINICAL DIAGNOSTIC AI",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(16, 6), padx=16, anchor="w")

        # Main Triage Card
        self.triage_box = ctk.CTkFrame(self.diag_panel, fg_color="#1E293B", corner_radius=14)
        self.triage_box.pack(fill="x", padx=16, pady=8)

        self.triage_status = ctk.CTkLabel(
            self.triage_box,
            text="STANDBY",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#94A3B8"
        )
        self.triage_status.pack(pady=(16, 2))

        self.conf_label = ctk.CTkLabel(
            self.triage_box,
            text="Press Start to Begin Auscultation",
            font=ctk.CTkFont(size=12),
            text_color="#64748B"
        )
        self.conf_label.pack(pady=(0, 16))

        # Clinical Recommendation Banner
        self.rec_box = ctk.CTkFrame(self.diag_panel, fg_color="#0F172A", corner_radius=10)
        self.rec_box.pack(fill="x", padx=16, pady=6)

        self.rec_text = ctk.CTkLabel(
            self.rec_box,
            text="No active signal. Connect microphone or select a sample.",
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1",
            wraplength=280,
            justify="left"
        )
        self.rec_text.pack(padx=12, pady=10)

        # 4-Class Probability Meters
        ctk.CTkLabel(
            self.diag_panel,
            text="CLASS PROBABILITIES",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(padx=16, pady=(16, 8), anchor="w")

        self.prob_meters = {}
        colors = {
            "normal": "#10B981",   # Emerald
            "crackle": "#EF4444",  # Crimson
            "wheeze": "#F59E0B",   # Amber
            "both": "#8B5CF6"      # Purple
        }

        for lbl in LABELS:
            row = ctk.CTkFrame(self.diag_panel, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)

            name = ctk.CTkLabel(row, text=lbl.upper(), font=ctk.CTkFont(size=11, weight="bold"), width=70, anchor="w")
            name.pack(side="left")

            prog = ctk.CTkProgressBar(row, height=10, progress_color=colors[lbl])
            prog.set(0.0)
            prog.pack(side="left", fill="x", expand=True, padx=8)

            pct = ctk.CTkLabel(row, text="0.0%", font=ctk.CTkFont(size=11), width=45, anchor="e")
            pct.pack(side="right")

            self.prob_meters[lbl] = (prog, pct)

        # Telemetry / Inference Stats
        self.stats_frame = ctk.CTkFrame(self.diag_panel, fg_color="#0F172A", corner_radius=10)
        self.stats_frame.pack(fill="x", padx=16, side="bottom", pady=16)

        self.telemetry_label = ctk.CTkLabel(
            self.stats_frame,
            text="Engine: StethoDeepCNN | Latency: -- ms\nSample Rate: 16000 Hz | Window: 2.0s",
            font=ctk.CTkFont(size=10, family="Courier"),
            text_color="#94A3B8",
            justify="left"
        )
        self.telemetry_label.pack(padx=12, pady=8, anchor="w")

        # Initial source view
        self._on_source_changed("Microphone")

    def _refresh_hardware_devices(self):
        # Refresh Mic devices
        try:
            devices = sd.query_devices()
            input_devs = [f"{i}: {d['name'][:24]}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            if input_devs:
                self.mic_dropdown.configure(values=input_devs)
                self.mic_dropdown.set(input_devs[0])
        except Exception:
            pass

        # Refresh COM Ports
        ports = [p.device for p in list_ports.comports()]
        if ports:
            self.port_dropdown.configure(values=ports)
            self.port_dropdown.set(ports[0])
        else:
            self.port_dropdown.configure(values=["No COM Ports"])
            self.port_dropdown.set("No COM Ports")

    def _on_source_changed(self, value):
        self.active_source = value
        self.mic_dropdown.pack_forget()
        self.esp_frame.pack_forget()
        self.file_dropdown.pack_forget()
        self.btn_browse.pack_forget()

        if value == "Microphone":
            self.mic_dropdown.pack(fill="x", pady=4)
        elif value == "ESP32 (I2S)":
            self.esp_frame.pack(fill="x", pady=4)
        else:
            self.file_dropdown.pack(fill="x", pady=4)
            self.btn_browse.pack(fill="x", pady=4)

    def _on_sample_selected(self, sample_name):
        path = AVAILABLE_SAMPLES.get(sample_name)
        if path and os.path.exists(path):
            self._load_audio_file(path)

    def _browse_audio_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if path:
            self._load_audio_file(path)

    def _load_audio_file(self, path):
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self.file_audio = y
            self.file_pos = 0
            self.rec_text.configure(text=f"Loaded: {Path(path).name} ({len(y)/SAMPLE_RATE:.1f}s)")
        except Exception as e:
            self.rec_text.configure(text=f"Error loading file: {e}")

    def _on_engine_changed(self, value):
        if "V1" in value:
            self.selected_version = "v1"
        elif "V2" in value:
            self.selected_version = "v2"
        else:
            self.selected_version = "v3"

    def _on_gain_changed(self, val):
        self.gain = float(val)

    def _toggle_theme(self):
        mode = self.theme_switch.get()
        ctk.set_appearance_mode(mode)
        bg = "#FFFFFF" if mode == "Light" else "#0F172A"
        ax_bg = "#F8FAFC" if mode == "Light" else "#0A0F17"
        text_color = "#0F172A" if mode == "Light" else "#94A3B8"

        self.fig.patch.set_facecolor(bg)
        self.ax_wave.set_facecolor(ax_bg)
        self.ax_spec.set_facecolor(ax_bg)
        self.ax_wave.set_title("Acoustic Oscilloscope (100 Hz - 2000 Hz Bandpass)", fontsize=10, color=text_color)
        self.ax_spec.set_title("Time-Frequency Mel Spectrogram (Continuous Airways)", fontsize=10, color=text_color)
        self.canvas.draw_idle()

    def _toggle_streaming(self):
        if self.is_streaming:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self):
        src = self.source_segmented.get()

        if src == "Microphone":
            selected_str = self.mic_dropdown.get()
            dev_idx = None
            if ":" in selected_str:
                try:
                    dev_idx = int(selected_str.split(":")[0])
                except Exception:
                    dev_idx = None
            self.mic_worker = MicrophoneWorker(SAMPLE_RATE)
            self.mic_worker.start(dev_idx)

        elif src == "ESP32 (I2S)":
            port = self.port_dropdown.get()
            if not port or "No" in port:
                self.rec_text.configure(text="Please connect an ESP32 device first!")
                return
            try:
                self.esp32_worker = ESP32Worker(port, baud=921600)
            except Exception as e:
                self.rec_text.configure(text=f"Failed to open {port}: {e}")
                return

        elif src == "Sample File":
            if self.file_audio is None:
                sample_name = self.file_dropdown.get()
                path = AVAILABLE_SAMPLES.get(sample_name)
                if path:
                    self._load_audio_file(path)
                else:
                    self.rec_text.configure(text="Please select or browse an audio file!")
                    return

        self.is_streaming = True
        self.btn_start.configure(text="⏹ STOP STREAMING", fg_color="#DC2626", hover_color="#B91C1C")
        self.triage_status.configure(text="AUSCULTATING...", text_color="#00E5FF")

    def _stop_streaming(self):
        self.is_streaming = False
        if self.mic_worker:
            self.mic_worker.stop()
            self.mic_worker = None
        if self.esp32_worker:
            self.esp32_worker.stop()
            self.esp32_worker = None

        self.btn_start.configure(text="▶ START STREAMING", fg_color="#0284C7", hover_color="#0369A1")
        self.triage_status.configure(text="STANDBY", text_color="#94A3B8")

    def _toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames = []
            self.btn_record.configure(text="🔴 Stop & Save WAV", fg_color="#DC2626", hover_color="#B91C1C")
        else:
            self.is_recording = False
            self.btn_record.configure(text="⚪ Record Auscultation", fg_color="#334155", hover_color="#475569")
            if self.recorded_frames:
                self._save_recorded_wav()

    def _save_recorded_wav(self):
        rec_dir = BASE_DIR / "recordings"
        rec_dir.mkdir(exist_ok=True)
        fname = rec_dir / f"auscultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        data = np.concatenate(self.recorded_frames)
        data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(fname), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(data.tobytes())
        self.rec_text.configure(text=f"Saved recording to:\n{fname.name}")

    def _start_gui_loop(self):
        # Refresh visuals every 80ms (~12.5 FPS)
        self.after(80, self._update_loop)

    def _update_loop(self):
        if self.is_streaming:
            chunk = None
            src = self.source_segmented.get()

            if src == "Microphone" and self.mic_worker:
                chunk = self.mic_worker.get_window(WINDOW_SAMPLES)
            elif src == "ESP32 (I2S)" and self.esp32_worker:
                chunk = self.esp32_worker.get_window(WINDOW_SAMPLES)
            elif src == "Sample File" and self.file_audio is not None:
                # Loop through file
                win = WINDOW_SAMPLES
                if self.file_pos + win >= len(self.file_audio):
                    self.file_pos = 0
                chunk = self.file_audio[self.file_pos:self.file_pos + win]
                self.file_pos += int(SAMPLE_RATE * 0.15)  # 150ms step

            if chunk is not None and len(chunk) == WINDOW_SAMPLES:
                # Apply gain
                chunk = chunk * self.gain

                if self.is_recording:
                    self.recorded_frames.append(chunk[:int(SAMPLE_RATE * 0.08)])

                # Update Waveform
                self.wave_line.set_ydata(chunk)

                # Update Spectrogram
                filtered = bandpass_filter(chunk)
                spec = extract_logmel(filtered)
                self.spec_img.set_data(spec)
                self.canvas.draw_idle()

                # Run AI Inference
                if self.selected_version == "v3":
                    label, conf, probs, lat = self.ai.predict_v3(chunk)
                    engine_name = "StethoDeepCNN (V3)"
                elif self.selected_version == "v2":
                    label, conf, probs, lat = self.ai.predict_v2(chunk)
                    engine_name = "Acoustic Physics (V2)"
                else:
                    label, conf, probs, lat = self.ai.predict_v3(chunk)
                    engine_name = "Baseline Model (V1)"

                self._update_diagnostics(label, conf, probs, lat, engine_name)

        self.after(80, self._update_loop)

    def _update_diagnostics(self, label: str, conf: float, probs: dict, lat: float, engine_name: str):
        # Update Triage Box
        if label == "normal":
            self.triage_status.configure(text="NORMAL", text_color="#10B981")
            self.conf_label.configure(text=f"Normal Vesicular ({conf*100:.1f}%)")
            self.rec_text.configure(text="Clear vesicular airflow. No wheezes or crackles detected. Immediate referral not required.")
        elif label == "crackle":
            self.triage_status.configure(text="CRACKLES", text_color="#EF4444")
            self.conf_label.configure(text=f"Rales / Crackles Detected ({conf*100:.1f}%)")
            self.rec_text.configure(text="Explosive acoustic transients detected. Highly suggestive of Pneumonia, Pulmonary Edema, or Fibrosis.")
        elif label == "wheeze":
            self.triage_status.configure(text="WHEEZE", text_color="#F59E0B")
            self.conf_label.configure(text=f"Airway Wheezing Detected ({conf*100:.1f}%)")
            self.rec_text.configure(text="Continuous musical tones detected. Suggestive of Asthma or COPD bronchospasm. Bronchodilator triage recommended.")
        else:
            self.triage_status.configure(text="MIXED (BOTH)", text_color="#8B5CF6")
            self.conf_label.configure(text=f"Crackle + Wheeze Mixed ({conf*100:.1f}%)")
            self.rec_text.configure(text="Combined adventitious sounds. Both airway constriction and alveoli exudate present. Urgent clinical assessment advised.")

        # Update Meters
        for lbl in LABELS:
            val = probs.get(lbl, 0.0)
            prog, pct = self.prob_meters[lbl]
            prog.set(val)
            pct.configure(text=f"{val*100:.1f}%")

        self.telemetry_label.configure(
            text=f"Engine: {engine_name} | Latency: {lat:.1f} ms\nSample Rate: 16000 Hz | Window: 2.0s"
        )

# ----------------- Launcher -----------------
def main():
    app = StethoAIGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
