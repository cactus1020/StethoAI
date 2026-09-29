import os
import sys
import math
import random
import numpy as np
import librosa
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Fix random seeds for reproducible excellence
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

SAMPLE_RATE = 16000
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC) # 32000
N_MELS = 64
N_FRAMES = 128
HOP_LEN = 256
N_FFT = 1024

LABELS = ["normal", "crackle", "wheeze", "both"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

# ----------------- Audio Preprocessing & Spectrogram Extraction -----------------
_SOS = scipy.signal.butter(4, [100.0 / 8000.0, 2000.0 / 8000.0], btype="band", output="sos")

def bandpass_filter(y: np.ndarray) -> np.ndarray:
    if len(y) < 30:
        return y
    return scipy.signal.sosfiltfilt(_SOS, y).astype(np.float32)

def extract_logmel(y: np.ndarray) -> np.ndarray:
    # Ensure exact length
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    else:
        y = y[:WINDOW_SAMPLES]
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS,
        fmin=100, fmax=2000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    if mel_db.shape[1] < N_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - mel_db.shape[1])), mode="edge")
    else:
        mel_db = mel_db[:, :N_FRAMES]
        
    norm_spec = ((mel_db + 80.0) / 80.0).clip(0.0, 1.0).astype(np.float32)
    return norm_spec

# ----------------- Acoustic Simulation & Augmentation -----------------
def generate_crackles(duration_sec=2.0, sr=16000, density=18):
    """Simulate physiological crackles: explosive acoustic transients."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    signal = np.zeros_like(t)
    num_crackles = int(duration_sec * density)
    for _ in range(num_crackles):
        idx = np.random.randint(0, len(t) - 500)
        dur = np.random.randint(80, 250) # 5-15 ms
        tau = np.linspace(0, 0.015, dur)
        f_res = np.random.uniform(250, 850)
        amp = np.random.uniform(0.3, 0.9)
        pulse = amp * np.exp(-tau / 0.003) * np.sin(2 * np.pi * f_res * tau)
        signal[idx:idx + dur] += pulse
    return signal.astype(np.float32)

def generate_wheezes(duration_sec=2.0, sr=16000):
    """Simulate physiological wheezes: continuous musical tones in airways."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    f0 = np.random.uniform(180, 550)
    # Pitch modulation (breathing envelope)
    freq_curve = f0 + 25 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    harmonic1 = 0.6 * np.sin(phase)
    harmonic2 = 0.3 * np.sin(2 * phase)
    harmonic3 = 0.15 * np.sin(3 * phase)
    wheeze = harmonic1 + harmonic2 + harmonic3
    # Envelope
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.8 * t))
    return (wheeze * env).astype(np.float32)

def apply_spec_augment(spec: np.ndarray, num_masks=2, max_mask_f=8, max_mask_t=16):
    """SpecAugment on logmel matrix (64, 128)."""
    spec = spec.copy()
    for _ in range(num_masks):
        f = np.random.randint(1, max_mask_f)
        f0 = np.random.randint(0, N_MELS - f)
        spec[f0:f0+f, :] = 0.0

        t = np.random.randint(1, max_mask_t)
        t0 = np.random.randint(0, N_FRAMES - t)
        spec[:, t0:t0+t] = 0.0
    return spec

# ----------------- Deep Neural Network Architecture -----------------
class StethoDeepCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Conv Block 1: 1 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2), # 32 x 64
            nn.Dropout2d(0.10)
        )
        # Conv Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2), # 16 x 32
            nn.Dropout2d(0.15)
        )
        # Conv Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2), # 8 x 16
            nn.Dropout2d(0.20)
        )
        # Conv Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits

# ----------------- Dataset Builder -----------------
class RespiratorySpectrogramDataset(Dataset):
    def __init__(self, specs, labels, augment=False):
        self.specs = specs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                spec = apply_spec_augment(spec)
            # Add subtle gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, spec.shape).astype(np.float32)
                spec = np.clip(spec + noise, 0.0, 1.0)
        tensor_spec = torch.from_numpy(spec).unsqueeze(0) # (1, 64, 128)
        label = self.labels[idx]
        return tensor_spec, label

def load_audio_seeds():
    seed_paths = {
        "normal": [
            r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples\normal.wav",
            r"c:\Users\Dell\Ai Box\DSPproject_v2\Normal- Vesicular.wav"
        ],
        "crackle": [
            r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples\crackles.mp3",
            r"D:\Download\Downloads\Crackles- Bronchiectasis.mp3",
            r"D:\Download\Downloads\Crackles- Pulmonary Edema.mp3"
        ],
        "wheeze": [
            r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples\wheeze.mp3",
            r"D:\Download\Downloads\Wheeze- Asthma.mp3"
        ],
        "both": [
            r"c:\Users\Dell\Ai Box\test_vercel_stetho\samples\both.wav",
            r"c:\Users\Dell\Ai Box\DSPproject_v2\Both- Crackle and Wheeze.wav"
        ]
    }
    
    seeds = {k: [] for k in LABELS}
    for label, paths in seed_paths.items():
        for p in paths:
            if os.path.exists(p):
                y, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
                y_filt = bandpass_filter(y)
                seeds[label].append(y_filt)
    return seeds

def build_training_dataset(seeds, samples_per_class=450):
    print("Synthesizing multi-modal clinical training set...")
    all_specs = []
    all_labels = []

    for label_idx, label in enumerate(LABELS):
        audios = seeds[label]
        for i in range(samples_per_class):
            base_audio = random.choice(audios) if audios else np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            
            # Slice random 2-second window
            if len(base_audio) > WINDOW_SAMPLES:
                start = random.randint(0, len(base_audio) - WINDOW_SAMPLES)
                chunk = base_audio[start:start + WINDOW_SAMPLES].copy()
            else:
                chunk = np.pad(base_audio, (0, max(0, WINDOW_SAMPLES - len(base_audio))))[:WINDOW_SAMPLES].copy()

            # Random circular shift
            shift = random.randint(-8000, 8000)
            chunk = np.roll(chunk, shift)

            # Random amplitude gain
            gain = random.uniform(0.6, 1.4)
            chunk = chunk * gain

            # Random general stethoscope background noise applied across ALL classes
            if random.random() > 0.4:
                noise_lvl = random.uniform(0.0005, 0.005)
                chunk += np.random.normal(0, noise_lvl, len(chunk)).astype(np.float32)

            # Class-specific physiological additions
            if label == "crackle":
                if random.random() > 0.35:
                    crackles = generate_crackles(duration_sec=2.0, sr=SAMPLE_RATE, density=random.randint(12, 32))
                    chunk = 0.65 * chunk + 0.35 * crackles
            elif label == "wheeze":
                if random.random() > 0.35:
                    wheezes = generate_wheezes(duration_sec=2.0, sr=SAMPLE_RATE)
                    chunk = 0.65 * chunk + 0.35 * wheezes
            elif label == "both":
                if random.random() > 0.25:
                    crackles = generate_crackles(duration_sec=2.0, sr=SAMPLE_RATE, density=random.randint(10, 24))
                    wheezes = generate_wheezes(duration_sec=2.0, sr=SAMPLE_RATE)
                    chunk = 0.50 * chunk + 0.30 * crackles + 0.20 * wheezes

            # Extract logmel
            spec = extract_logmel(chunk)
            all_specs.append(spec)
            all_labels.append(label_idx)

    specs = np.array(all_specs, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    return specs, labels

# ----------------- Training Loop -----------------
def train():
    seeds = load_audio_seeds()
    for lbl, lst in seeds.items():
        print(f"Loaded {len(lst)} reference audio seeds for '{lbl}'")
        
    specs, labels = build_training_dataset(seeds, samples_per_class=450)
    print(f"Total dataset: {specs.shape[0]} spectrograms across 4 classes.")

    # Split train/val
    indices = np.arange(len(specs))
    np.random.shuffle(indices)
    split_pt = int(0.85 * len(specs))
    train_idx, val_idx = indices[:split_pt], indices[split_pt:]

    train_ds = RespiratorySpectrogramDataset(specs[train_idx], labels[train_idx], augment=True)
    val_ds = RespiratorySpectrogramDataset(specs[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training StethoDeepCNN on: {device}")

    model = StethoDeepCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)

    num_epochs = 18
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | Val Acc: {val_acc*100:5.1f}%")

    # Save PyTorch Model
    os.makedirs(r"c:\Users\Dell\Ai Box\test_vercel_stetho\models", exist_ok=True)
    pt_path = r"c:\Users\Dell\Ai Box\test_vercel_stetho\models\stetho_cnn_v3.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"PyTorch weights saved to {pt_path}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 128, device=device)
    onnx_path = r"c:\Users\Dell\Ai Box\test_vercel_stetho\models\stetho_cnn_v3.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    print(f"ONNX Model successfully exported to {onnx_path} ({os.path.getsize(onnx_path)/1e6:.2f} MB)")

    # Also copy to DSPproject_v2/models
    dsp_models = r"c:\Users\Dell\Ai Box\DSPproject_v2\models"
    os.makedirs(dsp_models, exist_ok=True)
    import shutil
    shutil.copy(onnx_path, os.path.join(dsp_models, "stetho_cnn_v3.onnx"))
    print("Copied ONNX model to DSPproject_v2/models/")

if __name__ == "__main__":
    train()
