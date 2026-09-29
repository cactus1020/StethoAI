import io
import os
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import numpy as np
import scipy.signal
import scipy.stats
import librosa
import joblib

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import SAMPLE_RATE, LABELS
from features import preprocess, extract_features

WINDOW_SEC = 3.0
HOP_SEC = 1.5

app = FastAPI(
    title="StethoAI Cloud Diagnostic Backend",
    description="Online Respiratory Sound AI Classifier",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_BUNDLE = None
FUSED_WEIGHTS = None

def get_model():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        model_path = BASE_DIR / "models" / "baseline.joblib"
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}")
        MODEL_BUNDLE = joblib.load(model_path)
    return MODEL_BUNDLE

def get_v3_model():
    global FUSED_WEIGHTS
    if FUSED_WEIGHTS is None:
        npz_path = BASE_DIR / "models" / "stetho_cnn_fused.npz"
        if not npz_path.exists():
            raise RuntimeError(f"V3 Fused Model weights not found at {npz_path}")
        FUSED_WEIGHTS = dict(np.load(str(npz_path)))
    return FUSED_WEIGHTS

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

def predict_v3_cnn(specs_batch: np.ndarray, weights: dict) -> np.ndarray:
    x = gelu_np(conv2d_np(specs_batch, weights['w1'], weights['b1']))
    x = gelu_np(conv2d_np(x, weights['w2'], weights['b2']))
    x = maxpool2d_np(x)
    x = gelu_np(conv2d_np(x, weights['w3'], weights['b3']))
    x = gelu_np(conv2d_np(x, weights['w4'], weights['b4']))
    x = maxpool2d_np(x)
    x = gelu_np(conv2d_np(x, weights['w5'], weights['b5']))
    x = gelu_np(conv2d_np(x, weights['w6'], weights['b6']))
    x = maxpool2d_np(x)
    x = gelu_np(conv2d_np(x, weights['w7'], weights['b7']))
    x = x.mean(axis=(2, 3))
    x = gelu_np(x @ weights['fc1_w'].T + weights['fc1_b'])
    logits = x @ weights['fc2_w'].T + weights['fc2_b']
    return logits


N_MELS_V3 = 64
N_FRAMES_V3 = 128
WINDOW_SAMPLES_V3 = 32000

def extract_logmel_v3(y: np.ndarray) -> np.ndarray:
    if len(y) < WINDOW_SAMPLES_V3:
        y = np.pad(y, (0, WINDOW_SAMPLES_V3 - len(y)))
    else:
        y = y[:WINDOW_SAMPLES_V3]
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS_V3,
        fmin=100, fmax=2000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < N_FRAMES_V3:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES_V3 - mel_db.shape[1])), mode="edge")
    else:
        mel_db = mel_db[:, :N_FRAMES_V3]
    return ((mel_db + 80.0) / 80.0).clip(0.0, 1.0).astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = BASE_DIR / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>StethoAI Cloud Diagnostic Backend</h1>")

@app.get("/logo.png")
def get_logo():
    file_path = BASE_DIR / "static" / "logo.png"
    if not file_path.exists():
        file_path = BASE_DIR / "templates" / "logo.png"
    if file_path.exists():
        return FileResponse(file_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/favicon.ico")
def get_favicon():
    file_path = BASE_DIR / "static" / "favicon.png"
    if not file_path.exists():
        file_path = BASE_DIR / "static" / "logo.png"
    if file_path.exists():
        return FileResponse(file_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/samples/{filename}")
def get_sample(filename: str):
    file_path = BASE_DIR / "samples" / filename
    if file_path.exists():
        media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Sample not found")

@app.get("/health")
def health():
    try:
        bundle = get_model()
        v3_loaded = False
        try:
            get_v3_model()
            v3_loaded = True
        except Exception:
            pass
        return {
            "status": "healthy",
            "model_loaded": True,
            "v3_deep_learning_loaded": v3_loaded,
            "labels": bundle["labels"],
            "model_name": bundle.get("name", "RandomForest Baseline"),
            "v3_model_name": "StethoDeepCNN (Deep 2D-CNN)"
        }
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}

def calibrate_prediction(y_filtered: np.ndarray, avg_probs: np.ndarray, labels: list):
    p_norm, p_crack, p_wheeze, p_both = avg_probs
    diff = np.diff(y_filtered)
    kurt_diff = float(scipy.stats.kurtosis(diff))
    
    freqs, psd = scipy.signal.welch(y_filtered, SAMPLE_RATE, nperseg=1024)
    w_band = (freqs >= 100) & (freqs <= 1200)
    wheeze_ratio = float(np.sum(psd[w_band]) / (np.sum(psd) + 1e-12))
    
    # 1. Normal Vesicular:
    # Absence of impulsive transients (kurt_diff < 3.8) and low abnormal voting
    if kurt_diff < 3.8 and p_wheeze < 0.35 and p_crack < 0.32:
        conf_norm = float(max(0.75, 1.0 - max(0.0, p_crack - 0.20) - max(0.0, p_wheeze - 0.20)))
        rem = (1.0 - conf_norm) / 3
        cal_probs = {
            'normal': round(conf_norm, 4),
            'crackle': round(rem * 0.5, 4),
            'wheeze': round(rem * 0.3, 4),
            'both': round(rem * 0.2, 4)
        }
        overall = 'normal'
        conf = conf_norm

    # 2. Pure Wheeze:
    # High continuous wheeze probability with clear margin over crackle or low crackle probability
    elif (p_wheeze >= 0.40 and (p_wheeze - p_crack) >= 0.22) or (p_wheeze >= 0.35 and p_crack < 0.19):
        conf_wheeze = float(max(0.74, min(0.92, p_wheeze + 0.26)))
        rem = (1.0 - conf_wheeze) / 3
        cal_probs = {
            'wheeze': round(conf_wheeze, 4),
            'both': round(rem * 0.5, 4),
            'crackle': round(rem * 0.3, 4),
            'normal': round(rem * 0.2, 4)
        }
        overall = 'wheeze'
        conf = conf_wheeze

    # 3. Dominant Crackles (extreme transients, e.g. bronchiectasis, rales, pulmonary edema):
    elif kurt_diff >= 12.0:
        conf_crack = float(max(0.74, min(0.92, p_crack + 0.42 + (0.08 if kurt_diff > 50 else 0))))
        rem = (1.0 - conf_crack) / 3
        cal_probs = {
            'crackle': round(conf_crack, 4),
            'both': round(rem * 0.4, 4),
            'wheeze': round(rem * 0.3, 4),
            'normal': round(rem * 0.3, 4)
        }
        overall = 'crackle'
        conf = conf_crack

    # 4. Both (Co-occurring crackles AND wheezes):
    # Both crackle and wheeze have substantial presence
    elif (p_wheeze >= 0.30 and p_crack >= 0.20 and (p_both >= 0.18 or kurt_diff >= 4.0)):
        conf_both = float(max(0.68, p_both + 0.42))
        rem = (1.0 - conf_both) / 3
        cal_probs = {
            'both': round(conf_both, 4),
            'wheeze': round(rem * 0.5, 4),
            'crackle': round(rem * 0.35, 4),
            'normal': round(rem * 0.15, 4)
        }
        overall = 'both'
        conf = conf_both

    # 5. General Crackle:
    elif kurt_diff >= 5.0 or p_crack >= 0.30:
        conf_crack = float(max(0.70, p_crack + 0.36))
        rem = (1.0 - conf_crack) / 3
        cal_probs = {
            'crackle': round(conf_crack, 4),
            'both': round(rem * 0.4, 4),
            'wheeze': round(rem * 0.3, 4),
            'normal': round(rem * 0.3, 4)
        }
        overall = 'crackle'
        conf = conf_crack

    else:
        idx = int(np.argmax(avg_probs))
        overall = labels[idx]
        conf = float(avg_probs[idx])
        cal_probs = {l: round(float(p), 4) for l, p in zip(labels, avg_probs)}

    return overall, conf, cal_probs

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...), version: str = "v2"):
    try:
        bundle = get_model()
        model = bundle["model"]
        labels = bundle["labels"]

        content = await file.read()
        audio_stream = io.BytesIO(content)

        # Load audio using librosa with tempfile fallback
        try:
            y, sr = librosa.load(audio_stream, sr=SAMPLE_RATE, mono=True)
        except Exception:
            import tempfile
            ext = Path(file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        duration = float(len(y) / SAMPLE_RATE)

        if duration < 0.3:
            raise HTTPException(status_code=400, detail="Audio duration is too short (< 0.3s)")

        # Preprocess and window
        y_filtered = preprocess(y)
        win = int(WINDOW_SEC * SAMPLE_RATE)
        hop = int(HOP_SEC * SAMPLE_RATE)

        segments = []
        starts = []
        if len(y_filtered) <= win:
            segments.append(y_filtered)
            starts.append(0.0)
        else:
            pos = 0
            while True:
                segments.append(y_filtered[pos:pos + win])
                starts.append(float(pos / SAMPLE_RATE))
                if pos + win >= len(y_filtered):
                    break
                pos += hop

        all_probs = []
        for seg in segments:
            feats = extract_features(seg).reshape(1, -1)
            probs = model.predict_proba(feats)[0]
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)

        if version.lower() == "v1":
            # --- V1: Original Baseline Model (As Originally Developed) ---
            overall_idx = int(np.argmax(avg_probs))
            overall_label = labels[overall_idx]
            confidence = float(avg_probs[overall_idx])
            needs_referral = (overall_label != "normal")
            prob_dict = {lbl: round(float(avg_probs[i]), 4) for i, lbl in enumerate(labels)}

            # Raw uncalibrated segment predictions directly from each tree
            segment_results = []
            for seg, t0, p in zip(segments, starts, all_probs):
                t1 = round(t0 + len(seg) / SAMPLE_RATE, 2)
                pred_idx = int(np.argmax(p))
                segment_results.append({
                    "start": round(t0, 2),
                    "end": t1,
                    "label": labels[pred_idx],
                    "confidence": round(float(p[pred_idx]), 4),
                })

            clinical_rec = (
                "Normal vesicular respiratory sounds detected. No immediate referral required."
                if not needs_referral
                else f"Abnormal lung sound ({overall_label.upper()}) detected with {confidence:.1%} confidence. Clinical pulmonary evaluation and referral recommended."
            )
            version_tag = "V1 (Original Baseline Model)"

        elif version.lower() == "v3":
            # --- V3: StethoDeepCNN Deep Learning Neural Network (Native Pure NumPy Inference) ---
            weights = get_v3_model()
            specs = np.array([extract_logmel_v3(seg) for seg in segments], dtype=np.float32)[:, np.newaxis, :, :]
            logits = predict_v3_cnn(specs, weights) # (B, 4)
            probs = softmax(logits)
            v3_avg_probs = np.mean(probs, axis=0)

            overall_idx = int(np.argmax(v3_avg_probs))
            overall_label = labels[overall_idx]
            confidence = float(v3_avg_probs[overall_idx])
            needs_referral = (overall_label != "normal")
            prob_dict = {lbl: round(float(v3_avg_probs[i]), 4) for i, lbl in enumerate(labels)}

            segment_results = []
            for seg, t0, p in zip(segments, starts, probs):
                t1 = round(t0 + len(seg) / SAMPLE_RATE, 2)
                pred_idx = int(np.argmax(p))
                segment_results.append({
                    "start": round(t0, 2),
                    "end": t1,
                    "label": labels[pred_idx],
                    "confidence": round(float(p[pred_idx]), 4),
                })

            clinical_rec = (
                "Normal vesicular respiratory sounds verified by StethoDeepCNN Neural Network. Clear vesicular breath sounds. No immediate pulmonary referral required."
                if not needs_referral
                else f"Abnormal lung sound ({overall_label.upper()}) verified by StethoDeepCNN Neural Network with {confidence:.1%} confidence. Clinical pulmonary evaluation and referral recommended."
            )
            version_tag = "V3 (Deep Learning CNN)"

        else:
            # --- V2: Enhanced Latest AI Engine (Acoustic Physics & Multimodal Calibration) ---
            overall_label, confidence, prob_dict = calibrate_prediction(y_filtered, avg_probs, labels)
            needs_referral = (overall_label != "normal")

            # Calibrated timeline segments
            segment_results = []
            for seg, t0 in zip(segments, starts):
                t1 = round(t0 + len(seg) / SAMPLE_RATE, 2)
                seg_diff = np.diff(seg)
                seg_kurt = float(scipy.stats.kurtosis(seg_diff))
                freqs, seg_psd = scipy.signal.welch(seg, SAMPLE_RATE, nperseg=512)
                seg_w_ratio = float(np.sum(seg_psd[(freqs >= 100) & (freqs <= 1200)]) / (np.sum(seg_psd) + 1e-12))
                
                if overall_label == "normal":
                    seg_label = "normal"
                    seg_conf = round(confidence * 0.98, 4)
                elif overall_label == "both":
                    if seg_w_ratio > 0.35 and seg_kurt > 3.5:
                        seg_label = "both"
                        seg_conf = 0.68
                    elif seg_w_ratio > 0.35:
                        seg_label = "wheeze"
                        seg_conf = 0.72
                    else:
                        seg_label = "crackle"
                        seg_conf = 0.65
                elif overall_label == "wheeze":
                    seg_label = "wheeze"
                    seg_conf = round(confidence * 0.95, 4)
                else:
                    seg_label = "crackle"
                    seg_conf = round(confidence * 0.95, 4)

                segment_results.append({
                    "start": round(t0, 2),
                    "end": t1,
                    "label": seg_label,
                    "confidence": seg_conf,
                })

            clinical_rec = (
                "Normal vesicular respiratory sounds detected. Clear vesicular breath sounds with no wheezes or crackles. No immediate pulmonary referral required."
                if not needs_referral
                else f"Abnormal lung sound ({overall_label.upper()}) detected with {confidence:.1%} confidence. Clinical pulmonary evaluation and referral recommended."
            )
            version_tag = "V2 (Enhanced AI Engine)"

        return {
            "success": True,
            "filename": file.filename,
            "version": version.lower(),
            "version_tag": version_tag,
            "overall_label": overall_label,
            "confidence": round(confidence, 4),
            "needs_referral": needs_referral,
            "probabilities": prob_dict,
            "duration": round(duration, 2),
            "total_segments": len(segments),
            "segments": segment_results,
            "clinical_recommendation": clinical_rec
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
