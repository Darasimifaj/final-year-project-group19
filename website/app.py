"""
app.py — Flask backend for EEG Alzheimer's Classification Dashboard
Dataset: AD_all_patients.xls (Kaggle ucimachinelearning)
Classes: 0 = Healthy/Control, 1 = Alzheimer's Disease
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json

# Optional TF import — graceful fallback to demo mode
try:
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

from scipy.signal import welch
from scipy.stats import skew, kurtosis

app = Flask(__name__, static_folder='.')
CORS(app)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\oluwa\OneDrive\Desktop\final year project confirmed\models\best_model.keras"
SCALER_PATH = r"C:\Users\oluwa\OneDrive\Desktop\final year project confirmed\models\scaler.pkl"
SFREQ       = 256
EPOCH_SEC   = 4
EPOCH_SAMP  = SFREQ * EPOCH_SEC

CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T3','C3','Cz','C4','T4',
            'T5','P3','Pz','P4']
N_CH = len(CHANNELS)

BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8, 12),
    'beta' : (12,30),
    'gamma': (30,45),
}

CLASS_NAMES = ["Healthy (CN)", "Alzheimer's (AD)"]
CLASS_COLORS = ["#00C896", "#FF4757"]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model, scaler = None, None
if MODEL_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model  = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✓ Model and scaler loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_features(epoch):
    features = []
    for ch in range(N_CH):
        sig = epoch[:, ch]
        freqs, psd = welch(sig, fs=SFREQ, nperseg=min(256, len(sig)))
        for band, (lo, hi) in BANDS.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            features.append(float(np.mean(psd[idx])) if idx.any() else 0.0)
        features.extend([
            float(np.mean(sig)), float(np.var(sig)),
            float(skew(sig)),    float(kurtosis(sig))
        ])
        activity = np.var(sig)
        diff1    = np.diff(sig)
        diff2    = np.diff(diff1)
        mobility   = np.sqrt(np.var(diff1) / (activity + 1e-10))
        complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
                      / (mobility + 1e-10))
        features.extend([float(activity), float(mobility), float(complexity)])
    return np.array(features, dtype=np.float32)

# ── DEMO DATA GENERATOR ───────────────────────────────────────────────────────
def generate_demo_eeg(label=None):
    """Generate synthetic EEG epoch with realistic AD/CN patterns."""
    t = np.linspace(0, EPOCH_SEC, EPOCH_SAMP)
    epoch = np.zeros((EPOCH_SAMP, N_CH))
    if label is None:
        label = np.random.randint(0, 2)
    for ch in range(N_CH):
        if label == 1:  # AD: elevated delta/theta, reduced alpha
            sig = (8  * np.sin(2*np.pi*2*t)   # delta
                 + 6  * np.sin(2*np.pi*6*t)   # theta
                 + 2  * np.sin(2*np.pi*10*t)  # reduced alpha
                 + np.random.randn(EPOCH_SAMP) * 3)
        else:           # CN: strong alpha, lower slow waves
            sig = (2  * np.sin(2*np.pi*2*t)
                 + 2  * np.sin(2*np.pi*6*t)
                 + 10 * np.sin(2*np.pi*10*t)  # strong alpha
                 + 4  * np.sin(2*np.pi*20*t)  # beta
                 + np.random.randn(EPOCH_SAMP) * 2)
        epoch[:, ch] = sig * (1 + 0.1 * np.random.randn())
    return epoch, label

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'channels': CHANNELS,
        'n_channels': N_CH,
        'classes': CLASS_NAMES,
        'sfreq': SFREQ,
        'epoch_sec': EPOCH_SEC,
    })

@app.route('/api/predict/demo', methods=['GET'])
def predict_demo():
    """Run prediction on synthetic EEG data."""
    label_param = request.args.get('label', None)
    true_label  = int(label_param) if label_param is not None else None
    epoch, true_label = generate_demo_eeg(true_label)

    features = extract_features(epoch).reshape(1, -1)

    if model is not None and scaler is not None:
        features_scaled = scaler.transform(features)
        features_3d     = features_scaled.reshape(1, -1, 1)
        probs = model.predict(features_3d, verbose=0)[0].tolist()
        pred  = int(np.argmax(probs))
    else:
        # Fallback demo probabilities
        if true_label == 1:
            probs = [float(np.random.uniform(0.05, 0.25)),
                     float(np.random.uniform(0.75, 0.95))]
        else:
            probs = [float(np.random.uniform(0.75, 0.95)),
                     float(np.random.uniform(0.05, 0.25))]
        pred = int(np.argmax(probs))

    # Band powers for waveform display (first channel)
    sig = epoch[:, 0]
    freqs, psd = welch(sig, fs=SFREQ, nperseg=256)
    band_powers = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        band_powers[band] = float(np.mean(psd[idx])) if idx.any() else 0.0

    # Downsample waveform to 512 points for display
    step = max(1, EPOCH_SAMP // 512)
    waveform = epoch[::step, 0].tolist()

    return jsonify({
        'prediction'  : pred,
        'class_name'  : CLASS_NAMES[pred],
        'true_label'  : true_label,
        'true_name'   : CLASS_NAMES[true_label],
        'probabilities': probs,
        'band_powers' : band_powers,
        'waveform'    : waveform,
        'correct'     : pred == true_label,
    })

@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """Predict from uploaded CSV file containing one epoch."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    try:
        df = pd.read_csv(f)
        # Ensure correct channels present
        missing = [c for c in CHANNELS if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400
        data = df[CHANNELS].values[:EPOCH_SAMP]
        if len(data) < EPOCH_SAMP:
            return jsonify({'error': f'Need at least {EPOCH_SAMP} rows'}), 400
        epoch    = data[:EPOCH_SAMP]
        features = extract_features(epoch).reshape(1, -1)

        if model is not None and scaler is not None:
            features_scaled = scaler.transform(features)
            features_3d     = features_scaled.reshape(1, -1, 1)
            probs = model.predict(features_3d, verbose=0)[0].tolist()
        else:
            probs = [0.5, 0.5]

        pred = int(np.argmax(probs))
        sig  = epoch[:, 0]
        freqs, psd = welch(sig, fs=SFREQ, nperseg=256)
        band_powers = {}
        for band, (lo, hi) in BANDS.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            band_powers[band] = float(np.mean(psd[idx])) if idx.any() else 0.0
        step     = max(1, EPOCH_SAMP // 512)
        waveform = epoch[::step, 0].tolist()

        return jsonify({
            'prediction'   : pred,
            'class_name'   : CLASS_NAMES[pred],
            'probabilities': probs,
            'band_powers'  : band_powers,
            'waveform'     : waveform,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results')
def get_results():
    """Return saved model performance metrics."""
    results = {}
    models_dir = "../models"
    for name in ["1D_CNN", "BiLSTM", "CNN_LSTM"]:
        path = os.path.join(models_dir, f"classification_report_{name}.txt")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = f.read()
    return jsonify(results)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  EEG Alzheimer's Classification Dashboard")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
