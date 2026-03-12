# ============================================================
# train_model.py
# EEG-Based Alzheimer's Disease Classification
# Dataset: AD_all_patients.xls (Kaggle ucimachinelearning)
# Classes: 0 = Healthy/Control, 1 = Alzheimer's Disease
# Channels: 16 (Fp1,Fp2,F7,F3,Fz,F4,F8,T3,C3,Cz,C4,T4,T5,P3,Pz,P4)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch
from scipy.stats import skew, kurtosis

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                     Dropout, BatchNormalization, Flatten,
                                     Bidirectional, Reshape, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH   = "AD_all_patients.csv"   # place this file in the same folder
OUTPUT_DIR  = "models"
SFREQ       = 256          # assumed sampling frequency (Hz)
EPOCH_SEC   = 4            # epoch length in seconds
EPOCH_SAMP  = SFREQ * EPOCH_SEC   # = 1024 samples per epoch
N_CLASSES   = 2
BATCH_SIZE  = 32
MAX_EPOCHS  = 100
PATIENCE    = 10
N_FOLDS     = 5
RANDOM_SEED = 42

CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T3','C3','Cz','C4','T4',
            'T5','P3','Pz','P4']
N_CH = len(CHANNELS)  # 16

# Frequency bands
BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  12),
    'beta' : (12, 30),
    'gamma': (30, 45),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── STEP 1: LOAD DATA ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading EEG data from Excel...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Status distribution:\n{df['status'].value_counts().to_string()}")

# ── STEP 2: EPOCH EXTRACTION ─────────────────────────────────────────────────
print("\nSTEP 2: Extracting 4-second epochs...")

X_raw = df[CHANNELS].values        # (848640, 16)
y_raw = df['status'].values         # (848640,)

# Determine majority label per epoch
n_rows   = X_raw.shape[0]
n_epochs = n_rows // EPOCH_SAMP
X_raw    = X_raw[:n_epochs * EPOCH_SAMP]
y_raw    = y_raw[:n_epochs * EPOCH_SAMP]

# Reshape to (n_epochs, EPOCH_SAMP, N_CH)
X_epochs = X_raw.reshape(n_epochs, EPOCH_SAMP, N_CH)
y_epochs = y_raw.reshape(n_epochs, EPOCH_SAMP)
y_labels = np.round(y_epochs.mean(axis=1)).astype(int)  # majority vote

print(f"  Total epochs : {n_epochs:,}")
print(f"  Samples/epoch: {EPOCH_SAMP}")
print(f"  Class distribution:")
unique, counts = np.unique(y_labels, return_counts=True)
for u, c in zip(unique, counts):
    label = "Healthy (CN)" if u == 0 else "Alzheimer's (AD)"
    print(f"    {label}: {c:,} ({100*c/n_epochs:.1f}%)")

# ── STEP 3: FEATURE EXTRACTION ───────────────────────────────────────────────
print("\nSTEP 3: Extracting features (Welch PSD + time-domain + Hjorth)...")

def extract_features(epoch):
    """
    epoch: (EPOCH_SAMP, N_CH)
    Returns: 1D feature vector of length N_CH * 12 = 192
    """
    features = []
    for ch in range(N_CH):
        sig = epoch[:, ch]

        # --- Welch PSD: 5 bands (mean power per band) ---
        freqs, psd = welch(sig, fs=SFREQ, nperseg=min(256, len(sig)))
        for band, (lo, hi) in BANDS.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            features.append(np.mean(psd[idx]) if idx.any() else 0.0)

        # --- Time-domain: mean, variance, skewness, kurtosis ---
        features.append(float(np.mean(sig)))
        features.append(float(np.var(sig)))
        features.append(float(skew(sig)))
        features.append(float(kurtosis(sig)))

        # --- Hjorth parameters: activity, mobility, complexity ---
        activity   = np.var(sig)
        diff1      = np.diff(sig)
        diff2      = np.diff(diff1)
        mobility   = np.sqrt(np.var(diff1) / (activity + 1e-10))
        complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
                      / (mobility + 1e-10))
        features.extend([activity, mobility, complexity])

    return np.array(features, dtype=np.float32)

# Extract all epochs (with progress)
N_FEAT = N_CH * 12   # 192 features
X_feat = np.zeros((n_epochs, N_FEAT), dtype=np.float32)
for i in range(n_epochs):
    if i % 5000 == 0:
        print(f"  Processing epoch {i:,}/{n_epochs:,}...")
    X_feat[i] = extract_features(X_epochs[i])

print(f"  Feature matrix shape: {X_feat.shape}")
print(f"  Features per epoch  : {N_FEAT} "
      f"({len(BANDS)} bands × {N_CH} ch + 4 time-domain × {N_CH} ch + "
      f"3 Hjorth × {N_CH} ch)")

# ── STEP 4: TRAIN-TEST SPLIT & NORMALISATION ─────────────────────────────────
print("\nSTEP 4: Splitting data and normalising features...")

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_labels, test_size=0.20, random_state=RANDOM_SEED,
    stratify=y_labels
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print(f"  Train: {X_train.shape[0]:,} epochs")
print(f"  Test : {X_test.shape[0]:,} epochs")

# Reshape for Conv1D / LSTM: (samples, features, 1)
X_train_3d = X_train.reshape(-1, N_FEAT, 1)
X_test_3d  = X_test.reshape(-1, N_FEAT, 1)

# ── STEP 5: MODEL DEFINITIONS ────────────────────────────────────────────────
print("\nSTEP 5: Building model architectures...")

def build_cnn1d(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name="1D_CNN")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_bilstm(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name="BiLSTM")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_lstm(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ], name="CNN_LSTM")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

BUILDERS = {
    "1D_CNN"  : build_cnn1d,
    "BiLSTM"  : build_bilstm,
    "CNN_LSTM": build_cnn_lstm,
}

# ── STEP 6: 5-FOLD CROSS-VALIDATION ─────────────────────────────────────────
print("\nSTEP 6: 5-Fold Stratified Cross-Validation...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, verbose=0)
]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_results = {}

for name, builder in BUILDERS.items():
    print(f"\n  [{name}] Cross-validating...")
    fold_acc, fold_f1, fold_auc = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(X_train_3d, y_train), 1):
        Xtr, Xval = X_train_3d[tr_idx], X_train_3d[val_idx]
        ytr, yval = y_train[tr_idx],    y_train[val_idx]

        model = builder((N_FEAT, 1), N_CLASSES)
        model.fit(Xtr, ytr,
                  validation_data=(Xval, yval),
                  epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=callbacks, verbose=0)

        y_pred = np.argmax(model.predict(Xval, verbose=0), axis=1)
        y_prob = model.predict(Xval, verbose=0)[:, 1]

        from sklearn.metrics import f1_score
        f1  = f1_score(yval, y_pred, average='macro')
        auc_val = roc_auc_score(yval, y_prob)
        acc = np.mean(y_pred == yval)

        fold_acc.append(acc);  fold_f1.append(f1);  fold_auc.append(auc_val)
        print(f"    Fold {fold}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")

    cv_results[name] = {
        'acc': np.mean(fold_acc),
        'f1' : np.mean(fold_f1),
        'auc': np.mean(fold_auc),
    }
    print(f"  [{name}] CV avg → Acc={cv_results[name]['acc']:.4f}  "
          f"F1={cv_results[name]['f1']:.4f}  "
          f"AUC={cv_results[name]['auc']:.4f}")

# ── STEP 7: FINAL TRAINING & TEST EVALUATION ─────────────────────────────────
print("\nSTEP 7: Final training on full train set + test evaluation...")

test_results = {}
histories    = {}

for name, builder in BUILDERS.items():
    print(f"\n  [{name}] Training final model...")
    model = builder((N_FEAT, 1), N_CLASSES)
    history = model.fit(
        X_train_3d, y_train,
        validation_split=0.1,
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )
    histories[name] = history

    y_pred = np.argmax(model.predict(X_test_3d, verbose=0), axis=1)
    y_prob = model.predict(X_test_3d, verbose=0)[:, 1]
    acc    = np.mean(y_pred == y_test)

    from sklearn.metrics import f1_score, precision_score, recall_score
    report = classification_report(
        y_test, y_pred,
        target_names=["Healthy (CN)", "Alzheimer's (AD)"],
        output_dict=True
    )
    auc_score = roc_auc_score(y_test, y_prob)

    test_results[name] = {
        'accuracy' : acc,
        'precision': report['macro avg']['precision'],
        'recall'   : report['macro avg']['recall'],
        'f1'       : report['macro avg']['f1-score'],
        'auc'      : auc_score,
        'report'   : report,
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'model'    : model,
    }
    print(f"  [{name}] Test → Acc={acc:.4f}  "
          f"F1={report['macro avg']['f1-score']:.4f}  AUC={auc_score:.4f}")

    # Save classification report
    rpt_str = classification_report(
        y_test, y_pred,
        target_names=["Healthy (CN)", "Alzheimer's (AD)"]
    )
    with open(os.path.join(OUTPUT_DIR, f"classification_report_{name}.txt"), "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Test Accuracy : {acc:.4f}\n")
        f.write(f"Test AUC      : {auc_score:.4f}\n\n")
        f.write(rpt_str)

# ── STEP 8: SAVE BEST MODEL ───────────────────────────────────────────────────
print("\nSTEP 8: Saving best model...")

best_name = max(test_results, key=lambda k: test_results[k]['f1'])
best_model = test_results[best_name]['model']
best_model.save(os.path.join(OUTPUT_DIR, "best_model.keras"))
print(f"  Best model: {best_name} (F1={test_results[best_name]['f1']:.4f})")
print(f"  Saved to  : {OUTPUT_DIR}/best_model.keras")

# ── STEP 9: PLOTS ─────────────────────────────────────────────────────────────
print("\nSTEP 9: Generating plots...")

CLASS_NAMES = ["Healthy (CN)", "Alzheimer's (AD)"]

# 9a — Confusion matrices (one per model)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, test_results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    acc = res['accuracy']
    ax.set_title(f"{name}\nAccuracy: {acc:.2%}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()
print("  Saved: confusion_matrices.png")

# 9b — ROC curves
plt.figure(figsize=(8, 6))
colors = ['#2196F3', '#FF5722', '#4CAF50']
for (name, res), color in zip(test_results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {res['auc']:.4f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlim([0,1]); plt.ylim([0,1.02])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models", fontweight='bold')
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
plt.close()
print("  Saved: roc_curves.png")

# 9c — Training history (best model)
history = histories[best_name]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'],     label='Train Acc', color='#2196F3')
ax1.plot(history.history['val_accuracy'], label='Val Acc',   color='#FF5722')
ax1.set_title(f"{best_name} — Training Accuracy", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history.history['loss'],     label='Train Loss', color='#2196F3')
ax2.plot(history.history['val_loss'], label='Val Loss',   color='#FF5722')
ax2.set_title(f"{best_name} — Training Loss", fontweight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)
plt.close()
print("  Saved: training_history.png")

# 9d — Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics   = ['accuracy', 'f1', 'auc']
titles    = ['Accuracy', 'Macro F1-Score', 'ROC-AUC']
bar_color = ['#2196F3', '#FF5722', '#4CAF50']
names     = list(test_results.keys())
for ax, metric, title, color in zip(axes, metrics, titles, bar_color):
    vals = [test_results[n][metric] for n in names]
    bars = ax.bar(names, vals, color=color, alpha=0.85, edgecolor='black')
    ax.set_ylim(0, 1.05); ax.set_title(title, fontweight='bold')
    ax.set_ylabel(title); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
plt.suptitle("Model Comparison — Test Set Performance", fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: model_comparison.png")

# 9e — Gradient saliency map (best model)
print("\nSTEP 10: Computing gradient saliency map (best model)...")

sample_idx = 0
X_sample = tf.constant(X_test_3d[sample_idx:sample_idx+1], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(X_sample)
    preds  = best_model(X_sample, training=False)
    pred_class = tf.argmax(preds[0]).numpy()
    score  = preds[0][pred_class]
grads  = tape.gradient(score, X_sample)
saliency = tf.abs(grads).numpy().squeeze()   # (N_FEAT,)

# Average across feature groups per channel
saliency_per_ch = saliency.reshape(N_CH, 12).mean(axis=1)
plt.figure(figsize=(10, 4))
bars = plt.bar(CHANNELS, saliency_per_ch, color='#7B1FA2', alpha=0.85,
               edgecolor='black')
plt.title(f"Feature Importance — Gradient Saliency ({best_name})\n"
          f"Predicted class: {CLASS_NAMES[pred_class]}",
          fontweight='bold')
plt.xlabel("EEG Channel"); plt.ylabel("Mean |Gradient|")
plt.xticks(rotation=45); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("  Saved: feature_importance.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 60)
for name, res in test_results.items():
    marker = " ★" if name == best_name else ""
    print(f"{name:<12} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['auc']:>10.4f}{marker}")
print("=" * 60)
print(f"\nAll outputs saved to: ./{OUTPUT_DIR}/")
print("  best_model.keras")
print("  scaler.pkl")
print("  confusion_matrices.png")
print("  roc_curves.png")
print("  training_history.png")
print("  model_comparison.png")
print("  feature_importance.png")
print("  classification_report_<model>.txt  (x3)")
print("\nDone.")
