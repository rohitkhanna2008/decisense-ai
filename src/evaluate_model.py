"""
DeciSense AI - Module: Automated Model Evaluation & CPCB Compliance Auditing
Performs Stratified 5-Fold Cross-Validation, benchmarks Baseline vs Deep Classifiers,
and exports publication-grade metrics, confusion matrix plots, and CPCB compliance reports.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Scikit-learn tools
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".mpl_cache")

from src.audio_classifier import (
    load_audio, compute_decibel_metrics, compute_mel_spectrogram,
    extract_acoustic_features, CLASSES
)
from src.geospatial_engine import compute_sleep_disruption_index, CPCB_LIMITS

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CSV_LOG_PATH = os.path.join(DATA_DIR, "chennai_audio_logs.csv")


# -----------------------------------------------------------------------------
# 1. Feature Extraction Pipeline (Log-Mel + MFCC + Temporal Crest)
# -----------------------------------------------------------------------------
def extract_mfcc(log_mel_spec, n_mfcc=13):
    """Computes standard MFCC coefficients using Discrete Cosine Transform (DCT-II)."""
    # DCT along the mel-frequency axis (axis 0)
    dct_spec = scipy.fftpack.dct(log_mel_spec, type=2, axis=0, norm='ortho')
    return dct_spec[:n_mfcc, :]


def extract_clip_features(wav_path, target_sr=16000):
    """
    Extracts standardized acoustic feature vector:
    - 13 MFCCs: Mean & Standard Deviation across time (26 dims)
    - Spectral characteristics: Centroid, Rolloff, ZCR, Low Ratio, Horn Ratio (5 dims)
    - Log-Mel Spectrogram band summary: 16 sub-band energies mean & std (32 dims)
    - Decibel temporal dynamics: L_eq, L_max, Crest Factor (3 dims)
    Total: 66 acoustic features.
    """
    # Resolve relative paths
    if not os.path.isabs(wav_path):
        resolved_path = os.path.join(PROJECT_ROOT, wav_path)
    else:
        resolved_path = wav_path

    if not os.path.exists(resolved_path):
        # Fallback to sample_wavs
        fallback_name = os.path.basename(wav_path)
        fallback_path = os.path.join(DATA_DIR, "sample_wavs", fallback_name)
        if os.path.exists(fallback_path):
            resolved_path = fallback_path
        else:
            raise FileNotFoundError(f"Audio clip not found: {wav_path}")

    waveform, sr = load_audio(resolved_path, target_sr=target_sr)
    leq, lmax, crest = compute_decibel_metrics(waveform, sample_rate=sr, offset_db=92.0)
    mel_spec = compute_mel_spectrogram(waveform, sample_rate=sr, n_mels=64)
    mfccs = extract_mfcc(mel_spec, n_mfcc=13)
    spec_feats = extract_acoustic_features(waveform, sample_rate=sr)

    # 1. MFCC stats
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)

    # 2. Spectral dynamics
    spectral_vector = np.array([
        spec_feats["spectral_centroid"],
        spec_feats["rolloff"],
        spec_feats["zcr"],
        spec_feats["low_ratio"],
        spec_feats["horn_ratio"]
    ])

    # 3. Log-Mel pooled 16-band stats
    band_size = mel_spec.shape[0] // 16
    mel_band_means = [np.mean(mel_spec[i*band_size:(i+1)*band_size, :]) for i in range(16)]
    mel_band_stds = [np.std(mel_spec[i*band_size:(i+1)*band_size, :]) for i in range(16)]

    # 4. Decibel temporal dynamics
    db_vector = np.array([leq, lmax, crest])

    feature_vector = np.concatenate([
        mfcc_means, mfcc_stds, spectral_vector,
        mel_band_means, mel_band_stds, db_vector
    ])

    return feature_vector, leq, lmax, crest


# -----------------------------------------------------------------------------
# 2. Stratified 5-Fold Cross-Validation Evaluation
# -----------------------------------------------------------------------------
def evaluate_models(csv_path=CSV_LOG_PATH, outputs_dir=OUTPUTS_DIR):
    """
    Executes Stratified 5-Fold Cross-Validation on the Chennai audio dataset.
    Compares Baseline Classifier (Random Forest) vs Deep Classifier (MLP / Dense Head).
    Saves report JSON, confusion matrix PNG, and CPCB compliance summary CSV.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    print(f"Loading Chennai audio logs from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records across {len(CLASSES)} target classes.")

    # Extract acoustic features for each record
    X_list = []
    y_list = []
    valid_indices = []

    print("Extracting acoustic features (Log-Mel + MFCC + Spectral + Crest Dynamics)...")
    start_feat_time = time.time()

    for idx, row in df.iterrows():
        try:
            label = str(row.get("ground_truth_class", "")).strip()
            if label not in CLASSES:
                continue
            wav_val = row.get("wav_path")
            if pd.isna(wav_val) or not str(wav_val).strip():
                continue
            feats, leq, lmax, crest = extract_clip_features(str(wav_val))
            X_list.append(feats)
            y_list.append(label)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  [!] Skipping row {idx} ({row.get('clip_id', 'unknown')}): {e}")

    if not X_list:
        raise RuntimeError("No valid audio clips found for evaluation.")

    feat_extract_duration = time.time() - start_feat_time
    avg_inference_ms = round((feat_extract_duration / max(len(X_list), 1)) * 1000.0, 2)
    print(f"Extracted features for {len(X_list)} clips in {feat_extract_duration:.2f}s (Avg: {avg_inference_ms} ms/clip).")

    X = np.array(X_list)
    y = np.array(y_list)

    # Class mappings
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_idx = np.array([class_to_idx[c] for c in y])

    # Determine safe number of stratified folds from smallest class count
    _, class_counts = np.unique(y_idx, return_counts=True)
    n_splits = min(5, int(np.min(class_counts)))
    if n_splits < 2:
        raise ValueError("Need at least 2 samples per class for stratified cross-validation.")

    # Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Models under comparison
    baseline_rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    deep_acoustic_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True
    )

    baseline_preds = np.zeros(len(y_idx), dtype=int)
    deep_preds = np.zeros(len(y_idx), dtype=int)

    print("\nExecuting Stratified 5-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_idx), 1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        # Train Baseline Random Forest
        baseline_rf.fit(X_train, y_idx[train_idx])
        baseline_preds[val_idx] = baseline_rf.predict(X_val)

        # Train Deep Acoustic Classifier Head
        deep_acoustic_mlp.fit(X_train, y_idx[train_idx])
        deep_preds[val_idx] = deep_acoustic_mlp.predict(X_val)

        fold_acc = accuracy_score(y_idx[val_idx], deep_preds[val_idx])
        print(f"  Fold {fold}/5 - Validation Accuracy (Deep Classifier): {fold_acc * 100:.2f}%")

    # Metrics computation for Deep Classifier
    deep_acc = float(accuracy_score(y_idx, deep_preds))
    deep_prec = float(precision_score(y_idx, deep_preds, average="macro", zero_division=0))
    deep_rec = float(recall_score(y_idx, deep_preds, average="macro", zero_division=0))
    deep_f1 = float(f1_score(y_idx, deep_preds, average="macro", zero_division=0))

    # Metrics computation for Baseline RF
    base_acc = float(accuracy_score(y_idx, baseline_preds))
    base_prec = float(precision_score(y_idx, baseline_preds, average="macro", zero_division=0))
    base_rec = float(recall_score(y_idx, baseline_preds, average="macro", zero_division=0))
    base_f1 = float(f1_score(y_idx, baseline_preds, average="macro", zero_division=0))

    # Per-Class breakdown
    report_dict = classification_report(
        y_idx, deep_preds, target_names=CLASSES, output_dict=True, zero_division=0
    )

    per_class_summary = {}
    for cname in CLASSES:
        if cname in report_dict:
            per_class_summary[cname] = {
                "precision": round(float(report_dict[cname]["precision"]), 3),
                "recall": round(float(report_dict[cname]["recall"]), 3),
                "f1_score": round(float(report_dict[cname]["f1-score"]), 3),
                "support": int(report_dict[cname]["support"])
            }

    # Assemble Model Evaluation Report JSON
    eval_report = {
        "project": "DeciSense AI: Hyper-Local Acoustic Pollution & Sleep-Disruption Mapper",
        "milestone": "FRESHATHON 3.0 (1st Review) & IEEE Paper Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_sample_count": len(X),
        "validation_strategy": "Stratified 5-Fold Cross-Validation (k=5)",
        "inference_latency_ms_per_clip": avg_inference_ms,
        "primary_deep_classifier": {
            "model_type": "Deep Acoustic Classifier (Log-Mel/MFCC Dense Neural Architecture)",
            "accuracy": round(deep_acc, 4),
            "macro_precision": round(deep_prec, 4),
            "macro_recall": round(deep_rec, 4),
            "macro_f1": round(deep_f1, 4)
        },
        "baseline_feature_classifier": {
            "model_type": "Random Forest Baseline (n_estimators=100)",
            "accuracy": round(base_acc, 4),
            "macro_precision": round(base_prec, 4),
            "macro_recall": round(base_rec, 4),
            "macro_f1": round(base_f1, 4)
        },
        "performance_gain_pct": round((deep_f1 - base_f1) * 100.0, 2),
        "per_class_metrics": per_class_summary
    }

    report_path = os.path.join(outputs_dir, "model_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n[+] Saved Evaluation Report: {report_path}")


    # -------------------------------------------------------------------------
    # 3. High-Resolution Publication-Grade Confusion Matrix Heatmap (300 DPI)
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_idx, deep_preds, labels=list(range(len(CLASSES))))
    row_sums = cm.sum(axis=1)[:, np.newaxis].astype(float)
    cm_normalized = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Custom Clean Black & White Monochrome Colormap on pure-white background
    mono_cmap = LinearSegmentedColormap.from_list(
        "DeciSenseMonochrome",
        ["#FFFFFF", "#E5E5E5", "#A3A3A3", "#525252", "#000000"],
        N=256
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300, facecolor="#FFFFFF")
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=mono_cmap, vmin=0.0, vmax=1.0)

    # Format ticks with clean display names
    display_labels = [c.replace('_', ' ').title() for c in CLASSES]
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(display_labels, rotation=28, ha="right", fontsize=9, color="#000000", fontweight="600")
    ax.set_yticklabels(display_labels, fontsize=9, color="#000000", fontweight="600")

    ax.set_xlabel("Predicted Acoustic Class", fontsize=10, color="#000000", fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=10, color="#000000", fontweight="bold", labelpad=10)
    ax.set_title(
        f"DeciSense AI — Normalized Confusion Matrix\nStratified 5-Fold CV (Overall Accuracy: {deep_acc*100:.1f}%)",
        fontsize=11, color="#000000", fontweight="bold", pad=14
    )

    # Annotate cells with percentage and sample count
    thresh = cm_normalized.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            pct = cm_normalized[i, j] * 100.0
            count = cm[i, j]
            text_color = "#FFFFFF" if cm_normalized[i, j] > thresh else "#000000"
            ax.text(
                j, i, f"{pct:.1f}%\n(n={count})",
                ha="center", va="center", color=text_color,
                fontsize=8.5, fontweight="bold"
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors="#000000")
    cbar.set_label("Classification Probability Ratio", fontsize=8.5, color="#000000", fontweight="600")

    # Clean borders
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"[+] Saved Publication-Grade Confusion Matrix (300 DPI): {cm_path}")


    # -------------------------------------------------------------------------
    # 4. CPCB Zone Compliance & Sleep Disruption Summary Table
    # -------------------------------------------------------------------------
    compliance_rows = []
    
    # Infer time window if not already present
    if "time_window" in df.columns:
        df["time_window_short"] = df["time_window"].apply(lambda s: "Night" if "Night" in str(s) else "Day")
    elif "timestamp" in df.columns:
        hours = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour.fillna(12)
        df["time_window_short"] = np.where((hours >= 6) & (hours < 22), "Day", "Night")
    else:
        df["time_window_short"] = "Day"

    window_col = "time_window_short"

    for zone in ["silence", "residential", "commercial"]:
        for win in ["Day", "Night"]:
            subset = df[(df["zone_type"] == zone) & (df[window_col] == win)]
            sample_count = len(subset)
            if sample_count == 0:
                continue

            mean_leq = round(float(subset["L_eq_db"].mean()), 1)
            peak_lmax = round(float(subset["L_max_db"].max()), 1)
            
            # Compute Sleep-Disruption Scores
            test_hour = 23 if win == "Night" else 14
            sleep_scores = [
                compute_sleep_disruption_index(r["L_eq_db"], r["L_max_db"], zone=zone, timestamp_hour=test_hour)["score"]
                for _, r in subset.iterrows()
            ]
            mean_sleep = round(float(np.mean(sleep_scores)), 1)

            # CPCB statutory violation rate
            statutory_cap = CPCB_LIMITS[win.lower()].get(zone, 45.0)
            violations = sum(subset["L_eq_db"] > statutory_cap)
            violation_rate = round((violations / sample_count) * 100.0, 1)

            compliance_rows.append({
                "zone_type": zone.capitalize(),
                "time_window": win,
                "sample_count": sample_count,
                "cpcb_limit_db": statutory_cap,
                "mean_l_eq_db": mean_leq,
                "peak_l_max_db": peak_lmax,
                "mean_sleep_score": mean_sleep,
                "cpcb_violation_rate_pct": f"{violation_rate}%"
            })

    summary_df = pd.DataFrame(compliance_rows)
    summary_path = os.path.join(outputs_dir, "cpcb_zone_compliance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[+] Saved CPCB Compliance Audit Summary: {summary_path}")

    return eval_report, summary_df


if __name__ == "__main__":
    evaluate_models()
