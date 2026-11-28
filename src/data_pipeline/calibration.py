"""
DeciSense AI - Module: Microphone Calibration & Calibrated SPL Computation
Implements smartphone microphone calibration against Class-2 Reference SLMs
and IEC 61672-1 A-weighted Sound Pressure Level (SPL) metrics (L_eq and L_max).
"""

import os
import sys
import json
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.audio_classifier import (
    apply_a_weighting,
    design_a_weighting_filter,
    load_audio as load_audio_waveform,
)

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
CALIBRATION_FILE = os.path.join(DEFAULT_DATA_DIR, "device_calibration.json")


# -----------------------------------------------------------------------------
# 1. Device Calibration Management
# -----------------------------------------------------------------------------
def load_calibration_registry(filepath=CALIBRATION_FILE):
    """Loads device calibration mapping from JSON."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_calibration_registry(registry, filepath=CALIBRATION_FILE):
    """Saves device calibration mapping to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def calibrate_device(device_model, reference_db, raw_measured_db, filepath=CALIBRATION_FILE):
    """
    Computes and saves device-specific hardware offset constant (offset_db)
    against a Class-2 reference Sound Level Meter (SLM) or calibrated acoustic source.
    
    offset_db = reference_db - raw_measured_db
    
    Parameters:
        device_model (str): e.g. "OnePlus 11R", "Samsung Galaxy A54", "Redmi Note 12"
        reference_db (float): Ground truth reading from Class-2 SLM (e.g. 74.0 dB(A))
        raw_measured_db (float): Raw digital reading from smartphone mic without offset
        filepath (str): Path to device_calibration.json
        
    Returns:
        dict: Device calibration entry containing offset_db and timestamp
    """
    clean_model = str(device_model or "").strip()
    if not clean_model:
        raise ValueError("Device model name cannot be empty.")

    offset_db = round(float(reference_db - raw_measured_db), 2)
    registry = load_calibration_registry(filepath)
    
    entry = {
        "device_model": clean_model,
        "reference_db": round(float(reference_db), 2),
        "raw_measured_db": round(float(raw_measured_db), 2),
        "offset_db": offset_db
    }
    registry[clean_model] = entry
    save_calibration_registry(registry, filepath)
    return entry


def get_device_offset(device_model, default_offset=92.0, filepath=CALIBRATION_FILE):
    """
    Retrieves the calibrated offset_db for a given device model.
    Falls back to default_offset if not calibrated.
    """
    registry = load_calibration_registry(filepath)
    if device_model and device_model.strip() in registry:
        return float(registry[device_model.strip()]["offset_db"])
    return float(default_offset)


def list_calibrated_devices(filepath=CALIBRATION_FILE):
    """Lists all registered devices and calibration offsets."""
    return load_calibration_registry(filepath)


# -----------------------------------------------------------------------------
# 2. IEC 61672-1 Digital A-Weighting Filter & Calibrated SPL Computation
# -----------------------------------------------------------------------------
# Re-exported from src.audio_classifier (design_a_weighting_filter,
# apply_a_weighting, load_audio_waveform) to keep a single DSP implementation.

def compute_calibrated_spl(audio_input, sample_rate=16000, offset_db=0.0):
    """
    Computes exact calibrated A-weighted SPL metrics:
      - L_eq_db: Equivalent continuous sound level over full duration
      - L_max_db: Maximum impulse peak dB over 125 ms fast time-weighting windows
      - crest_factor_db: L_max_db - L_eq_db
      
    Parameters:
        audio_input: WAV file path, bytes, BytesIO, or 1D numpy array
        sample_rate: sampling rate (default: 16000 Hz)
        offset_db: hardware calibration offset constant in dB
        
    Returns:
        dict: {
            "L_eq_db": float,
            "L_max_db": float,
            "crest_factor_db": float,
            "duration_sec": float,
            "sample_rate": int
        }
    """
    waveform, sr = load_audio_waveform(audio_input, target_sr=sample_rate)

    if len(waveform) == 0:
        return {
            "L_eq_db": 0.0,
            "L_max_db": 0.0,
            "crest_factor_db": 0.0,
            "duration_sec": 0.0,
            "sample_rate": sr
        }

    duration_sec = round(len(waveform) / float(sr), 3)

    # 1. Apply A-weighting filter
    a_weighted = apply_a_weighting(waveform, sr)

    # 2. Compute Equivalent Continuous Sound Level (L_eq)
    rms_full = np.sqrt(np.mean(a_weighted**2) + 1e-12)
    leq_raw = 20.0 * np.log10(max(rms_full, 1e-6))
    l_eq = round(float(leq_raw + offset_db), 1)

    # 3. Compute Fast Time-Weighting Peak (L_max over 125 ms windows per IEC 61672-1)
    # 125 ms window length; 25 ms hop for smooth peak capture
    window_samples = int(sr * 0.125)
    hop_samples = int(sr * 0.025)

    if len(a_weighted) <= window_samples:
        l_max = l_eq
    else:
        num_windows = (len(a_weighted) - window_samples) // hop_samples + 1
        window_rms = [
            np.sqrt(np.mean(a_weighted[i * hop_samples : i * hop_samples + window_samples]**2) + 1e-12)
            for i in range(num_windows)
        ]
        peak_rms = max(window_rms) if len(window_rms) > 0 else rms_full
        lmax_raw = 20.0 * np.log10(max(peak_rms, 1e-6))
        l_max = round(float(lmax_raw + offset_db), 1)

    crest_factor = round(float(max(0.0, l_max - l_eq)), 1)

    return {
        "L_eq_db": l_eq,
        "L_max_db": l_max,
        "crest_factor_db": crest_factor,
        "duration_sec": duration_sec,
        "sample_rate": sr
    }


if __name__ == "__main__":
    # Self-test calibration
    print("Testing Device Calibration API...")
    cal_res = calibrate_device("OnePlus 11R (Test)", reference_db=74.0, raw_measured_db=-18.5)
    print("  [+] Calibrated Entry:", cal_res)
    offset = get_device_offset("OnePlus 11R (Test)")
    print(f"  [+] Retrieved Offset: {offset} dB")
    
    # Self-test SPL on synthetic test wave
    t = np.linspace(0, 5.0, 16000 * 5)
    test_wave = 0.5 * np.sin(2 * np.pi * 1000 * t)
    spl_metrics = compute_calibrated_spl(test_wave, sample_rate=16000, offset_db=offset)
    print("  [+] Test Waveform SPL Metrics:", spl_metrics)
