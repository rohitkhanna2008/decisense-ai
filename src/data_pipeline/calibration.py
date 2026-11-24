"""
DeciSense AI - Module: Microphone Calibration & Calibrated SPL Computation
Implements smartphone microphone calibration against Class-2 Reference SLMs
and IEC 61672-1 A-weighted Sound Pressure Level (SPL) metrics (L_eq and L_max).
"""

import os
import json
import wave
import io
import numpy as np
import scipy.signal as signal

# Optional librosa import
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

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
    offset_db = round(float(reference_db - raw_measured_db), 2)
    registry = load_calibration_registry(filepath)
    
    entry = {
        "device_model": device_model.strip(),
        "reference_db": round(float(reference_db), 2),
        "raw_measured_db": round(float(raw_measured_db), 2),
        "offset_db": offset_db
    }
    registry[device_model.strip()] = entry
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
# 2. IEC 61672-1 Digital A-Weighting Filter
# -----------------------------------------------------------------------------
def design_a_weighting_filter(sample_rate=16000):
    """
    Designs an IEC 61672-1 compliant digital IIR filter approximating
    standard A-weighting frequency response for human ear perception.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217

    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    w3 = 2 * np.pi * f3
    w4 = 2 * np.pi * f4

    # Analog poles and zeros
    b_analog = [(2 * np.pi * f4)**2 * (10**(1.9997 / 20)), 0, 0, 0, 0]
    p = [w1, w1, w2, w3, w4, w4]
    a_analog = np.poly([-x for x in p])

    # Bilinear transformation to digital IIR coefficients
    b_dig, a_dig = signal.bilinear(b_analog, a_analog, fs=sample_rate)
    return b_dig, a_dig


def apply_a_weighting(waveform, sample_rate=16000):
    """Applies standard A-weighting digital IIR filter to audio waveform."""
    try:
        b, a = design_a_weighting_filter(sample_rate)
        return signal.lfilter(b, a, waveform)
    except Exception:
        return waveform


# -----------------------------------------------------------------------------
# 3. Calibrated SPL Metrics Computation (L_eq and Fast L_max)
# -----------------------------------------------------------------------------
def load_audio_waveform(audio_input, target_sr=16000):
    """
    Helper to load audio into normalized float32 [-1.0, 1.0] array.
    Accepts:
      - File path (str)
      - Bytes / BytesIO
      - numpy array (1D)
    """
    if isinstance(audio_input, np.ndarray):
        waveform = audio_input.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, target_sr

    if isinstance(audio_input, (bytes, io.BytesIO)):
        bio = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else audio_input
        with wave.open(bio, 'rb') as wf:
            n_ch = wf.getnchannels()
            width = wf.getsampwidth()
            src_sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if width == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if n_ch > 1:
            arr = arr.reshape(-1, n_ch).mean(axis=1)

        if src_sr != target_sr and len(arr) > 0:
            target_len = int(len(arr) * float(target_sr) / src_sr)
            arr = signal.resample(arr, target_len)
        return arr, target_sr

    # Audio input is a string file path
    if HAS_LIBROSA:
        try:
            y, sr = librosa.load(audio_input, sr=target_sr, mono=True)
            return y, sr
        except Exception:
            pass

    # Standard library fallback for WAV files
    with wave.open(audio_input, 'rb') as wf:
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        src_sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1)

    if src_sr != target_sr and len(arr) > 0:
        target_len = int(len(arr) * float(target_sr) / src_sr)
        arr = signal.resample(arr, target_len)

    return arr, target_sr


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
