"""
DeciSense AI - Module: Audio Preprocessing & Quality-Control Validator
Converts raw smartphone field recordings into standardized 16 kHz, mono, 16-bit PCM
5.0-second WAV clips, with automatic splitting and QC filtering (clipping & silence detection).
"""

import os
import sys
import wave
import io
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.audio_classifier import _normalize_audio_array

# Optional imports
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
RAW_DIR = os.path.join(DEFAULT_DATA_DIR, "raw_recordings")
PROCESSED_DIR = os.path.join(DEFAULT_DATA_DIR, "processed_wavs")

TARGET_SR = 16000
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 80,000 samples


# -----------------------------------------------------------------------------
# 1. Quality Control & Signal Validation
# -----------------------------------------------------------------------------
def validate_audio_quality(waveform, sample_rate=TARGET_SR):
    """
    Validates audio signal quality against clipping and silence thresholds.
    
    Returns:
        tuple: (is_valid: bool, reason: str, metrics: dict)
    """
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.size == 0:
        return False, "EMPTY_SIGNAL", {"rms": 0.0, "clipped_ratio": 0.0}
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    # 1. Check for silence (RMS < 1e-4)
    rms = float(np.sqrt(np.mean(arr**2) + 1e-12))
    if rms < 1e-4:
        return False, "REJECT_SILENT", {"rms": rms, "clipped_ratio": 0.0}

    # 2. Check for severe clipping (amplitude >= 0.99 for > 5% of frames)
    clipped_count = np.sum(np.abs(arr) >= 0.99)
    clipped_ratio = float(clipped_count / len(arr))
    if clipped_ratio > 0.05:
        return False, "REJECT_CLIPPED", {"rms": rms, "clipped_ratio": clipped_ratio}

    return True, "PASSED", {"rms": rms, "clipped_ratio": clipped_ratio}


# -----------------------------------------------------------------------------
# 2. Audio Ingestion & Resampling
# -----------------------------------------------------------------------------
def load_and_resample(file_input, target_sr=TARGET_SR):
    """
    Decodes audio from file path, bytes/BytesIO, or ndarray and resamples to 16 kHz mono float32.
    """
    if isinstance(file_input, np.ndarray):
        arr = np.nan_to_num(file_input.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr, target_sr

    if isinstance(file_input, (bytes, bytearray, io.BytesIO)):
        raw_bytes = bytes(file_input) if isinstance(file_input, (bytes, bytearray)) else file_input.getvalue()
        try:
            src_sr, data = wavfile.read(io.BytesIO(raw_bytes))
            return _normalize_audio_array(data, src_sr, target_sr=target_sr)
        except Exception:
            if HAS_SOUNDFILE:
                arr, src_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                return _normalize_audio_array(arr, src_sr, target_sr=target_sr)
            if HAS_LIBROSA:
                arr, _ = librosa.load(io.BytesIO(raw_bytes), sr=target_sr, mono=True)
                return np.asarray(arr, dtype=np.float32), target_sr
            raise

    # File path loading
    if HAS_LIBROSA:
        try:
            arr, sr = librosa.load(file_input, sr=target_sr, mono=True)
            return np.asarray(arr, dtype=np.float32), target_sr
        except Exception:
            pass

    if HAS_SOUNDFILE:
        try:
            arr, src_sr = sf.read(file_input, dtype="float32")
            return _normalize_audio_array(arr, src_sr, target_sr=target_sr)
        except Exception:
            pass

    try:
        src_sr, data = wavfile.read(file_input)
        return _normalize_audio_array(data, src_sr, target_sr=target_sr)
    except Exception:
        pass

    # Standard wave reader fallback
    with wave.open(file_input, 'rb') as wf:
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        src_sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        arr = np.frombuffer(raw, dtype=np.int16)
    elif width == 4:
        arr = np.frombuffer(raw, dtype=np.int32)
    else:
        arr = np.frombuffer(raw, dtype=np.uint8)

    if n_ch > 1 and arr.size > 0:
        arr = arr.reshape(-1, n_ch)

    return _normalize_audio_array(arr, src_sr, target_sr=target_sr)


def save_pcm16_wav(filepath, waveform, sample_rate=TARGET_SR):
    """Saves normalized float32 waveform as standard 16-bit PCM Mono WAV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Clip to prevent overflow
    clipped = np.clip(waveform, -1.0, 1.0)
    scaled = np.int16(clipped * 32767)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())


# -----------------------------------------------------------------------------
# 3. Audio Standardization & 5-Second Segment Splitting
# -----------------------------------------------------------------------------
def process_and_standardize_audio(file_input, output_dir=PROCESSED_DIR, base_filename="clip"):
    """
    Takes an input audio file (path, bytes, or waveform), validates quality,
    and splits into standardized exact 5.0-second 16 kHz Mono WAV files.
    
    Returns:
        list of dict: Information on each generated 5s clip.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        waveform, sr = load_and_resample(file_input, target_sr=TARGET_SR)
    except Exception as e:
        return [{"status": "ERROR_CORRUPTED", "error": str(e), "wav_path": None}]

    total_samples = len(waveform)
    if total_samples < int(TARGET_SR * 3.0):
        # Discard clips shorter than 3 seconds
        return [{
            "status": "REJECT_TOO_SHORT",
            "duration_sec": round(total_samples / TARGET_SR, 2),
            "wav_path": None
        }]

    # If audio is between 3.0s and 5.0s, pad with gentle silence/mirroring to exact 5.0s
    if total_samples < TARGET_SAMPLES:
        pad_len = TARGET_SAMPLES - total_samples
        padded = np.pad(waveform, (0, pad_len), mode="constant")
        segments = [padded]
    else:
        # Split into non-overlapping 5-second chunks (80,000 samples each)
        num_full_segments = total_samples // TARGET_SAMPLES
        segments = []
        for i in range(num_full_segments):
            start = i * TARGET_SAMPLES
            end = start + TARGET_SAMPLES
            segments.append(waveform[start:end])

    generated_clips = []
    base_name = os.path.splitext(os.path.basename(base_filename))[0]

    for seg_idx, seg in enumerate(segments, start=1):
        is_valid, reason, metrics = validate_audio_quality(seg, sample_rate=TARGET_SR)
        
        if not is_valid:
            generated_clips.append({
                "status": reason,
                "segment_index": seg_idx,
                "metrics": metrics,
                "wav_path": None
            })
            continue

        # Format output filename: e.g. base_name_seg01.wav or base_name.wav
        if len(segments) > 1:
            out_name = f"{base_name}_seg{seg_idx:02d}.wav"
        else:
            out_name = f"{base_name}.wav"

        out_path = os.path.join(output_dir, out_name)
        save_pcm16_wav(out_path, seg, sample_rate=TARGET_SR)

        generated_clips.append({
            "status": "VALID",
            "segment_index": seg_idx,
            "duration_sec": 5.0,
            "sample_rate": TARGET_SR,
            "metrics": metrics,
            "wav_path": out_path
        })

    return generated_clips


def process_raw_recordings_batch(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    """
    Batch processor scanning data/raw_recordings/ for smartphone field audio files
    (.m4a, .mp3, .ogg, .wav) and outputting standardized 5s WAVs to data/processed_wavs/.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    supported_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    raw_files = [
        f for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    print(f"Scanning raw field recordings in: {raw_dir}")
    print(f"Found {len(raw_files)} audio file(s) for batch preprocessing.")

    results = []
    accepted_count = 0
    rejected_count = 0

    for fname in raw_files:
        fpath = os.path.join(raw_dir, fname)
        clips = process_and_standardize_audio(fpath, output_dir=processed_dir, base_filename=fname)
        
        for c in clips:
            if c["status"] == "VALID":
                accepted_count += 1
                print(f"  [+] Valid 5.0s Clip: {os.path.basename(c['wav_path'])}")
            else:
                rejected_count += 1
                print(f"  [-] Rejected ({c['status']}): {fname} (seg {c.get('segment_index', 1)})")
            results.append(c)

    print(f"Batch processing complete: {accepted_count} clips accepted, {rejected_count} rejected.")
    return results


if __name__ == "__main__":
    # Test validator with synthetic wave
    print("Testing Audio Validator...")
    t = np.linspace(0, 11.2, int(TARGET_SR * 11.2))
    # 11.2 second synthetic tone (should yield two 5.0s segments)
    test_signal = 0.4 * np.sin(2 * np.pi * 440 * t)
    clips = process_and_standardize_audio(test_signal, base_filename="test_multisegment")
    print(f"Produced {len(clips)} segment(s):")
    for c in clips:
        print(" ", c)
