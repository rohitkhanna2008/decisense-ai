"""
DeciSense AI - Unit Tests for Field Data Acquisition & Calibration Pipeline
Verifies A-weighting filter response, SPL math, audio QC standardization,
Chennai coordinate bounds checking, and CSV schema integrity.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.calibration import (
    calibrate_device, get_device_offset, compute_calibrated_spl,
    apply_a_weighting, design_a_weighting_filter
)
from src.data_pipeline.audio_validator import (
    validate_audio_quality, process_and_standardize_audio,
    TARGET_SR, TARGET_SAMPLES
)
from src.data_pipeline.log_manager import (
    validate_coordinates, validate_log_entry, infer_time_window,
    register_audio_log, load_audio_logs, SCHEMA_COLUMNS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)


class TestAcousticCalibration(unittest.TestCase):
    """Test suite for smartphone microphone calibration & IEC 61672-1 SPL math."""

    def test_calibration_offset_math(self):
        """Verify device offset calculation: offset_db = reference_db - raw_measured_db."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            res = calibrate_device("TestPhone_X", reference_db=74.0, raw_measured_db=-18.0, filepath=tmp_path)
            self.assertEqual(res["offset_db"], 92.0)
            retrieved = get_device_offset("TestPhone_X", filepath=tmp_path)
            self.assertEqual(retrieved, 92.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_a_weighting_filter_attenuation(self):
        """
        IEC 61672-1 A-weighting specification:
        - At 1 kHz: Gain is ~0 dB (attenuation ≈ 0 dB).
        - At 100 Hz: Severe negative attenuation (approx -19 dB to -20 dB).
        """
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)

        # 1. Test 1000 Hz pure sine wave
        sig_1k = np.sin(2 * np.pi * 1000 * t)
        filt_1k = apply_a_weighting(sig_1k, sr)
        rms_in_1k = np.sqrt(np.mean(sig_1k**2))
        rms_out_1k = np.sqrt(np.mean(filt_1k[int(sr * 0.1):]**2))  # skip transient
        gain_1k_db = 20 * np.log10(rms_out_1k / rms_in_1k)
        # Gain at 1 kHz should be close to 0 dB (+/- 2 dB digital filter tolerance)
        self.assertAlmostEqual(gain_1k_db, 0.0, delta=2.0)

        # 2. Test 100 Hz pure sine wave (should be significantly attenuated by ~ -19 dB)
        sig_100hz = np.sin(2 * np.pi * 100 * t)
        filt_100hz = apply_a_weighting(sig_100hz, sr)
        rms_in_100 = np.sqrt(np.mean(sig_100hz**2))
        rms_out_100 = np.sqrt(np.mean(filt_100hz[int(sr * 0.1):]**2))
        gain_100_db = 20 * np.log10(rms_out_100 / rms_in_100)
        self.assertLess(gain_100_db, -15.0)

    def test_calibrated_spl_bounds(self):
        """Verify that L_max >= L_eq and crest factor is non-negative."""
        sr = 16000
        t = np.linspace(0, 5.0, sr * 5)
        # Signal with an impulse in the middle
        sig = 0.1 * np.sin(2 * np.pi * 500 * t)
        sig[int(sr * 2.0) : int(sr * 2.1)] = 0.95  # 100ms horn spike

        spl = compute_calibrated_spl(sig, sample_rate=sr, offset_db=90.0)
        self.assertGreaterEqual(spl["L_max_db"], spl["L_eq_db"])
        self.assertGreaterEqual(spl["crest_factor_db"], 0.0)
        self.assertEqual(spl["duration_sec"], 5.0)


class TestAudioValidator(unittest.TestCase):
    """Test suite for audio QC validation, silence/clipping detection, and splitting."""

    def test_silence_detection(self):
        """Silent audio (RMS < 1e-4) must be rejected."""
        silent_sig = np.zeros(16000 * 5, dtype=np.float32)
        is_valid, reason, _ = validate_audio_quality(silent_sig, sample_rate=16000)
        self.assertFalse(is_valid)
        self.assertEqual(reason, "REJECT_SILENT")

    def test_clipping_detection(self):
        """Severely clipped audio (>= 0.99 for > 5% frames) must be rejected."""
        clipped_sig = np.ones(16000 * 5, dtype=np.float32)
        is_valid, reason, metrics = validate_audio_quality(clipped_sig, sample_rate=16000)
        self.assertFalse(is_valid)
        self.assertEqual(reason, "REJECT_CLIPPED")
        self.assertGreater(metrics["clipped_ratio"], 0.05)

    def test_audio_segmentation_5_seconds(self):
        """An 11.5 second recording must be split into exact 5.0s (80,000 sample) chunks."""
        sr = TARGET_SR
        t = np.linspace(0, 11.5, int(sr * 11.5))
        test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        with tempfile.TemporaryDirectory() as tmp_dir:
            clips = process_and_standardize_audio(test_audio, output_dir=tmp_dir, base_filename="field_test")
            valid_clips = [c for c in clips if c["status"] == "VALID"]
            # 11.5 seconds / 5.0s yields 2 full 5.0s segments
            self.assertEqual(len(valid_clips), 2)
            for c in valid_clips:
                self.assertEqual(c["duration_sec"], 5.0)
                self.assertTrue(os.path.exists(c["wav_path"]))


class TestLogManagerAndSchema(unittest.TestCase):
    """Test suite for coordinate bounds checking, time inference, and CSV schema."""

    def test_chennai_corridor_coordinates_bounds(self):
        """Validates coordinates strictly within Greater Chennai / Poonamallee bounding box."""
        # Valid: RIT Chennai Campus
        ok, _ = validate_coordinates(13.0478, 80.0245)
        self.assertTrue(ok)

        # Valid: Saveetha Medical Center
        ok, _ = validate_coordinates(13.0450, 80.0150)
        self.assertTrue(ok)

        # Invalid: Mumbai
        bad, msg = validate_coordinates(19.0760, 72.8777)
        self.assertFalse(bad)
        self.assertIn("bounds", msg)

        # Invalid: New Delhi
        bad, msg = validate_coordinates(28.6139, 77.2090)
        self.assertFalse(bad)

    def test_time_window_inference(self):
        """Verifies Day (06:00-22:00) vs Night (22:00-06:00) categorization."""
        self.assertEqual(infer_time_window("2026-11-20 14:30:00"), "Day: 06:00-22:00")
        self.assertEqual(infer_time_window("2026-11-20 06:00:00"), "Day: 06:00-22:00")
        self.assertEqual(infer_time_window("2026-11-20 22:00:00"), "Night: 22:00-06:00")
        self.assertEqual(infer_time_window("2026-11-20 01:15:00"), "Night: 22:00-06:00")

    def test_log_entry_validation_rules(self):
        """Verifies invalid zone types and classes are rejected."""
        valid_entry = {
            "latitude": 13.0478,
            "longitude": 80.0245,
            "zone_type": "silence",
            "ground_truth_class": "vehicular_honking"
        }
        ok, _ = validate_log_entry(valid_entry)
        self.assertTrue(ok)

        # Invalid zone
        invalid_zone = dict(valid_entry, zone_type="jungle")
        bad, msg = validate_log_entry(invalid_zone)
        self.assertFalse(bad)
        self.assertIn("zone_type", msg)

        # Invalid class
        invalid_class = dict(valid_entry, ground_truth_class="dog_bark")
        bad, msg = validate_log_entry(invalid_class)
        self.assertFalse(bad)
        self.assertIn("ground_truth_class", msg)

    def test_register_and_csv_schema_integrity(self):
        """Registers a sample into a temporary CSV and asserts all 13 schema columns exist."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_csv = tmp.name

        try:
            rec = register_audio_log(
                wav_path="data/sample_wavs/vehicular_honking.wav",
                latitude=13.0478,
                longitude=80.0245,
                zone_type="silence",
                ground_truth_class="vehicular_honking",
                L_eq_db=74.5,
                L_max_db=82.1,
                device_model="OnePlus 11R",
                collector_name="Saipalaniappan T",
                timestamp="2026-11-24 23:15:00",
                csv_path=tmp_csv
            )

            df = pd.read_csv(tmp_csv)
            self.assertEqual(list(df.columns), SCHEMA_COLUMNS)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["time_window"], "Night: 22:00-06:00")
            self.assertEqual(df.iloc[0]["crest_factor_db"], 7.6)
            self.assertEqual(df.iloc[0]["collector_name"], "Saipalaniappan T")
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)


if __name__ == "__main__":
    unittest.main()
