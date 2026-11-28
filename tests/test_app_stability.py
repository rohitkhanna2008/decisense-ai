"""
DeciSense AI - Stability & Streamlit Regression Tests
Covers audio decoding edge cases, geospatial IDW/CPCB bounds,
and headless Streamlit AppTest runs for app.py and field_capture_ui.py.
"""

import io
import json
import os
import sys
import threading
import unittest
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pandas as pd
from scipy.io import wavfile
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.apilayer_geo import (
    build_freegeoip_record,
    format_freegeoip_csv,
    get_country_metadata,
    list_country_codes,
    lookup_apilayer_geolocation,
    validate_within_country_bounds,
)
from src.audio_classifier import (
    analyze_audio_file,
    get_classifier,
    load_audio,
    compute_decibel_metrics,
    compute_mel_spectrogram,
)
from src.data_pipeline.audio_validator import load_and_resample
from src.data_pipeline.log_manager import infer_time_window
from src.geospatial_engine import (
    CPCB_LIMITS,
    compute_sleep_disruption_index,
    run_idw_interpolation,
)


class TestAudioAndGeospatialStability(unittest.TestCase):
    """Edge-case resilience tests for DSP and geospatial modules."""

    def test_load_audio_float32_and_stereo_wav_bytes(self):
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        stereo = np.stack([0.4 * np.sin(2 * np.pi * 440 * t), 0.2 * np.cos(2 * np.pi * 880 * t)], axis=1)
        buf = io.BytesIO()
        wavfile.write(buf, sr, stereo)

        y1, out_sr1 = load_audio(io.BytesIO(buf.getvalue()), target_sr=16000)
        y2, out_sr2 = load_and_resample(buf.getvalue(), target_sr=16000)
        self.assertEqual(out_sr1, 16000)
        self.assertEqual(out_sr2, 16000)
        self.assertEqual(y1.ndim, 1)
        self.assertEqual(y2.ndim, 1)
        self.assertGreater(len(y1), 7000)
        self.assertTrue(np.all(np.isfinite(y1)))

    def test_mel_spectrogram_empty_and_short_waveforms(self):
        empty_mel = compute_mel_spectrogram(np.array([], dtype=np.float32), sample_rate=16000)
        self.assertEqual(empty_mel.shape[0], 64)
        self.assertTrue(np.all(np.isfinite(empty_mel)))

        short_mel = compute_mel_spectrogram(np.ones(200, dtype=np.float32) * 0.1, sample_rate=16000)
        self.assertEqual(short_mel.shape[0], 64)
        self.assertTrue(np.all(np.isfinite(short_mel)))

    def test_decibel_metrics_handles_nan_and_empty(self):
        leq, lmax, crest = compute_decibel_metrics(np.array([], dtype=np.float32))
        self.assertEqual((leq, lmax, crest), (0.0, 0.0, 0.0))

        sig = np.array([0.1, np.nan, -0.2, np.inf] * 1000, dtype=np.float32)
        leq2, lmax2, crest2 = compute_decibel_metrics(sig, sample_rate=16000)
        self.assertTrue(np.isfinite(leq2))
        self.assertTrue(np.isfinite(lmax2))
        self.assertGreaterEqual(crest2, 0.0)

    def test_idw_interpolation_handles_empty_small_and_nan_dataframes(self):
        empty_fc = run_idw_interpolation(pd.DataFrame(), is_night=True)
        self.assertEqual(empty_fc["features"], [])

        small_df = pd.DataFrame([
            {
                "latitude": 13.0478,
                "longitude": 80.0245,
                "L_eq_db": 72.0,
                "L_max_db": 84.0,
                "timestamp": "invalid-date",
                "ground_truth_class": None,
            },
            {
                "latitude": np.nan,
                "longitude": 80.0245,
                "L_eq_db": 70.0,
                "L_max_db": 80.0,
                "timestamp": "2026-11-24 23:00:00",
                "ground_truth_class": "vehicular_honking",
            },
        ])
        fc = run_idw_interpolation(small_df, is_night=True, k=5)
        self.assertGreater(len(fc["features"]), 0)
        for feat in fc["features"]:
            self.assertTrue(np.isfinite(feat["properties"]["L_eq_db"]))
            self.assertTrue(np.isfinite(feat["properties"]["sleep_disruption_score"]))

    def test_industrial_cpcb_limits_and_invalid_time(self):
        self.assertEqual(CPCB_LIMITS["day"]["industrial"], 75.0)
        self.assertEqual(CPCB_LIMITS["night"]["industrial"], 70.0)
        res = compute_sleep_disruption_index(72.0, 78.0, zone="industrial", timestamp_hour=14)
        self.assertFalse(res["is_violation"])
        self.assertEqual(res["statutory_limit"], 75.0)

        nan_res = compute_sleep_disruption_index(np.nan, np.nan, zone="silence", timestamp_hour=23)
        self.assertEqual(nan_res["score"], 0)

        tw = infer_time_window(None)
        self.assertIn(tw, {"Day: 06:00-22:00", "Night: 22:00-06:00"})


class TestAcousticModelPredictions(unittest.TestCase):
    """Verifies the 5-class acoustic classifier and evaluation pipeline."""

    def test_sample_wav_predictions_all_five_classes(self):
        expected = [
            "ambient_baseline",
            "construction_piling",
            "continuous_traffic",
            "pa_loudspeaker",
            "vehicular_honking",
        ]
        for cname in expected:
            wav_path = os.path.join(PROJECT_ROOT, "data", "sample_wavs", f"{cname}.wav")
            res = analyze_audio_file(wav_path, offset_db=92.0)
            self.assertEqual(res["predicted_class"], cname)
            self.assertGreater(res["confidence"], 0.90)

    def test_classifier_handles_empty_and_midlevel_inputs(self):
        clf = get_classifier()
        empty_pred = clf.predict(np.array([], dtype=np.float32), sample_rate=16000, leq=0.0, crest=0.0)
        self.assertEqual(empty_pred["predicted_class"], "ambient_baseline")

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        pa_mid = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        mid_pred = clf.predict(pa_mid, sample_rate=sr, leq=58.0, crest=1.5)
        self.assertEqual(mid_pred["predicted_class"], "pa_loudspeaker")
        self.assertGreater(mid_pred["confidence"], 0.5)


class TestApilayerAndMonochromeTheme(unittest.TestCase):
    """Tests for apilayer/geolocationapi, apilayer/freegeoip, and B&W theme."""

    def test_apilayer_countries_and_bounds(self):
        codes = list_country_codes()
        self.assertGreaterEqual(len(codes), 249)
        in_meta = get_country_metadata("IN")
        self.assertEqual(in_meta["alpha2"], "IN")
        self.assertEqual(in_meta["alpha3"], "IND")
        self.assertEqual(in_meta["country_code"], "91")
        self.assertEqual(in_meta["currency_code"], "INR")
        self.assertEqual(in_meta["subregion"], "Southern Asia")

        ok, _ = validate_within_country_bounds(13.0475, 80.0250, "IN")
        self.assertTrue(ok)
        bad, _ = validate_within_country_bounds(51.5074, -0.1278, "IN")
        self.assertFalse(bad)

    def test_freegeoip_schema_and_local_server_lookup(self):
        rec = build_freegeoip_record(ip="8.8.8.8", country_code="US", city="Mountain View", region_code="CA")
        self.assertEqual(
            list(rec.keys()),
            [
                "ip", "country_code", "country_name", "region_code", "region_name",
                "city", "zip_code", "time_zone", "latitude", "longitude", "metro_code",
            ],
        )
        csv_line = format_freegeoip_csv(rec)
        self.assertEqual(len(csv_line.split(",")), 11)

        class _MockFreegeoipHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({
                    "ip": "1.2.3.4",
                    "country_code": "IN",
                    "country_name": "India",
                    "region_code": "TN",
                    "region_name": "Tamil Nadu",
                    "city": "Chennai",
                    "zip_code": "600056",
                    "time_zone": "Asia/Kolkata",
                    "latitude": 13.0475,
                    "longitude": 80.025,
                    "metro_code": 0,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _MockFreegeoipHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        prev = os.environ.get("DECISENSE_FREEGEOIP_URL")
        try:
            os.environ["DECISENSE_FREEGEOIP_URL"] = f"http://127.0.0.1:{port}"
            ctx = lookup_apilayer_geolocation("1.2.3.4", country_code="IN", allow_network=True)
            self.assertEqual(ctx["freegeoip"]["city"], "Chennai")
            self.assertEqual(ctx["country"]["alpha3"], "IND")
            self.assertTrue(ctx["within_country_bounds"])
        finally:
            server.shutdown()
            server.server_close()
            if prev is None:
                os.environ.pop("DECISENSE_FREEGEOIP_URL", None)
            else:
                os.environ["DECISENSE_FREEGEOIP_URL"] = prev

    def test_monochrome_theme_tokens(self):
        cfg_path = os.path.join(PROJECT_ROOT, ".streamlit", "config.toml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = f.read()
        self.assertIn('primaryColor = "#000000"', cfg)
        self.assertIn('backgroundColor = "#FFFFFF"', cfg)
        self.assertIn('textColor = "#000000"', cfg)

        for leq, lmax, expected in [
            (85.0, 96.0, "#ff3366"),
            (58.0, 62.0, "#ffd600"),
            (36.0, 39.0, "#00e676"),
        ]:
            res = compute_sleep_disruption_index(leq, lmax, zone="silence", timestamp_hour=23)
            self.assertEqual(res["color"], expected)

        with open(os.path.join(PROJECT_ROOT, "app.py"), "r", encoding="utf-8") as f:
            app_src = f.read()
        self.assertNotIn("filter: grayscale", app_src)
        self.assertIn('key="chennai_corridor_map"', app_src)


class TestStreamlitApps(unittest.TestCase):
    """Headless Streamlit AppTest verification for both UIs."""

    def test_main_dashboard_loads_and_filters_cleanly(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30).run()
            self.assertEqual(len(at.exception), 0, f"Initial load errors: {[e.value for e in at.exception]}")

            # Filter through time windows, zones, and apilayer country selector
            at.selectbox(key="time_window").select("Night: 22:00–06:00").run()
            self.assertEqual(len(at.exception), 0)

            at.selectbox(key="zone_filter").select("Silence").run()
            self.assertEqual(len(at.exception), 0)

            at.selectbox(key="apilayer_country").select("US").run()
            self.assertEqual(len(at.exception), 0)

            at.selectbox(key="apilayer_country").select("IN").run()
            self.assertEqual(len(at.exception), 0)

            at.slider(key="calib_offset").set_value(95.0).run()
            self.assertEqual(len(at.exception), 0)

            at.selectbox(key="sample_choice").select("ambient_baseline.wav").run()
            self.assertEqual(len(at.exception), 0)

            at.selectbox(key="sample_choice").select("— select preset —").run()
            self.assertEqual(len(at.exception), 0)

            cartodb_warnings = [w for w in caught if "CartoDB" in str(w.message)]
            self.assertEqual(cartodb_warnings, [])

    def test_field_capture_ui_loads_and_interacts_cleanly(self):
        at = AppTest.from_file(
            os.path.join(PROJECT_ROOT, "src", "data_pipeline", "field_capture_ui.py"),
            default_timeout=30,
        ).run()
        self.assertEqual(len(at.exception), 0, f"Field UI errors: {[e.value for e in at.exception]}")

        at.selectbox(key="selected_anchor").select("Saveetha Hospital Entrance (Silence)").run()
        self.assertEqual(len(at.exception), 0)

        at.selectbox(key="zone_type").select("industrial").run()
        self.assertEqual(len(at.exception), 0)


if __name__ == "__main__":
    unittest.main()
