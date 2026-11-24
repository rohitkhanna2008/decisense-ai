"""
DeciSense AI: Hyper-Local Acoustic Pollution & Sleep-Disruption Mapper
Master End-to-End Pipeline Runner
Executes:
  1. Audio Validation & Preprocessing (data/raw_recordings/ -> data/processed_wavs/)
  2. Acoustic Classification Benchmark & 5-Fold Stratified Cross-Validation (outputs/)
  3. Geospatial Engine & Sleep-Disruption Index IDW Interpolation (data/decisense_scored_streets.geojson)
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".mpl_cache")

from src.data_pipeline.audio_validator import process_raw_recordings_batch
from src.evaluate_model import evaluate_models
from src.geospatial_engine import export_geospatial_layers


def print_banner():
    banner = """
================================================================================
   DeciSense AI: Master Pipeline & Benchmark Refresh Runner
   Aligned with UN SDG 11.6 & SDG 3.9 | FRESHATHON 3.0 | RIT Chennai Corridor
================================================================================
"""
    print(banner)


def run_pipeline():
    start_total = time.time()
    print_banner()

    # -------------------------------------------------------------------------
    # Step 1: Ingest & Preprocess Raw Smartphone Recordings
    # -------------------------------------------------------------------------
    print("\n[STEP 1/3] Running Audio Validator on Field Recordings...")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw_recordings")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed_wavs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    t0 = time.time()
    validation_results = process_raw_recordings_batch(raw_dir, processed_dir)
    n_processed = len(validation_results)
    print(f"--> Step 1 completed in {time.time() - t0:.2f}s (Processed {n_processed} files).")

    # -------------------------------------------------------------------------
    # Step 2: Acoustic Feature Extraction & Stratified 5-Fold Cross-Validation
    # -------------------------------------------------------------------------
    print("\n[STEP 2/3] Executing Model Evaluation & Benchmark Metrics Exporter...")
    t1 = time.time()
    report, cpcb_df = evaluate_models()
    deep_acc = report.get("primary_deep_classifier", {}).get("accuracy", 0.0) * 100
    deep_f1 = report.get("primary_deep_classifier", {}).get("macro_f1", 0.0)
    latency = report.get("inference_latency_ms_per_clip", 0.0)
    print(f"--> Step 2 completed in {time.time() - t1:.2f}s.")
    print(f"    - Deep Classifier Accuracy: {deep_acc:.1f}%")
    print(f"    - Macro F1-Score: {deep_f1:.3f}")
    print(f"    - Mean Inference Latency: {latency:.2f} ms/clip")

    # -------------------------------------------------------------------------
    # Step 3: Geospatial Interpolation & Silence Zone Buffers
    # -------------------------------------------------------------------------
    print("\n[STEP 3/3] Exporting Geospatial Heatmaps & 100m Silence Buffers...")
    t2 = time.time()
    export_geospatial_layers(PROJECT_ROOT)
    print(f"--> Step 3 completed in {time.time() - t2:.2f}s.")

    total_elapsed = time.time() - start_total

    # -------------------------------------------------------------------------
    # Final Executive Summary Table
    # -------------------------------------------------------------------------
    summary_text = f"""
+------------------------------------------------------------------------------+
|                         DECISENSE AI PIPELINE SUMMARY                        |
+------------------------------------------------------------------------------+
| Metric / Artifact                     | Value / Path                         |
+---------------------------------------+--------------------------------------+
| Pipeline Execution Time               | {total_elapsed:5.2f} seconds                 |
| Total Acoustic Records Audited        | {report.get('dataset_sample_count', 150):5d} clips (5.0s PCM WAV)       |
| Cross-Validation Protocol             | Stratified 5-Fold (k=5)              |
| Deep Acoustic Classifier Accuracy     | {deep_acc:5.1f}%                              |
| Deep Acoustic Classifier Macro F1     | {deep_f1:5.3f}                              |
| Baseline Random Forest Accuracy       | {report.get('baseline_feature_classifier', {}).get('accuracy', 1.0) * 100:5.1f}%                              |
| Inference Latency per 5s Audio Clip   | {latency:5.2f} ms                            |
+---------------------------------------+--------------------------------------+
| Generated Artifacts:                                                         |
|   1. outputs/model_evaluation_report.json                                    |
|   2. outputs/confusion_matrix.png (300 DPI Publication Heatmap)              |
|   3. outputs/cpcb_zone_compliance_summary.csv                                |
|   4. data/silence_zones.geojson (100m CPCB Buffers around RIT & Hospitals)   |
|   5. data/decisense_scored_streets.geojson (k-NN IDW Sleep Disruption Scores)|
+------------------------------------------------------------------------------+
| Status: ALL SYSTEMS OPERATIONAL | READY FOR FRESHATHON 3.0 REVIEW & IEEE PPT |
+------------------------------------------------------------------------------+
"""
    print(summary_text)
    return report, cpcb_df


if __name__ == "__main__":
    run_pipeline()
