"""
DeciSense AI - Module: Mobile Field Data Capture & Acoustic Logging UI
Streamlit-based mobile web recorder and field logger for Saipalaniappan T and team
during acoustic surveys along Poonamallee High Road, Kuthambakkam, and RIT Campus.
"""

import os
import sys
import wave
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.calibration import (
    calibrate_device, get_device_offset, compute_calibrated_spl, list_calibrated_devices
)
from src.data_pipeline.audio_validator import (
    process_and_standardize_audio, validate_audio_quality, PROCESSED_DIR
)
from src.data_pipeline.log_manager import (
    register_audio_log, load_audio_logs, VALID_ZONES, VALID_CLASSES,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CSV_LOG_PATH
)
from src.geospatial_engine import CPCB_LIMITS
from src.apilayer_geo import lookup_apilayer_geolocation, validate_within_country_bounds

# Page Setup - Mobile Optimized
st.set_page_config(
    page_title="DeciSense AI — Mobile Field Capture",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Touch / Smartphone Displays (Black & White Monochrome)
st.markdown("""
<style>
    html, body, .main, .stApp {
        background-color: #FFFFFF !important;
        color: #000000;
    }
    .mobile-header {
        background: #FFFFFF;
        border: 1.5px solid #000000;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 0 #000000;
    }
    .mobile-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #000000;
        margin-bottom: 2px;
    }
    .mobile-sub {
        font-size: 0.85rem;
        color: #525252;
        font-weight: 500;
    }
    .tag-badge {
        background: #000000;
        color: #FFFFFF;
        border: 1px solid #000000;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .metric-box {
        background: #FAFAFA;
        border: 1px solid #000000;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mobile-header">
    <div class="mobile-title">🎙️ DeciSense AI — Field Logger</div>
    <div class="mobile-sub">
        Mobile Acoustic Survey & SLM Calibration Interface
        <span class="tag-badge">Poonamallee–RIT Corridor</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top Navigation Tabs
# -----------------------------------------------------------------------------
tab_capture, tab_calibrate, tab_history = st.tabs([
    "📍 Field Audio Logger",
    "⚖️ Device SLM Calibration",
    "📋 Monitored Logs"
])


# =============================================================================
# TAB 1: FIELD AUDIO LOGGER
# =============================================================================
with tab_capture:
    st.markdown("#### 1. Collector & Hardware Profile")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        collector_name = st.text_input("Collector Name", value="Saipalaniappan T", key="collector_name")
    with col_u2:
        calibrated_devs = list_calibrated_devices()
        known_models = list(calibrated_devs.keys()) if calibrated_devs else ["OnePlus 11R", "Samsung Galaxy A54", "Redmi Note 12", "iPhone 14"]
        device_model = st.selectbox("Smartphone Model", known_models + ["Other / Custom Phone"], key="device_select")
        if device_model == "Other / Custom Phone":
            device_model = st.text_input("Enter Device Model", value="Custom Field Phone", key="custom_device_name") or "Custom Field Phone"

    offset_db = get_device_offset(device_model, default_offset=92.0)
    st.caption(f"🔧 Active Hardware Calibration Offset: **+{offset_db:.1f} dB**")

    st.markdown("---")
    st.markdown("#### 2. GPS Location & Corridor Anchors")

    # Quick Anchor Buttons for Chennai Corridor
    FIELD_ANCHORS = {
        "RIT Chennai Main Campus (Silence)": (13.0478, 80.0245, "silence"),
        "Saveetha Hospital Entrance (Silence)": (13.0450, 80.0150, "silence"),
        "Panimalar Hospital Buffer (Silence)": (13.0520, 80.0380, "silence"),
        "Kuthambakkam Bus Terminus (Commercial)": (13.0410, 80.0090, "commercial"),
        "Bangalore Highway NH48 Flyover (Commercial)": (13.0510, 80.0250, "commercial"),
        "Poonamallee Trunk Road Junction (Commercial)": (13.0490, 80.0350, "commercial"),
        "Nemam Residential Layout (Residential)": (13.0560, 80.0210, "residential")
    }

    selected_anchor = st.selectbox(
        "Quick Corridor Anchors",
        ["Custom GPS Coordinates"] + list(FIELD_ANCHORS.keys()),
        key="selected_anchor",
    )

    if selected_anchor != "Custom GPS Coordinates":
        default_lat, default_lon, default_zone = FIELD_ANCHORS[selected_anchor]
    else:
        default_lat, default_lon, default_zone = 13.0478, 80.0245, "silence"

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        latitude = st.number_input(
            f"Latitude ({LAT_MIN} to {LAT_MAX})",
            min_value=float(LAT_MIN), max_value=float(LAT_MAX),
            value=float(default_lat), format="%.6f", key="latitude"
        )
    with col_g2:
        longitude = st.number_input(
            f"Longitude ({LON_MIN} to {LON_MAX})",
            min_value=float(LON_MIN), max_value=float(LON_MAX),
            value=float(default_lon), format="%.6f", key="longitude"
        )

    apilayer_in = lookup_apilayer_geolocation(country_code="IN")
    in_ok, in_msg = validate_within_country_bounds(latitude, longitude, "IN")
    c_in = apilayer_in["country"]
    fg_in = apilayer_in["freegeoip"]
    st.caption(
        f"🌐 APILayer Context: **{c_in['alpha2']}/{c_in['alpha3']}** · {fg_in['time_zone']} · "
        f"{c_in['calling_code']} · {c_in['currency_code']} | {'✓' if in_ok else '!'} {in_msg}"
    )

    col_z1, col_z2 = st.columns(2)
    with col_z1:
        zone_type = st.selectbox(
            "CPCB Zone Type",
            ["silence", "residential", "commercial", "industrial"],
            index=["silence", "residential", "commercial", "industrial"].index(default_zone),
            key="zone_type",
        )
    with col_z2:
        ground_truth_class = st.selectbox(
            "Observed Noise Source",
            ["vehicular_honking", "continuous_traffic", "construction_piling", "pa_loudspeaker", "ambient_baseline"],
            key="ground_truth_class",
        )

    st.markdown("---")
    st.markdown("#### 3. Audio Ingestion")

    # Audio input support: Streamlit 1.40+ audio_input or file upload
    audio_file = None
    if hasattr(st, "audio_input"):
        try:
            recorded_audio = st.audio_input("Record 5s Field Sample", key="recorded_audio")
            if recorded_audio:
                audio_file = recorded_audio
        except Exception:
            pass

    if audio_file is None:
        audio_file = st.file_uploader(
            "Upload Field Recording (.wav, .m4a, .mp3, .ogg)",
            type=["wav", "m4a", "mp3", "ogg"],
            key="uploaded_field_audio",
        )

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("🚀 Process, Calibrate & Log Sample", type="primary", width="stretch", key="process_log_btn"):
            with st.spinner("Processing audio through QC validator and calibration pipeline..."):
                raw_bytes = audio_file.getvalue()
                
                # 1. Standardize and split into 5.0s PCM WAV
                base_fname = f"fld_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                clips = process_and_standardize_audio(
                    raw_bytes,
                    output_dir=PROCESSED_DIR,
                    base_filename=base_fname
                )

                valid_clips = [c for c in clips if c.get("status") == "VALID"]

                if not valid_clips:
                    reason = clips[0].get("status", "UNKNOWN") if clips else "NO_SEGMENTS"
                    reason_messages = {
                        "REJECT_TOO_SHORT": "Recording must be at least 3.0 seconds long.",
                        "REJECT_SILENT": "Audio signal is silent (RMS below threshold).",
                        "REJECT_CLIPPED": "Audio signal is severely clipped (>= 0.99 for > 5% of samples).",
                        "ERROR_CORRUPTED": f"Could not decode audio file ({clips[0].get('error', 'unsupported format')}).",
                    }
                    st.error(f"❌ Audio rejected by QC validator ({reason}): {reason_messages.get(reason, 'Validation failed.')}")
                else:
                    try:
                        registered_records = []
                        for c in valid_clips:
                            # 2. Compute Calibrated SPL
                            spl_metrics = compute_calibrated_spl(c["wav_path"], offset_db=offset_db)

                            # 3. Register in CSV
                            rel_wav_path = os.path.relpath(c["wav_path"], PROJECT_ROOT).replace("\\", "/")
                            rec = register_audio_log(
                                wav_path=rel_wav_path,
                                latitude=latitude,
                                longitude=longitude,
                                zone_type=zone_type,
                                ground_truth_class=ground_truth_class,
                                L_eq_db=spl_metrics["L_eq_db"],
                                L_max_db=spl_metrics["L_max_db"],
                                crest_factor_db=spl_metrics["crest_factor_db"],
                                device_model=device_model,
                                collector_name=collector_name,
                                csv_path=CSV_LOG_PATH
                            )
                            registered_records.append((rec, spl_metrics))

                        st.success(f"✅ Successfully validated & logged {len(registered_records)} standardized 5.0-second clip(s) to dataset!")

                        # Display metrics for the first logged clip
                        rec, spl = registered_records[0]
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Sound Level (L_eq)", f"{spl['L_eq_db']} dB(A)")
                        with m2:
                            st.metric("Peak Impulse (L_max)", f"{spl['L_max_db']} dB(A)")
                        with m3:
                            st.metric("Crest Factor", f"+{spl['crest_factor_db']} dB")

                        # Check statutory compliance
                        now_hour = datetime.now().hour
                        is_night = (now_hour >= 22 or now_hour < 6)
                        limit = CPCB_LIMITS["night" if is_night else "day"].get(zone_type, 45.0)
                        
                        if spl['L_eq_db'] > limit:
                            st.error(f"🚨 **CPCB Breach Detected:** Exceeds {zone_type} limit ({limit} dB) by +{round(spl['L_eq_db'] - limit, 1)} dB(A)!")
                        else:
                            st.success(f"✅ **Compliant:** Within statutory CPCB {zone_type} limit ({limit} dB).")
                    except Exception as exc:
                        st.error(f"❌ Failed to log sample: {exc}")


# =============================================================================
# TAB 2: DEVICE SLM CALIBRATION
# =============================================================================
with tab_calibrate:
    st.markdown("#### ⚖️ Smartphone Microphone SLM Calibration")
    st.markdown("""
    Calibrate your mobile phone microphone against a **Class-2 reference Sound Level Meter (SLM)** 
    or a standardized 1 kHz 74 dB(A) acoustic reference tone.
    """)

    cal_model = st.text_input("Device Model to Calibrate", value="OnePlus 11R", key="cal_model")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        ref_db = st.number_input("Class-2 Reference SLM Reading (dB(A))", value=74.0, step=0.5, key="ref_db")
    with col_c2:
        raw_db = st.number_input("Phone Raw Uncalibrated Reading (dBFS)", value=-18.0, step=0.5, key="raw_db")

    computed_offset = round(ref_db - raw_db, 2)
    st.info(f"Calculated Hardware Offset: **offset_db = {ref_db:.1f} - ({raw_db:.1f}) = +{computed_offset:.2f} dB**")

    if st.button("💾 Save Device Calibration", width="stretch", key="save_calib_btn"):
        if not cal_model.strip():
            st.error("Please enter a valid device model name.")
        else:
            entry = calibrate_device(cal_model, reference_db=ref_db, raw_measured_db=raw_db)
            st.success(f"✅ Calibration saved for **{entry['device_model']}** with offset **+{entry['offset_db']} dB**.")

    st.markdown("---")
    st.markdown("#### Registered Calibration Profiles")
    all_devs = list_calibrated_devices()
    if all_devs:
        st.json(all_devs)
    else:
        st.caption("No custom calibrated devices found. System using standard default (+92.0 dB).")


# =============================================================================
# TAB 3: MONITORED LOGS
# =============================================================================
with tab_history:
    st.markdown("#### 📋 Chennai Corridor Audio Logs")
    df_logs = load_audio_logs(CSV_LOG_PATH)
    st.caption(f"Total Registered Samples: **{len(df_logs)}** | Dataset: `data/chennai_audio_logs.csv`")
    
    if not df_logs.empty:
        display_cols = [
            c for c in [
                "clip_id", "timestamp", "time_window", "zone_type",
                "L_eq_db", "L_max_db", "crest_factor_db", "ground_truth_class", "device_model"
            ] if c in df_logs.columns
        ]
        st.dataframe(
            df_logs[display_cols].tail(20),
            width="stretch"
        )

        st.download_button(
            label="📥 Download CSV Dataset",
            data=df_logs.to_csv(index=False),
            file_name="chennai_audio_logs.csv",
            mime="text/csv",
            width="stretch"
        )
