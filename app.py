"""
DeciSense AI — Chennai Acoustic Pollution & Sleep-Disruption Mapper
Responsive Cross-Device Dashboard (Mobile · Tablet · Desktop Projector)
RIT Chennai | FRESHATHON 3.0 | UN SDG 11 (Target 11.6) & SDG 3 (Target 3.9)
"""

import os
import sys
import json
import wave
import io
import numpy as np
import pandas as pd

# Matplotlib config must be set before importing pyplot on restricted Windows envs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is in sys.path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.audio_classifier import (
    load_audio, compute_decibel_metrics, compute_mel_spectrogram,
    get_classifier, CLASSES
)
from src.geospatial_engine import (
    compute_sleep_disruption_index, CHENNAI_SILENCE_POIS,
    CHENNAI_CORRIDOR_STREETS, CPCB_LIMITS, run_idw_interpolation
)

# Optional Plotly / Folium imports with graceful fallbacks
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="DeciSense AI — Chennai Acoustic Pollution & Sleep-Disruption Mapper",
    page_icon="🔇",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# Responsive Pure-White CSS (Mobile → Desktop)
# =============================================================================
st.markdown("""
<style>
/* ---- Reset & Base Canvas ------------------------------------------------- */
html, body, .main, .stApp {
    background-color: #FFFFFF !important;
    color: #0F172A;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
                 Arial, sans-serif;
}

/* ---- Hide default Streamlit deploy / toolbar clutter --------------------- */
header[data-testid="stHeader"],
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}

/* ---- Sidebar polish ------------------------------------------------------ */
section[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #0F172A;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ---- Header Card --------------------------------------------------------- */
.ds-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.ds-header h1 {
    color: #0F172A;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.4px;
    margin: 0 0 4px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.ds-header p {
    color: #475569;
    font-size: 0.88rem;
    font-weight: 500;
    margin: 0;
    line-height: 1.5;
}
.ds-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 0.68rem;
    font-weight: 700;
    vertical-align: middle;
}
.ds-badge-green {
    background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0;
}
.ds-badge-blue {
    background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE;
}

/* ---- KPI Grid ------------------------------------------------------------ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 100px;
}
.kpi-label {
    color: #64748B;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    font-feature-settings: "tnum";
    line-height: 1.15;
}
.kpi-unit {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 600;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 4px;
    font-weight: 500;
}

/* ---- Semantic Badge Colors ----------------------------------------------- */
.c-navy   { color: #0F172A; }
.c-teal   { color: #0D9488; }
.c-red    { color: #DC2626; }
.c-amber  { color: #D97706; }
.c-green  { color: #16A34A; }
.c-blue   { color: #2563EB; }
.c-muted  { color: #64748B; }

/* ---- Section Header ------------------------------------------------------ */
.sec-hdr {
    color: #0F172A;
    font-size: 1.08rem;
    font-weight: 700;
    margin: 10px 0 14px;
}

/* ---- Diagnostic Result Card ---------------------------------------------- */
.diag-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}

/* ---- Pass / Fail Badge --------------------------------------------------- */
.badge-pass {
    display: inline-block;
    background: #ECFDF5; color: #16A34A; border: 1px solid #BBF7D0;
    padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;
}
.badge-fail {
    display: inline-block;
    background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA;
    padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;
}
.badge-warn {
    display: inline-block;
    background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A;
    padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;
}

/* ---- Touch-friendly targets >= 44 px ------------------------------------- */
.stButton > button,
.stDownloadButton > button {
    min-height: 44px !important;
    min-width: 44px !important;
    border-radius: 8px !important;
}

/* ---- Floating Map Legend ------------------------------------------------- */
.map-legend {
    background: rgba(255,255,255,0.95);
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    font-size: 0.78rem;
    line-height: 1.65;
}

/* ==== MOBILE BREAKPOINT (<768 px) ========================================= */
@media (max-width: 768px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    .kpi-value {
        font-size: 1.35rem;
    }
    .ds-header h1 {
        font-size: 1.25rem;
    }
    .ds-header p {
        font-size: 0.8rem;
    }
    .sec-hdr {
        font-size: 0.95rem;
    }
}

/* ==== TABLET BREAKPOINT (768–1024 px) ===================================== */
@media (min-width: 769px) and (max-width: 1024px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    .kpi-value {
        font-size: 1.55rem;
    }
}

/* ==== PROJECTOR / LARGE (>1600 px) ======================================== */
@media (min-width: 1600px) {
    .kpi-value {
        font-size: 2.1rem;
    }
    .ds-header h1 {
        font-size: 2.0rem;
    }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached Data Loaders (Streamlit Community Cloud ready)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_corridor_data():
    csv_path = os.path.join(BASE_DIR, "data", "chennai_audio_logs.csv")
    if not os.path.exists(csv_path):
        from src.generate_synthetic_data import generate_dataset
        generate_dataset(BASE_DIR)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = (df["hour"] >= 22) | (df["hour"] < 6)
    return df


@st.cache_data(show_spinner=False)
def load_scored_streets_geojson():
    p = os.path.join(BASE_DIR, "data", "decisense_scored_streets.geojson")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_silence_zones_geojson():
    p = os.path.join(BASE_DIR, "data", "silence_zones.geojson")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_evaluation_report():
    p = os.path.join(BASE_DIR, "outputs", "model_evaluation_report.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_cpcb_audit():
    p = os.path.join(BASE_DIR, "outputs", "cpcb_zone_compliance_summary.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


# =============================================================================
# Data Initialization
# =============================================================================
df_all = load_corridor_data()


# =============================================================================
# Sidebar — Collapsible Global Filters
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Observation Filters")

    time_window = st.selectbox(
        "Time Window",
        ["All Hours (24 h)",
         "Day: 06:00–22:00",
         "Night: 22:00–06:00"],
        index=0,
    )

    zone_options = ["All Zones", "Silence", "Residential", "Commercial"]
    zone_filter = st.selectbox("CPCB Zone Type", zone_options, index=0)

    st.markdown("---")
    st.markdown("### 📱 Mic Calibration")
    calib_offset = st.slider(
        "Offset dB (SPL ref.)",
        min_value=70.0,
        max_value=110.0,
        value=92.0,
        step=0.5,
        help="Offset calibrated against a Class-2 sound level meter.",
    )

    st.markdown("---")
    st.markdown("### ⚖️ CPCB Statutory Limits")
    st.markdown("""
| Zone | Day | Night |
|------|----:|------:|
| Silence | 50 | 40 |
| Residential | 55 | 45 |
| Commercial | 65 | 55 |
""")
    st.caption("WHO Bedroom Cap: < 30–40 dB(A)")
    st.caption("Chennai Poonamallee–Kuthambakkam–Porur Corridor · RIT Innovation Project 1")


# =============================================================================
# Apply Filters
# =============================================================================
filtered_df = df_all.copy()

if "Night" in time_window:
    filtered_df = filtered_df[filtered_df["is_night"]]
elif "Day" in time_window:
    filtered_df = filtered_df[~filtered_df["is_night"]]

zone_map = {"Silence": "silence", "Residential": "residential", "Commercial": "commercial"}
if zone_filter in zone_map:
    filtered_df = filtered_df[filtered_df["zone_type"] == zone_map[zone_filter]]

# Recompute Sleep-Disruption Scores
scores_list = []
violations_count = 0
for _, row in filtered_df.iterrows():
    sd = compute_sleep_disruption_index(
        row["L_eq_db"], row["L_max_db"],
        zone=row["zone_type"], timestamp_hour=row["hour"],
    )
    scores_list.append(sd["score"])
    if sd["is_violation"]:
        violations_count += 1

filtered_df = filtered_df.copy()
filtered_df["sleep_score"] = scores_list

n_records = len(filtered_df)
avg_leq = round(filtered_df["L_eq_db"].mean(), 1) if n_records > 0 else 0.0
peak_lmax = round(filtered_df["L_max_db"].max(), 1) if n_records > 0 else 0.0
mean_sleep_score = int(np.mean(scores_list)) if scores_list else 0
is_night_mode = "Night" in time_window


# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="ds-header">
    <h1>
        🔇 DeciSense AI
        <span class="ds-badge ds-badge-green">UN SDG 11.6 &amp; SDG 3</span>
        <span class="ds-badge ds-badge-blue">FRESHATHON 3.0</span>
    </h1>
    <p>
        <b>Hyper-Local Acoustic Pollution &amp; Sleep-Disruption Mapper</b><br/>
        Chennai, India · Poonamallee–Kuthambakkam–Porur Highway Corridor · Rajalakshmi Institute of Technology
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# KPI Cards (responsive 2×2 mobile / 4-col desktop)
# =============================================================================
score_color_cls = (
    "c-red" if mean_sleep_score >= 65
    else ("c-amber" if mean_sleep_score >= 40 else "c-green")
)
score_label = (
    "Severe Awakenings" if mean_sleep_score >= 65
    else ("Moderate Disturbance" if mean_sleep_score >= 40 else "Undisturbed Baseline")
)

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Avg L<sub>eq</sub></div>
        <div class="kpi-value c-navy">{avg_leq} <span class="kpi-unit">dB(A)</span></div>
        <div class="kpi-sub">Equivalent continuous sound energy</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Peak L<sub>max</sub></div>
        <div class="kpi-value c-red">{peak_lmax} <span class="kpi-unit">dB(A)</span></div>
        <div class="kpi-sub">Maximum impulse crest recorded</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Sleep-Disruption Score</div>
        <div class="kpi-value {score_color_cls}">{mean_sleep_score} <span class="kpi-unit">/ 100</span></div>
        <div class="kpi-sub">{score_label}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">CPCB Breaches</div>
        <div class="kpi-value c-red">{violations_count} <span class="kpi-unit">pts</span></div>
        <div class="kpi-sub">Points exceeding statutory limits</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================
tab_map, tab_diag, tab_analytics, tab_bench = st.tabs([
    "🗺️ Live Chennai Map",
    "🎙️ Instant Audio Diagnostic",
    "📊 Source & Temporal Analytics",
    "📈 Model Benchmarks & CPCB Audit",
])


# =============================================================================
# TAB 1 — Live Chennai Map
# =============================================================================
with tab_map:
    st.markdown('<div class="sec-hdr">Corridor Acoustic Heatmap & 100 m Silence Buffers</div>',
                unsafe_allow_html=True)

    if HAS_FOLIUM:
        center_lat, center_lon = 13.0475, 80.0250
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="CartoDB positron",
            control_scale=True,
        )

        # 1 — 100 m Silence Zone Buffers
        for poi in CHENNAI_SILENCE_POIS:
            folium.Circle(
                location=[poi["lat"], poi["lon"]],
                radius=poi["buffer_radius_m"],
                color="#DC2626",
                dash_array="6, 6",
                weight=2,
                fill=True,
                fill_color="#FEE2E2",
                fill_opacity=0.40,
                popup=folium.Popup(
                    f'<div style="font-family:sans-serif;font-size:12px;line-height:1.4">'
                    f'<b style="color:#DC2626">🏥 100 m CPCB Silence Zone</b><br/>'
                    f'<b>{poi["name"]}</b><br/>'
                    f'Category: {poi["category"]}<br/>'
                    f'Night: <b>40 dB(A)</b> · Day: <b>50 dB(A)</b></div>',
                    max_width=250,
                ),
            ).add_to(m)
            folium.CircleMarker(
                location=[poi["lat"], poi["lon"]],
                radius=4,
                color="#991B1B",
                fill=True,
                fill_color="#991B1B",
            ).add_to(m)

        # 2 — k-NN IDW Street Polylines
        scored_streets_geojson = run_idw_interpolation(filtered_df, is_night=is_night_mode)
        for feat in scored_streets_geojson["features"]:
            props = feat["properties"]
            coords = [[c[1], c[0]] for c in feat["geometry"]["coordinates"]]
            color = props["color_code"]
            weight = 6 if props["is_violation"] else 4
            status_html = (
                '<span style="color:#DC2626;font-weight:700">🚨 CPCB Breach</span>'
                if props["is_violation"]
                else '<span style="color:#16A34A;font-weight:700">✅ Compliant</span>'
            )
            popup_html = (
                f'<div style="font-family:sans-serif;font-size:12px;line-height:1.45;min-width:190px">'
                f'<b style="font-size:13px;color:#0F172A">{props["street_name"]}</b><br/>'
                f'Zone: <b>{props["zone_type"].upper()}</b><br/>'
                f'L<sub>eq</sub>: <b>{props["L_eq_db"]} dB(A)</b> · '
                f'L<sub>max</sub>: <b>{props["L_max_db"]} dB(A)</b><br/>'
                f'Noise: <span style="color:#2563EB">{props["dominant_noise_source"]}</span><br/>'
                f'<hr style="margin:5px 0;border:0;border-top:1px solid #E2E8F0"/>'
                f'Sleep Score: <b style="color:{color};font-size:14px">{props["sleep_disruption_score"]}/100</b><br/>'
                f'Status: {status_html}</div>'
            )
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(m)

        st_folium(m, width="100%", height=560, returned_objects=[])

        # Compact floating legend + download
        col_leg, col_dl = st.columns([2, 1])
        with col_leg:
            st.markdown("""
<div class="map-legend">
<b>Sleep-Disruption Risk</b><br/>
<span style="color:#DC2626;font-weight:700">■</span> Critical (> 65) · 
<span style="color:#D97706;font-weight:700">■</span> Moderate (40–65) · 
<span style="color:#16A34A;font-weight:700">■</span> Compliant (< 40)<br/>
<span style="color:#DC2626">⬚ Dashed Red:</span> 100 m CPCB Silence Zone
</div>
""", unsafe_allow_html=True)

        with col_dl:
            st.download_button(
                label="📥 Download Scored GeoJSON",
                data=json.dumps(scored_streets_geojson, indent=2),
                file_name="decisense_scored_streets.geojson",
                mime="application/json",
            )

    else:
        st.warning("Folium / streamlit-folium not installed. Showing tabular GIS data.")
        st.dataframe(
            filtered_df[["clip_id", "latitude", "longitude", "zone_type",
                          "L_eq_db", "L_max_db", "ground_truth_class"]],
        )


# =============================================================================
# TAB 2 — Instant Audio Diagnostic
# =============================================================================
with tab_diag:
    st.markdown('<div class="sec-hdr">Live Acoustic Classifier & Spectrogram Diagnostic</div>',
                unsafe_allow_html=True)
    st.caption("Upload a 5-second WAV or select a Chennai corridor benchmark preset.")

    sample_wavs_dir = os.path.join(BASE_DIR, "data", "sample_wavs")
    available_samples = (
        sorted([f for f in os.listdir(sample_wavs_dir) if f.endswith(".wav")])
        if os.path.isdir(sample_wavs_dir) else []
    )

    # Preset selector
    presets = ["— select preset —"] + available_samples
    sample_choice = st.selectbox(
        "Benchmark Preset",
        presets,
        index=1 if available_samples else 0,
        help="Pre-generated 5 s clips covering the five target classes.",
    )

    uploaded_audio = st.file_uploader("Or upload a custom WAV file", type=["wav"])

    audio_bytes = None
    current_audio_label = ""
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        current_audio_label = uploaded_audio.name
    elif sample_choice != "— select preset —" and available_samples:
        sample_path = os.path.join(sample_wavs_dir, sample_choice)
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()
        current_audio_label = sample_choice

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

        # Parse WAV
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        leq_calc, lmax_calc, crest_calc = compute_decibel_metrics(
            waveform, sample_rate=sr, offset_db=calib_offset
        )

        classifier = get_classifier()
        pred = classifier.predict(waveform, sample_rate=sr, leq=leq_calc, crest=crest_calc)

        sleep_res = compute_sleep_disruption_index(
            leq_calc, lmax_calc, zone="silence", timestamp_hour=23
        )

        # Metrics row
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("L_eq  (dB A)", f"{leq_calc}")
        with mc2:
            st.metric("L_max (dB A)", f"{lmax_calc}")
        with mc3:
            st.metric("Crest Factor", f"+{crest_calc} dB")

        # Classification result card
        pred_cls = pred["predicted_class"].replace("_", " ").title()
        conf_pct = pred["confidence"] * 100

        st.markdown(f"""
<div class="diag-card">
    <div class="kpi-label">Predicted Noise Event</div>
    <div style="font-size:1.35rem;font-weight:800;color:#2563EB;margin-top:4px">{pred_cls}</div>
    <div style="font-size:0.82rem;color:#64748B;margin-top:2px">
        Confidence: <b>{conf_pct:.1f} %</b>
    </div>
</div>
""", unsafe_allow_html=True)

        # Confidence bar chart for all 5 classes
        if HAS_PLOTLY:
            probs = pred["probabilities"]
            fig_conf = go.Figure(go.Bar(
                x=[probs.get(c, 0) * 100 for c in CLASSES],
                y=[c.replace("_", " ").title() for c in CLASSES],
                orientation="h",
                marker_color=["#0D9488" if c == pred["predicted_class"] else "#CBD5E1" for c in CLASSES],
            ))
            fig_conf.update_layout(
                xaxis_title="Confidence %",
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # Sleep-Disruption Index
        score_val = sleep_res["score"]
        st.progress(min(score_val / 100.0, 1.0))

        badge_cls = (
            "badge-fail" if score_val >= 65
            else ("badge-warn" if score_val >= 40 else "badge-pass")
        )
        badge_text = (
            "CPCB Breach" if sleep_res["is_violation"]
            else ("Moderate Risk" if score_val >= 40 else "Compliant")
        )
        st.markdown(
            f'**Sleep-Disruption Index:** `{score_val} / 100` '
            f'<span class="{badge_cls}">{badge_text}</span>',
            unsafe_allow_html=True,
        )

        if sleep_res["is_violation"]:
            st.error(
                f"🚨 Exceeds CPCB Silence Limit (40 dB) by **+{sleep_res['delta_db']} dB(A)**!"
            )

        # Log-Mel Spectrogram
        st.markdown("#### Log-Mel Spectrogram")
        log_mel = compute_mel_spectrogram(waveform, sample_rate=sr)
        fig_mel, ax_mel = plt.subplots(figsize=(8, 2.6), facecolor="#FFFFFF")
        img_mel = ax_mel.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        ax_mel.set_title(
            f"Log-Mel Spectrogram  ({current_audio_label})",
            fontsize=10, color="#0F172A", fontweight="bold",
        )
        ax_mel.set_xlabel("Time Frames", fontsize=8, color="#475569")
        ax_mel.set_ylabel("Mel Bins", fontsize=8, color="#475569")
        ax_mel.tick_params(colors="#475569", labelsize=7)
        fig_mel.colorbar(img_mel, ax=ax_mel, format="%+2.0f dB")
        fig_mel.tight_layout()
        st.pyplot(fig_mel)
        plt.close(fig_mel)

    else:
        st.info("Select a benchmark preset or upload a WAV file to begin diagnostic analysis.")


# =============================================================================
# TAB 3 — Source & Temporal Analytics
# =============================================================================
with tab_analytics:
    st.markdown('<div class="sec-hdr">Noise Source Distribution & CPCB Threshold Compliance</div>',
                unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    # 3a — Source Breakdown Donut
    with col_c1:
        st.markdown("#### Dominant Noise Source Breakdown")
        source_counts = filtered_df["ground_truth_class"].value_counts().reset_index()
        source_counts.columns = ["Noise Source", "Count"]

        if HAS_PLOTLY:
            fig_pie = px.pie(
                source_counts,
                values="Count",
                names="Noise Source",
                hole=0.45,
                color="Noise Source",
                color_discrete_map={
                    "vehicular_honking": "#EF4444",
                    "continuous_traffic": "#F59E0B",
                    "construction_piling": "#8B5CF6",
                    "pa_loudspeaker": "#EC4899",
                    "ambient_baseline": "#10B981",
                },
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                height=350,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.bar_chart(source_counts.set_index("Noise Source"))

    # 3b — Observed vs CPCB Limits
    with col_c2:
        st.markdown("#### Observed vs CPCB Statutory Limits by Zone")
        zone_summary = filtered_df.groupby("zone_type")["L_eq_db"].mean().reset_index()
        zone_summary["Statutory Limit"] = zone_summary["zone_type"].map(
            lambda z: CPCB_LIMITS["night" if is_night_mode else "day"].get(z, 45.0)
        )
        zone_summary.columns = ["Zone", "Observed Leq", "CPCB Limit"]

        if HAS_PLOTLY:
            fig_bar = go.Figure(data=[
                go.Bar(
                    name="Observed L_eq dB(A)",
                    x=zone_summary["Zone"],
                    y=zone_summary["Observed Leq"],
                    marker_color="#3B82F6",
                ),
                go.Bar(
                    name="CPCB Statutory Limit",
                    x=zone_summary["Zone"],
                    y=zone_summary["CPCB Limit"],
                    marker_color="#EF4444",
                ),
            ])
            fig_bar.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="dB(A)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.table(zone_summary)

    # 3c — Day vs Night Distribution (box plot)
    st.markdown("---")
    st.markdown("#### Day vs Night Acoustic Distribution")
    if HAS_PLOTLY and n_records > 0:
        tw_col = "time_window"
        if tw_col not in filtered_df.columns:
            filtered_df[tw_col] = filtered_df["is_night"].map(
                {True: "Night: 22:00–06:00", False: "Day: 06:00–22:00"}
            )
        fig_box = px.box(
            filtered_df,
            x=tw_col,
            y="L_eq_db",
            color=tw_col,
            color_discrete_map={
                "Night: 22:00–06:00": "#6366F1",
                "Day: 06:00–22:00": "#F59E0B",
                "Night": "#6366F1",
                "Day": "#F59E0B",
            },
            labels={"L_eq_db": "L_eq  dB(A)", tw_col: "Time Window"},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Insufficient data for Day vs Night comparison or Plotly unavailable.")

    st.markdown("---")
    st.markdown("#### Raw Monitored Log Records")
    st.dataframe(filtered_df, use_container_width=True)


# =============================================================================
# TAB 4 — Model Benchmarks & CPCB Audit
# =============================================================================
with tab_bench:
    st.markdown(
        '<div class="sec-hdr">Acoustic Classifier Benchmarks & Statutory CPCB Compliance Audit</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Stratified 5-Fold CV results across the Chennai Poonamallee–Kuthambakkam–Porur "
        "corridor dataset (RIT Chennai / FRESHATHON 3.0)."
    )

    report_data = load_evaluation_report()
    cpcb_df = load_cpcb_audit()
    cm_path = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")

    if report_data is None:
        st.warning(
            "⚠️ Benchmark artifacts not found in `outputs/`. "
            "Run `python src/run_full_pipeline.py` to generate them."
        )
    else:
        deep = report_data.get("primary_deep_classifier", {})
        base = report_data.get("baseline_feature_classifier", {})
        latency = report_data.get("inference_latency_ms_per_clip", 0.0)

        # KPI cards for benchmarks
        st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Deep Classifier Accuracy</div>
        <div class="kpi-value c-teal">{deep.get('accuracy', 0) * 100:.1f} <span class="kpi-unit">%</span></div>
        <div class="kpi-sub">Stratified 5-Fold CV</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Macro F1-Score</div>
        <div class="kpi-value c-navy">{deep.get('macro_f1', 0):.3f}</div>
        <div class="kpi-sub">Balanced across 5 classes</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Inference Latency</div>
        <div class="kpi-value c-blue">{latency:.1f} <span class="kpi-unit">ms</span></div>
        <div class="kpi-sub">Per 5 s audio clip</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Baseline RF F1</div>
        <div class="kpi-value c-muted">{base.get('macro_f1', 1.0):.3f}</div>
        <div class="kpi-sub">Random Forest (n=100)</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Two-column layout: confusion matrix + compliance
        b_left, b_right = st.columns([1.1, 1.2])

        with b_left:
            st.markdown("#### Normalized Confusion Matrix (300 DPI)")
            if os.path.exists(cm_path):
                st.image(
                    cm_path,
                    caption="Stratified 5-Fold CV Normalized Confusion Matrix "
                            "(IEEE / FRESHATHON PPT ready)",
                    use_container_width=True,
                )
            else:
                st.info("Confusion matrix image pending generation.")

            # Per-class table
            st.markdown("##### Per-Class Performance")
            per_class = report_data.get("per_class_metrics", {})
            if per_class:
                rows = [
                    {
                        "Class": c.replace("_", " ").title(),
                        "Precision": f"{v.get('precision', 0):.3f}",
                        "Recall": f"{v.get('recall', 0):.3f}",
                        "F1": f"{v.get('f1_score', 0):.3f}",
                        "n": v.get("support", 0),
                    }
                    for c, v in per_class.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with b_right:
            st.markdown("#### CPCB Zone Compliance Summary")
            st.caption("CPCB Noise Pollution (Regulation & Control) Rules, 2000.")

            if cpcb_df is not None:
                st.dataframe(cpcb_df, use_container_width=True, hide_index=True)

                st.markdown("""
**Key Observations:**
- **Silence Zones:** Exceed Day limits by **+12.4 dB(A)** and Night by **+7.9 dB(A)**
  (heavy vehicle horns on the Poonamallee bypass).
- **Commercial Arterials:** **96.7 %** nocturnal breach rate; peak L_max hits **96.0 dB(A)**.
- **Peak Sleep Fragmentation:** Clusters 22:00–04:00 along the highway corridor.
""")

                # Download buttons
                cpcb_csv = cpcb_df.to_csv(index=False)
                report_json = json.dumps(report_data, indent=2)
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "📥 Download Audit CSV",
                        data=cpcb_csv,
                        file_name="cpcb_zone_compliance_summary.csv",
                        mime="text/csv",
                    )
                with dl2:
                    st.download_button(
                        "📥 Download Report JSON",
                        data=report_json,
                        file_name="model_evaluation_report.json",
                        mime="application/json",
                    )
            else:
                st.info("CPCB compliance CSV pending generation.")
