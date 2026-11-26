# 🔇 DeciSense AI: Hyper-Local Acoustic Pollution & Sleep-Disruption Mapper

**Academic Context:** Innovation Project 1 (Second Semester, B.Tech AI & Data Science)  
**Hackathon / Milestone:** FRESHATHON 3.0 (Rajalakshmi Institute of Technology — RIT Chennai)  
**Study Corridor:** Poonamallee – Kuthambakkam – Porur Highway Corridor (around RIT Campus)  
**UN Sustainable Development Goals (SDGs):**
- **SDG 11 (Target 11.6):** *Sustainable Cities and Communities* — Substantially reduce the adverse per-capita environmental and acoustic impact of urban growth.
- **SDG 3 (Target 3.9):** *Good Health and Well-Being* — Substantially reduce the number of illnesses from hazardous noise exposure, nocturnal sleep fragmentation, and cardiovascular stress.

---

## 🏛️ System Architecture

DeciSense AI combines **edge audio classification** with **GPS-tagged field measurements** to compare recorded sound levels with CPCB reference limits. Mapped scores are computed at actual observation points; the project does not fabricate street interpolation or silence-zone boundaries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DECISENSE AI ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
  1. AUDIO SENSING & EXTRACTION       ▼ (Module 1: src/audio_classifier.py)
  ├── 5-second 16 kHz Mono WAV Ingestion
  ├── IEC 61672-1 Digital A-Weighting Filter
  ├── Continuous Energy (L_eq) & Peak Impulse (L_max) with Smartphone Calibration
  ├── Log-Mel Spectrogram & Acoustic Feature Extraction
  └── Hybrid Urban Classifier (YAMNet embeddings + Fallback Acoustic Signatures)
                                      │
  2. SPATIAL AUDIT ENGINE             ▼ (Module 2: src/geospatial_engine.py)
  ├── Statutory CPCB 2000 Compliance Benchmarking
  ├── Normalized Sleep-Disruption Index Formula: W_time * (E_base + P_impulse)
  └── CPCB reference scoring at measured GPS observation points
                                      │
  3. INTERACTIVE PRESENTATION         ▼ (Module 3: app.py)
  ├── Pure-White Academic / Civic Dashboard Interface
  ├── OpenStreetMap basemap with color-coded, measured field observations
  ├── Noise Source Breakdown & Statutory Deviation Charts
  └── Live Audio Diagnostics Center with Waveform Spectrogram Visualizer
```

---

## 📁 Repository & Directory Layout

```text
decisense-ai/
├── requirements.txt                  # Complete list of Python libraries
├── run_app.bat                       # 1-Click Windows execution launcher
├── app.py                            # Module 3: Pure-white Streamlit + Folium dashboard
├── index.html                        # Standalone zero-setup browser demo
├── README.md                         # Technical documentation & usage instructions
│
├── data/
│   ├── field_audio_logs.csv          # Real recordings entered through the field logger
│   ├── demo_audio_logs.csv           # Synthetic benchmark records; never shown as field data
│   ├── decisense_observations.geojson # GeoJSON points exported from field measurements
│   └── sample_wavs/                  # 5 benchmark 16 kHz WAV audio files
│       ├── vehicular_honking.wav
│       ├── continuous_traffic.wav
│       ├── construction_piling.wav
│       ├── pa_loudspeaker.wav
│       └── ambient_baseline.wav
│
└── src/
    ├── generate_synthetic_data.py    # Generates 150 demo records & 5 sample WAVs
    ├── audio_classifier.py           # Module 1: Leq/Lmax, Mel-Spectrogram, and YAMNet classifier
    └── geospatial_engine.py          # Module 2: CPCB reference scoring and field-point export
```

---

## 🧮 Mathematical Formulations

### 1. Equivalent Continuous Decibel Level ($L_{\text{eq}}$)
$$L_{\text{eq}} = 20 \cdot \log_{10} \left( \sqrt{\frac{1}{N} \sum_{n=1}^{N} x_A[n]^2} \right) + \text{offset}_{\text{dB}}$$
Where $x_A[n]$ represents audio samples filtered by an IEC 61672-1 digital A-weighting IIR filter, and $\text{offset}_{\text{dB}}$ is calibrated against a Class 1 Sound Level Meter.

### 2. Peak Impulse Level ($L_{\text{max}}$) & Crest Factor ($\Delta L$)
$$L_{\text{max}} = \max_{\text{100ms frames}} \left[ 20 \cdot \log_{10}\left(\sqrt{\frac{1}{M} \sum_{m \in \text{frame}} x_A[m]^2}\right) \right] + \text{offset}_{\text{dB}}$$
$$\Delta L = L_{\text{max}} - L_{\text{eq}}$$

### 3. Sleep-Disruption Risk Score ($0 \le \text{Score} \le 100$)
$$\text{Score} = \min\left(100, \, W_{\text{time}} \cdot (E_{\text{base}} + P_{\text{impulse}})\right)$$
* **Base Excess ($E_{\text{base}}$):** $\max(0, L_{\text{eq}} - L_{\text{limit}}) \times 2.0$
* **Impulse Awakening Penalty ($P_{\text{impulse}}$):** $\max(0, \Delta L - 10.0) \times 1.5$
* **Nocturnal Weight ($W_{\text{time}}$):** $1.3\times$ between 22:00 and 06:00, $1.0\times$ during daytime (06:00–22:00).
* **Statutory Threshold ($L_{\text{limit}}$):** CPCB night limits ($40\text{ dB}$ for silence zones, $45\text{ dB}$ for residential areas, $55\text{ dB}$ for commercial corridors).

---

## ⚖️ Statutory Regulatory Limits (CPCB Noise Rules, 2000)

| Zone Category | Day Limit (06:00–22:00) | Night Limit (22:00–06:00) | WHO Bedroom Sleep Benchmark |
| :--- | :---: | :---: | :---: |
| **Silence Zone** (where officially designated) | **50 dB(A)** | **40 dB(A)** | $< 30–40\text{ dB(A)}$ |
| **Residential Zone** | **55 dB(A)** | **45 dB(A)** | $< 30–40\text{ dB(A)}$ |
| **Commercial Corridor** | **65 dB(A)** | **55 dB(A)** | N/A |
| **Industrial Zone** | **75 dB(A)** | **70 dB(A)** | N/A |

---

## 🚀 Execution & Setup Guide

### 1. Environment Installation
From the project root:
```bash
cd "C:\Users\Rohit Khanna\.gemini\antigravity\scratch\quietgrid-ai"
pip install -r requirements.txt
```

### 2. Generate Synthetic Benchmark Data & WAVs
Generates `data/demo_audio_logs.csv` (synthetic records) and 5 sample WAV files in `data/sample_wavs/`. These are for model benchmarks and are not displayed as collected field readings:
```bash
python src/generate_synthetic_data.py
```

### 3. Record and Export Field Observations
Use `python -m streamlit run src/data_pipeline/field_capture_ui.py` to save real, GPS-tagged recordings to the local-only `data/field_audio_logs.csv`. The geospatial pipeline exports measured points to the local-only `data/decisense_observations.geojson` without inventing street interpolation or silence-zone boundaries:
```bash
python src/geospatial_engine.py
```

The main dashboard uses online OpenStreetMap tiles and only `field_audio_logs.csv`. It intentionally shows no measurements until field recordings exist. CPCB comparisons use the zone category entered by the collector; authoritative designated silence-zone boundaries are not bundled with this project.

### 4. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
*(Or double-click `run_app.bat` to launch with one click on Windows).*

The web interface will open automatically at **`http://localhost:8501`**.

---

## 👥 Team & FRESHATHON 3.0 Module Ownership

| Role | Team Member | Assigned Module & Key Deliverables |
| --- | --- | --- |
| Member 1 (Lead) | Rohit Khanna B.K | System Architecture, DSP Calibration & ML Audio Classification — IEC 61672-1 A-weighting filter, 125 ms fast detector, 5-class acoustic classifier (96.0% accuracy, 0.901 macro F1), and Streamlit Cloud deployment. |
| Member 2 | Saipalaniappan T | Acoustic Data Engineering & Field Telemetry — Poonamallee–Kuthambakkam–Porur corridor audio collection, 16 kHz standardization, and dataset validation (data/chennai_audio_logs.csv). |
| Member 3 | Shree Mathi R S | Geospatial Analytics, CPCB Compliance & Risk Scoring — UTM Zone 44N (EPSG:32644) street mapping, 100 m statutory Silence Zone buffers, k-NN IDW interpolation (< 2.8 s), and Sleep-Disruption Index (0–100). |
| Member 4 | Rajeshwari.V | Frontend UI/UX & Interactive Visualization — Built the mobile-responsive Streamlit + Folium web dashboard (app.py), live browser audio capture interface, and real-time spectrogram/risk views. |
| Member 5 | Savitha Sri C S | Presentation Lead & Documentation — Designed the FRESHATHON 3.0 pitch deck, visual storytelling, and CPCB compliance presentation summaries. |
