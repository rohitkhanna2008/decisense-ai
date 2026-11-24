# 🔇 DeciSense AI: Hyper-Local Acoustic Pollution & Sleep-Disruption Mapper

**Academic Context:** Innovation Project 1 (Second Semester, B.Tech AI & Data Science)  
**Hackathon / Milestone:** FRESHATHON 3.0 (Rajalakshmi Institute of Technology — RIT Chennai)  
**Study Corridor:** Poonamallee – Kuthambakkam – Porur Highway Corridor (around RIT Campus)  
**UN Sustainable Development Goals (SDGs):**
- **SDG 11 (Target 11.6):** *Sustainable Cities and Communities* — Substantially reduce the adverse per-capita environmental and acoustic impact of urban growth.
- **SDG 3 (Target 3.9):** *Good Health and Well-Being* — Substantially reduce the number of illnesses from hazardous noise exposure, nocturnal sleep fragmentation, and cardiovascular stress.

---

## 🏛️ System Architecture

DeciSense AI moves beyond sparse, static decibel boards (which report only time-averaged $L_{\text{eq}}$) by combining **edge audio classification** with **hyper-local GIS spatial interpolation** to quantify acute sleep-disruption risks in growing urban transport corridors:

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
  2. SPATIAL ML & AUDIT ENGINE        ▼ (Module 2: src/geospatial_engine.py)
  ├── Statutory CPCB 2000 Compliance Benchmarking
  ├── Normalized Sleep-Disruption Index Formula: W_time * (E_base + P_impulse)
  ├── 100-meter CPCB Silence Zone Buffer Generation (UTM Zone 44N EPSG:32644)
  └── k-Nearest Neighbor Inverse Distance Weighting (IDW) Street Interpolation
                                      │
  3. INTERACTIVE PRESENTATION         ▼ (Module 3: app.py)
  ├── Pure-White Academic / Civic Dashboard Interface
  ├── Interactive Folium Heatmap with Color-Coded Segments & POI Rings
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
│   ├── chennai_audio_logs.csv        # 150 geotagged corridor logs (lat: 13.025-13.065, lon: 80.005-80.055)
│   ├── decisense_scored_streets.geojson # IDW-interpolated street network with sleep scores
│   ├── silence_zones.geojson         # 100m statutory buffer polygons around hospitals/colleges
│   └── sample_wavs/                  # 5 benchmark 16 kHz WAV audio files
│       ├── vehicular_honking.wav
│       ├── continuous_traffic.wav
│       ├── construction_piling.wav
│       ├── pa_loudspeaker.wav
│       └── ambient_baseline.wav
│
└── src/
    ├── generate_synthetic_data.py    # Generates 150 geotagged records & 5 sample WAVs
    ├── audio_classifier.py           # Module 1: Leq/Lmax, Mel-Spectrogram, and YAMNet classifier
    └── geospatial_engine.py          # Module 2: CPCB audit, Sleep Index, 100m buffers, and IDW
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
| **Silence Zone** (Within 100m of Hospitals/Colleges) | **50 dB(A)** | **40 dB(A)** | $< 30–40\text{ dB(A)}$ |
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

### 2. Generate Synthetic Dataset & Benchmark WAVs
Generates `data/chennai_audio_logs.csv` (150 geotagged records) and 5 sample WAV files in `data/sample_wavs/`:
```bash
python src/generate_synthetic_data.py
```

### 3. Generate Geospatial Heatmap & 100m Silence Buffers
Computes IDW street network interpolation and produces `data/decisense_scored_streets.geojson` and `data/silence_zones.geojson`:
```bash
python src/geospatial_engine.py
```

### 4. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
*(Or double-click `run_app.bat` to launch with one click on Windows).*

The web interface will open automatically at **`http://localhost:8501`**.

---

## 👥 Team & FRESHATHON 3.0 Module Ownership

| Role | Primary Deliverable | Script / Module |
| :--- | :--- | :--- |
| **Rohit Khanna B.K (Team Lead & Audio ML)** | Feature Extraction, $L_{\text{eq}}/L_{\text{max}}$, YAMNet classification | `src/audio_classifier.py` |
| **Member 2 (Data & Calibration Lead)** | Data protocol, 150-point dataset, smartphone dB calibration | `src/generate_synthetic_data.py` |
| **Member 3 (Geospatial & Spatial ML)** | Sleep-Disruption Index, 100m Silence POI buffers, k-NN IDW | `src/geospatial_engine.py` |
| **Member 4 (Frontend UI/UX)** | Pure-white academic dashboard, Folium heatmap, Plotly charts | `app.py` |
| **Member 5 (SDG Impact & Pitch)** | CPCB/WHO statutory framing, survey validation, slide deck | `README.md` / Pitch Deck |
