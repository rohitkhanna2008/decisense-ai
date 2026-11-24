"""
DeciSense AI - Synthetic Acoustic Data Generator
Location: Chennai, India (Poonamallee - Kuthambakkam - Porur Corridor around RIT)
Generates 150 geotagged audio logs + 5 benchmark 16kHz WAV files.
"""

import os
import wave
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

SAMPLE_RATE = 16000
DURATION_SEC = 5.0

CLASSES = [
    "vehicular_honking",
    "continuous_traffic",
    "construction_piling",
    "pa_loudspeaker",
    "ambient_baseline"
]

# Key Landmark Anchors in Chennai Poonamallee-Kuthambakkam Corridor
CORRIDOR_ANCHORS = [
    {"name": "RIT Chennai Campus Academic Zone", "lat": 13.0478, "lon": 80.0245, "zone": "silence"},
    {"name": "Saveetha Medical Center & Hospital", "lat": 13.0450, "lon": 80.0150, "zone": "silence"},
    {"name": "Panimalar Hospital Buffer", "lat": 13.0520, "lon": 80.0380, "zone": "silence"},
    {"name": "Kuthambakkam Bus Terminus", "lat": 13.0410, "lon": 80.0090, "zone": "commercial"},
    {"name": "Bangalore National Highway (NH48) Flyover", "lat": 13.0510, "lon": 80.0250, "zone": "commercial"},
    {"name": "Poonamallee Trunk Road Junction", "lat": 13.0490, "lon": 80.0350, "zone": "commercial"},
    {"name": "Porur Bypass Link Corridor", "lat": 13.0320, "lon": 80.0510, "zone": "commercial"},
    {"name": "Nemam Residential Layout", "lat": 13.0560, "lon": 80.0210, "zone": "residential"},
    {"name": "Kuthambakkam Village Residential Green", "lat": 13.0380, "lon": 80.0120, "zone": "residential"},
    {"name": "Chettipedu Residential Colony", "lat": 13.0600, "lon": 80.0300, "zone": "residential"}
]


def synthesize_class_audio(class_name, duration=DURATION_SEC, sample_rate=SAMPLE_RATE):
    """
    Synthesizes authentic acoustic waveforms for the 5 target noise classes.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if class_name == "vehicular_honking":
        # Dual-frequency high-intensity Indian horn with resonant harmonics (440Hz, 880Hz, 1760Hz, 2640Hz)
        carrier = (0.50 * np.sin(2 * np.pi * 480 * t) + 
                   0.40 * np.sin(2 * np.pi * 820 * t) + 
                   0.35 * np.sin(2 * np.pi * 1640 * t) + 
                   0.25 * np.sin(2 * np.pi * 2460 * t))
        envelope = np.zeros_like(t)
        # Blast 1: 1.2s - 1.55s (350ms sharp blast)
        envelope[(t >= 1.2) & (t <= 1.55)] = 1.0
        # Blast 2: 2.8s - 3.15s (350ms sharp blast)
        envelope[(t >= 2.8) & (t <= 3.15)] = 1.0
        # Light ambient background during silence
        ambient_road = 0.02 * np.sin(2 * np.pi * 200 * t)
        waveform = carrier * envelope + ambient_road

    elif class_name == "continuous_traffic":
        # Heavy low-frequency diesel rumble (45-180 Hz) and rolling tire friction
        rumble = (0.55 * np.sin(2 * np.pi * 55 * t) + 
                  0.40 * np.sin(2 * np.pi * 95 * t) + 
                  0.35 * np.sin(2 * np.pi * 145 * t) + 
                  0.25 * np.sin(2 * np.pi * 210 * t))
        # Amplitude modulation from passing vehicles
        amp_mod = 0.75 + 0.25 * np.sin(2 * np.pi * 0.4 * t)
        waveform = rumble * amp_mod

    elif class_name == "construction_piling":
        # Rhythmic high-impact pile-driver thuds (90 Hz deep impact)
        waveform = np.zeros_like(t)
        strike_times = [0.8, 1.8, 2.8, 3.8]
        for st in strike_times:
            idx = int(st * sample_rate)
            decay_len = min(int(0.25 * sample_rate), len(t) - idx)
            decay_curve = np.exp(-np.linspace(0, 7.0, decay_len))
            strike_tone = (0.90 * np.sin(2 * np.pi * 80 * t[:decay_len]) + 
                           0.30 * np.sin(2 * np.pi * 240 * t[:decay_len])) * decay_curve
            waveform[idx : idx + decay_len] += strike_tone
        # Very low ambient between strikes
        waveform += 0.01 * np.sin(2 * np.pi * 60 * t)

    elif class_name == "pa_loudspeaker":
        # Nasal band-limited amplified speech/chanting (580Hz, 920Hz, 1380Hz)
        speech_synth = (0.45 * np.sin(2 * np.pi * 580 * t) + 
                        0.40 * np.sin(2 * np.pi * 920 * t) + 
                        0.30 * np.sin(2 * np.pi * 1380 * t))
        distortion = np.tanh(speech_synth * 2.5)
        waveform = distortion * (0.8 + 0.2 * np.sin(2 * np.pi * 3.0 * t))

    else:  # ambient_baseline
        # Quiet natural background, low amplitude peak ~ 0.015 (yielding ~ 36-40 dB(A))
        waveform = 0.02 * np.sin(2 * np.pi * 120 * t) + 0.01 * np.random.normal(0, 0.05, len(t))

    # Scale audio amplitudes realistically according to target sound level
    if class_name == "vehicular_honking":
        # Sharp bursts peaking at 0.92, low baseline
        peak = np.max(np.abs(waveform))
        waveform = 0.92 * (waveform / max(peak, 1e-6))
    elif class_name == "construction_piling":
        # Sharp impact strikes peaking at 0.90
        peak = np.max(np.abs(waveform))
        waveform = 0.90 * (waveform / max(peak, 1e-6))
    elif class_name == "continuous_traffic":
        # Moderate sustained level peaking around 0.40
        peak = np.max(np.abs(waveform))
        waveform = 0.40 * (waveform / max(peak, 1e-6))
    elif class_name == "pa_loudspeaker":
        # Sustained amplified speech peaking around 0.65
        peak = np.max(np.abs(waveform))
        waveform = 0.65 * (waveform / max(peak, 1e-6))
    else:  # ambient_baseline
        # Quiet natural background, low amplitude peak ~ 0.015 (yielding ~ 36-40 dB(A))
        peak = np.max(np.abs(waveform))
        waveform = 0.015 * (waveform / max(peak, 1e-6))
        
    return waveform, sample_rate


def save_pcm16_wav(filepath, waveform, sample_rate=SAMPLE_RATE):
    """Saves waveform as 16-bit Mono PCM WAV without external dependencies."""
    scaled = np.int16(waveform * 32767)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())


def generate_dataset(base_dir="."):
    data_dir = os.path.join(base_dir, "data")
    sample_wavs_dir = os.path.join(data_dir, "sample_wavs")
    os.makedirs(sample_wavs_dir, exist_ok=True)

    print("Generating 5 master benchmark WAV files...")
    master_wav_paths = {}
    for cname in CLASSES:
        wav_filename = f"{cname}.wav"
        wav_path = os.path.join(sample_wavs_dir, wav_filename)
        waveform, sr = synthesize_class_audio(cname)
        save_pcm16_wav(wav_path, waveform, sr)
        master_wav_paths[cname] = os.path.relpath(wav_path, base_dir).replace("\\", "/")
        print(f"  [+] Saved benchmark: {wav_path}")

    # Generate 150 realistic geotagged records along the corridor
    print("\nSynthesizing 150 geotagged corridor audio logs...")
    random.seed(42)
    np.random.seed(42)

    records = []
    base_time = datetime(2026, 11, 15, 0, 0, 0)

    for i in range(1, 151):
        clip_id = f"DS_CHN_{i:03d}"
        
        # Pick an anchor with some spatial jitter
        anchor = random.choice(CORRIDOR_ANCHORS)
        lat = round(anchor["lat"] + random.gauss(0, 0.0035), 6)
        lon = round(anchor["lon"] + random.gauss(0, 0.0035), 6)

        # Clamp within corridor boundaries: lat [13.025, 13.065], lon [80.005, 80.055]
        lat = max(13.025, min(13.065, lat))
        lon = max(80.005, min(80.055, lon))
        zone_type = anchor["zone"]

        # Random timestamp over 7 days with realistic day/night distribution (40% night, 60% day)
        day_offset = random.randint(0, 6)
        if random.random() < 0.40:
            # Night slot: 22:00 to 05:59
            hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
        else:
            # Day slot: 06:00 to 21:59
            hour = random.randint(6, 21)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        timestamp = base_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        is_night = (hour >= 22 or hour < 6)

        # Noise class probabilities conditioned on zone & time
        if zone_type == "silence":
            if is_night:
                weights = [0.15, 0.10, 0.05, 0.05, 0.65] # mostly ambient, occasional horn
            else:
                weights = [0.25, 0.30, 0.10, 0.05, 0.30]
        elif zone_type == "commercial":
            if is_night:
                weights = [0.35, 0.40, 0.15, 0.05, 0.05] # highway trucks & honking
            else:
                weights = [0.45, 0.35, 0.10, 0.08, 0.02]
        else: # residential
            if is_night:
                weights = [0.10, 0.10, 0.05, 0.15, 0.60] # temple/wedding speaker or ambient
            else:
                weights = [0.20, 0.25, 0.15, 0.20, 0.20]

        ground_truth = random.choices(CLASSES, weights=weights, k=1)[0]
        rel_wav_path = master_wav_paths[ground_truth]

        # Generate realistic Leq and Lmax corresponding to class and zone
        if ground_truth == "vehicular_honking":
            leq = round(random.uniform(72.0, 84.5), 1)
            crest = round(random.uniform(7.5, 14.0), 1)
        elif ground_truth == "continuous_traffic":
            leq = round(random.uniform(66.0, 77.0), 1)
            crest = round(random.uniform(3.5, 6.5), 1)
        elif ground_truth == "construction_piling":
            leq = round(random.uniform(74.0, 86.0), 1)
            crest = round(random.uniform(6.0, 11.0), 1)
        elif ground_truth == "pa_loudspeaker":
            leq = round(random.uniform(68.0, 78.5), 1)
            crest = round(random.uniform(4.0, 8.0), 1)
        else: # ambient_baseline
            leq = round(random.uniform(34.0, 43.5), 1)
            crest = round(random.uniform(2.0, 4.5), 1)

        lmax = round(leq + crest, 1)

        records.append({
            "clip_id": clip_id,
            "wav_path": rel_wav_path,
            "latitude": lat,
            "longitude": lon,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "zone_type": zone_type,
            "L_eq_db": leq,
            "L_max_db": lmax,
            "ground_truth_class": ground_truth
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "chennai_audio_logs.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [+] Saved 150 audio logs to: {csv_path}")
    print(f"  Dataset Zone Breakdown:\n{df['zone_type'].value_counts().to_string()}")
    print(f"  Dataset Class Breakdown:\n{df['ground_truth_class'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_dataset(current_dir)
