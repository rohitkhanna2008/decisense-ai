"""
DeciSense AI - Demo Acoustic Data Generator
Generates sample .wav audio clips for testing audio classification and Leq extraction.
"""

import os
import wave
import numpy as np
from audio_engine import synthesize_test_tone

def save_wav(filename, waveform, sample_rate=16000):
    # Scale to 16-bit PCM
    scaled = np.int16(waveform * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)       # 2 bytes = 16 bit
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())

def generate_all_samples(output_dir="sample_audio"):
    os.makedirs(output_dir, exist_ok=True)
    
    presets = [
        ("honk", "horn_blast_82db.wav"),
        ("traffic", "highway_traffic_68db.wav"),
        ("construction", "metro_drilling_78db.wav"),
        ("quiet", "night_ambient_36db.wav")
    ]
    
    print("Generating demo .wav audio files...")
    for source_type, fname in presets:
        waveform, sr = synthesize_test_tone(source_type=source_type, duration=4.0)
        out_path = os.path.join(output_dir, fname)
        save_wav(out_path, waveform, sr)
        print(f"  [+] Saved: {out_path}")
        
    print("All demo samples generated successfully!")

if __name__ == "__main__":
    generate_all_samples()
