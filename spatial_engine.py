"""
DeciSense AI - Geospatial Analytics & Sleep-Disruption Engine
Authored for Member 3 (Geospatial & Spatial-ML Engineer)
FRESHATHON 3.0 - Innovation Project 1
"""

import json
import numpy as np

# CPCB Statutory Decibel Limits (Regulation and Control Rules, 2000)
CPCB_LIMITS = {
    "day": {
        "silence": 50.0,
        "residential": 55.0,
        "commercial": 65.0,
        "industrial": 75.0
    },
    "night": {
        "silence": 40.0,
        "residential": 45.0,
        "commercial": 55.0,
        "industrial": 70.0
    }
}

# 100m Silence POIs around RIT Chennai / Poonamallee Corridor
SILENCE_POIS = [
    {
        "id": "POI-01",
        "name": "RIT Chennai Campus Academic Complex",
        "coords": [13.0478, 80.0245],
        "buffer_radius_m": 100
    },
    {
        "id": "POI-02",
        "name": "Saveetha Medical & Dental Hospital",
        "coords": [13.0450, 80.0150],
        "buffer_radius_m": 100
    },
    {
        "id": "POI-03",
        "name": "Panimalar Hospital Silence Zone",
        "coords": [13.0520, 80.0380],
        "buffer_radius_m": 100
    },
    {
        "id": "POI-04",
        "name": "Kuthambakkam Primary Health Center",
        "coords": [13.0410, 80.0090],
        "buffer_radius_m": 100
    }
]

# Baseline Poonamallee Corridor Road Segments
SAMPLE_STREETS = [
    {
        "id": "ST-01",
        "name": "Bangalore National Highway (NH48) Flyover Stretch",
        "zone": "commercial",
        "coords": [[13.0495, 80.0120], [13.0510, 80.0250], [13.0535, 80.0400]],
        "base_leq": 76.5,
        "base_lmax": 85.0,
        "dominant_source": "Continuous_Traffic"
    },
    {
        "id": "ST-02",
        "name": "RIT Campus Approach Road (Saveetha Junction)",
        "zone": "silence",
        "coords": [[13.0470, 80.0220], [13.0478, 80.0245], [13.0485, 80.0280]],
        "base_leq": 64.2,
        "base_lmax": 78.4,
        "dominant_source": "Vehicular_Honking"
    },
    {
        "id": "ST-03",
        "name": "Kuthambakkam Bus Terminus Arterial",
        "zone": "commercial",
        "coords": [[13.0430, 80.0100], [13.0420, 80.0180], [13.0415, 80.0260]],
        "base_leq": 72.1,
        "base_lmax": 79.5,
        "dominant_source": "Continuous_Traffic"
    },
    {
        "id": "ST-04",
        "name": "Saveetha Medical Center Main Entrance",
        "zone": "silence",
        "coords": [[13.0440, 80.0140], [13.0450, 80.0150], [13.0460, 80.0170]],
        "base_leq": 68.9,
        "base_lmax": 82.3,
        "dominant_source": "Vehicular_Honking"
    },
    {
        "id": "ST-05",
        "name": "Nemam Residential Colony 1st Main",
        "zone": "residential",
        "coords": [[13.0540, 80.0190], [13.0560, 80.0210], [13.0580, 80.0240]],
        "base_leq": 42.0,
        "base_lmax": 48.0,
        "dominant_source": "Ambient_Baseline"
    },
    {
        "id": "ST-06",
        "name": "Poonamallee Metro Corridor Construction Stretch",
        "zone": "commercial",
        "coords": [[13.0490, 80.0350], [13.0510, 80.0380], [13.0530, 80.0420]],
        "base_leq": 79.8,
        "base_lmax": 88.2,
        "dominant_source": "Construction_Drilling"
    },
    {
        "id": "ST-07",
        "name": "RIT Student Hostels Internal Quad",
        "zone": "silence",
        "coords": [[13.0485, 80.0255], [13.0490, 80.0270]],
        "base_leq": 37.5,
        "base_lmax": 42.1,
        "dominant_source": "Ambient_Baseline"
    }
]


def compute_sleep_disruption_index(l_eq, l_max, zone="residential", is_night=True):
    """
    Computes normalized 0-100 Sleep-Disruption Risk Score:
    Score = min(100, W_time * (E_base + P_impulse))
    
    E_base: Base Excess over CPCB night statutory threshold (40 dB silence, 45 dB residential)
    P_impulse: Sudden crest jump penalty (L_max - L_eq - 10)
    W_time: 1.3x multiplier during nighttime (22:00 - 06:00), 1.0x during daytime
    """
    mode = "night" if is_night else "day"
    limit = CPCB_LIMITS[mode].get(zone, 45.0)

    # 1. Base Excess Energy Score
    base_excess = max(0.0, l_eq - limit) * 2.0

    # 2. Impulse Crest Penalty (Awakening Trigger)
    crest = max(0.0, (l_max - l_eq) - 10.0)
    impulse_penalty = crest * 1.5

    # 3. Night Weight Multiplier
    w_time = 1.3 if is_night else 1.0

    raw_score = w_time * (base_excess + impulse_penalty)
    final_score = int(min(100, round(raw_score)))
    
    is_violation = l_eq > limit
    delta_db = round(l_eq - limit, 1)

    return {
        "sleep_score": final_score,
        "is_violation": is_violation,
        "delta_db": delta_db,
        "statutory_limit_db": limit,
        "category": "Critical" if final_score > 65 else ("Moderate" if final_score >= 40 else "Low / Quiet")
    }


def generate_corridor_geojson(is_night=True):
    """
    Exports continuous street segments as valid GeoJSON FeatureCollection
    ready for Folium, Leaflet, or Mapbox.
    """
    features = []
    for st in SAMPLE_STREETS:
        score_info = compute_sleep_disruption_index(
            st["base_leq"], st["base_lmax"], zone=st["zone"], is_night=is_night
        )
        # Flip coordinates to GeoJSON standard [longitude, latitude]
        geojson_coords = [[c[1], c[0]] for c in st["coords"]]

        feature = {
            "type": "Feature",
            "properties": {
                "street_id": st["id"],
                "name": st["name"],
                "zone_type": st["zone"],
                "l_eq_db": st["base_leq"],
                "l_max_db": st["base_lmax"],
                "dominant_source": st["dominant_source"],
                "sleep_disruption_score": score_info["sleep_score"],
                "is_violation": score_info["is_violation"],
                "delta_db": score_info["delta_db"],
                "statutory_limit_db": score_info["statutory_limit_db"]
            },
            "geometry": {
                "type": "LineString",
                "coordinates": geojson_coords
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


if __name__ == "__main__":
    geojson = generate_corridor_geojson(is_night=True)
    with open("decisense_corridor.geojson", "w") as f:
        json.dump(geojson, f, indent=2)
    print("Exported decisense_corridor.geojson successfully.")
