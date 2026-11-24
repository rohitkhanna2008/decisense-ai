"""
DeciSense AI - Module 2: Sleep-Disruption Index & Chennai GIS Interpolation
Computes CPCB statutory violations, Sleep-Disruption Scores (0-100),
builds 100m Silence Zone buffers, performs k-NN IDW street interpolation,
and exports data/decisense_scored_streets.geojson and data/silence_zones.geojson.
"""

import os
import json
import numpy as np
import pandas as pd

# Optional GIS dependencies with automatic fallback
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon, mapping
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

# CPCB Statutory Limits dB(A) under Noise Pollution (Regulation and Control) Rules, 2000
CPCB_LIMITS = {
    "day": {
        "silence": 50.0,
        "residential": 55.0,
        "commercial": 65.0
    },
    "night": {
        "silence": 40.0,
        "residential": 45.0,
        "commercial": 55.0
    }
}

# Key Hospital & Educational Silence POIs in Poonamallee-Kuthambakkam Corridor
CHENNAI_SILENCE_POIS = [
    {
        "id": "POI_RIT",
        "name": "Rajalakshmi Institute of Technology (RIT) Campus",
        "category": "Educational Institution",
        "lat": 13.0478,
        "lon": 80.0245,
        "buffer_radius_m": 100
    },
    {
        "id": "POI_SAVEETHA",
        "name": "Saveetha Medical College & Super-Specialty Hospital",
        "category": "Hospital",
        "lat": 13.0450,
        "lon": 80.0150,
        "buffer_radius_m": 100
    },
    {
        "id": "POI_PANIMALAR",
        "name": "Panimalar Hospital & Medical Research Institute",
        "category": "Hospital",
        "lat": 13.0520,
        "lon": 80.0380,
        "buffer_radius_m": 100
    },
    {
        "id": "POI_KUTHAMBAKKAM_PHC",
        "name": "Kuthambakkam Primary Health Center",
        "category": "Hospital / Clinic",
        "lat": 13.0410,
        "lon": 80.0090,
        "buffer_radius_m": 100
    },
    {
        "id": "POI_SASTHA",
        "name": "Sri Sastha Institute of Engineering & Technology",
        "category": "Educational Institution",
        "lat": 13.0485,
        "lon": 80.0315,
        "buffer_radius_m": 100
    },
    {
        "id": "POI_POONAMALLEE_GH",
        "name": "Poonamallee Government Hospital",
        "category": "Hospital",
        "lat": 13.0495,
        "lon": 80.0460,
        "buffer_radius_m": 100
    }
]

# Primary Corridor Street Network (Chennai Poonamallee-Kuthambakkam-Porur Corridor)
CHENNAI_CORRIDOR_STREETS = [
    {
        "id": "CHN_ST_01",
        "name": "Bangalore National Highway (NH48) Expressway - RIT Flyover Stretch",
        "zone": "commercial",
        "coords": [[13.0495, 80.0100], [13.0505, 80.0180], [13.0515, 80.0260], [13.0525, 80.0360], [13.0535, 80.0450]]
    },
    {
        "id": "CHN_ST_02",
        "name": "RIT College Main Campus Approach Road",
        "zone": "silence",
        "coords": [[13.0465, 80.0230], [13.0475, 80.0245], [13.0485, 80.0260]]
    },
    {
        "id": "CHN_ST_03",
        "name": "Saveetha Medical College Entrance Arterial",
        "zone": "silence",
        "coords": [[13.0435, 80.0135], [13.0448, 80.0150], [13.0460, 80.0165]]
    },
    {
        "id": "CHN_ST_04",
        "name": "Kuthambakkam Mofussil Bus Terminus Access Corridor",
        "zone": "commercial",
        "coords": [[13.0395, 80.0075], [13.0410, 80.0090], [13.0425, 80.0115]]
    },
    {
        "id": "CHN_ST_05",
        "name": "Nemam Lake Road Residential Corridor",
        "zone": "residential",
        "coords": [[13.0535, 80.0180], [13.0555, 80.0205], [13.0575, 80.0230]]
    },
    {
        "id": "CHN_ST_06",
        "name": "Poonamallee Trunk Road - Metro Phase 2 Elevated Corridor",
        "zone": "commercial",
        "coords": [[13.0480, 80.0330], [13.0490, 80.0380], [13.0500, 80.0440]]
    },
    {
        "id": "CHN_ST_07",
        "name": "Panimalar Hospital Silence Buffer Service Lane",
        "zone": "silence",
        "coords": [[13.0508, 80.0365], [13.0518, 80.0380], [13.0528, 80.0395]]
    },
    {
        "id": "CHN_ST_08",
        "name": "Porur - Kundrathur Link Road",
        "zone": "commercial",
        "coords": [[13.0310, 80.0480], [13.0335, 80.0510], [13.0360, 80.0540]]
    },
    {
        "id": "CHN_ST_09",
        "name": "Kuthambakkam Village Gramani Residential Street",
        "zone": "residential",
        "coords": [[13.0365, 80.0105], [13.0380, 80.0125], [13.0395, 80.0145]]
    },
    {
        "id": "CHN_ST_10",
        "name": "Sastha College Academic Approach",
        "zone": "silence",
        "coords": [[13.0475, 80.0300], [13.0485, 80.0315], [13.0495, 80.0330]]
    },
    {
        "id": "CHN_ST_11",
        "name": "Poonamallee Bypass Residential Enclave",
        "zone": "residential",
        "coords": [[13.0450, 80.0410], [13.0465, 80.0435], [13.0480, 80.0460]]
    },
    {
        "id": "CHN_ST_12",
        "name": "Chettipedu Industrial Link Highway",
        "zone": "commercial",
        "coords": [[13.0580, 80.0270], [13.0600, 80.0300], [13.0620, 80.0330]]
    }
]


# -----------------------------------------------------------------------------
# 1. Sleep-Disruption Index Formula
# -----------------------------------------------------------------------------
def compute_sleep_disruption_index(leq, lmax, zone="residential", timestamp_hour=23):
    """
    Computes normalized 0-100 Sleep-Disruption Risk Score according to exact specification:
    - CPCB Night Limits: 40 dB(A) for silence, 45 dB(A) for residential, 55 dB(A) for commercial.
    - Base Excess: E_base = max(0, L_eq - L_limit) * 2.0
    - Impulse Penalty: P_impulse = max(0, (L_max - L_eq) - 10) * 1.5
    - Night Multiplier: 1.3 if timestamp hour is between 22:00 and 06:00, else 1.0
    - Final Score: min(100, W_time * (E_base + P_impulse))
    """
    is_night = (timestamp_hour >= 22 or timestamp_hour < 6)
    mode = "night" if is_night else "day"
    
    # Statutory threshold
    l_limit = CPCB_LIMITS[mode].get(zone, 45.0)

    # Base Excess
    e_base = max(0.0, leq - l_limit) * 2.0

    # Impulse Crest Penalty (penalizes sudden awakenings)
    crest = lmax - leq
    p_impulse = max(0.0, crest - 10.0) * 1.5

    # Night Multiplier
    w_time = 1.3 if is_night else 1.0

    raw_score = w_time * (e_base + p_impulse)
    final_score = int(min(100, round(raw_score)))

    is_violation = (leq > l_limit)
    delta_db = round(leq - l_limit, 1)

    if final_score >= 65:
        category = "Critical (High Sleep Awakenings)"
        color = "#ff3366"  # Red
    elif final_score >= 40:
        category = "Moderate (Sleep Fragmentation)"
        color = "#ffd600"  # Amber
    else:
        category = "Low / Undisturbed Sleep"
        color = "#00e676"  # Green

    return {
        "score": final_score,
        "category": category,
        "color": color,
        "is_violation": is_violation,
        "delta_db": delta_db,
        "statutory_limit": l_limit,
        "is_night": is_night
    }


# -----------------------------------------------------------------------------
# 2. 100m Silence Zone Buffer Geometry (UTM Zone 44N EPSG:32644)
# -----------------------------------------------------------------------------
def build_silence_buffers():
    """
    Generates 100m circular buffer geometries for all silence POIs.
    Uses GeoPandas/Shapely with UTM Zone 44N projection if available,
    or geodesic circular polygon vertices approximation.
    """
    features = []
    
    for poi in CHENNAI_SILENCE_POIS:
        lat, lon = poi["lat"], poi["lon"]
        radius_m = poi["buffer_radius_m"]

        if HAS_GEOPANDAS:
            try:
                pt_wgs84 = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                # Project to UTM Zone 44N (EPSG:32644) for Chennai metric coordinates
                pt_utm = pt_wgs84.to_crs(epsg=32644)
                buf_utm = pt_utm.buffer(radius_m)
                buf_wgs84 = buf_utm.to_crs(epsg=4326).iloc[0]
                geom = mapping(buf_wgs84)
            except Exception:
                geom = None
        else:
            geom = None

        if geom is None:
            # High-precision geodesic circular polygon fallback (36 vertices)
            lat_deg = radius_m / 111320.0
            lon_deg = radius_m / (111320.0 * np.cos(np.radians(lat)))
            angles = np.linspace(0, 2 * np.pi, 36)
            ring = [[round(lon + lon_deg * np.sin(a), 6), round(lat + lat_deg * np.cos(a), 6)] for a in angles]
            geom = {"type": "Polygon", "coordinates": [ring]}

        feature = {
            "type": "Feature",
            "properties": {
                "id": poi["id"],
                "name": poi["name"],
                "category": poi["category"],
                "buffer_radius_meters": radius_m,
                "cpcb_day_limit": 50.0,
                "cpcb_night_limit": 40.0
            },
            "geometry": geom
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


# -----------------------------------------------------------------------------
# 3. k-NN Inverse Distance Weighting (IDW) Interpolation onto Streets
# -----------------------------------------------------------------------------
def compute_haversine_distance(lat1, lon1, lat2, lon2):
    """Computes great circle distance in meters between two lat/lon points."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0)**2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def run_idw_interpolation(audio_df, is_night=True, k=5, power=2.0):
    """
    Interpolates sensor point readings onto each corridor street segment
    using k-Nearest Neighbor Inverse Distance Weighting.
    """
    # Filter points by time window if available
    if "timestamp" in audio_df.columns:
        hours = pd.to_datetime(audio_df["timestamp"]).dt.hour
        if is_night:
            pts_df = audio_df[(hours >= 22) | (hours < 6)]
        else:
            pts_df = audio_df[(hours >= 6) & (hours < 22)]
        if len(pts_df) < 5:
            pts_df = audio_df
    else:
        pts_df = audio_df

    pt_lats = pts_df["latitude"].values
    pt_lons = pts_df["longitude"].values
    pt_leqs = pts_df["L_eq_db"].values
    pt_lmaxs = pts_df["L_max_db"].values
    pt_classes = pts_df["ground_truth_class"].values

    eval_hour = 23 if is_night else 14
    scored_features = []

    for st in CHENNAI_CORRIDOR_STREETS:
        # Compute street centroid
        st_coords = np.array(st["coords"])
        mid_lat = float(np.mean(st_coords[:, 0]))
        mid_lon = float(np.mean(st_coords[:, 1]))

        # Calculate distances to all audio sensor points
        distances = compute_haversine_distance(mid_lat, mid_lon, pt_lats, pt_lons)
        
        # Select k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_dists = np.maximum(distances[k_indices], 10.0)  # clamp to 10m minimum to avoid division by 0
        weights = 1.0 / (k_dists ** power)
        weight_sum = np.sum(weights)

        # IDW interpolation for Leq and Lmax
        interp_leq = round(float(np.sum(weights * pt_leqs[k_indices]) / weight_sum), 1)
        interp_lmax = round(float(np.sum(weights * pt_lmaxs[k_indices]) / weight_sum), 1)
        
        # Most frequent dominant class among nearest neighbors
        k_classes = list(pt_classes[k_indices])
        dominant_class = max(set(k_classes), key=k_classes.count)

        # Compute sleep score
        score_res = compute_sleep_disruption_index(
            interp_leq, interp_lmax, zone=st["zone"], timestamp_hour=eval_hour
        )

        # Flip coords to GeoJSON format: [lon, lat]
        geojson_line = [[round(c[1], 6), round(c[0], 6)] for c in st["coords"]]

        feat = {
            "type": "Feature",
            "properties": {
                "street_id": st["id"],
                "street_name": st["name"],
                "zone_type": st["zone"],
                "L_eq_db": interp_leq,
                "L_max_db": interp_lmax,
                "dominant_noise_source": dominant_class,
                "sleep_disruption_score": score_res["score"],
                "risk_category": score_res["category"],
                "color_code": score_res["color"],
                "is_violation": score_res["is_violation"],
                "statutory_limit_db": score_res["statutory_limit"],
                "delta_db": score_res["delta_db"]
            },
            "geometry": {
                "type": "LineString",
                "coordinates": geojson_line
            }
        }
        scored_features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": scored_features
    }


def export_geospatial_layers(base_dir="."):
    """
    Generates and saves:
    - data/silence_zones.geojson
    - data/decisense_scored_streets.geojson
    """
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Export 100m Silence Zone Buffers
    silence_geojson = build_silence_buffers()
    silence_path = os.path.join(data_dir, "silence_zones.geojson")
    with open(silence_path, "w", encoding="utf-8") as f:
        json.dump(silence_geojson, f, indent=2)
    print(f"  [+] Saved Silence Zone Buffers: {silence_path}")

    # 2. Ingest sensor points and perform IDW
    csv_path = os.path.join(data_dir, "chennai_audio_logs.csv")
    if os.path.exists(csv_path):
        audio_df = pd.read_csv(csv_path)
    else:
        # Fallback dummy points if CSV not yet created
        audio_df = pd.DataFrame([{
            "latitude": 13.0478, "longitude": 80.0245,
            "L_eq_db": 74.0, "L_max_db": 82.0, "ground_truth_class": "vehicular_honking"
        }])

    scored_streets = run_idw_interpolation(audio_df, is_night=True)
    streets_path = os.path.join(data_dir, "decisense_scored_streets.geojson")
    with open(streets_path, "w", encoding="utf-8") as f:
        json.dump(scored_streets, f, indent=2)
    print(f"  [+] Saved Scored Street Heatmap: {streets_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export_geospatial_layers(current_dir)
