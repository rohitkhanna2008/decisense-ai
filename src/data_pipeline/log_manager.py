"""
DeciSense AI - Module: Dataset Log Manager & Validation API
Maintains data/chennai_audio_logs.csv schema, enforces Chennai geospatial bounding box,
and validates CPCB statutory zone types and acoustic classes.
"""

import os
import math
import argparse
from datetime import datetime
import pandas as pd

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
CSV_LOG_PATH = os.path.join(DEFAULT_DATA_DIR, "chennai_audio_logs.csv")

# Exact Schema Definition
SCHEMA_COLUMNS = [
    "clip_id",
    "wav_path",
    "device_model",
    "latitude",
    "longitude",
    "timestamp",
    "time_window",
    "zone_type",
    "L_eq_db",
    "L_max_db",
    "crest_factor_db",
    "ground_truth_class",
    "collector_name"
]

# Validation Constraints
VALID_ZONES = {"silence", "residential", "commercial", "industrial"}

VALID_CLASSES = {
    "vehicular_honking",
    "continuous_traffic",
    "construction_piling",
    "pa_loudspeaker",
    "ambient_baseline"
}

# Greater Chennai / Poonamallee-Kuthambakkam-Porur Corridor Bounding Box
LAT_MIN, LAT_MAX = 12.85, 13.20
LON_MIN, LON_MAX = 79.95, 80.32


# -----------------------------------------------------------------------------
# 1. Validation Logic
# -----------------------------------------------------------------------------
def infer_time_window(timestamp_str_or_dt):
    """
    Infers CPCB operational time window from timestamp:
      - 'Day: 06:00-22:00'
      - 'Night: 22:00-06:00'
    """
    if isinstance(timestamp_str_or_dt, datetime):
        dt = timestamp_str_or_dt
    elif timestamp_str_or_dt is None or (isinstance(timestamp_str_or_dt, float) and math.isnan(timestamp_str_or_dt)):
        dt = datetime.now()
    else:
        try:
            dt = pd.to_datetime(timestamp_str_or_dt, errors="coerce")
            if pd.isna(dt):
                dt = datetime.now()
        except Exception:
            dt = datetime.now()

    hour = dt.hour
    if 6 <= hour < 22:
        return "Day: 06:00-22:00"
    else:
        return "Night: 22:00-06:00"


def validate_coordinates(lat, lon):
    """Checks if latitude and longitude fall within the Chennai corridor bounding box."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return False, f"Coordinates must be numeric (received lat={lat}, lon={lon})."

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False, f"Coordinates must be finite numbers (received lat={lat}, lon={lon})."

    if not (LAT_MIN <= lat <= LAT_MAX):
        return False, f"Latitude {lat:.4f} is outside Chennai study bounds [{LAT_MIN}, {LAT_MAX}]."

    if not (LON_MIN <= lon <= LON_MAX):
        return False, f"Longitude {lon:.4f} is outside Chennai study bounds [{LON_MIN}, {LON_MAX}]."

    return True, "VALID"


def validate_log_entry(entry):
    """
    Validates a complete audio log entry against schema rules.
    
    Returns:
        tuple: (is_valid: bool, error_msg: str)
    """
    # Check required fields
    for col in ["latitude", "longitude", "zone_type", "ground_truth_class"]:
        if col not in entry or entry[col] is None:
            return False, f"Missing required field: {col}"

    # Check coordinates
    coords_ok, coords_msg = validate_coordinates(entry["latitude"], entry["longitude"])
    if not coords_ok:
        return False, coords_msg

    # Check zone_type
    zone = str(entry["zone_type"]).strip().lower()
    if zone not in VALID_ZONES:
        return False, f"Invalid zone_type '{zone}'. Must be one of: {sorted(list(VALID_ZONES))}."

    # Check ground_truth_class
    cname = str(entry["ground_truth_class"]).strip().lower()
    if cname not in VALID_CLASSES:
        return False, f"Invalid ground_truth_class '{cname}'. Must be one of: {sorted(list(VALID_CLASSES))}."

    return True, "VALID"


# -----------------------------------------------------------------------------
# 2. CSV Ingestion & Log Registration API
# -----------------------------------------------------------------------------
def get_next_clip_id(df, prefix="DS_CHN_"):
    """Generates the next sequential unique clip_id."""
    if df.empty or "clip_id" not in df.columns:
        return f"{prefix}001"

    existing_ids = df["clip_id"].dropna().astype(str).tolist()
    numeric_ids = []
    for cid in existing_ids:
        parts = cid.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            numeric_ids.append(int(parts[-1]))
        elif len(parts) >= 2 and parts[-1].isdigit():
            numeric_ids.append(int(parts[-1]))

    next_num = max(numeric_ids, default=0) + 1
    return f"{prefix}{next_num:03d}"


def load_audio_logs(csv_path=CSV_LOG_PATH):
    """Loads existing audio logs DataFrame with schema compliance."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, Exception):
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    # Automatically upgrade / migrate schema if older columns present
    modified = False
    if "time_window" not in df.columns and "timestamp" in df.columns:
        df["time_window"] = df["timestamp"].apply(infer_time_window)
        modified = True

    if "crest_factor_db" not in df.columns and "L_eq_db" in df.columns and "L_max_db" in df.columns:
        df["crest_factor_db"] = (df["L_max_db"] - df["L_eq_db"]).round(1)
        modified = True

    if "device_model" not in df.columns:
        df["device_model"] = "Generic Android SLM"
        modified = True

    if "collector_name" not in df.columns:
        df["collector_name"] = "Saipalaniappan T (Field Team)"
        modified = True

    # Reorder columns to match canonical schema
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
            modified = True

    df = df[SCHEMA_COLUMNS]

    if modified:
        try:
            df.to_csv(csv_path, index=False)
        except OSError:
            pass

    return df


def register_audio_log(
    wav_path,
    latitude,
    longitude,
    zone_type,
    ground_truth_class,
    L_eq_db,
    L_max_db,
    device_model="Field Smartphone",
    collector_name="Saipalaniappan T",
    timestamp=None,
    crest_factor_db=None,
    clip_id=None,
    csv_path=CSV_LOG_PATH
):
    """
    Validates and registers a new 5-second acoustic recording into chennai_audio_logs.csv.
    
    Returns:
        dict: The registered record dictionary.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = load_audio_logs(csv_path)

    # Format timestamp
    if timestamp is None:
        dt = datetime.now()
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(timestamp, datetime):
        dt = timestamp
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp_str = str(timestamp)
        try:
            dt = pd.to_datetime(timestamp_str, errors="coerce")
            if pd.isna(dt):
                dt = datetime.now()
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.now()

    time_win = infer_time_window(dt)

    # Calculate crest factor
    leq = round(float(L_eq_db), 1)
    lmax = round(float(L_max_db), 1)
    if not (math.isfinite(leq) and math.isfinite(lmax)):
        raise ValueError("L_eq_db and L_max_db must be finite numbers.")
    if crest_factor_db is None:
        crest = round(float(max(0.0, lmax - leq)), 1)
    else:
        crest = round(float(crest_factor_db), 1)

    # Generate clip_id if not provided
    if not clip_id:
        clip_id = get_next_clip_id(df, prefix="DS_CHN_")

    record = {
        "clip_id": clip_id,
        "wav_path": str(wav_path).replace("\\", "/"),
        "device_model": str(device_model).strip(),
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "timestamp": timestamp_str,
        "time_window": time_win,
        "zone_type": str(zone_type).strip().lower(),
        "L_eq_db": leq,
        "L_max_db": lmax,
        "crest_factor_db": crest,
        "ground_truth_class": str(ground_truth_class).strip().lower(),
        "collector_name": str(collector_name).strip()
    }

    # Validate
    is_valid, error_msg = validate_log_entry(record)
    if not is_valid:
        raise ValueError(f"Log validation failed: {error_msg}")

    # Append to DataFrame and save
    new_row = pd.DataFrame([record])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_csv(csv_path, index=False)

    return record


# -----------------------------------------------------------------------------
# 3. CLI Interface
# -----------------------------------------------------------------------------
def build_cli():
    parser = argparse.ArgumentParser(description="DeciSense AI Audio Log Manager CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: list
    subparsers.add_parser("list", help="List recent acoustic logs")

    # Command: register
    reg_parser = subparsers.add_parser("register", help="Register a new acoustic sample")
    reg_parser.add_argument("--wav", required=True, help="Path to 5s WAV file")
    reg_parser.add_argument("--lat", type=float, required=True, help="Latitude (12.85 to 13.20)")
    reg_parser.add_argument("--lon", type=float, required=True, help="Longitude (79.95 to 80.32)")
    reg_parser.add_argument("--zone", required=True, choices=list(VALID_ZONES), help="CPCB Zone Type")
    reg_parser.add_argument("--class-name", required=True, choices=list(VALID_CLASSES), help="Noise Class")
    reg_parser.add_argument("--leq", type=float, required=True, help="L_eq dB(A)")
    reg_parser.add_argument("--lmax", type=float, required=True, help="L_max dB(A)")
    reg_parser.add_argument("--device", default="OnePlus Field Phone", help="Smartphone Model")
    reg_parser.add_argument("--collector", default="Saipalaniappan T", help="Collector Name")

    return parser


if __name__ == "__main__":
    parser = build_cli()
    args = parser.parse_args()

    if args.command == "register":
        rec = register_audio_log(
            wav_path=args.wav,
            latitude=args.lat,
            longitude=args.lon,
            zone_type=args.zone,
            ground_truth_class=args.class_name,
            L_eq_db=args.leq,
            L_max_db=args.lmax,
            device_model=args.device,
            collector_name=args.collector
        )
        print(f"Successfully registered record {rec['clip_id']} to {CSV_LOG_PATH}")
    else:
        df_logs = load_audio_logs()
        print(f"Loaded {len(df_logs)} logs from {CSV_LOG_PATH}:")
        if not df_logs.empty:
            print(df_logs[["clip_id", "zone_type", "L_eq_db", "L_max_db", "ground_truth_class"]].tail(5))
