"""
DeciSense AI - APILayer Geolocation & Country Context Integration
Combines schemas and datasets from:
  - apilayer/geolocationapi (data/countries.json + /api/countries & /api/geolocate schema)
  - apilayer/freegeoip (responseRecord JSON/CSV schema)
"""

from functools import lru_cache
import json
import math
import os
import urllib.parse
import urllib.request

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)
APILAYER_COUNTRIES_PATH = os.path.join(DATA_DIR, "apilayer_countries.json")

# Default Chennai / RIT Corridor geolocation record in apilayer/freegeoip schema
DEFAULT_CHENNAI_FREEGEOIP = {
    "ip": "127.0.0.1",
    "country_code": "IN",
    "country_name": "India",
    "region_code": "TN",
    "region_name": "Tamil Nadu",
    "city": "Chennai",
    "zip_code": "600056",
    "time_zone": "Asia/Kolkata",
    "latitude": 13.0475,
    "longitude": 80.0250,
    "metro_code": 0,
}


@lru_cache(maxsize=2)
def load_apilayer_countries(filepath=APILAYER_COUNTRIES_PATH):
    """Loads the 249-country metadata dictionary from apilayer/geolocationapi."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def list_country_codes(filepath=APILAYER_COUNTRIES_PATH):
    """Returns sorted ISO alpha-2 country codes from apilayer/geolocationapi."""
    countries = load_apilayer_countries(filepath)
    return sorted(countries.keys())


def get_country_metadata(country_code="IN", filepath=APILAYER_COUNTRIES_PATH):
    """
    Returns country metadata matching apilayer/geolocationapi /api/countries/{country_code}.
    Falls back to minimal IN defaults if the JSON file is absent.
    """
    code = str(country_code or "IN").strip().upper()
    countries = load_apilayer_countries(filepath)
    if code in countries:
        return countries[code]
    if "IN" in countries:
        return countries["IN"]
    return {
        "alpha2": "IN",
        "alpha3": "IND",
        "name": "India",
        "continent": "Asia",
        "region": "Asia",
        "subregion": "Southern Asia",
        "world_region": "APAC",
        "country_code": "91",
        "currency_code": "INR",
        "languages_official": ["hi", "en"],
        "start_of_week": "monday",
        "geo": {
            "latitude": 20.593684,
            "longitude": 78.96288,
            "bounds": {
                "northeast": {"lat": 35.5087008, "lng": 97.39535869999999},
                "southwest": {"lat": 6.4626999, "lng": 68.1097},
            },
        },
    }


def validate_within_country_bounds(lat, lon, country_code="IN"):
    """
    Validates coordinates against apilayer/geolocationapi country geo.bounds.
    """
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False, "Coordinates must be numeric."

    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return False, "Coordinates must be finite."

    meta = get_country_metadata(country_code)
    bounds = (meta.get("geo") or {}).get("bounds") or {}
    sw = bounds.get("southwest") or {}
    ne = bounds.get("northeast") or {}

    if not all(k in sw and k in ne for k in ("lat", "lng")):
        return True, f"No bounding box defined for {meta.get('alpha2', country_code)}."

    min_lat, max_lat = float(sw["lat"]), float(ne["lat"])
    min_lon, max_lon = float(sw["lng"]), float(ne["lng"])
    inside = (min_lat <= lat_f <= max_lat) and (min_lon <= lon_f <= max_lon)
    name = meta.get("name", country_code)
    if inside:
        return True, f"Within {name} bounds ([{min_lat:.2f}, {max_lat:.2f}], [{min_lon:.2f}, {max_lon:.2f}])."
    return False, f"Outside {name} bounds ([{min_lat:.2f}, {max_lat:.2f}], [{min_lon:.2f}, {max_lon:.2f}])."


def build_freegeoip_record(
    ip="127.0.0.1",
    country_code="IN",
    country_name=None,
    region_code="",
    region_name="",
    city="",
    zip_code="",
    time_zone="",
    latitude=None,
    longitude=None,
    metro_code=0,
):
    """
    Builds a dict conforming to apilayer/freegeoip's responseRecord JSON schema.
    """
    meta = get_country_metadata(country_code)
    geo = meta.get("geo") or {}
    resolved_code = str(meta.get("alpha2") or country_code or "IN").upper()
    resolved_name = country_name or meta.get("name", "India")

    if resolved_code == "IN" and latitude is None and longitude is None:
        base = dict(DEFAULT_CHENNAI_FREEGEOIP)
        base["ip"] = str(ip or "127.0.0.1")
        return base

    lat_val = float(latitude if latitude is not None else geo.get("latitude", 0.0))
    lon_val = float(longitude if longitude is not None else geo.get("longitude", 0.0))
    return {
        "ip": str(ip or "127.0.0.1"),
        "country_code": resolved_code,
        "country_name": resolved_name,
        "region_code": str(region_code or ""),
        "region_name": str(region_name or ""),
        "city": str(city or ""),
        "zip_code": str(zip_code or ""),
        "time_zone": str(time_zone or ("Asia/Kolkata" if resolved_code == "IN" else "UTC")),
        "latitude": round(lat_val, 4),
        "longitude": round(lon_val, 4),
        "metro_code": int(metro_code or 0),
    }


def format_freegeoip_csv(record):
    """Formats a responseRecord in apilayer/freegeoip CSV field order."""
    cols = [
        "ip",
        "country_code",
        "country_name",
        "region_code",
        "region_name",
        "city",
        "zip_code",
        "time_zone",
        "latitude",
        "longitude",
        "metro_code",
    ]
    return ",".join(str(record.get(c, "")) for c in cols)


def lookup_apilayer_geolocation(
    ip_or_host="",
    country_code="IN",
    allow_network=False,
    timeout=2.0,
):
    """
    Resolves geolocation using an optional configured apilayer/freegeoip server
    (DECISENSE_FREEGEOIP_URL) and enriches with apilayer/geolocationapi country values.
    Always falls back cleanly to local apilayer/geolocationapi data.
    """
    freegeoip_record = None
    source = "apilayer/geolocationapi (local countries.json)"

    host = str(ip_or_host or "").strip()
    endpoint_base = os.environ.get("DECISENSE_FREEGEOIP_URL", "").strip().rstrip("/")

    if allow_network and host and endpoint_base:
        safe_host = urllib.parse.quote(host, safe=".:")
        url = f"{endpoint_base}/json/{safe_host}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("country_code"):
                freegeoip_record = build_freegeoip_record(
                    ip=payload.get("ip", host),
                    country_code=payload.get("country_code", country_code),
                    country_name=payload.get("country_name"),
                    region_code=payload.get("region_code", ""),
                    region_name=payload.get("region_name", ""),
                    city=payload.get("city", ""),
                    zip_code=payload.get("zip_code", ""),
                    time_zone=payload.get("time_zone", ""),
                    latitude=payload.get("latitude"),
                    longitude=payload.get("longitude"),
                    metro_code=payload.get("metro_code", 0),
                )
                source = f"apilayer/freegeoip ({endpoint_base})"
        except Exception:
            freegeoip_record = None

    if freegeoip_record is None:
        freegeoip_record = build_freegeoip_record(
            ip=host or "127.0.0.1",
            country_code=country_code,
        )

    country_meta = get_country_metadata(freegeoip_record["country_code"])
    geo = country_meta.get("geo") or {}
    bounds = geo.get("bounds") or {}
    within_bounds, bounds_msg = validate_within_country_bounds(
        freegeoip_record["latitude"],
        freegeoip_record["longitude"],
        freegeoip_record["country_code"],
    )

    return {
        "source": source,
        "freegeoip": freegeoip_record,
        "freegeoip_csv": format_freegeoip_csv(freegeoip_record),
        "country": {
            "alpha2": country_meta.get("alpha2", freegeoip_record["country_code"]),
            "alpha3": country_meta.get("alpha3", ""),
            "name": country_meta.get("name", freegeoip_record["country_name"]),
            "continent": country_meta.get("continent", ""),
            "region": country_meta.get("region", ""),
            "subregion": country_meta.get("subregion", ""),
            "world_region": country_meta.get("world_region", ""),
            "calling_code": f"+{country_meta['country_code']}" if country_meta.get("country_code") else "",
            "currency_code": country_meta.get("currency_code", ""),
            "languages_official": country_meta.get("languages_official", []),
            "start_of_week": country_meta.get("start_of_week", ""),
            "centroid_lat": geo.get("latitude"),
            "centroid_lon": geo.get("longitude"),
            "bounds": bounds,
        },
        "within_country_bounds": within_bounds,
        "bounds_message": bounds_msg,
    }
