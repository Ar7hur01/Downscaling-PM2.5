import io
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests as rq

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Output folder relative to script location
OUTPUT_DIR = BASE_DIR.parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ICON-D2 Extent
LON_MIN, LON_MAX = 9.85, 10.05
LAT_MIN, LAT_MAX = 49.74, 49.85


# =========================================================
# 2. Helper: Fetch Sensors in Bounding Box
# =========================================================
def get_sensors_in_extent(lon_min, lon_max, lat_min, lat_max):
    """
    Queries Sensor.Community API for active SDS011 fine dust sensors
    and filters them using a coordinate bounding box.
    """
    url = "https://data.sensor.community/static/v2/data.json"

    try:
        response = rq.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to fetch sensor location index: {e}")
        return pd.DataFrame(columns=["sensor_id", "lon", "lat"])

    sensor_list = []
    for entry in data:
        sensor_info = entry.get("sensor", {})
        sensor_type = sensor_info.get("sensor_type", {}).get("name", "")

        # Filter for SDS011 particulate matter sensors
        if "SDS011" in sensor_type:
            loc = entry.get("location", {})
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            sensor_id = sensor_info.get("id")

            if lat and lon and sensor_id:
                sensor_list.append({
                    "sensor_id": int(sensor_id),
                    "lat": float(lat),
                    "lon": float(lon)
                })

    if not sensor_list:
        return pd.DataFrame(columns=["sensor_id", "lon", "lat"])

    df = pd.DataFrame(sensor_list).drop_duplicates(subset=["sensor_id"])

    # Filter by bounding box
    mask = (
            (df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
            (df["lat"] >= lat_min) & (df["lat"] <= lat_max)
    )

    return df[mask].copy()


# =========================================================
# 3. Download & Process Sensor Data
# =========================================================
def download_bbox_raw_data(start_date, end_date):
    print("Fetching active SDS011 sensors within the specified bounding box...")
    sensors_df = get_sensors_in_extent(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    if sensors_df.empty:
        print("No active SDS011 sensors found in the bounding box.")
        return None

    print(f"Found {len(sensors_df)} SDS011 sensors in the bounding box.")

    location_map = sensors_df[['sensor_id', 'lon', 'lat']].copy()
    bbox_ids = sensors_df['sensor_id'].tolist()
    all_data = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        year_str = current_date.strftime("%Y")

        for s_id in bbox_ids:
            file_url = f"https://archive.sensor.community/{year_str}/{date_str}/{date_str}_sds011_sensor_{s_id}.csv.gz"
            try:
                res = rq.get(file_url, timeout=5)
                if res.status_code == 200:
                    df = pd.read_csv(io.BytesIO(res.content), sep=";", compression='gzip')
                    df = df[['timestamp', 'sensor_id', 'P1', 'P2']]
                    all_data.append(df)
            except Exception:
                continue

        print(f"Processed date: {date_str}")
        current_date += timedelta(days=1)

    if not all_data:
        print("No archive data downloaded.")
        return None

    full_df = pd.concat(all_data, ignore_index=True)

    # Convert timestamp to UTC and adjust to German local time
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df['timestamp'] = full_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    full_df['timestamp'] = full_df['timestamp'].dt.tz_localize(None)

    final_df = pd.merge(full_df, location_map, on='sensor_id', how='left')
    final_df = final_df.rename(columns={
        'timestamp': 'timestamp',
        'P1': 'PM10',
        'P2': 'PM2.5'
    })

    final_df = final_df.sort_values(by=['sensor_id', 'timestamp'])
    return final_df[['timestamp', 'sensor_id', 'lon', 'lat', 'PM2.5', 'PM10']]


# =========================================================
# 4. Execution
# =========================================================
start = datetime(2024, 5, 1)
end = datetime(2024, 8, 31)

result = download_bbox_raw_data(start, end)

if result is not None:
    output_file = OUTPUT_DIR / f"sensor_community_complete_may_august.csv".lower()
    result.to_csv(output_file, index=False, sep=";", decimal=",", encoding='utf-8-sig')
    print(f"\nSuccessfully saved {len(result)} records to:\n{output_file}")