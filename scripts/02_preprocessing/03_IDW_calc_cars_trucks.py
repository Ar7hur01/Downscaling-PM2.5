# This script extracts traffic features for the 13:00 (1 PM) timestamp.
# It calculates both the cumulative Inverse Distance Weighting (IDW) sum
# and the maximum single-station IDW value for cars and trucks.
# It exports a standalone feature table for later merging into the ML dataset.

import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# Define data directories
RAW_DIR = BASE_DIR.parent / "data" / "raw"
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input files (Adjust subfolders if necessary)
traffic_file = RAW_DIR / "Traffic" / "Verkehrsdaten_mit_Koordinaten_merged_cars_n_trucks.csv"
sensor_raw_file = RAW_DIR / "sensor_community_complete_may_august.csv"

# Output file
output_file = PROCESSED_DIR / "13Uhr_Verkehrs_Proxys_Komplett.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping calculation.")
    exit()

def calculate_distance_matrix_m(lat1, lon1, lat2_series, lon2_series):
    """Calculates the Haversine distance in meters from one point to a series of coordinates."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_series)
    dphi = np.radians(lat2_series - lat1)
    dlambda = np.radians(lon2_series - lon1)

    a = (np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    # Fixed syntax errors from copy/paste
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# =========================================================
# 2. Load & Clean Traffic Data (13:00 / 1 PM)
# =========================================================
print("Loading and cleaning traffic data (13:00)...")
df_traffic = pd.read_csv(traffic_file, sep=";", decimal=",")

df_traffic["Time"] = pd.to_datetime(df_traffic["Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
df_traffic_13 = df_traffic[df_traffic["Time"].dt.hour == 13].copy()
df_traffic_13["date"] = df_traffic_13["Time"].dt.strftime("%Y-%m-%d")

for col in ["Lat", "Lon", "processed_car_vol", "processed_truck_vol"]:
    df_traffic_13[col] = df_traffic_13[col].astype(str).str.replace(",", ".", regex=False)
    df_traffic_13[col] = pd.to_numeric(df_traffic_13[col], errors="coerce")

df_traffic_13 = df_traffic_13.dropna(subset=["Lat", "Lon", "processed_car_vol"])
print(f"-> Loaded {len(df_traffic_13)} traffic observations at 13:00.")

# =========================================================
# 3. Extract Sensor Coordinates
# =========================================================
print("Extracting sensor coordinates...")
df_sc = pd.read_csv(sensor_raw_file, sep=";", decimal=",")

for col in ["lat", "lon"]:
    df_sc[col] = df_sc[col].astype(str).str.replace(",", ".", regex=False)
    df_sc[col] = pd.to_numeric(df_sc[col], errors="coerce")

sensor_coords = df_sc[["sensor_id", "lat", "lon"]].drop_duplicates(subset=["sensor_id"]).dropna()
sensor_coords["station_id"] = sensor_coords["sensor_id"].astype(float).astype(int).astype(str)
sensor_coords = sensor_coords[["station_id", "lat", "lon"]]

# Ensure Kopfklinik (Station ID "1") is included as an anchor
kopfklinik_entry = pd.DataFrame([{
    "station_id": "1",
    "lat": 49.8047,
    "lon": 9.95642,
}])
all_sensors = pd.concat([sensor_coords, kopfklinik_entry], ignore_index=True).drop_duplicates(subset=["station_id"])
print(f"-> Locked {len(all_sensors)} sensor locations.")

# =========================================================
# 4. Calculate Proxies (Cumulative & Max IDW)
# =========================================================
print("Calculating proxy values for all days and stations...")
cumulative_results = []

unique_dates = df_traffic_13["date"].unique()

for current_date in unique_dates:
    daily_traffic = df_traffic_13[df_traffic_13["date"] == current_date]

    for _, sensor in all_sensors.iterrows():
        s_id = sensor["station_id"]
        distances = calculate_distance_matrix_m(
            sensor["lat"], sensor["lon"], daily_traffic["Lat"], daily_traffic["Lon"]
        )

        # Calculate IDW series
        idw_car_series = daily_traffic["processed_car_vol"] / (distances + 1.0)
        idw_truck_series = daily_traffic["processed_truck_vol"].fillna(0.0) / (distances + 1.0)

        cumulative_results.append({
            "date": current_date,
            "station_id": s_id,
            "prox_kumulativ_car": round(idw_car_series.sum(), 4),
            "prox_kumulativ_truck": round(idw_truck_series.sum(), 4),
            "IDW_car": round(idw_car_series.max(), 4),
            "IDW_truck": round(idw_truck_series.max(), 4),
        })

df_proxys_final = pd.DataFrame(cumulative_results)
print(f"-> Calculated {len(df_proxys_final)} feature combinations.")

# =========================================================
# 5. Export Standalone Feature Dataset
# =========================================================
df_proxys_final.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! The standalone traffic feature table is located at:\n{output_file}")
print("-" * 60)
print(df_proxys_final.head(10))