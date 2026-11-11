# This script filters PM2.5 particulate matter measurements for the midday
# satellite overpass window (12:30 - 14:30), calculates the daily mean PM2.5 target
# per station, and merges it with the master feature dataset for machine learning.

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
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input files
sensor_file = PROCESSED_DIR / "sensor_community_with_kopfklinik_may_august.csv"
master_file = PROCESSED_DIR / "Features_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# Output file - The final training-ready Machine Learning dataset
output_file = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping target generation process.")
    exit()

# =========================================================
# 2. Load and Time-Filter Particulate Matter Data
# =========================================================
print("Loading fine dust measurement data...")
df_sensors = pd.read_csv(sensor_file, sep=";", decimal=",")

# Unify timestamp column name
if 'Zeitstempel' in df_sensors.columns:
    df_sensors.rename(columns={'Zeitstempel': 'timestamp'}, inplace=True)

df_sensors['timestamp'] = pd.to_datetime(df_sensors['timestamp'], errors='coerce')

# Convert PM2.5 to numeric floats
df_sensors['PM2.5'] = df_sensors['PM2.5'].astype(str).str.replace(",", ".", regex=False)
df_sensors['PM2.5'] = pd.to_numeric(df_sensors['PM2.5'], errors='coerce')

# Extract date and station_id for matching
df_sensors['date'] = df_sensors['timestamp'].dt.strftime('%Y-%m-%d')

if 'station_id' not in df_sensors.columns and 'sensor_id' in df_sensors.columns:
    df_sensors['station_id'] = df_sensors['sensor_id']

df_sensors['station_id'] = df_sensors['station_id'].astype(str).str.strip()

# --- SATELLITE OVERPASS TIME FILTER (12:30 to 14:30) ---
print("Filtering for the satellite overpass window (12:30 - 14:30)...")
time_series = df_sensors['timestamp'].dt.time
start_time = pd.to_datetime('12:30:00').time()
end_time = pd.to_datetime('14:30:00').time()

# Keep only rows within the 2-hour window
df_filtered = df_sensors[(time_series >= start_time) & (time_series <= end_time)].copy()

# =========================================================
# 3. Aggregation (Midday Mean per Station and Day)
# =========================================================
print("Calculating midday PM2.5 mean per station and day...")
daily_pm25 = df_filtered.groupby(['date', 'station_id'], as_index=False)['PM2.5'].mean()

# Rename to target variable
daily_pm25.rename(columns={'PM2.5': 'PM25_Target'}, inplace=True)
daily_pm25 = daily_pm25.dropna(subset=['PM25_Target'])

print(f"-> Generated {len(daily_pm25)} valid (Date x Station) target combinations.")

# =========================================================
# 4. Merge with Master Feature File
# =========================================================
print("Merging PM2.5 Target into the Master Feature Dataset...")
df_features = pd.read_csv(master_file, sep=";", decimal=",")
df_features['station_id'] = df_features['station_id'].astype(str).str.strip()

# Inner merge: keeps only records that have BOTH features AND target measurements
df_final = pd.merge(
    df_features,
    daily_pm25,
    on=['date', 'station_id'],
    how='inner'
)

# Convert remaining 0.0 values in S5P_NO2 to NaN if present
if 'S5P_NO2' in df_final.columns:
    df_final.loc[df_final['S5P_NO2'] == 0.0, 'S5P_NO2'] = np.nan

# =========================================================
# 5. Save Final Dataset
# =========================================================
df_final.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"MISSION ACCOMPLISHED! ML dataset ready for training:\n{output_file}")
print("-" * 60)