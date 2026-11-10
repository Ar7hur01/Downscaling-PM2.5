# This script merges Sensor.Community data with official station data (Kopfklinik).
# It fixes a previous issue where an "inner" merge caused missing data in May.
# We now use an "outer" merge to retain all available measurements.

import pandas as pd
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Define data directories relative to the script
RAW_DIR = BASE_DIR.parent / "data" / "raw"
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input files
path_sc = RAW_DIR / "sensor_community_complete_may_august.csv"
# Note: Ensure both Kopfklinik files are placed in a 'Kopfklinik' subfolder inside 'data/raw'
path_kk_pm25 = RAW_DIR / "Kopfklinik" / "PM2.5_Wuerzburg_2024.csv"
path_kk_pm10 = RAW_DIR / "Kopfklinik" / "Kopfklinik_PM10_2024.xlsx"

# Output file
path_out = PROCESSED_DIR / "sensor_community_with_kopfklinik_may_august.csv"

# STRATEGY 1: Skip if already processed
if path_out.exists():
    print(f"File {path_out.name} already exists. Skipping processing.")
else:
    print("Loading datasets...")

    # =========================================================
    # 2. Load Sensor.Community Data
    # =========================================================
    # Use semicolon separator and comma decimal
    df_sc = pd.read_csv(path_sc, sep=";", decimal=",")

    # Unify column name (handles both 'Zeitstempel' and 'timestamp')
    if 'Zeitstempel' in df_sc.columns:
        df_sc.rename(columns={'Zeitstempel': 'timestamp'}, inplace=True)

    # Parse timestamps safely
    df_sc['timestamp'] = pd.to_datetime(df_sc['timestamp'], errors='coerce')

    # Safely convert coordinates to float
    for col in ['lon', 'lat']:
        df_sc[col] = df_sc[col].astype(str).str.replace(',', '.', regex=False)
        df_sc[col] = pd.to_numeric(df_sc[col], errors='coerce')

    # =========================================================
    # 3. Load & Prepare Kopfklinik Data
    # =========================================================
    # Load PM2.5 CSV
    df_kk_pm25 = pd.read_csv(path_kk_pm25, sep=";", decimal=".")
    df_kk_pm25['Zeitpunkt'] = pd.to_datetime(df_kk_pm25['Zeitpunkt'], format='%d.%m.%Y %H:%M', errors='coerce')

    # Load PM10 Excel
    df_kk_pm10 = pd.read_excel(path_kk_pm10)
    df_kk_pm10.rename(columns={'Würzburg/Kopfklinik': 'PM10'}, inplace=True)
    df_kk_pm10['Zeitpunkt'] = pd.to_datetime(df_kk_pm10['Zeitpunkt'], errors='coerce')

    # Merge Kopfklinik data on time
    # BUGFIX: Changed to "outer" merge to prevent dropping data if one parameter is missing
    df_kk = pd.merge(df_kk_pm25, df_kk_pm10, on="Zeitpunkt", how="outer")

    # =========================================================
    # 4. Filter Kopfklinik and Match SC Structure
    # =========================================================
    print("Filtering Kopfklinik data (May - August 2024) and formatting columns...")

    # Filter exact time range
    mask = (df_kk['Zeitpunkt'] >= '2024-05-01') & (df_kk['Zeitpunkt'] <= '2024-08-31 23:59:59')
    df_kk = df_kk.loc[mask].copy()

    # Rename columns to match SC structure
    df_kk.rename(columns={
        'Zeitpunkt': 'timestamp',
        'Lon': 'lon',
        'Lat': 'lat'
    }, inplace=True)

    # Assign official ID "1"
    df_kk['sensor_id'] = "1"

    # Keep only relevant columns in exact order
    df_kk = df_kk[['timestamp', 'sensor_id', 'lon', 'lat', 'PM2.5', 'PM10']]

    # =========================================================
    # 5. Concatenate Datasets
    # =========================================================
    print("Merging Sensor.Community and Kopfklinik data...")
    df_final = pd.concat([df_sc, df_kk], ignore_index=True)

    # Sort cleanly by sensor ID and timestamp
    df_final.sort_values(by=['sensor_id', 'timestamp'], inplace=True)

    # Revert timestamp to clean string format 'YYYY-MM-DD HH:MM:SS'
    df_final['timestamp'] = df_final['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV (using comma as decimal, matching SC original format)
    df_final.to_csv(path_out, sep=";", decimal=",", index=False)

    print("-" * 60)
    print(f"SUCCESS! Combined dataset saved to:\n{path_out}")
    print("-" * 60)