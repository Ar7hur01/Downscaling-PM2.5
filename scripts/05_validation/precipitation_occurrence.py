# This script analyzes the availability of satellite data (MAIAC AOD and S5P NO2)
# during rainy days to quantify how often cloud cover blocks satellite observations
# when precipitation occurs.

import pandas as pd
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Define data directory
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"

# Input file (Final master dataset)
file_path = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# =========================================================
# 2. Load Data
# =========================================================
print("Loading master dataset for rain/cloud analysis...")
if not file_path.exists():
    print(f"ERROR: File not found at {file_path}")
    exit()

df = pd.read_csv(file_path, sep=";", decimal=",")

# =========================================================
# 3. Filter and Analyze
# =========================================================
# Filter all rows where it rained (precipitation > 0)
df_rain = df[df["icon_precip_12h"] > 0]
total_rain_days = len(df_rain)

# Count how often satellite data is available during these rainy days.
# We check if values are NOT missing (notna) and NOT exactly 0.
valid_sat_during_rain = df_rain[
    (df_rain["MAIAC_AOD"].notna()) & (df_rain["MAIAC_AOD"] != 0) &
    (df_rain["S5P_NO2"].notna()) & (df_rain["S5P_NO2"] != 0)
]
count_valid = len(valid_sat_during_rain)
count_missing = total_rain_days - count_valid

# =========================================================
# 4. Output Results
# =========================================================
print("\n--- Satellite Observation Availability During Rain ---")
print(f"Total observations with rain (> 0 mm):               {total_rain_days}")
print(f"Of those, WITH available satellite data (AOD & NO2): {count_valid}")
print(f"Of those, WITHOUT satellite data (Cloud blockage):   {count_missing}")
print("-" * 60)