# This script loads the final master dataset, scales the S5P NO2 values
# for better readability, and calculates summary statistics (Min, Max, Mean)
# for the selected environmental and traffic features.

import pandas as pd
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# Define data directory
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"

# Input file
file_path = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# =========================================================
# 2. Load and Transform Data
# =========================================================
print("Loading dataset for statistical summary...")
df = pd.read_csv(file_path, sep=";", decimal=",")

# Multiply S5P NO2 by 10^6 to convert it to a more meaningful unit (µmol/m²)
if 'S5P_NO2' in df.columns:
    df['S5P_NO2'] = df['S5P_NO2'] * 1e6
    df.rename(columns={'S5P_NO2': 'S5P_NO2 (µmol/m²)'}, inplace=True)

# =========================================================
# 3. Select Features to Analyze
# =========================================================
features_to_check = [
    'icon_temp_2m',
    'S5P_NO2 (µmol/m²)',
    'icon_relative_humidity',
    'S5P_UVAI',
    'IDW_car',
    'ERA5_BLH',
    'icon_wind_speed',
    'icon_wind_direction',
    'MAIAC_AOD',
    'IDW_truck',
    'tpi_300m',
    'tpi_1500m',
    'icon_precip_12h'
]

# Safety check: Only keep features that actually exist in the DataFrame
available_features = [f for f in features_to_check if f in df.columns]

if not available_features:
    print("WARNING: None of the specified features were found in the dataset.")
    exit()

# =========================================================
# 4. Calculate Statistics (Min, Max, Mean)
# =========================================================
# .describe() computes multiple stats; we filter for min, max, and mean
summary_df = df[available_features].describe().loc[['min', 'max', 'mean']]

# Transpose (.T) flips the table to make it easier to read in the terminal
summary_df = summary_df.T

# Round the results for a cleaner look
summary_df = summary_df.round(4)

print("\n--- Value Ranges in the Entire Summer Dataset ---")
print(summary_df.to_string())