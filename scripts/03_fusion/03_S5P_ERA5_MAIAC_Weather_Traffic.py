# This script merges the Sentinel-5P (NO2, UVAI) and ERA5 (Boundary Layer Height)
# data directly from the Google Earth Engine CSV export into the master feature dataset.

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

# Input files
master_file = PROCESSED_DIR / "Features_MAIAC_Traffic_Weather.csv"

## This file needs to be downloaded from Google Drive and moved into the "data/raw"-folder.
gee_export_file = RAW_DIR / "cloud0_5_Wuerzburg_S5P_ERA5_Extraction_2024.csv"

# Output file
output_file = PROCESSED_DIR / "Features_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping extraction process.")
    exit()

# =========================================================
# 2. Load Datasets
# =========================================================
print("Loading previous master dataset...")
df_master = pd.read_csv(master_file, sep=";", decimal=",")

print("Loading Google Earth Engine CSV export...")
if not gee_export_file.exists():
    print(f"ERROR: Earth Engine CSV not found at {gee_export_file}")
    exit()

df_gee = pd.read_csv(gee_export_file)

# =========================================================
# 3. Clean and Prepare Station IDs for Merging
# =========================================================
print("Preparing data for merge...")

# Ensure station_id in master is a clean string
df_master['station_id'] = df_master['station_id'].astype(str).str.strip()

# In the GEE CSV, station_id might be a float (e.g., 5778.0). We need to make it a string (e.g., '5778')
df_gee['station_id'] = pd.to_numeric(df_gee['station_id'], errors='coerce')
# Convert to integer (ignores NaNs) then to string
df_gee['station_id'] = df_gee['station_id'].astype('Int64').astype(str).str.strip()

# =========================================================
# 4. Rename GEE Columns to match our pipeline
# =========================================================
rename_mapping = {
    'absorbing_aerosol_index': 'S5P_UVAI',
    'boundary_layer_height': 'ERA5_BLH',
    'tropospheric_NO2_column_number_density': 'S5P_NO2'
}
df_gee = df_gee.rename(columns=rename_mapping)

# Select only the columns we actually need to merge
gee_columns_to_keep = ['date', 'station_id', 'S5P_UVAI', 'ERA5_BLH', 'S5P_NO2']
df_gee_subset = df_gee[gee_columns_to_keep]

# Remove possible duplicates in GEE export (same station and date)
df_gee_subset = df_gee_subset.drop_duplicates(subset=['date', 'station_id'])

# =========================================================
# 5. Merge Data
# =========================================================
print("Merging Satellite and Reanalysis data...")

# Left merge: Keeps all rows from df_master, attaches matching data from df_gee
df_merged = pd.merge(
    df_master,
    df_gee_subset,
    on=['date', 'station_id'],
    how='left'
)

# =========================================================
# 6. Data Cleaning
# =========================================================
# Replace exact 0.0 values in S5P NO2 with real missing values (NaN)
if 'S5P_NO2' in df_merged.columns:
    df_merged.loc[df_merged['S5P_NO2'] == 0.0, 'S5P_NO2'] = np.nan

# =========================================================
# 7. Save Final Dataset
# =========================================================
df_merged.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! Your complete machine learning dataset is located at:\n{output_file}")
print("-" * 60)