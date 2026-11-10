# This script extracts Sentinel-5P (NO2, UVAI) and ERA5 (Boundary Layer Height)
# satellite/reanalysis data from daily GeoTIFFs and appends them to the
# master feature dataset. It also cleans anomalous 0.0 values in NO2 data.

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Define data directories
RAW_DIR = BASE_DIR.parent / "data" / "raw"
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input files
master_file = PROCESSED_DIR / "Features_MAIAC_Traffic_Weather.csv"
raster_base_dir = RAW_DIR

# Output file (The ultimate, complete master file!)
output_file = PROCESSED_DIR / "Features_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping extraction process.")
    exit()

# =========================================================
# 2. Load Master Dataset
# =========================================================
print("Loading previous master dataset...")
df_master = pd.read_csv(master_file, sep=";", decimal=",")

# Ensure UTM coordinates are numeric floats
for col in ["x_utm", "y_utm"]:
    if df_master[col].dtype == object:
        df_master[col] = df_master[col].str.replace(",", ".", regex=False)
    df_master[col] = pd.to_numeric(df_master[col], errors="coerce")

# =========================================================
# 3. Configure Raster Parameters
# =========================================================
# Dictionary for the three new feature folders
raster_mappings = {
    "S5P_NO2": {"folder": "S5P_NO2", "prefix": "S5P_NO2_"},
    "S5P_UVAI": {"folder": "S5P_UVAI", "prefix": "S5P_UVAI_"},
    "ERA5_BLH": {"folder": "ERA5_BLH", "prefix": "ERA5_BLH_"}
}

# Initialize new columns with NaN
for col in raster_mappings.keys():
    df_master[col] = np.nan

# =========================================================
# 4. Extract Data from GeoTIFFs
# =========================================================
print("Extracting S5P and ERA5 satellite data...")

unique_dates = df_master["date"].dropna().unique()

for current_date in unique_dates:
    # Filter rows for the current day
    mask = df_master["date"] == current_date
    day_points = df_master.loc[mask, ["x_utm", "y_utm"]].values

    # Skip if no valid coordinates exist for this day
    if len(day_points) == 0 or np.isnan(day_points).all():
        continue

    # Format as list of tuples: [(x1, y1), (x2, y2), ...]
    coords_list = [(x, y) for x, y in day_points]

    # Extract values for all 3 rasters (NO2, UVAI, BLH)
    for col_name, mapping in raster_mappings.items():
        tif_filename = f"{mapping['prefix']}{current_date}.tif"
        tif_path = raster_base_dir / mapping["folder"] / tif_filename

        if tif_path.exists():
            with rasterio.open(tif_path) as src:
                nodata_val = src.nodata

                sampled_values = []
                for val in src.sample(coords_list):
                    v = val[0]

                    # SAFETY FILTER:
                    # Exclude official NoData values or unnatural extremes (> 1 billion)
                    if v == nodata_val or v > 1e9 or v < -1e9:
                        sampled_values.append(np.nan)
                    else:
                        sampled_values.append(v)

                # Assign extracted values to the dataset
                df_master.loc[mask, col_name] = sampled_values
        else:
            # Optional: Add feedback here if you want to track missing days/clouds
            pass

print("Extraction complete.")

# =========================================================
# 5. Data Cleaning
# =========================================================
# Replace exact 0.0 values in S5P NO2 with real missing values (NaN)
df_master.loc[df_master['S5P_NO2'] == 0.0, 'S5P_NO2'] = np.nan

# =========================================================
# 6. Save Final Dataset
# =========================================================
# Helper columns x_utm and y_utm can be dropped if they are no longer needed (optional)
# df_master = df_master.drop(columns=["x_utm", "y_utm"], errors="ignore")

df_master.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! Your complete machine learning dataset is located at:\n{output_file}")
print("-" * 60)