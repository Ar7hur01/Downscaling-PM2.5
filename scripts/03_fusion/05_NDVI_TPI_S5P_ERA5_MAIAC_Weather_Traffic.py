# This script extracts Normalized Difference Vegetation Index (NDVI) values
# from a Sentinel-2 GeoTIFF and appends them to the final master dataset.

import numpy as np
import pandas as pd
import rasterio
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
master_file = PROCESSED_DIR / "Features_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# NDVI Raster (Placed in the specific NDVI folder)
NDVI_DIR = RAW_DIR / "NDVI"
ndvi_path = NDVI_DIR / "50m_AOI_NDVI_S2_11082024.tif"

# Output file - Your ultimate, final feature master file!
output_file = PROCESSED_DIR / "Features_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping extraction process.")
    exit()

# =========================================================
# 2. Load Master Dataset
# =========================================================
print("Loading master dataset...")
df_master = pd.read_csv(master_file, sep=";", decimal=",")

# Ensure UTM coordinates are numeric floats
for col in ["x_utm", "y_utm"]:
    if df_master[col].dtype == object:
        df_master[col] = df_master[col].str.replace(",", ".", regex=False)
    df_master[col] = pd.to_numeric(df_master[col], errors="coerce")

# Initialize new NDVI column with NaN
df_master["ndvi"] = np.nan

# =========================================================
# 3. Extract NDVI Data
# =========================================================
if not ndvi_path.exists():
    print(f"WARNING: NDVI file not found -> {ndvi_path}")
else:
    print(f"Extracting NDVI values from {ndvi_path.name}...")

    # Filter only rows that have valid coordinates
    valid_mask = df_master["x_utm"].notna() & df_master["y_utm"].notna()

    # Create a list of (x, y) coordinate tuples
    coords_list = list(zip(df_master.loc[valid_mask, "x_utm"], df_master.loc[valid_mask, "y_utm"]))

    if len(coords_list) > 0:
        with rasterio.open(ndvi_path) as src:
            nodata_val = src.nodata
            sampled_values = []

            # Sample values at coordinates
            for val in src.sample(coords_list):
                v = val[0]
                # Filter out NoData and unnatural extremes
                # (NDVI is normally between -1 and 1, but sometimes scaled to +/- 10000)
                if v == nodata_val or v > 1e6 or v < -1e6:
                    sampled_values.append(np.nan)
                else:
                    sampled_values.append(v)

            # Assign values exactly to the valid rows
            df_master.loc[valid_mask, "ndvi"] = sampled_values

# =========================================================
# 4. Save Final Dataset
# =========================================================
df_master.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! Your dataset is now absolutely complete (incl. NDVI) and saved to:\n{output_file}")
print("-" * 60)