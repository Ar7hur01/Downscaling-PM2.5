# The two TPI files (300m and 1500m) are assigned to the sensor coordinates.
# Note: The Area of Interest (AOI) of the TPI files / DOM is the smallest
# in the dataset. The final Random Forest algorithm will be trained based
# on this specific extent.

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
master_file = PROCESSED_DIR / "Features_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# TPI Rasters
DOM_DIR = RAW_DIR / "TPI"
tpi_300_path = DOM_DIR / "wuerzburg_tpi_300m.tif"
tpi_1500_path = DOM_DIR / "wuerzburg_tpi_1500m.tif"

# Output file
output_file = PROCESSED_DIR / "Features_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

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

# Initialize new columns with NaN
df_master["tpi_300m"] = np.nan
df_master["tpi_1500m"] = np.nan

# =========================================================
# 3. Function to Extract Static Rasters Safely
# =========================================================
def extract_static_raster(raster_path, col_name):
    if not raster_path.exists():
        print(f"WARNING: File not found -> {raster_path}")
        return

    print(f"Extracting data from {raster_path.name}...")

    # Filter only rows that have valid coordinates
    valid_mask = df_master["x_utm"].notna() & df_master["y_utm"].notna()

    # Create a list of (x, y) coordinate tuples
    coords_list = list(zip(df_master.loc[valid_mask, "x_utm"], df_master.loc[valid_mask, "y_utm"]))

    if len(coords_list) == 0:
        return

    with rasterio.open(raster_path) as src:
        nodata_val = src.nodata
        sampled_values = []

        # Sample values at coordinates
        for val in src.sample(coords_list):
            v = val[0]
            # Filter out NoData and absurd placeholders
            if v == nodata_val or v > 1e9 or v < -1e9:
                sampled_values.append(np.nan)
            else:
                sampled_values.append(v)

        # Assign values exactly to the valid rows
        df_master.loc[valid_mask, col_name] = sampled_values

# =========================================================
# 4. Extract TPI Data
# =========================================================
extract_static_raster(tpi_300_path, "tpi_300m")
extract_static_raster(tpi_1500_path, "tpi_1500m")

# =========================================================
# 5. Save Final Dataset
# =========================================================
df_master.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! The dataset now includes TPI features and is saved to:\n{output_file}")
print("-" * 60)