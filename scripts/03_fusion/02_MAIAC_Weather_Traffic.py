# This script extracts MAIAC Aerosol Optical Depth (AOD) satellite data
# from daily GeoTIFFs and appends it to the master dataset containing
# traffic and weather features. It includes a filter for clouds/NoData.

import numpy as np
import pandas as pd
import geopandas as gpd
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
master_file = PROCESSED_DIR / "Features_Traffic_and_Weather.csv"
sensor_file = PROCESSED_DIR / "sensor_community_with_kopfklinik_may_august.csv"
raster_base_dir = RAW_DIR / "MAIAC_AOD"

# Output file
output_file = PROCESSED_DIR / "Features_MAIAC_Traffic_Weather.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping extraction process.")
    exit()

# =========================================================
# 2. Load Master Data & Ensure Coordinates
# =========================================================
print("Loading master dataset...")
df_merged = pd.read_csv(master_file, sep=";", decimal=",")
df_merged["station_id"] = df_merged["station_id"].astype(str)

# SAFETY CHECK: If lat/lon are missing in the master, fetch them from the sensor file
if "lon" not in df_merged.columns or "lat" not in df_merged.columns:
    print("Coordinates missing in master. Fetching them from the sensor file...")
    df_sensors = pd.read_csv(sensor_file, sep=";", decimal=",")

    for col in ["lat", "lon"]:
        df_sensors[col] = df_sensors[col].astype(str).str.replace(",", ".", regex=False)
        df_sensors[col] = pd.to_numeric(df_sensors[col], errors="coerce")

    unique_sensors = df_sensors[["sensor_id", "lat", "lon"]].drop_duplicates(subset=["sensor_id"]).dropna()
    unique_sensors["station_id"] = unique_sensors["sensor_id"].astype(float).astype(int).astype(str)
    unique_sensors = unique_sensors[["station_id", "lat", "lon"]]

    df_merged = pd.merge(df_merged, unique_sensors, on="station_id", how="left")
else:
    print("Coordinates (lat/lon) are already present in the master. Merge skipped!")

# Ensure coordinates are numeric
df_merged["lat"] = pd.to_numeric(df_merged["lat"], errors="coerce")
df_merged["lon"] = pd.to_numeric(df_merged["lon"], errors="coerce")

# =========================================================
# 3. Adjust Coordinate System for Raster Extraction
# =========================================================
print("Transforming coordinates to the metric system (EPSG:25832)...")
gdf = gpd.GeoDataFrame(
    df_merged,
    geometry=gpd.points_from_xy(df_merged["lon"], df_merged["lat"]),
    crs="EPSG:4326"
)

# Convert to GeoTIFF system (UTM 32N)
gdf = gdf.to_crs("EPSG:25832")
df_merged["x_utm"] = gdf.geometry.x
df_merged["y_utm"] = gdf.geometry.y

# =========================================================
# 4. Extract MAIAC AOD from Rasters
# =========================================================
print("Extracting MAIAC AOD satellite data...")

df_merged["MAIAC_AOD"] = np.nan
unique_dates = df_merged["date"].unique()

for current_date in unique_dates:
    mask = df_merged["date"] == current_date
    day_points = df_merged.loc[mask, ["x_utm", "y_utm"]].values

    coords_list = [(x, y) for x, y in day_points]

    tif_filename = f"MAIAC_AOD_{current_date}.tif"
    tif_path = raster_base_dir / tif_filename

    if tif_path.exists():
        with rasterio.open(tif_path) as src:
            nodata_val = src.nodata  # Reads official NoData value from metadata

            sampled_values = []
            for val in src.sample(coords_list):
                v = val[0]

                # FILTER: If value is NoData, absurdly large (> 10), or negative (< 0),
                # it is not a valid AOD measurement (e.g. cloud cover)
                if v == nodata_val or v < 0 or v > 10:
                    sampled_values.append(np.nan)
                else:
                    sampled_values.append(v)

            df_merged.loc[mask, "MAIAC_AOD"] = sampled_values
    else:
        print(f"Warning: No MAIAC data found for {current_date}")

# =========================================================
# 5. Cleanup and Save
# =========================================================
# Drop metric helper coordinates to keep the file clean (optional)
# df_merged = df_merged.drop(columns=["x_utm", "y_utm"], errors="ignore")

df_merged.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! Dataset including MAIAC AOD saved to:\n{output_file}")
print("-" * 60)