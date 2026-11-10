# This script merges the extracted traffic proxies with the sensor coordinates,
# reprojects them to EPSG:25832, and extracts the meteorological features
# (temperature, wind, precipitation, humidity) from the ICON-D2 GeoTIFF rasters.

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
traffic_file = PROCESSED_DIR / "13Uhr_Verkehrs_Proxys_Komplett.csv"
sensor_file = PROCESSED_DIR / "sensor_community_with_kopfklinik_may_august.csv"
raster_base_dir = RAW_DIR / "New_ICON_D2_Rasters"

# Output file
output_file = PROCESSED_DIR / "Features_Traffic_and_Weather.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping extraction process.")
    exit()

# =========================================================
# 2. Extract and Clean Sensor Coordinates
# =========================================================
print("Loading sensor data and extracting coordinates...")
df_sensors = pd.read_csv(sensor_file, sep=";", decimal=",")

# Replace commas with dots and convert to numeric
for col in ["lat", "lon"]:
    df_sensors[col] = df_sensors[col].astype(str).str.replace(",", ".", regex=False)
    df_sensors[col] = pd.to_numeric(df_sensors[col], errors="coerce")

# Keep only unique stations
unique_sensors = df_sensors[["sensor_id", "lat", "lon"]].drop_duplicates(subset=["sensor_id"]).dropna()
unique_sensors["station_id"] = unique_sensors["sensor_id"].astype(float).astype(int).astype(str)
unique_sensors = unique_sensors[["station_id", "lat", "lon"]]

print(f"-> Found {len(unique_sensors)} unique sensor locations.")

# =========================================================
# 3. Load Traffic Data and Merge with Coordinates
# =========================================================
print("Loading traffic proxies and appending coordinates...")
df_traffic = pd.read_csv(traffic_file, sep=";", decimal=",")
df_traffic["station_id"] = df_traffic["station_id"].astype(str)

# Merge sensor coordinates to traffic data
df_merged = pd.merge(df_traffic, unique_sensors, on="station_id", how="left")

# =========================================================
# 4. Adjust Coordinate System for Raster Extraction
# =========================================================
print("Transforming coordinates to the metric system (EPSG:25832)...")
# Create GeoDataFrame from WGS84 coordinates
gdf = gpd.GeoDataFrame(
    df_merged,
    geometry=gpd.points_from_xy(df_merged["lon"], df_merged["lat"]),
    crs="EPSG:4326"
)

# Reproject to the GeoTIFF coordinate system
gdf = gdf.to_crs("EPSG:25832")

# Prepare X and Y coordinates for rasterio
df_merged["x_utm"] = gdf.geometry.x
df_merged["y_utm"] = gdf.geometry.y

# =========================================================
# 5. Extract Weather Data from Rasters
# =========================================================
print("Extracting weather data from ICON-D2 GeoTIFFs...")

# Initialize new columns
df_merged["icon_temp_2m"] = np.nan
df_merged["icon_wind_speed"] = np.nan
df_merged["icon_wind_direction"] = np.nan
df_merged["icon_precip_12h"] = np.nan
df_merged["icon_relative_humidity"] = np.nan

# Mapping for raster folders and filename prefixes
raster_mappings = {
    "icon_temp_2m": {"folder": "Temp_2m", "prefix": "ICON_D2_Temp_"},
    "icon_wind_speed": {"folder": "Wind_Speed", "prefix": "ICON_D2_WindSpeed_"},
    "icon_wind_direction": {"folder": "Wind_Direction", "prefix": "wind_dir_"},
    "icon_precip_12h": {"folder": "Precipitation_12h", "prefix": "ICON_D2_Precip_12h_"},
    "icon_relative_humidity": {"folder": "Relative_Humidity_2m", "prefix": "ICON_D2_RelHumid_"}
}

unique_dates = df_merged["date"].unique()

for current_date in unique_dates:
    # Filter for the current day
    mask = df_merged["date"] == current_date
    day_points = df_merged.loc[mask, ["x_utm", "y_utm"]].values

    # Format as list of tuples: [(x1, y1), (x2, y2), ...]
    coords_list = [(x, y) for x, y in day_points]

    for col_name, mapping in raster_mappings.items():
        # Construct file path
        tif_filename = f"{mapping['prefix']}{current_date}.tif"
        tif_path = raster_base_dir / mapping["folder"] / tif_filename

        # Check if file exists to prevent crashes on missing days
        if tif_path.exists():
            with rasterio.open(tif_path) as src:
                # rasterio.sample extracts values at the given coordinates
                sampled_values = [val[0] for val in src.sample(coords_list)]
                df_merged.loc[mask, col_name] = sampled_values
        else:
            print(f"Warning: Missing file {tif_filename}")

# =========================================================
# 6. Cleanup and Save
# =========================================================
# Drop temporary coordinate columns
df_final = df_merged.drop(columns=["x_utm", "y_utm"])

# Save to CSV
df_final.to_csv(output_file, index=False, sep=";", decimal=",")

print("-" * 60)
print(f"SUCCESS! Dataset with Traffic AND Weather features saved to:\n{output_file}")
print("-" * 60)