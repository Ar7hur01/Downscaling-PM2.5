# This notebook downloads all ICON-D2 Data, which is:
# Wind Direction (in u/v), Relative Humidity, Wind_Speed, Precipitation und Temp (2m),
# however Precipitation is accumulated to 12 hours and all other values are extracted only for 1 pm (satellite overpass time).

import os
from pathlib import Path
import time
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import requests
from scipy.interpolate import griddata

# 1. Settings & relative paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent.parent / "data" / "raw" / "New_ICON_D2_Rasters"

# Added Relative_Humidity_2m to the output folders
for folder in ["Temp_2m", "Relative_Humidity_2m", "Wind_Speed", "U_Wind", "V_Wind", "Precipitation_12h"]:
    os.makedirs(OUTPUT_DIR / folder, exist_ok=True)

START_DATE = "2024-05-01"
END_DATE = "2024-08-31"

LON_MIN, LON_MAX = 9.85, 10.05
LAT_MIN, LAT_MAX = 49.74, 49.85

lons = np.arange(LON_MIN, LON_MAX, 0.02)
lats = np.arange(LAT_MIN, LAT_MAX, 0.02)
grid_lons, grid_lats = np.meshgrid(lons, lats)
points_wgs84 = pd.DataFrame({"lon": grid_lons.flatten(), "lat": grid_lats.flatten()})

print(f"Creating grid for Würzburg with {len(points_wgs84)} points...")

# 2. Fetch and filter weather data
all_grid_data = []

for idx, row in points_wgs84.iterrows():
    lat, lon = row["lat"], row["lon"]
    print(f" -> Loading point {idx+1}/{len(points_wgs84)} ({lat:.4f}, {lon:.4f})...")

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        # Added relative_humidity_2m to the hourly request
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"],
        "models": "icon_d2",
        "timezone": "Europe/Berlin",
    }

    try:
        res = requests.get(url, params=params).json()
        if "error" in res:
            print(f"    API Error at {lat}, {lon}")
            continue

        # Extract all requested parameters
        df_pt = pd.DataFrame({
            "time": pd.to_datetime(res["hourly"]["time"]),
            "temp": res["hourly"]["temperature_2m"],
            "humidity": res["hourly"]["relative_humidity_2m"],
            "wind_kmh": res["hourly"]["wind_speed_10m"],
            "wind_dir": res["hourly"]["wind_direction_10m"],
            "precip": res["hourly"]["precipitation"],
        })

        df_pt["date"] = df_pt["time"].dt.date
        df_pt["hour"] = df_pt["time"].dt.hour

        # Snapshot at 13:00 (Temp, Humidity & Wind)
        df_13 = df_pt[df_pt["hour"] == 13].copy()
        df_13["wind_ms"] = df_13["wind_kmh"] / 3.6
        rads = np.radians(df_13["wind_dir"])
        df_13["u_10"] = -df_13["wind_ms"] * np.sin(rads)
        df_13["v_10"] = -df_13["wind_ms"] * np.cos(rads)

        # 12-hour precipitation sum (01:00 - 13:00)
        df_precip = df_pt[(df_pt["hour"] >= 1) & (df_pt["hour"] <= 13)]
        precip_sum = df_precip.groupby("date")["precip"].sum().reset_index()
        precip_sum.rename(columns={"precip": "precip_12h_sum"}, inplace=True)

        # Merge daily data
        df_daily = pd.merge(df_13, precip_sum, on="date", how="left")
        df_daily["lat"] = lat
        df_daily["lon"] = lon

        all_grid_data.append(df_daily)

    except Exception as e:
        print(f"Error at point {idx+1}: {e}")
    time.sleep(0.3)

df_all = pd.concat(all_grid_data, ignore_index=True)

# 3. Reproject to EPSG:25832 (UTM 32N)
print("\nReprojecting grid to EPSG:25832...")
gdf_all = gpd.GeoDataFrame(
    df_all,
    geometry=gpd.points_from_xy(df_all["lon"], df_all["lat"]),
    crs="EPSG:4326",
).to_crs("EPSG:25832")

gdf_all["x"] = gdf_all.geometry.x
gdf_all["y"] = gdf_all.geometry.y

# 4. Define raster parameters
X_MIN, X_MAX = 560000.0, 575000.0
Y_MIN, Y_MAX = 5510000.0, 5522000.0
PIXEL_SIZE = 100.0

width = int((X_MAX - X_MIN) / PIXEL_SIZE)
height = int((Y_MAX - Y_MIN) / PIXEL_SIZE)
transform = from_origin(X_MIN, Y_MAX, PIXEL_SIZE, PIXEL_SIZE)

grid_x, grid_y = np.meshgrid(
    np.linspace(X_MIN, X_MAX, width), np.linspace(Y_MAX, Y_MIN, height)
)

# 5. Generate daily GeoTIFFs
unique_days = gdf_all["time"].dt.strftime("%Y-%m-%d").unique()
print(f"\nGenerating {len(unique_days)} daily GeoTIFFs per parameter...")

for day in unique_days:
    df_day = gdf_all[gdf_all["time"].dt.strftime("%Y-%m-%d") == day]
    points = df_day[["x", "y"]].values

    params_to_grid = {
        "Temp_2m": ("temp", f"ICON_D2_Temp_{day}.tif"),
        "Relative_Humidity_2m": ("humidity", f"ICON_D2_RelHumid_{day}.tif"),
        "Wind_Speed": ("wind_ms", f"ICON_D2_WindSpeed_{day}.tif"),
        "U_Wind": ("u_10", f"ICON_D2_U_{day}.tif"),
        "V_Wind": ("v_10", f"ICON_D2_V_{day}.tif"),
        "Precipitation_12h": ("precip_12h_sum", f"ICON_D2_Precip_12h_{day}.tif"),
    }

    for folder, (col, filename) in params_to_grid.items():
        out_path = OUTPUT_DIR / folder / filename

        # ---> HIER IST DIE KORREKTE ÜBERPRÜFUNG <---
        if out_path.exists():
            print(f"[{day}] File {filename} already exists. Skipping...")
            continue
        # ------------------------------------------

        values = df_day[col].values

        raster_array = griddata(
            points, values, (grid_x, grid_y), method="nearest"
        )

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=raster_array.dtype,
            crs="EPSG:25832",
            transform=transform,
        ) as dst:
            dst.write(raster_array, 1)

print(f"\nDONE! Raster files saved to:\n{OUTPUT_DIR}")