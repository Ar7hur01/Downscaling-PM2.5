# This script trains the Random Forest model on cloud-free days and then runs
# mass inference across a 100x100m grid for all available summer days (01.05 - 31.08.2024).
# It exports downscaled raster maps representing the Mean, Minimum, and Maximum PM2.5 concentrations.

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Define directories
RAW_DIR = BASE_DIR.parent / "data" / "raw"
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
MAPS_DIR = BASE_DIR.parent / "reports" / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
csv_file = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"
traffic_file = RAW_DIR / "Traffic" / "Verkehrsdaten_mit_Koordinaten_merged_cars_n_trucks.csv"

# Output files
mean_map_path = MAPS_DIR / "PM25_Summer_Mean_Map_2024.tif"
max_map_path = MAPS_DIR / "PM25_Summer_Max_Map_2024.tif"
min_map_path = MAPS_DIR / "PM25_Summer_Min_Map_2024.tif"

# STRATEGY 1: Skip if maps already generated
if mean_map_path.exists() and max_map_path.exists() and min_map_path.exists():
    print("Downscaled summer maps already exist in reports/maps/. Skipping mass inference.")
    exit()

# =========================================================
# 2. Train Model for Inference
# =========================================================
print("--- Training model for mass inference ---")
df = pd.read_csv(csv_file, sep=";", decimal=",")
df['station_id'] = df['station_id'].astype(str).str.strip()

# Exclude bad sensors
bad_sensors = ['5456', '15449', '46505', '9094']
df = df[~df['station_id'].isin(bad_sensors)].copy()

features = [
    'IDW_car', 'IDW_truck',
    'icon_temp_2m', 'icon_wind_speed', 'icon_wind_direction',
    'icon_precip_12h', 'icon_relative_humidity',
    'MAIAC_AOD', 'S5P_NO2', 'S5P_UVAI', 'ERA5_BLH',
    'tpi_300m', 'tpi_1500m'
]

# Complete-case analysis for training
df = df.dropna(subset=['PM25_Target'] + features)

# Baseline Calibration (Kopfklinik anchor)
ANCHOR_ID = "1"
mean_kopf = df[df['station_id'] == ANCHOR_ID]['PM25_Target'].mean()
mean_low = df[df['station_id'] != ANCHOR_ID]['PM25_Target'].mean()
df.loc[df['station_id'] != ANCHOR_ID, 'PM25_Target'] += (mean_kopf - mean_low)

# Retrieve maximum IDW values seen during training to prevent spatial overfitting later
MAX_IDW_CAR_TRAIN = df['IDW_car'].max()
MAX_IDW_TRUCK_TRAIN = df['IDW_truck'].max()

# Fit the Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(df[features], df['PM25_Target'])

# =========================================================
# 3. Create 100m Grid & Load Static Features
# =========================================================
print("\nCreating 100x100m grid and loading static features...")
tpi_300_path = RAW_DIR / "DOM" / "wuerzburg_tpi_300m.tif"

with rasterio.open(tpi_300_path) as ref_src:
    bounds = ref_src.bounds
    crs = ref_src.crs
    x_min, x_max = bounds.left, bounds.right
    y_min, y_max = bounds.bottom, bounds.top

resolution = 100
grid_x, grid_y = np.meshgrid(
    np.arange(x_min, x_max, resolution),
    np.arange(y_max, y_min, -resolution)
)
flat_x, flat_y = grid_x.flatten(), grid_y.flatten()
coords_list = list(zip(flat_x, flat_y))

grid_df_static = pd.DataFrame({'x_utm': flat_x, 'y_utm': flat_y})


def extract_raster(tif_path):
    """Extracts raster values for the 100m grid coordinates."""
    if not tif_path.exists():
        return np.full(len(coords_list), np.nan)
    with rasterio.open(tif_path) as src:
        vals = np.array([val[0] for val in src.sample(coords_list)], dtype=float)
        invalid = (vals == src.nodata) | (vals > 1e6) | (vals < -1e6) | np.isnan(vals)
        vals[invalid] = np.nanmedian(vals[~invalid]) if np.any(~invalid) else 0.0
        return vals


grid_df_static['tpi_300m'] = extract_raster(tpi_300_path)
grid_df_static['tpi_1500m'] = extract_raster(RAW_DIR / "DOM" / "wuerzburg_tpi_1500m.tif")

# Preload Traffic Data
df_traffic_all = pd.read_csv(traffic_file, sep=";", decimal=",")
df_traffic_all["Time"] = pd.to_datetime(df_traffic_all["Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
for col in ["Lat", "Lon", "processed_car_vol", "processed_truck_vol"]:
    df_traffic_all[col] = df_traffic_all[col].astype(str).str.replace(",", ".", regex=False)
    df_traffic_all[col] = pd.to_numeric(df_traffic_all[col], errors="coerce")

transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)

# =========================================================
# 4. LOOP OVER ALL AVAILABLE CLOUD-FREE DAYS
# =========================================================
valid_dates = df['date'].unique()
all_daily_maps = []

print(f"\nStarting mass inference for {len(valid_dates)} cloud-free days...")

for current_date in valid_dates:
    print(f" -> Calculating map for: {current_date}")
    grid_df = grid_df_static.copy()

    # 4.1 Load dynamic weather and satellite data
    tifs = {
        'ERA5_BLH': RAW_DIR / "ERA5_BLH" / f"ERA5_BLH_{current_date}.tif",
        'icon_temp_2m': RAW_DIR / "New_ICON_D2_Rasters" / "Temp_2m" / f"ICON_D2_Temp_{current_date}.tif",
        'icon_wind_speed': RAW_DIR / "New_ICON_D2_Rasters" / "Wind_Speed" / f"ICON_D2_WindSpeed_{current_date}.tif",
        'icon_wind_direction': RAW_DIR / "New_ICON_D2_Rasters" / "Wind_Direction" / f"wind_dir_{current_date}.tif",
        'icon_precip_12h': RAW_DIR / "New_ICON_D2_Rasters" / "Precipitation_12h" / f"ICON_D2_Precip_12h_{current_date}.tif",
        'icon_relative_humidity': RAW_DIR / "New_ICON_D2_Rasters" / "Relative_Humidity_2m" / f"ICON_D2_RelHumid_{current_date}.tif",
        'MAIAC_AOD': RAW_DIR / "MAIAC_AOD" / f"MAIAC_AOD_{current_date}.tif",
        'S5P_NO2': RAW_DIR / "S5P_NO2" / f"S5P_NO2_{current_date}.tif",
        'S5P_UVAI': RAW_DIR / "S5P_UVAI" / f"S5P_UVAI_{current_date}.tif"
    }

    for feat, path in tifs.items():
        grid_df[feat] = extract_raster(path)

    # 4.2 Calculate traffic proxies for 13:00 on this specific day
    df_traffic_13 = df_traffic_all[
        (df_traffic_all["Time"].dt.hour == 13) &
        (df_traffic_all["Time"].dt.strftime("%Y-%m-%d") == current_date)
        ].copy()
    df_traffic_13 = df_traffic_13.dropna(subset=["Lat", "Lon", "processed_car_vol"])

    max_car = np.zeros(len(grid_df))
    max_truck = np.zeros(len(grid_df))
    grid_x_arr, grid_y_arr = grid_df['x_utm'].values, grid_df['y_utm'].values

    for _, t_stat in df_traffic_13.iterrows():
        t_x, t_y = transformer.transform(t_stat['Lon'], t_stat['Lat'])
        dist = np.sqrt((grid_x_arr - t_x) ** 2 + (grid_y_arr - t_y) ** 2)

        idw_car = t_stat['processed_car_vol'] / ((dist + 1.0) ** 2)
        idw_truck = t_stat['processed_truck_vol'] / ((dist + 1.0) ** 2) if pd.notna(
            t_stat['processed_truck_vol']) else 0.0

        max_car = np.maximum(max_car, idw_car)
        max_truck = np.maximum(max_truck, idw_truck)

    # FIX FOR SPATIAL OVERFITTING (CLIPPING)
    # Clip extreme grid values to the maximum IDW value the model saw during training
    grid_df['IDW_car'] = np.clip(max_car, 0, MAX_IDW_CAR_TRAIN)
    grid_df['IDW_truck'] = np.clip(max_truck, 0, MAX_IDW_TRUCK_TRAIN)

    # 4.3 Impute missing spatial pixels and predict
    grid_df.fillna(grid_df.median(), inplace=True)
    pm25_pred = rf_model.predict(grid_df[features])
    all_daily_maps.append(pm25_pred.reshape(grid_y.shape))


# =========================================================
# 5. Export Downscaled Raster Maps (Mean, Max, Min)
# =========================================================
def save_map(matrix, path, title):
    with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=matrix.shape[0], width=matrix.shape[1],
            count=1, dtype=matrix.dtype, crs=crs,
            transform=rasterio.transform.from_origin(x_min, y_max, resolution, resolution)
    ) as new_dataset:
        new_dataset.write(matrix, 1)
    print(f"\nSUCCESS! High-resolution {title} map saved to:\n{path}")


# Calculate Mean (Average Summer Map)
print("\nCalculating the mean over all daily maps...")
summer_mean_matrix = np.mean(all_daily_maps, axis=0)
save_map(summer_mean_matrix, mean_map_path, "Summer Mean")

# Calculate Maximum (Worst-Case Map)
print("\nCalculating the maximum over all daily maps...")
summer_max_matrix = np.max(all_daily_maps, axis=0)
save_map(summer_max_matrix, max_map_path, "Summer Maximum")

# Calculate Minimum (Best-Case Map)
print("\nCalculating the minimum over all daily maps...")
summer_min_matrix = np.min(all_daily_maps, axis=0)
save_map(summer_min_matrix, min_map_path, "Summer Minimum")

print("-" * 60)
print("Mass inference and mapping complete.")
print("-" * 60)