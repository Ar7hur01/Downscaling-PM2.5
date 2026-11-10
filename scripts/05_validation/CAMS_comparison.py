# This script compares the PM2.5 measurements from the official anchor station (Kopfklinik)
# and the low-cost sensor network against the regional Copernicus CAMS background model.
# It also includes an optional offset adjustment line for the low-cost sensors and
# plots the data as a comprehensive time-series validation graph.

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
FIGURES_DIR = BASE_DIR.parent / "reports" / "figures"
MAPS_DIR = BASE_DIR.parent / "reports" / "maps"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input files
ml_file = PROCESSED_DIR / "allValues_Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# Static rasters for the RF Grid
aoi_ref_path = RAW_DIR / "DOM" / "wuerzburg_tpi_1500m.tif"
tpi_300_path = RAW_DIR / "DOM" / "wuerzburg_tpi_300m.tif"
ndvi_path = RAW_DIR / "NDVI" / "50m_AOI_NDVI_S2_11082024.tif"

# Multi-Band Copernicus GeoTIFF
cams_path = MAPS_DIR / "ENS_ANALYSIS_reprojected.tif"

# Output figure
output_plot = FIGURES_DIR / "RF_vs_Copernicus_vs_Sensors_Validation.png"

START_DATE = pd.to_datetime('2024-05-01')

# STRATEGY 1: Skip if plot already generated
if output_plot.exists():
    print(f"Validation plot {output_plot.name} already exists. Skipping plot generation.")
    exit()

# =========================================================
# 2. X-Ray Scan of Copernicus File (Metadata Check)
# =========================================================
print("Starting X-ray scan of Copernicus CAMS file...\n")
if cams_path.exists():
    with rasterio.open(cams_path) as src:
        print("--- 1. METADATA ---")
        print(f"Coordinate Reference System (CRS): {src.crs}")
        print(f"Spatial Bounds:                    {src.bounds}")
        print(f"Number of Days (Bands):            {src.count}")
        print(f"Pixel Data Type:                   {src.dtypes[0]}")
        print(f"NoData Value:                      {src.nodata}\n")

        band1 = src.read(1)
        print("--- 2. PIXEL VALUES IN BAND 1 (May 1st) ---")
        print(f"Minimum Value: {np.nanmin(band1):.2f}")
        print(f"Maximum Value: {np.nanmax(band1):.2f}")
        print(f"Mean Value:    {np.nanmean(band1):.2f}\n")
else:
    print(f"WARNING: CAMS file not found at {cams_path}")

# =========================================================
# 3. Load Data, Clean & Separate Ground Measurements
# =========================================================
print("Step 1: Loading data & calculating sensor time series...")
df_raw = pd.read_csv(ml_file, sep=";", decimal=",")

df_raw['station_id'] = df_raw['station_id'].astype(str).str.strip()

if df_raw['PM25_Target'].dtype == object:
    df_raw['PM25_Target'] = df_raw['PM25_Target'].str.replace(",", ".", regex=False)
df_raw['PM25_Target'] = pd.to_numeric(df_raw['PM25_Target'], errors='coerce')

bad_sensors = ['5456', '15449', '9094', '46505']
df = df_raw[~df_raw['station_id'].isin(bad_sensors)].copy()

# A) Time series of official anchor station ("1" = Kopfklinik)
df_kopfklinik = df[df['station_id'] == "1"].groupby('date')['PM25_Target'].mean().reset_index()
df_kopfklinik.rename(columns={'PM25_Target': 'Kopfklinik_Target'}, inplace=True)

# B) Time series of low-cost sensors mean (excluding station "1")
df_lc = df[df['station_id'] != "1"].groupby('date')['PM25_Target'].mean().reset_index()
df_lc.rename(columns={'PM25_Target': 'LC_Sensors_Mean'}, inplace=True)

# =========================================================
# 4. Train RF Model (Clear-Sky Approach)
# =========================================================
print("Step 2: Training RF model on complete-case days...")
features = ['prox_kumulativ_car', 'prox_kumulativ_truck', 'IDW_car', 'IDW_truck',
            'icon_temp_2m', 'icon_wind_speed', 'icon_wind_direction', 'icon_precip_12h',
            'MAIAC_AOD', 'S5P_NO2', 'S5P_UVAI', 'ERA5_BLH', 'tpi_300m', 'tpi_1500m', 'ndvi']

df_clean = df.dropna(subset=['PM25_Target'] + features).copy()

rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1)
sample_weights = np.where(df_clean['station_id'] == "1", 2.0, 1.0)
rf.fit(df_clean[features], df_clean['PM25_Target'], sample_weight=sample_weights)

# =========================================================
# 5. Prepare 100m Raster Grid for Würzburg
# =========================================================
print("Step 3: Generating spatial grid...")
with rasterio.open(aoi_ref_path) as ref_src:
    aoi_bounds_utm = ref_src.bounds

resolution = 100
x_coords = np.arange(aoi_bounds_utm.left, aoi_bounds_utm.right, resolution)
y_coords = np.arange(aoi_bounds_utm.top, aoi_bounds_utm.bottom, -resolution)
grid_x, grid_y = np.meshgrid(x_coords, y_coords)
flat_x, flat_y = grid_x.flatten(), grid_y.flatten()

grid_df = pd.DataFrame({'x_utm': flat_x, 'y_utm': flat_y})
coords_list = list(zip(flat_x, flat_y))

def sample_raster(raster_path):
    with rasterio.open(raster_path) as src:
        vals = np.array([val[0] for val in src.sample(coords_list)], dtype=float)
        invalid = (vals == src.nodata) | (vals > 1e6) | (vals < -1e6) | np.isnan(vals)
        vals[invalid] = np.nanmedian(vals[~invalid]) if np.any(~invalid) else 0.0
        return vals

grid_df['tpi_300m'] = sample_raster(tpi_300_path)
grid_df['tpi_1500m'] = sample_raster(aoi_ref_path)
grid_df['ndvi'] = sample_raster(ndvi_path)

# =========================================================
# 6. Daily Prediction & Merging
# =========================================================
print("Step 4: Starting daily prediction loop...")

unique_dates = sorted(df_clean['date'].unique())
results = []

# Würzburg Center Coordinates for CAMS Point-Sampling
WUE_LON = 9.93
WUE_LAT = 49.79

if cams_path.exists():
    with rasterio.open(cams_path) as cams_src:
        for day in unique_dates:
            day_data = df_clean[df_clean['date'] == day]
            if len(day_data) == 0: continue

            # a) RF Model Prediction on the city grid
            grid_day = grid_df.copy()
            for feat in features:
                if feat not in ['tpi_300m', 'tpi_1500m', 'ndvi']:
                    grid_day[feat] = day_data[feat].median()

            rf_preds = rf.predict(grid_day[features])
            rf_mean = np.mean(rf_preds)
            rf_min = np.min(rf_preds)
            rf_max = np.max(rf_preds)

            # b) Extract CAMS value via Point-Sampling
            current_date = pd.to_datetime(day)
            day_diff = (current_date - START_DATE).days
            band_idx = day_diff + 1

            cams_val = np.nan
            if 1 <= band_idx <= cams_src.count:
                try:
                    sample_gen = cams_src.sample([(WUE_LON, WUE_LAT)], indexes=[band_idx])
                    val = list(sample_gen)[0][0]
                    nodata = cams_src.nodata
                    if val != nodata and -1e6 < val < 1e6:
                        cams_val = float(val)
                except Exception:
                    pass

            results.append({
                'date': day,
                'Date_dt': pd.to_datetime(day),
                'RF_Mean': rf_mean,
                'RF_Min': rf_min,
                'RF_Max': rf_max,
                'CAMS_Mean': cams_val
            })
else:
    print(f"Skipping CAMS extraction, file not found: {cams_path}")

results_df = pd.DataFrame(results)

# c) Merge Kopfklinik & LC Sensor measurements
plot_df = pd.merge(results_df, df_kopfklinik, on='date', how='left')
plot_df = pd.merge(plot_df, df_lc, on='date', how='left')

if 'CAMS_Mean' in plot_df.columns:
    plot_df = plot_df.dropna(subset=['CAMS_Mean']).sort_values('Date_dt').copy()
else:
    plot_df = plot_df.sort_values('Date_dt').copy()

# =========================================================
# 7. Visualization: Comprehensive Validation Chart
# =========================================================
print("Step 5: Creating comprehensive time-series plot...")

if len(plot_df) > 0:
    fig, ax = plt.subplots(figsize=(15, 7))

    # Optional: Make figure background transparent for reports
    # fig.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0)

    # 1. Copernicus CAMS Background
    if 'CAMS_Mean' in plot_df.columns:
        ax.plot(plot_df['Date_dt'], plot_df['CAMS_Mean'],
                label='Copernicus CAMS Background (~10km)',
                color='#ff7f0e', linewidth=2.0, linestyle='--', marker='o', markersize=3)

    # 2. Low-Cost Sensors City Mean (Raw Measurement)
    ax.plot(plot_df['Date_dt'], plot_df['LC_Sensors_Mean'],
            label='Low-Cost Sensors (Raw Measurement Mean)',
            color='#2ca02c', linewidth=2.0, linestyle=':', marker='^', markersize=4, alpha=0.3)

    # 2.5. Low-Cost Sensors City Mean (+ Hardware Offset)
    ax.plot(plot_df['Date_dt'], plot_df['LC_Sensors_Mean'] + 4.88,
            label='Low-Cost Sensors (+4.88 Hardware Offset)',
            color='#2ca02c', linewidth=2.0, linestyle=':', marker='^', markersize=4)

    # 3. Kopfklinik Anchor Station (Official Ground Truth)
    ax.plot(plot_df['Date_dt'], plot_df['Kopfklinik_Target'],
            label='Official Anchor Station (Ground Truth)',
            color='#d62728', linewidth=2.5, linestyle='-', marker='d', markersize=4)

    # RF Model bounds (Currently commented out in user code, left for reference)
    # ax.plot(plot_df['Date_dt'], plot_df['RF_Mean'],
    #         label='RF Downscaling Model (100m Grid Mean)',
    #         color='#1f77b4', linewidth=2.5, marker='s', markersize=4)
    # ax.fill_between(plot_df['Date_dt'], plot_df['RF_Min'], plot_df['RF_Max'],
    #                 color='#1f77b4', alpha=0.18,
    #                 label='RF Model Spatial Variance (Min - Max)')

    # Axes & Formatting
    ax.set_ylabel('PM2.5 Concentration [µg/m³]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Time-Series Validation: Measurements vs. Copernicus',
                 fontsize=14, fontweight='bold', pad=15)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)

    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='grey', framealpha=0.92, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()

    print(f"-> Final validation plot saved to:\n   {output_plot}")
else:
    print("No data available to generate the plot.")