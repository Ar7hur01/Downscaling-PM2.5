# This script trains a Random Forest Regressor using Leave-Location-Out (LLO)
# cross-validation on complete-case data (cloud-free days). It includes
# hardware baseline calibration relative to the anchor station (Kopfklinik),
# computes performance metrics (R2, RMSE, Bias), and exports IEEE-styled
# feature importance plots.

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# Define data and output figure directories
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
FIGURES_DIR = BASE_DIR.parent / "data" / "output" / "Validationgraphs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset
file_path = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# =========================================================
# 2. Data Loading & Cleaning
# =========================================================
print("Loading dataset...")
df = pd.read_csv(file_path, sep=";", decimal=",")

df['station_id'] = df['station_id'].astype(str).str.strip()

# Remove known defective / distorting sensors
bad_sensors = ['5456', '15449', '46505']
df = df[~df['station_id'].isin(bad_sensors)].copy()

features = [
    'IDW_car', 'IDW_truck',
    'icon_temp_2m', 'icon_wind_speed', 'icon_wind_direction',
    'icon_precip_12h', 'icon_relative_humidity',
    'MAIAC_AOD', 'S5P_NO2', 'S5P_UVAI', 'ERA5_BLH',
    'tpi_300m', 'tpi_1500m'
]

# Complete-Case Filter: Exclude rows with missing values
print("Step 1: Removing rows with missing values (complete-case analysis)...")
initial_len = len(df)
df = df.dropna(subset=['PM25_Target'] + features)
cleaned_len = len(df)
print(f"-> Retained {cleaned_len} complete observations out of {initial_len} initial records.\n")

# Retain only sensors with at least 10 valid observations
sensor_counts = df['station_id'].value_counts()
valid_sensors = sensor_counts[sensor_counts >= 10].index
df = df[df['station_id'].isin(valid_sensors)].copy()

# =========================================================
# 3. Hardware Baseline Calibration
# =========================================================
print("Performing hardware baseline calibration relative to anchor station...")

ANCHOR_ID = "1"  # Kopfklinik official reference station

# Compute baseline offsets
mean_kopfklinik = df[df['station_id'] == ANCHOR_ID]['PM25_Target'].mean()
mean_lowcost = df[df['station_id'] != ANCHOR_ID]['PM25_Target'].mean()

offset = mean_kopfklinik - mean_lowcost
print(f"-> Systematic hardware offset: {offset:.2f} µg/m³")

# Adjust low-cost sensor values by calculated offset
df.loc[df['station_id'] != ANCHOR_ID, 'PM25_Target'] += offset
print("-> Low-cost sensors adjusted to official reference level.\n")

# =========================================================
# 4. Leave-Location-Out (LLO) Cross-Validation
# =========================================================
print("Step 2: Starting Leave-Location-Out (LLO) Cross-Validation...")

test_stations = [s for s in df['station_id'].unique() if s != ANCHOR_ID]
fold_results = []
feature_importances_list = []

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

for fold, test_sensor in enumerate(test_stations, 1):
    test_mask = df['station_id'] == test_sensor
    df_test = df[test_mask]
    df_train = df[~test_mask]

    if len(df_test) == 0:
        continue

    X_train = df_train[features]
    y_train = df_train['PM25_Target']
    X_test = df_test[features]
    y_test = df_test['PM25_Target']

    sample_weights = np.where(df_train['station_id'] == ANCHOR_ID, 1.0, 1.0)
    rf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_pred - y_test)

    fold_results.append({'Sensor': test_sensor, 'R2': r2, 'RMSE': rmse, 'Bias': bias})
    feature_importances_list.append(rf.feature_importances_)

    print(
        f"Fold {fold:2d} (Test Sensor {test_sensor:>6s}): R² = {r2:5.2f} | RMSE = {rmse:4.2f} µg/m³ | Bias = {bias:5.2f} µg/m³")

# =========================================================
# 5. Summary Evaluation
# =========================================================
print("\n" + "=" * 50)
print("OVERALL PERFORMANCE (Complete-Case Analysis)")
print("=" * 50)
results_df = pd.DataFrame(fold_results)
print(f"Mean R² Score:   {results_df['R2'].mean():.2f}")
print(f"Mean RMSE:       {results_df['RMSE'].mean():.2f} µg/m³")
print(f"Mean Bias:       {results_df['Bias'].mean():.2f} µg/m³")
print("-" * 50)

if feature_importances_list:
    mean_importances = np.mean(feature_importances_list, axis=0)
    importance_df = pd.DataFrame({'Feature': features, 'Importance (%)': mean_importances * 100})
    importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)

    print("FEATURE IMPORTANCE:")
    for idx, row in importance_df.iterrows():
        print(f"{row['Feature']:>22}: {row['Importance (%)']:5.1f} %")
else:
    print("Insufficient data for feature importance analysis.")
print("=" * 50)

# =========================================================
# 6. Visualization & IEEE Figure Export
# =========================================================
if feature_importances_list:
    save_path_pdf = FIGURES_DIR / "feature_importance.pdf"
    save_path_png = FIGURES_DIR / "feature_importance.png"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Transparent background settings
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Feature display labels
    feature_names = {
        "IDW_car": "Traffic intensity (cars)",
        "IDW_truck": "Traffic intensity (trucks)",
        "icon_temp_2m": "Air temperature (2 m)",
        "icon_wind_speed": "Wind speed",
        "icon_wind_direction": "Wind direction",
        "icon_precip_12h": "Precipitation (12 h)",
        "icon_relative_humidity": "Relative humidity",
        "MAIAC_AOD": "MAIAC AOD",
        "S5P_NO2": "Sentinel-5P NO$_2$",
        "S5P_UVAI": "Sentinel-5P UVAI",
        "ERA5_BLH": "ERA5 PBLH",
        "tpi_300m": "TPI 300 m",
        "tpi_1500m": "TPI 1500 m"
    }

    importance_df["Feature_Label"] = (
        importance_df["Feature"]
        .map(feature_names)
        .fillna(importance_df["Feature"])
    )

    sns.barplot(
        x='Importance (%)',
        y='Feature_Label',
        data=importance_df,
        palette='viridis',
        ax=ax
    )

    ax.set_title(
        "Relative Feature Importance (Random Forest)",
        fontsize=16,
        fontweight="bold",
        pad=15,
        color="black"
    )
    ax.set_xlabel("Importance [%]", fontsize=12, color="black")
    ax.set_ylabel("Predictor Variable", fontsize=12, color="black")

    ax.tick_params(colors="black", which="both")
    plt.xticks(color="black")
    plt.yticks(color="black")
    ax.grid(axis='x', linestyle='--', alpha=0.4, color="gray")

    plt.tight_layout()

    plt.savefig(save_path_pdf, format="pdf", transparent=True, bbox_inches="tight")
    plt.savefig(save_path_png, format="png", transparent=True, dpi=300, bbox_inches="tight")

    print(f"\nFeature Importance figure successfully saved to:\n PDF: {save_path_pdf}\n PNG: {save_path_png}")
    plt.show()