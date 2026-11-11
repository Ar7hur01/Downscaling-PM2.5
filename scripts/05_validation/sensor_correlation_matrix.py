# This script computes a Pearson correlation matrix for PM2.5 time series
# across all sensors. It identifies outliers and generates an IEEE-styled,
# transparent heatmap for quality assurance and paper publication.

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# Define directories
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
FIGURES_DIR = BASE_DIR.parent / "data" /"output" / "Validationgraphs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset
file_path = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"

# Output figures
save_path_pdf = FIGURES_DIR / "sensor_correlation_matrix.pdf"
save_path_png = FIGURES_DIR / "sensor_correlation_matrix.png"

# STRATEGY 1: Skip if figures already exist
if save_path_pdf.exists() and save_path_png.exists():
    print(f"Figure {save_path_pdf.name} already exists. Skipping matrix generation.")
    exit()

# =========================================================
# 2. Load and Prepare Data
# =========================================================
print("Loading master dataset for quality assurance...")
df = pd.read_csv(file_path, sep=";", decimal=",")

df["station_id"] = df["station_id"].astype(str).str.strip()

# =========================================================
# 3. Restructure Dataset (Pivot)
# =========================================================
# Create a matrix where rows are dates, columns are sensors, and values are PM2.5
df_pivot = df.pivot_table(
    index="date", columns="station_id", values="PM25_Target"
)
print(f"Data structure pivoted. Number of shared measurement days: {len(df_pivot)}")

# =========================================================
# 4. Calculate and Sort Correlation Matrix
# =========================================================
corr_matrix = df_pivot.corr()

# Define desired visual sorting order
start_sensor = ['1']  # Anchor station (Kopfklinik)
end_sensors = ['9094', '46505', '15449', '5456']  # Problematic/Outlier sensors
middle_sensors = [col for col in corr_matrix.columns if col not in start_sensor + end_sensors]

desired_order = start_sensor + middle_sensors + end_sensors

# Reindex matrix to match the new order (ignores sensors missing from the dataset)
valid_order = [col for col in desired_order if col in corr_matrix.columns]
corr_matrix = corr_matrix.reindex(index=valid_order, columns=valid_order)

# =========================================================
# 5. Rename Columns and Indices for the Plot
# =========================================================
rename_mapping = {
    '1': 'AS',          # Anchor Station
    '9094': 'Group 1',
    '46505': 'Group 1',
    '15449': 'Group 2',
    '5456': 'Group 2'
}
corr_matrix = corr_matrix.rename(index=rename_mapping, columns=rename_mapping)

# =========================================================
# 6. Visualization (Heatmap)
# =========================================================
print("Generating correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

# Transparent background for IEEE format
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=0.3,
    vmax=1.0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Pearson Correlation Coefficient"},
)

# Titles and labels
ax.set_title(
    r"Correlation matrix $\mathrm{PM}_{2.5}$-time series",
    fontsize=16,
    fontweight="bold",
    pad=15,
    color="black",
)
ax.set_xlabel("Sensor ID", fontsize=12, color="black")
ax.set_ylabel("Sensor ID", fontsize=12, color="black")

# Styling: force black text
ax.tick_params(colors="black", which="both")
plt.xticks(rotation=45, color="black")
plt.yticks(rotation=0, color="black")

cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.label.set_color("black")
cbar.ax.tick_params(colors="black")

plt.tight_layout()

# =========================================================
# Highlight Specific Sensor IDs (Outliers)
# =========================================================
sensor_colors = {
    "Group 1": "red",
    "Group 2": "red"
}

# Color X-axis labels
for label in ax.get_xticklabels():
    sensor_name = label.get_text()
    label.set_color(sensor_colors.get(sensor_name, "black"))

# Color Y-axis labels
for label in ax.get_yticklabels():
    sensor_name = label.get_text()
    label.set_color(sensor_colors.get(sensor_name, "black"))

# =========================================================
# 7. Export and Display
# =========================================================
plt.savefig(save_path_pdf, format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(save_path_png, format="png", transparent=True, dpi=300, bbox_inches="tight")

print("-" * 60)
print(f"SUCCESS! Correlation matrix saved to:\n PDF: {save_path_pdf}\n PNG: {save_path_png}")
print("-" * 60)

plt.show()