# This script visualizes the Random Forest model's prediction precision (RMSE)
# in the context of natural PM2.5 data variance across the study area.
# It generates an IEEE-styled distribution plot comparing RMSE error margins
# against the dataset range and standard deviation.

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# Define directories relative to the script
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
FIGURES_DIR = BASE_DIR.parent / "data" / "output" / "Validationgraphs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset and output figure paths
file_path = PROCESSED_DIR / "Features_PM25_NDVI_TPI_S5P_ERA5_MAIAC_Traffic_Weather.csv"
output_pdf = FIGURES_DIR / "rmse_context_distribution.pdf"
output_png = FIGURES_DIR / "rmse_context_distribution.png"

# STRATEGY 1: Skip if already generated
if output_pdf.exists() and output_png.exists():
    print(f"Figure {output_pdf.name} already exists. Skipping figure generation.")
    exit()

# =========================================================
# 2. Data Loading & Statistical Calculations
# =========================================================
print("Loading dataset for variance analysis...")
df = pd.read_csv(file_path, sep=";", decimal=",")

# Clean target column and remove missing values
df['PM25_Target'] = pd.to_numeric(df['PM25_Target'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
df = df.dropna(subset=['PM25_Target']).copy()

# Set calculated LLO-CV RMSE value (from Random Forest script)
rmse_wert = 1.77

# Calculate distribution statistics
mean_pm25 = df['PM25_Target'].mean()
std_pm25 = df['PM25_Target'].std()
rrmse = (rmse_wert / mean_pm25) * 100
pm25_min = df['PM25_Target'].min()
pm25_max = df['PM25_Target'].max()

print(f"Mean PM2.5 Concentration: {mean_pm25:.2f} µg/m³")
print(f"Standard Deviation:        {std_pm25:.2f} µg/m³")
print(f"Relative RMSE (rRMSE):     {rrmse:.1f} %")
print(f"Data Range:                {pm25_min:.1f} to {pm25_max:.1f} µg/m³\n")

# =========================================================
# 3. IEEE-Style Plotting
# =========================================================
print("Generating PM2.5 error context visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Set background transparency for IEEE publication format
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 1. Density distribution of actual PM2.5 observations
sns.histplot(
    df['PM25_Target'],
    kde=True,
    color="#1f77b4",
    stat="density",
    alpha=0.4,
    linewidth=1,
    ax=ax,
    label="PM2.5 Data Distribution"
)

# 2. Mean reference line
ax.axvline(
    mean_pm25,
    color='black',
    linestyle='-',
    linewidth=2.5,
    label=f'Mean ({mean_pm25:.1f} µg/m³)'
)

# 3. Model RMSE error tolerance band
ax.axvspan(
    mean_pm25 - rmse_wert,
    mean_pm25 + rmse_wert,
    color='red',
    alpha=0.25,
    label=f'Model Error (RMSE: ±{rmse_wert:.2f})'
)

# 4. Standard deviation lines
ax.axvline(
    mean_pm25 - std_pm25,
    color='orange',
    linestyle='--',
    linewidth=2,
    label=f'Std Dev (±{std_pm25:.2f})'
)
ax.axvline(
    mean_pm25 + std_pm25,
    color='orange',
    linestyle='--',
    linewidth=2
)

# 5. Summary text box
textstr = f"Data Range: {pm25_min:.1f} to {pm25_max:.1f} µg/m³\nrRMSE: {rrmse:.1f} %"
props = dict(boxstyle="square", facecolor='grey', edgecolor='black', alpha=0.8)
ax.text(
    0.95, 0.5,
    textstr,
    transform=ax.transAxes,
    fontsize=11,
    color="black",
    verticalalignment='center',
    horizontalalignment='right',
    bbox=props
)

# 6. Styling and Axis Formatting
ax.set_title(
    "Model Precision vs. Natural PM2.5 Variance",
    fontsize=15,
    fontweight="bold",
    pad=15,
    color="black"
)
ax.set_xlabel(r"Observed $\mathrm{PM}_{2.5}$ Concentration [µg/m³]", fontsize=12, color="black")
ax.set_ylabel("Probability Density", fontsize=12, color="black")
ax.tick_params(colors="black", which="both")

# Legend formatting
legend = ax.legend(loc='upper right', facecolor='grey', framealpha=0.8, edgecolor="black")
for text in legend.get_texts():
    text.set_color("black")

plt.tight_layout()

# =========================================================
# 4. Export Figure
# =========================================================
plt.savefig(output_pdf, format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(output_png, format="png", transparent=True, dpi=300, bbox_inches="tight")

print("-" * 60)
print(f"SUCCESS! RMSE context distribution plot saved to:\n PDF: {output_pdf}\n PNG: {output_png}")
print("-" * 60)

plt.show()