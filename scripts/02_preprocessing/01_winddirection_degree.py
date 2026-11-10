# In this script ICON-D2-Data will be processed,
# to convert "u" and "v" wind components into meteorological
# wind direction in degrees (1° - 360°).

import numpy as np
import rasterio
from datetime import datetime, timedelta
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Define paths entirely with pathlib
# Note: Adjust .parent if your script is nested deeper
FILE_PATH = BASE_DIR.parent / "data" / "raw" / "New_ICON_D2_Rasters"

u_dir = FILE_PATH / "U_Wind"
v_dir = FILE_PATH / "V_Wind"
out_dir = FILE_PATH / "Wind_Direction"

out_dir.mkdir(parents=True, exist_ok=True)

start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 8, 31)
days_count = (end_date - start_date).days + 1

print(f"Starting batch calculation of wind direction for {days_count} days...")

# =========================================================
# 2. Daily Calculation Loop
# =========================================================
for i in range(days_count):
    current_date = start_date + timedelta(days=i)
    date_str = current_date.strftime("%Y-%m-%d")

    u_file = u_dir / f"ICON_D2_U_{date_str}.tif"
    v_file = v_dir / f"ICON_D2_V_{date_str}.tif"
    out_file = out_dir / f"wind_dir_{date_str}.tif"

    # STRATEGY 1: Skip if already processed
    if out_file.exists():
        if (i + 1) % 10 == 0 or (i + 1) == days_count:
            print(f"[{i+1}/{days_count}] {date_str} already exists. Skipping...")
        continue

    # Check if both source files exist
    if not u_file.exists() or not v_file.exists():
        print(f"[{date_str}] Skipped: Source files missing.")
        continue

    try:
        # Open GeoTIFFs and read raster values as NumPy arrays
        with rasterio.open(u_file) as src_u, rasterio.open(v_file) as src_v:
            u_data = src_u.read(1)
            v_data = src_v.read(1)

            # Copy metadata from the original U-wind raster
            meta = src_u.meta.copy()

            # Calculate meteorological wind direction in degrees
            # atan2(-u, -v) gives angle in radians, deg converts to degrees, % 360 restricts to 0-360
            wind_dir = np.mod(np.degrees(np.arctan2(-u_data, -v_data)), 360.0)

            # Update metadata to Float32 for precise degree values
            meta.update({
                "dtype": "float32",
                "nodata": -9999.0,
            })

            # Handle NaN values (e.g., outside the model domain)
            wind_dir = np.nan_to_num(wind_dir, nan=-9999.0)

            # Write the new GeoTIFF
            with rasterio.open(out_file, "w", **meta) as dst:
                dst.write(wind_dir.astype("float32"), 1)

        if (i + 1) % 10 == 0 or (i + 1) == days_count:
            print(f"[{i+1}/{days_count}] Progress: {date_str} successfully processed.")

    except Exception as e:
        print(f"[{date_str}] Error processing: {e}")

print(f"\nDone! All wind direction rasters are located at:\n{out_dir}")