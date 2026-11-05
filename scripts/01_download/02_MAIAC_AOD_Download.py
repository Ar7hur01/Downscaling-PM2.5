import os
from pathlib import Path
import requests
import pandas as pd
import ee

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Output folder relative to script location
OUTPUT_DIR = BASE_DIR.parent.parent / "data" / "raw" / "MAIAC_AOD"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-05-01"
END_DATE = "2024-08-31"

# Bounding Box for Würzburg (ICON-D2 extent)
LON_MIN, LON_MAX = 9.85, 10.05
LAT_MIN, LAT_MAX = 49.74, 49.85

# =========================================================
# 2. Earth Engine Initialization
# =========================================================
EE_PROJECT_ID = os.environ.get("EE_PROJECT_ID")

if not EE_PROJECT_ID:
    print("WARNING: No Earth Engine Project ID found in environment variables.")
    EE_PROJECT_ID = input("Please enter your Google Earth Engine Project ID: ").strip()

print("Authenticating and initializing Earth Engine...")
try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT_ID)

region = ee.Geometry.BBox(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

# =========================================================
# 3. Load MAIAC AOD ImageCollection
# =========================================================
print("Loading MCD19A2.061 (MAIAC) dataset...")

# Use 'Optical_Depth_047' (Blue Band) as PM2.5 proxy
collection = (
    ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
    .filterBounds(region)
    .select('Optical_Depth_047')
)

# =========================================================
# 4. Daily Aggregation & Download
# =========================================================
days = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
print(f"Starting download process for {len(days)} days...\n")

for date in days:
    date_str = date.strftime("%Y-%m-%d")
    filename = f"MAIAC_AOD_{date_str}.tif"
    filepath = OUTPUT_DIR / filename

    if filepath.exists():
        print(f"[{date_str}] File already exists. Skipping...")
        continue

    next_day_str = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    daily_coll = collection.filterDate(date_str, next_day_str)
    count = daily_coll.size().getInfo()

    if count == 0:
        print(f"[{date_str}] No data available (e.g., cloud cover). Skipping...")
        continue

    try:
        # Calculate daily mean (combines Terra + Aqua) and apply 0.001 scale factor
        daily_mean = daily_coll.mean().multiply(0.001).clip(region)

        # Request download URL (directly projected to EPSG:25832)
        url = daily_mean.getDownloadURL({
            'scale': 1000,
            'crs': 'EPSG:25832',
            'region': region,
            'format': 'GEO_TIFF'
        })

        filename = f"MAIAC_AOD_{date_str}.tif"
        filepath = OUTPUT_DIR / filename

        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(f"[{date_str}] SUCCESS: Downloaded {filename}")

    except Exception as e:
        print(f"[{date_str}] ERROR downloading: {e}")

print("\n" + "-" * 60)
print(f"DOWNLOAD COMPLETE! All raster files are located at:\n{OUTPUT_DIR}")
print("-" * 60)