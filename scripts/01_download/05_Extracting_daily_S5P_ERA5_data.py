import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd
import ee

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    # Works when executing a standard .py script file
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments (Jupyter, IPython, PyCharm selection)
    BASE_DIR = Path.cwd()

# Define the relative path to the Sensor Community CSV file.
# Adjust the exact folder structure (e.g., 'data' / 'raw') to match your repository.
FILE_PATH = BASE_DIR.parent.parent / "data" / "raw" / "sensor_community_with_kopfklinik_may_august.csv"

# =========================================================
# 2. Earth Engine Initialization
# =========================================================
project_id = input("Type in your Google Earth Engine project ID: ").strip()
ee.Authenticate()
ee.Initialize(project=project_id)

# =========================================================
# 3. Define Parameters
# =========================================================
START_DATE = '2024-05-01'
END_DATE = '2024-08-31'

# =========================================================
# 4. Read CSV and build FeatureCollection
# =========================================================
df = pd.read_csv(FILE_PATH, sep=';', decimal=',')
stations_df = df[['sensor_id', 'lat', 'lon']].drop_duplicates().dropna()

ee_features = []
for index, row in stations_df.iterrows():
    try:
        lon_val = float(str(row['lon']).replace(',', '.'))
        lat_val = float(str(row['lat']).replace(',', '.'))
        geom = ee.Geometry.Point([lon_val, lat_val])
        feat = ee.Feature(geom, {'station_id': str(row['sensor_id'])})
        ee_features.append(feat)
    except ValueError:
        pass

# This replaces the old stations_list!
stations_fc = ee.FeatureCollection(ee_features)


# =========================================================
# 5. Quality Filters
# =========================================================
def filter_no2(image):
    # Since qa_value is missing, use cloud_fraction (< 0.5 means largely cloud-free)
    cloud = image.select('cloud_fraction')
    mask = cloud.lt(0.5)
    return image.updateMask(mask).select('tropospheric_NO2_column_number_density')


def filter_uvai(image):
    # The AER_AI product in GEE lacks quality or cloud bands.
    # Directly select the absorbing aerosol index.
    return image.select('absorbing_aerosol_index')


# =========================================================
# 6. Daily Extraction Function
# =========================================================
def get_daily_data(date):
    current_date = ee.Date(date)
    next_date = current_date.advance(1, 'day')

    # 1. Fetch satellite data (S5P)
    no2_col = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2") \
        .filterBounds(stations_fc) \
        .filterDate(current_date, next_date) \
        .map(filter_no2)

    uvai_col = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI") \
        .filterBounds(stations_fc) \
        .filterDate(current_date, next_date) \
        .map(filter_uvai)

    # 2. Fetch ground meteorology (ERA5-Land: 9 km resolution)
    era5_land_col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
        .filterBounds(stations_fc) \
        .filterDate(current_date, next_date) \
        .select(['temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m'])

    # 3. Fetch atmospheric boundary layer (ERA5 Global: 31 km resolution)
    era5_atm_col = ee.ImageCollection("ECMWF/ERA5/HOURLY") \
        .filterBounds(stations_fc) \
        .filterDate(current_date, next_date) \
        .select(['boundary_layer_height'])

    # Calculate daily averages
    no2_daily = no2_col.mean()
    uvai_daily = uvai_col.mean()
    era5_land_daily = era5_land_col.mean()
    era5_atm_daily = era5_atm_col.mean()

    # Combine into a single multi-band image
    combined_image = ee.Image.cat([no2_daily, uvai_daily, era5_land_daily, era5_atm_daily])

    # Extract data exactly at the station coordinates
    extracted_data = combined_image.reduceRegions(
        collection=stations_fc,
        reducer=ee.Reducer.first(),
        scale=1000
    )

    # Append date
    def set_date(feature):
        return feature.set('date', current_date.format('YYYY-MM-dd'))

    return extracted_data.map(set_date)


# =========================================================
# 7. Iteration & Execution
# =========================================================
start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
date_list = [(start_dt + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_dt - start_dt).days + 1)]

print(f"Extracting data for {len(date_list)} days and {len(stations_df)} stations. This may take a moment...")

ee_dates = ee.List(date_list)

# Flatten the daily collections into one large FeatureCollection
daily_features = ee.FeatureCollection(ee_dates.map(get_daily_data)).flatten()

# =========================================================
# 8. Export as a Background Task to Google Drive
# =========================================================
print("Preparing the export task...")

# Filter out empty results to keep the CSV clean
clean_features = daily_features.filter(ee.Filter.notNull(['tropospheric_NO2_column_number_density']))

task = ee.batch.Export.table.toDrive(
    collection=clean_features,
    description='cloud0_5_Wuerzburg_S5P_ERA5_Extraction_2024',
    folder='GEE_Exports',  # GEE creates this folder in your Google Drive
    fileFormat='CSV'
)

# Start computation on Google servers
task.start()
print("Task started successfully! GEE is now processing in the background.")

# =========================================================
# 9. Check Calculation Status
# =========================================================
# A small loop to update on GEE's progress every 30 seconds
while task.active():
    print(f"Status: {task.status()['state']}... (Please wait)")
    time.sleep(30)

print(f"Done! Final status: {task.status()['state']}")
print("You can now download the finished CSV file from your Google Drive ('GEE_Exports' folder).")