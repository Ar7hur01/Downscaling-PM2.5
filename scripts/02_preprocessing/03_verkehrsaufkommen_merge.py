# This script merges raw traffic volume data with stationary coordinate data.
# It uses a custom string normalization function to fix inconsistencies
# in station names and handles double-IDs cleanly.

import re
import pandas as pd
from pathlib import Path

# =========================================================
# 1. Settings & Relative Paths
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Define the traffic data directory
TRAFFIC_DIR = BASE_DIR.parent / "data" / "raw" / "Traffic"

# Input files
traffic_input_file = TRAFFIC_DIR / "Detektor-Messquerschnitt-Verkehrsmesswerte - Verarbeitet-240501_240831.csv"
coord_input_file = TRAFFIC_DIR / "TEU_Koordinaten.csv"

# Output file
output_file = TRAFFIC_DIR / "Verkehrsdaten_mit_Koordinaten_merged_cars_n_trucks.csv"

# STRATEGY 1: Skip if already processed
if output_file.exists():
    print(f"File {output_file.name} already exists. Skipping merge process.")
    exit()

print("Loading traffic and coordinate datasets...")

# =========================================================
# 2. Load Datasets
# =========================================================
# Load traffic volumes (Requires car_vol and truck_vol!)
df_traffic = pd.read_csv(
    traffic_input_file,
    sep=';',
    decimal=".",
    usecols=['Name', 'Time', 'Long Name', 'processed_car_vol', 'processed_truck_vol']
)

# Load coordinates
df_teu = pd.read_csv(
    coord_input_file,
    sep=r'[;,]',
    engine='python',
    decimal='.',
    header=None,
    usecols=[0, 1, 2, 3],
    names=['Name', 'Lat', 'Lon', 'Angle_Arrow']
)
df_teu = df_teu.dropna()

# =========================================================
# 3. Robust Mapping Functions
# =========================================================
def normalize_traffic_name(row):
    """Normalizes the station name to a clean join key (e.g., TEU_XX)."""
    name = str(row['Name']).upper()
    long_name = str(row['Long Name']).upper()

    # 1. Handle hardcoded edge cases
    if 'MQ-TEU-24-1' in name or ('TEU-24' in name and 'SÜD' in long_name):
        return 'TEU_24_S'
    if 'TEU-24' in name and 'NORD' in long_name:
        return 'TEU_24_N'
    if 'TEU-59 SA' in name or 'TEU-59-SA' in name:
        return 'TEU_59_SA'
    if 'TEU-68-VMZ' in name:
        return 'TEU_68'
    if 'TEU-69-VMZ' in name:
        return 'TEU_69'

    # 2. General cleanup for the rest
    clean_name = name.replace('MQ-', '').replace('-', '_').replace(' ', '_')

    # Fix double-IDs like TEU_109110_109 -> TEU_109
    match = re.search(r'TEU_\d+_(\d+)(_[A-Z]+)?$', clean_name)
    if match:
        station_id = match.group(1)
        suffix = match.group(2) if match.group(2) else ''
        return f"TEU_{station_id}{suffix}"

    return clean_name

def normalize_coord_name(name):
    """Normalizes the coordinate name to match the traffic join key."""
    return str(name).upper().replace('-', '_').strip()

# =========================================================
# 4. Apply Mapping & Merge
# =========================================================
print("Applying naming normalization and merging data...")

# axis=1 is required so the function can access both 'Name' and 'Long Name' columns
df_traffic['Join_Key'] = df_traffic.apply(normalize_traffic_name, axis=1)
df_teu['Join_Key'] = df_teu['Name'].apply(normalize_coord_name)

# Perform Left Join
df_merged = pd.merge(
    df_traffic,
    df_teu,
    on='Join_Key',
    how='left',
    suffixes=('_Traffic', '_Coord')
)

# Validation Check: Did any traffic stations fail to find coordinates?
missing_coords = df_merged[df_merged['Lat'].isna()]['Join_Key'].unique()

if len(missing_coords) > 0:
    print("\nWARNING: No coordinates were found for the following Join_Keys:")
    print(missing_coords)
else:
    print("\nSUCCESS: All traffic stations successfully matched with coordinates!")

# =========================================================
# 5. Export
# =========================================================
# Export with decimal=',' to ensure downstream scripts parse it correctly
df_merged.to_csv(
    output_file,
    index=False,
    sep=';',
    decimal=',',
    encoding='utf-8-sig'
)

print("-" * 60)
print(f"File successfully saved to:\n{output_file}")
print("-" * 60)