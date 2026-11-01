## reran for the months May, June, July, August

## Dieses Skript zieht sich die Sensordaten aus dem Archiv der Sensor.Community über einen API
## und schreibt diese mit Datum, Sensornummer, Lan, Lot und den jeweiligen PM2.5 & PM10-Werten
## in eine CSV-Datei. In dem Fall zu unterschiedlichen Tageszeiten, aber auch als Durchschnitt.

import requests as rq
import pandas as pd
from datetime import datetime, timedelta
import io
import osmnx as ox
from monthly_download_community_sensors import get_wuerzburg_sensors_gdf


def download_wuerzburg_raw_data(start_date, end_date):
    # 1. GRENZEN & SENSOREN
    boundary_gdf = ox.geocode_to_gdf(["Würzburg, Germany"])
    wuerzburg_geometry = boundary_gdf.union_all()
    sensors_gdf = get_wuerzburg_sensors_gdf(wuerzburg_geometry)

    if sensors_gdf.empty:
        print("Keine Sensoren gefunden.")
        return None

    # Standort-Mapping vorbereiten
    location_map = sensors_gdf[['sensor_id', 'geometry']].copy()
    location_map['lon'] = location_map.geometry.x
    location_map['lat'] = location_map.geometry.y
    location_map = location_map[['sensor_id', 'lon', 'lat']]

    wuerzburg_ids = sensors_gdf['sensor_id'].tolist()

    all_data = []
    current_date = start_date

    # 2. DATEN DOWNLOAD
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        year_str = current_date.strftime("%Y")

        for s_id in wuerzburg_ids:
            file_url = f"https://archive.sensor.community/{year_str}/{date_str}/{date_str}_sds011_sensor_{s_id}.csv.gz"
            try:
                response = rq.get(file_url, timeout=5)
                if response.status_code == 200:
                    df = pd.read_csv(io.BytesIO(response.content), sep=";", compression='gzip')
                    # Nur relevante Spalten behalten
                    df = df[['timestamp', 'sensor_id', 'P1', 'P2']]
                    all_data.append(df)
            except Exception:
                continue

        print(f"Tag {date_str} verarbeitet...")
        current_date += timedelta(days=1)

    if not all_data:
        print("Keine Daten im Zeitraum gefunden.")
        return None

    # 3. DATEN ZUSAMMENFÜHREN (OHNE DURCHSCHNITTSBILDUNG)
    full_df = pd.concat(all_data, ignore_index=True)

    # Timestamp in Datetime-Objekt umwandeln für bessere Sortierung
    # 1. In Datetime umwandeln
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    # 2. Pandas sagen, dass die Originaldaten in UTC sind, und dann auf deutsche Zeit umrechnen
    full_df['timestamp'] = full_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    # 3. (Optional) Die Zeitzonen-Info (+02:00) wieder entfernen,
    # damit die CSV-Datei simple Zeitstempel wie "2024-06-20 14:00:00" behält
    full_df['timestamp'] = full_df['timestamp'].dt.tz_localize(None)

    # Verknüpfung mit den Geo-Koordinaten (lat/lon)
    final_df = pd.merge(full_df, location_map, on='sensor_id', how='left')

    # Spalten umbenennen für Klarheit
    final_df = final_df.rename(columns={
        'timestamp': 'Zeitstempel',
        'P1': 'PM10',
        'P2': 'PM2.5'
    })

    # Sortierung: Erst nach Sensor, dann chronologisch
    final_df = final_df.sort_values(by=['sensor_id', 'Zeitstempel'])

    return final_df[['Zeitstempel', 'sensor_id', 'lon', 'lat', 'PM2.5', 'PM10']]


# --- Ausführung ---
start = datetime(2024, 8, 1)
end = datetime(2024, 8, 31)

ergebnis = download_wuerzburg_raw_data(start, end)

if ergebnis is not None:
    print(f"\nErfolgreich {len(ergebnis)} Datensätze gesammelt.")
    print(ergebnis.head(10))

    # Speichern als CSV
    # Tipp: utf-8-sig hilft, damit Excel Umlaute und Formatierung direkt erkennt
    output_file = r"C:\Users\AD\Desktop\Uni Würzburg\EAGLE\InnoLab\Daten (Sensor_Community)\local_time_august.csv"
    ergebnis.to_csv(output_file, index=False, sep=";", decimal=",", encoding='utf-8-sig')
    print(f"\nDatei gespeichert unter: {output_file}")