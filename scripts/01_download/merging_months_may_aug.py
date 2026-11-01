## In this script I will merge the downloaded months from the sensor.community
## May to August to generate one complete csv-file

import os
import pandas as pd

# 1. Deinen Dateipfad definieren
base_path = r"C:\Users\AD\Desktop\Uni Würzburg\EAGLE\InnoLab\Daten (Sensor_Community)"

# 2. Die Dateinamen der 4 Monate auflisten 
# (Bitte passe die Endung .csv an, falls die Dateien anders heißen)
file_names = [
    "local_time_may.csv",
    "local_time_june.csv",
    "local_time_july.csv",
    "local_time_august.csv"]

# Leere Liste, um die einzelnen DataFrames temporär zu speichern
dataframes = []

print("Starte den Import...")

# 3. Schleife über alle Monats-Dateien
for file in file_names:
    file_path = os.path.join(base_path, file)

    # Fehlerabfrage: Prüfen, ob die Datei im Ordner wirklich existiert
    if os.path.exists(file_path):
        # Sensor.Community Daten nutzen standardmäßig oft das Semikolon als Trennzeichen
        df = pd.read_csv(file_path, sep=';')
        dataframes.append(df)
        print(f"Erfolgreich geladen: {file} (Zeilen: {len(df)})")
    else:
        print(f"WARNUNG: Datei nicht gefunden -> {file_path}")

# 4. Alle Monate zu einem einzigen DataFrame zusammenfügen
# ignore_index=True setzt den Index sauber von 0 bis zum Ende neu
merged_df = pd.concat(dataframes, ignore_index=True)

print(f"\nMerge erfolgreich! Gesamtdatensatz hat {len(merged_df)} Zeilen.")

# 5. Den kombinierten Datensatz als neue CSV exportieren
output_file = os.path.join(base_path, "sensor_community_complete_may_august.csv")
merged_df.to_csv(output_file, index=False, sep=';')

print(f"Fertige Datei exportiert unter: {output_file}")