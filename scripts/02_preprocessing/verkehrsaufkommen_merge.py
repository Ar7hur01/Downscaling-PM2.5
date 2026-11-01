import pandas as pd
import re

base_path = r"C:\Users\AD\Desktop\Uni Würzburg\EAGLE\InnoLab\Daten (Verkehrsmesswerte)"

Data_verkehrsaufkommen = f"{base_path}\\Detektor-Messquerschnitt-Verkehrsmesswerte - Verarbeitet-240501_240831.csv"
# WICHTIG: car_vol und truck_vol geladen!
df_traffic = pd.read_csv(Data_verkehrsaufkommen, sep=';', decimal=".",
                         usecols=['Name', 'Time', 'Long Name', 'processed_car_vol', 'processed_truck_vol'])

koord_verkehrsaufkommen_data = f"{base_path}\\TEU_Koordinaten.csv"
df_teu = pd.read_csv(
    koord_verkehrsaufkommen_data, sep=r'[;,]', engine='python', decimal='.',
    header=None, usecols=[0, 1, 2, 3], names=['Name', 'Lat', 'Lon', 'Angle_Arrow'])
df_teu = df_teu.dropna()


# --- DIE NEUE, ROBUSTE MAPPING-FUNKTION ---
def normalize_traffic_name(row):
    name = str(row['Name']).upper()
    long_name = str(row['Long Name']).upper()

    # 1. Die harten Ausnahmen (Edge Cases) abfangen
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

    # 2. Generelle Bereinigung für den Rest
    # 'MQ-' entfernen und Bindestriche/Leerzeichen durch Unterstriche ersetzen (Ziel: TEU_XX)
    clean_name = name.replace('MQ-', '')
    clean_name = clean_name.replace('-', '_').replace(' ', '_')

    # Behebt das Problem mit Doppel-IDs wie TEU_109110_109 -> TEU_109
    match = re.search(r'TEU_\d+_(\d+)(_[A-Z]+)?$', clean_name)
    if match:
        station_id = match.group(1)
        suffix = match.group(2) if match.group(2) else ''
        return f"TEU_{station_id}{suffix}"

    return clean_name


# Koordinaten-Namen einfach nur sauber formatieren (TEU_24_N etc.)
def normalize_coord_name(name):
    return str(name).upper().replace('-', '_').strip()


# --- Anwendung auf die DataFrames ---

# WICHTIG: axis=1 übergeben, damit die Funktion Zugriff auf die ganze Zeile (Name UND Long Name) hat!
df_traffic['Join_Key'] = df_traffic.apply(normalize_traffic_name, axis=1)
df_teu['Join_Key'] = df_teu['Name'].apply(normalize_coord_name)

# Jetzt können wir sinnvoll mergen
df_merged = pd.merge(
    df_traffic,
    df_teu,
    on='Join_Key',
    how='left',
    suffixes=('_Traffic', '_Coord')
)

# Kurzer Check: Gibt es noch Traffic-Stationen, die keine Koordinaten gefunden haben?
fehlende_koordinaten = df_merged[df_merged['Lat'].isna()]['Join_Key'].unique()
if len(fehlende_koordinaten) > 0:
    print("\nWARNUNG: Für folgende generierte Join_Keys wurden KEINE Koordinaten gefunden:")
    print(fehlende_koordinaten)
else:
    print("\nERFOLG: Alle Verkehrsstationen wurden erfolgreich mit Koordinaten gematcht!")

# Unnötige Join-Key Spalte wieder fallen lassen, wenn gewünscht
# df_merged = df_merged.drop(columns=['Join_Key'])

output_file = f"{base_path}\\Verkehrsdaten_mit_Koordinaten_merged_cars_n_trucks.csv"

# Exportieren
df_merged.to_csv(
    output_file,
    index=False,
    sep=';',
    decimal='.',
    encoding='utf-8-sig'
)

print(f"\nDatei erfolgreich gespeichert unter: {output_file}")