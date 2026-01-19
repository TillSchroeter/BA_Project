#%%-------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import functions_IMU
import functions_generell
import functions_PRESSURE

#%% Daten einlesen für alle Teilnehmer-------------------------------------------------------------------------------------------
# liste der Teilneher
participants = ['ID_1_Dabisch_Samuel', 'ID_2_Pohl_Jannis', 'ID_3_Kleber_Christian',
                'ID_4_Schröter_Till', 'ID_5_Zaschke_Lenard', 'ID_6_Petroll_Finn', 'ID_7_Gruber_Julius']

# Alle Daten der Teilnehmer einlesen und verarbeiten
all_data = {}

for participant in participants:
    folder_path = os.path.join('data_final', participant)
    
    # Lade die 6 CSV-Dateien für den Teilnehmer
    real_1, real_2, vr_1, vr_2, mvc_beine, mvc_hals = functions_generell.load_csvs(folder_path)
    
    # Speichere die DataFrames im Dictionary
    all_data[participant] = {
        'REAL_1': real_1,
        'REAL_2': real_2,
        'VR_1': vr_1,
        'VR_2': vr_2,
        'MVC_Beine': mvc_beine,
        'MVC_Hals': mvc_hals
    }
    
    print(f"Daten für {participant} geladen")

print(f"\nAlle Daten geladen! {len(all_data)} Teilnehmer erfolgreich eingelesen.")


#%% Verarbeite alle Teilnehmer und erstelle CSVs mit Sprung-Eigenschaften-------------------------------------------------------------------------------------------
measurements = ['REAL_1', 'REAL_2', 'VR_1', 'VR_2']
output_folder = 'jump_analysis_results'

# peak height für jeden teilneher für die jump detection
# peak_hight = [1200, 1080, 1200, 1100, 1250, 1100, 1200]

# Erstelle Ausgabeverzeichnis, falls nicht vorhanden
os.makedirs(output_folder, exist_ok=True)

for participant in participants:
    print(f"\nVerarbeite {participant}...")
    
    all_jumps_data = []
    
    # Durchlaufe alle 4 Messungen
    for measurement in measurements:
        df = all_data[participant][measurement]
        
        # Identifiziere Sprünge in dieser Messung
        jumps = functions_PRESSURE.identify_jumps(df, flight_threshold=155, min_flight_sec=0.2, buffer=0.75)
        functions_PRESSURE.visualize_jumps(df, jumps, participant, measurement)

        # Füge Measurement-Info zu jedem Sprung hinzu
        for jump in jumps:
            jump['messung'] = measurement
            all_jumps_data.append(jump)
        
        print(f"  {measurement}: {len(jumps)} Sprünge gefunden")
    
    # Erstelle DataFrame aus allen Sprüngen
    jumps_df = pd.DataFrame(all_jumps_data)
    
    # Speichere als CSV
    output_file = os.path.join(output_folder, f'{participant}_jumps.csv')
    jumps_df.to_csv(output_file, index=False)
    
    print(f"  ✓ Gesamt: {len(all_jumps_data)} Sprünge")
    print(f"  ✓ CSV gespeichert: {output_file}")

print(f"\n{'='*80}")
print(f"Verarbeitung abgeschlossen!")
print(f"Alle CSVs wurden im Ordner '{output_folder}' gespeichert.")
print(f"{'='*80}")

#%% Beispiel: Lade und normiere Sprungdaten für alle-------------------------------------------------------------------------------------------

print(f"\nStarte Zeit-Normierung...")

# Dieses Dictionary wird am Ende alle normierten Sprünge aller Probanden enthalten
# Struktur: all_normalized_data['ID_1_Samuel']['REAL_1_jump_1'] -> DataFrame
all_normalized_data = {}

for participant in participants:
    print(f"Normiere Daten für {participant}...")
    all_normalized_data[participant] = {}
    
    for measurement in measurements:
        df = all_data[participant][measurement]
        
        # Rufe die v2 deiner Funktion auf (inkl. Messung-Filter)
        # Wir normieren hier auf 100 Punkte (Standard)
        norm_jumps_dict = functions_generell.time_normalize_jumps(
            df, 
            participant, 
            measurement, 
            jumps_csv_folder=output_folder,
            normalize_points=100
        )
        
        # Die 6 normierten Sprünge in das Haupt-Dictionary speichern
        all_normalized_data[participant].update(norm_jumps_dict)
        
        print(f"  ✓ {measurement}: {len(norm_jumps_dict)} Sprünge normiert")

print(f"\nFertig! Alle Sprünge sind nun zeitnormiert verfügbar.")

# Beispiel-Abfrage: Zugriff auf die Daten
# Wenn du z.B. den 1. Sprung von Samuel in REAL_1 sehen willst:
# sample_jump = all_normalized_data['ID_1_Dabisch_Samuel']['REAL_1_jump_1']
# print(sample_jump.head())

# %% plotten und speicjhern des Knie Flexion winkels-------------------------------------------------------------------------------------------
print(f"\nStarte Generierung der Einzel-Plots...")

# Äußere Schleife: Geht jeden Teilnehmer durch
for participant in participants:
    print(f"Verarbeite Plots für: {participant}")
    
    # Innere Schleife: Geht jede der 4 Messungen pro Teilnehmer durch
    for measurement in ['REAL_1', 'REAL_2', 'VR_1', 'VR_2']:
        
        # Aufruf deiner Plot-Funktion (mit Subplots und Mean-Overlaid)
        # Hier wird pro Messung ein Grafikpaar erstellt
        functions_IMU.plot_normalized_jumps(
            all_normalized_data[participant], 
            participant, 
            measurement, 
            'RT Knee Flexion (deg)' 
        )
        
    print(f"  ✓ Alle 4 Messungen für {participant} geplottet.")

print(f"\n{'='*30}")
print("ALLE GRAFIKEN ERFOLGREICH ERSTELLT")
print(f"Speicherort: Pictures_Test/RT_Knee_Flexion_(deg)/")
print(f"{'='*30}")
# %%
