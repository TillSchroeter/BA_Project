#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import functions_generell
import functions_PRESSURE

# liste der Teilneher
participants = ['ID_1_Dabisch_Samuel', 'ID_2_Pohl_Jannis', 'ID_3_Kleber_Christian',
                'ID_4_Schröter_Till', 'ID_5_Zaschke_Lenard', 'ID_6_Petroll_Finn', 'ID_7_Gruber_Julius']
# peak height für jeden teilneher für die jump detection
peak_hight = [1200, 1080, 1200, 1100, 1250, 1100, 1200]

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

#%% Beispiel: REAL_1 von Samuel anzeigen
# print (all_data['ID_1_Dabisch_Samuel']['REAL_1'].head())

# #%% Test: Visualisiere Kraftdaten
# samuel_real_1 = all_data['ID_1_Dabisch_Samuel']['REAL_1']
# samuel_real_1['Total_Force'] = samuel_real_1['LT Force (N)'] + samuel_real_1['RT Force (N)']

# plt.figure(figsize=(15, 5))
# plt.plot(samuel_real_1['time'], samuel_real_1['Total_Force'])
# plt.xlabel('Zeit (s)')
# plt.ylabel('Gesamtkraft (N)')
# plt.title('REAL_1 - Gesamtkraft über Zeit')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#%%


#%% Verarbeite alle Teilnehmer und erstelle CSVs mit Sprung-Eigenschaften
measurements = ['REAL_1', 'REAL_2', 'VR_1', 'VR_2']
output_folder = 'jump_analysis_results'

# Erstelle Ausgabeverzeichnis, falls nicht vorhanden
os.makedirs(output_folder, exist_ok=True)

for participant in participants:
    print(f"\nVerarbeite {participant}...")
    
    all_jumps_data = []
    
    # Durchlaufe alle 4 Messungen
    for measurement in measurements:
        df = all_data[participant][measurement]
        
        # Identifiziere Sprünge in dieser Messung
        jumps = functions_PRESSURE.identify_jumps(df, height = peak_hight[participants.index(participant)])
        
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

#%%
