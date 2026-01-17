#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import functions_generell

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

#%% Beispiel: REAL_1 von Samuel anzeigen
# print (all_data['ID_1_Dabisch_Samuel']['REAL_1'].head())

#%% Test: Visualisiere Kraftdaten
samuel_real_1 = all_data['ID_1_Dabisch_Samuel']['REAL_1']
samuel_real_1['Total_Force'] = samuel_real_1['LT Force (N)'] + samuel_real_1['RT Force (N)']

plt.figure(figsize=(15, 5))
plt.plot(samuel_real_1['time'], samuel_real_1['Total_Force'])
plt.xlabel('Zeit (s)')
plt.ylabel('Gesamtkraft (N)')
plt.title('REAL_1 - Gesamtkraft über Zeit')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Test: Erkenne Sprungphasen
jumps = functions_generell.detect_jump_phases(samuel_real_1)

print(f"\n{len(jumps)} Sprünge erkannt:")
print("="*80)
for jump in jumps:
    print(f"Sprung {jump['jump_nr']}:")
    print(f"  Start Zeit (mit Buffer): {jump['start_time']:.3f} s")
    print(f"  Absprung Zeit:           {jump['takeoff_time']:.3f} s")
    print(f"  Landung Zeit:            {jump['landing_time']:.3f} s")
    print(f"  End Zeit (mit Buffer):   {jump['end_time']:.3f} s")
    print(f"  Flugzeit:                {jump['flight_time']:.3f} s")
    print()

#%% Visualisiere erkannte Sprünge
plt.figure(figsize=(15, 6))
plt.plot(samuel_real_1['time'], samuel_real_1['Total_Force'], label='Gesamtkraft', alpha=0.7)

# Markiere jeden Sprung
colors = plt.cm.tab10(np.linspace(0, 1, len(jumps)))
for i, jump in enumerate(jumps):
    # Markiere die Sprungphase
    plt.axvspan(jump['start_time'], jump['end_time'], alpha=0.2, color=colors[i], 
                label=f"Sprung {jump['jump_nr']}")
    # Markiere Absprung und Landung
    plt.axvline(jump['takeoff_time'], color=colors[i], linestyle='--', linewidth=2)
    plt.axvline(jump['landing_time'], color=colors[i], linestyle='--', linewidth=2)

plt.xlabel('Zeit (s)')
plt.ylabel('Gesamtkraft (N)')
plt.title('REAL_1 - Erkannte Sprungphasen')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%


#%%
