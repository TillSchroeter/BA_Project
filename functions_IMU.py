import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def plot_normalized_jumps(normalized_dict, participant_name, measurement_type, column_to_plot):
    """
    Erstellt zwei Grafiken für eine Messung: 
    1. Alle Sprünge als Subplots (Einzelansicht)
    2. Alle Sprünge übereinandergelegt mit MITTELWERT und SD
    """
    # 1. Vorbereitung Ordnerstruktur (Ordner nach Spalte benannt, dann Teilnehmer)
    target_dir = os.path.join('Pictures_Test', column_to_plot.replace(' ', '_'), participant_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Filtere die Sprünge für diesen Typ
    jump_keys = [k for k in normalized_dict.keys() if k.startswith(measurement_type)]
    num_jumps = len(jump_keys)
    
    if num_jumps == 0:
        return

    # Daten für Mittelwertberechnung sammeln
    all_values = []
    time_normalized = None

    # --- GRAFIK 1: SUBPLOTS ---
    fig_sub, axes = plt.subplots(num_jumps, 1, figsize=(10, 2.5 * num_jumps), sharex=True)
    if num_jumps == 1: axes = [axes]
    
    for i, key in enumerate(jump_keys):
        df_jump = normalized_dict[key]
        val = df_jump[column_to_plot].values
        all_values.append(val)
        if time_normalized is None:
            time_normalized = df_jump['time_normalized'].values
            
        axes[i].plot(time_normalized, val, color='black', lw=1)
        axes[i].set_title(f"{key}", fontsize=10)
        axes[i].grid(True, alpha=0.2)
    
    plt.xlabel("Normierte Zeit (0-100%)")
    plt.tight_layout()
    fig_sub.savefig(os.path.join(target_dir, f"{participant_name}_{measurement_type}_subplots.png"))
    plt.close(fig_sub)

    # --- GRAFIK 2: ÜBEREINANDERGELEGT MIT MEAN + SD ---
    plt.figure(figsize=(10, 6))
    
    # In Matrix umwandeln für Berechnung: (Anzahl_Sprünge, 100_Punkte)
    data_matrix = np.array(all_values)
    mean_val = np.mean(data_matrix, axis=0)
    std_val = np.std(data_matrix, axis=0)

    # Einzelne Sprünge (blasser im Hintergrund)
    for i, val in enumerate(all_values):
        plt.plot(time_normalized, val, alpha=0.3, lw=1, label=f"Sprung {i+1}")
    
    # Mittelwert (Fett)
    plt.plot(time_normalized, mean_val, color='black', lw=3, label='MITTELWERT')
    
    # Standardabweichung (Schatten)
    plt.fill_between(time_normalized, mean_val - std_val, mean_val + std_val, 
                     color='gray', alpha=0.2, label='Standardabweichung')
    
    plt.title(f"Vergleich: {participant_name} - {measurement_type}\n{column_to_plot}")
    plt.xlabel("Normierte Zeit (0-100%)")
    plt.ylabel(column_to_plot)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(target_dir, f"{participant_name}_{measurement_type}_overlaid_mean.png"))
    plt.close()

    print(f"  ✓ Grafiken gespeichert in: {target_dir}")