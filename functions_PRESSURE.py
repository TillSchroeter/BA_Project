import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def identify_jumps(df, height, distance_sec=0.3, fs=2000, buffer=0.5):
    """
    Identifiziert Sprünge basierend auf Paaren von Kraft-Peaks und fügt
    einen zeitlichen Puffer davor und danach hinzu.
    
    Parameters:
    -----------
    buffer : float
        Zeit in Sekunden, die vor dem Absprung und nach der Landung 
        zusätzlich eingeschlossen wird (Default: 0.5s).
    """
    # 1. Kraft berechnen
    total_force = df['LT Force (N)'] + df['RT Force (N)']
    time = df['time'].values
    max_time = time[-1]
    
    # # 2. Filterung (wie bewährt)
    # nyq = 0.5 * fs
    # b, a = butter(4, 20 / nyq, btype='low')
    # filtered_f = filtfilt(b, a, total_force)

    # 3. Peak Suche
    peak_indices, _ = find_peaks(total_force, height=height, distance=int(distance_sec * fs))
    
    jumps_list = []
    
    # 4. Paar-Logik mit Buffer
    for i in range(0, len(peak_indices) - 1, 2):
        idx_start_peak = peak_indices[i]
        idx_end_peak = peak_indices[i+1]
        
        # Die reinen Peak-zu-Peak Zeiten (fast wie Absprung bis Landung)
        t_absprung_peak = time[idx_start_peak]
        t_landung_peak = time[idx_end_peak]
        
        # Die reine Dauer zwischen den Kraftspitzen
        reine_peak_dauer = t_landung_peak - t_absprung_peak
        
        # Das Analyse-Fenster für das EMG (inkl. Puffer)
        start_with_buffer = max(0, t_absprung_peak - buffer)
        end_with_buffer = min(max_time, t_landung_peak + buffer)
        
        # Die Dauer des gesamten "Schnippels", den du ausschneidest
        fenster_dauer = end_with_buffer - start_with_buffer
        
        # Dictionary mit klarer Trennung
        jump_dict = {
            'sprung nr.': len(jumps_list) + 1,
            'start_ana': round(start_with_buffer, 4), # Start für EMG-Ausschnitt
            'end_ana': round(end_with_buffer, 4),     # Ende für EMG-Ausschnitt
            'ana_dauer': round(fenster_dauer, 4),     # Gesamtdauer des Ausschnitts
            'peak_to_peak_dauer': round(reine_peak_dauer, 4), # Wichtig für Bewegungsvergleich
            't_absprung': round(t_absprung_peak, 4),
            't_landung': round(t_landung_peak, 4)
            }   
        
        jumps_list.append(jump_dict)
        
    return jumps_list


def visualize_jumps(df, jumps_list):
    # Kraft berechnen
    total_force = df['LT Force (N)'] + df['RT Force (N)']
    time = df['time'].values
    
    plt.figure(figsize=(15, 7))
    plt.plot(time, total_force, color='black', alpha=0.3, label='Rohdaten Kraft')
    
    # Hilfsvariablen für die Legende (damit Labels nur 1x erscheinen)
    labeled_buffer = False
    labeled_jump_zone = False
    labeled_peaks = False
    
    for j in jumps_list:
        # 1. Den gesamten Analyse-Bereich (inkl. 0.5s Buffer) hinterlegen
        plt.axvspan(j['start_ana'], j['end_ana'], 
                    color='gray', alpha=0.15, 
                    label='Analyse-Fenster (Buffer)' if not labeled_buffer else "")
        labeled_buffer = True
        
        # 2. Den Bereich zwischen den Peaks (eigentlicher Sprung) markieren
        plt.axvspan(j['t_absprung'], j['t_landung'], 
                    color='orange', alpha=0.3, 
                    label='Sprung (Peak-to-Peak)' if not labeled_jump_zone else "")
        labeled_jump_zone = True
        
        # 3. Vertikale Linien für die exakten Peaks
        plt.axvline(j['t_absprung'], color='green', linestyle='--', alpha=0.7, 
                    label='Absprung-Peak' if not labeled_peaks else "")
        plt.axvline(j['t_landung'], color='red', linestyle='--', alpha=0.7, 
                    label='Landungs-Peak' if not labeled_peaks else "")
        labeled_peaks = True
        
        # 4. Sprungnummer über dem Sprung platzieren
        text_pos_x = j['t_absprung'] + (j['peak_to_peak_dauer'] / 2)
        plt.text(text_pos_x, max(total_force) * 0.85, f"Sprung {j['sprung nr.']}", 
                 horizontalalignment='center', fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 5. Dauer anzeigen
        plt.text(text_pos_x, max(total_force) * 0.78, f"{j['peak_to_peak_dauer']:.2f}s", 
                 horizontalalignment='center', fontsize=9, color='gray')

    plt.title(f"Identifizierte Sprungzyklen inkl. Pre- & Post-Buffer ({len(jumps_list)} Sprünge)")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Gesamtkraft (N)")
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()