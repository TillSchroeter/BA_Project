import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def identify_jumps(df, flight_threshold=50, min_flight_sec=0.2, buffer=0.75):
    """
    Identifiziert Sprünge rein über das Unterschreiten eines Kraft-Schwellenwerts.
    
    Parameters:
    -----------
    flight_threshold : float
        Kraftwert (N), unter dem eine Flugphase erkannt wird (Default: 50N).
    min_flight_sec : float
        Mindestdauer in der Luft, um als Sprung zu gelten (verhindert Fehltrigger).
    """
    total_force = (df['LT Force (N)'] + df['RT Force (N)']).values
    time = df['time'].values
    max_time = time[-1]
    
    # 1. Erstelle eine Maske: True wo wir "in der Luft" sind
    in_air = total_force < flight_threshold
    
    # 2. Finde die Zeitpunkte, an denen sich der Status ändert (0 auf 1 oder 1 auf 0)
    diff = np.diff(in_air.astype(int))
    takeoff_indices = np.where(diff == 1)[0]
    landing_indices = np.where(diff == -1)[0]
    
    # Sicherstellen, dass wir mit einem Take-off starten und einer Landung enden
    if len(takeoff_indices) > 0 and len(landing_indices) > 0:
        if landing_indices[0] < takeoff_indices[0]:
            landing_indices = landing_indices[1:]
        takeoff_indices = takeoff_indices[:len(landing_indices)]

    jumps_list = []
    
    # 3. Durch die gefundenen Phasen iterieren
    for i in range(len(takeoff_indices)):
        idx_off = takeoff_indices[i]
        idx_on = landing_indices[i]
        
        t_off = time[idx_off]
        t_on = time[idx_on]
        flug_dauer = t_on - t_off
        
        # Nur Sprünge nehmen, die eine plausible Flugzeit haben
        if flug_dauer > min_flight_sec:
            start_with_buffer = max(0, t_off - buffer)
            end_with_buffer = min(max_time, t_on + buffer)
            
            jump_dict = {
                'sprung nr.': len(jumps_list) + 1,
                'start_ana': round(start_with_buffer, 4),
                'end_ana': round(end_with_buffer, 4),
                'ana_dauer': round(end_with_buffer - start_with_buffer, 4),
                'peak_to_peak_dauer': round(flug_dauer, 4), # Hier: Echte Flugzeit
                't_absprung': round(t_off, 4),
                't_landung': round(t_on, 4)
            }
            jumps_list.append(jump_dict)
            
    return jumps_list


def visualize_jumps(df, jumps_list, participant, measurement):
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

    plt.title(f"Identifizierte Sprungzyklen inkl. Pre- & Post-Buffer ({len(jumps_list)} Sprünge) - {participant} - {measurement}")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Gesamtkraft (N)")
    plt.title(f"Identifizierte Sprungzyklen inkl. Pre- & Post-Buffer ({len(jumps_list)} Sprünge) - {participant} - {measurement}")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'Pictures_Test/Sprungzüglen Kraft/{participant}_{measurement}_jump_.png', dpi=300)
    plt.show()