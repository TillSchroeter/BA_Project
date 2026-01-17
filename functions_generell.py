import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

def load_csvs(folder_path):

    """
    Lädt 6 CSV-Dateien aus einem Ordner ein und gibt sie als separate DataFrames zurück.
    Genullt die Zeit-Spalte pro DataFrame (setzt sie auf 0 relativ zum Start).
    Parameters:
    -----------
    folder_path : str

        Pfad zum Ordner mit den CSV-Dateien

    Returns:
    --------
    tuple of pd.DataFrame
        (df_REAL_1, df_REAL_2, df_VR_1, df_VR_2, df_MVC_Beine, df_MVC_Hals)

    """
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
   
    # Gesamte Dateien in DataFrames separieren
    dfs = {}

    for file_name in file_list:
        # Überspringe info.csv und walking_calibration Dateien
        if 'info' in file_name.lower() or 'walking_calibration' in file_name.lower():
            continue
            
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, skiprows=3, sep=';', low_memory=False) 

        # Zeit-Spalte genullen (relativ zum Start)
        if 'time' in df.columns:
            df['time'] = df['time'] - df['time'].iloc[0]

        if 'REAL_1' in file_name:
            dfs['REAL_1'] = df
        elif 'REAL_2' in file_name:
            dfs['REAL_2'] = df
        elif 'VR_1' in file_name:
            dfs['VR_1'] = df
        elif 'VR_2' in file_name:
            dfs['VR_2'] = df
        elif 'MVC_Beine' in file_name:
            dfs['MVC_Beine'] = df
        elif 'MVC_Hals' in file_name:
            dfs['MVC_Hals'] = df


    return (dfs.get('REAL_1'), dfs.get('REAL_2'), dfs.get('VR_1'),
            dfs.get('VR_2'), dfs.get('MVC_Beine'), dfs.get('MVC_Hals'))


def detect_jump_phases(df, force_threshold=50, min_flight_time=0.1, buffer_time=0.5):
    """
    Erkennt Sprungphasen in einem DataFrame anhand der Drucksohlen-Kraftdaten.
    Berechnet die Gesamtkraft (links + rechts) und identifiziert Absprung und Landung.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame mit Kraftdaten ('LT Force (N)', 'RT Force (N)', 'time')
    force_threshold : float, optional
        Schwellenwert für Bodenkontakt in Newton (default: 50)
    min_flight_time : float, optional
        Minimale Flugzeit in Sekunden (default: 0.1)
    buffer_time : float, optional
        Zusätzliche Zeit vor und nach jedem Sprung in Sekunden (default: 0.5)
    
    Returns:
    --------
    list of dict
        Liste mit Dictionaries für jeden Sprung:
        [{'jump_nr': 1, 'start_time': t1, 'end_time': t2, 
          'takeoff_time': t_takeoff, 'landing_time': t_landing}, ...]
    """
    
    # Berechne Gesamtkraft (links + rechts)
    total_force = df['LT Force (N)'] + df['RT Force (N)']
    time = df['time'].values
    
    # Identifiziere Bodenkontakt (Kraft über Schwellenwert)
    ground_contact = total_force > force_threshold
    
    # Finde Übergänge (Absprung und Landung)
    contact_diff = np.diff(ground_contact.astype(int))
    
    # Absprung: Übergang von 1 zu 0 (contact_diff == -1)
    takeoff_indices = np.where(contact_diff == -1)[0] + 1
    
    # Landung: Übergang von 0 zu 1 (contact_diff == 1)
    landing_indices = np.where(contact_diff == 1)[0] + 1
    
    # Paare von Absprung und Landung bilden
    jumps = []
    
    for i, takeoff_idx in enumerate(takeoff_indices):
        # Finde die nächste Landung nach diesem Absprung
        landing_candidates = landing_indices[landing_indices > takeoff_idx]
        
        if len(landing_candidates) > 0:
            landing_idx = landing_candidates[0]
            
            # Prüfe, ob die Flugzeit lang genug ist
            flight_time = time[landing_idx] - time[takeoff_idx]
            
            if flight_time >= min_flight_time:
                # Berechne Start- und Endzeit mit Buffer
                takeoff_time = time[takeoff_idx]
                landing_time = time[landing_idx]
                
                start_time = max(0, takeoff_time - buffer_time)
                end_time = min(time[-1], landing_time + buffer_time)
                
                jumps.append({
                    'jump_nr': len(jumps) + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'takeoff_time': takeoff_time,
                    'landing_time': landing_time,
                    'flight_time': flight_time
                })
    
    return jumps

