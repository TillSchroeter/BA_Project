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



### Funktion zum Zeitnormieren 
def time_normalize_jumps(df, participant_name, measurement_type, 
                             jumps_csv_folder='jump_analysis_results', 
                             normalize_points=100):
    """
    Normiert nur die Sprünge, die zum angegebenen measurement_type passen.
    
    Parameters:
    -----------
    measurement_type : str
        Der Typ der Messung, der gerade verarbeitet wird (z.B. 'REAL_1', 'VR_2')
    """
    
    # Pfad zur CSV
    jumps_csv_path = os.path.join(jumps_csv_folder, f'{participant_name}_jumps.csv')
    
    if not os.path.exists(jumps_csv_path):
        raise FileNotFoundError(f"Jumps CSV nicht gefunden: {jumps_csv_path}")
    
    # 1. Gesamte CSV laden
    jumps_info = pd.read_csv(jumps_csv_path)
    
    # 2. CSV filtern: Nur die Zeilen behalten, wo die Spalte 'messung' dem Typ entspricht
    # Das verhindert die Fehlermeldungen für die anderen 18 Sprünge
    relevant_jumps = jumps_info[jumps_info['messung'] == measurement_type]
    
    normalized_jumps = {}
    
    # 3. Nur die relevanten 6 Sprünge durchlaufen
    for idx, row in relevant_jumps.iterrows():
        start_time = row['start_ana']
        # ACHTUNG: Hier entscheidest du, ob bis 'end_ana' (ganzer Sprung) 
        # oder 't_absprung' (nur Vorbereitung) normiert wird:
        end_time = row['t_absprung'] 
        
        jump_nr = int(row['sprung nr.'])
        
        # Extrahiere Daten
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        jump_data = df[mask].reset_index(drop=True)
        
        if len(jump_data) < 2:
            continue
            
        # Interpolations-Logik (bleibt gleich)
        original_indices = np.linspace(0, len(jump_data) - 1, len(jump_data))
        new_indices = np.linspace(0, len(jump_data) - 1, normalize_points)
        
        normalized_data = {}
        for col in jump_data.columns:
            if pd.api.types.is_numeric_dtype(jump_data[col]):
                normalized_data[col] = np.interp(new_indices, original_indices, jump_data[col].values)
            else:
                normalized_data[col] = [jump_data[col].iloc[0]] * normalize_points
        
        normalized_data['time_normalized'] = np.linspace(0, 1, normalize_points)
        normalized_df = pd.DataFrame(normalized_data)
        
        # Key generieren: z.B. "REAL_1_jump_1"
        key = f"{measurement_type}_jump_{jump_nr}"
        normalized_jumps[key] = normalized_df
    
    return normalized_jumps


