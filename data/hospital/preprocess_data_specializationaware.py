import pandas as pd
import os
import random

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "outpatient_merged.csv")

TOTAL = 20
TARGET_NUM_PER_SPECIALTY = TOTAL // 4

# Comprehensive Lat/Lon Centroids for all 254 Texas Counties
# Format: "48XXX": (Latitude, Longitude)
TX_COUNTY_CENTROIDS = {
    "48001": (31.81, -95.65), "48003": (32.30, -102.83), "48005": (31.25, -94.61), "48007": (28.10, -96.99),
    "48009": (33.61, -98.68), "48011": (35.02, -101.35), "48013": (28.89, -98.53), "48015": (29.88, -96.40),
    "48017": (34.07, -102.83), "48019": (29.75, -99.23), "48021": (30.10, -97.31), "48023": (33.62, -99.21),
    "48025": (28.42, -97.74), "48027": (31.04, -97.48), "48029": (29.45, -98.52), "48031": (30.27, -98.40),
    "48033": (32.74, -101.43), "48035": (31.90, -97.60), "48037": (33.45, -94.42), "48039": (29.17, -95.40),
    "48041": (30.66, -96.30), "48043": (29.81, -103.24), "48045": (34.53, -101.21), "48047": (27.04, -98.21),
    "48049": (31.78, -98.98), "48051": (30.54, -96.60), "48053": (30.71, -98.15), "48055": (29.84, -97.61),
    "48057": (28.57, -96.58), "48059": (32.33, -99.37), "48061": (26.15, -97.45), "48063": (33.02, -94.94),
    "48065": (35.40, -101.36), "48067": (33.07, -94.35), "48069": (34.53, -102.34), "48071": (29.82, -94.67),
    "48073": (31.84, -95.17), "48075": (34.53, -100.21), "48077": (33.79, -98.21), "48079": (33.61, -102.82),
    "48081": (31.88, -100.53), "48083": (31.78, -99.42), "48085": (33.18, -96.57), "48087": (34.87, -100.27),
    "48089": (29.73, -96.53), "48091": (29.81, -98.13), "48093": (31.94, -98.56), "48095": (31.33, -99.86),
    "48097": (33.64, -97.21), "48099": (31.39, -97.80), "48101": (34.07, -100.25), "48103": (31.41, -102.52),
    "48105": (30.73, -101.21), "48107": (33.61, -101.35), "48109": (31.45, -104.52), "48111": (36.28, -102.59),
    "48113": (32.77, -96.77), "48115": (32.74, -101.95), "48117": (34.84, -102.59), "48119": (33.39, -95.67),
    "48121": (33.20, -97.12), "48123": (29.08, -97.35), "48125": (33.62, -100.78), "48127": (28.42, -99.76),
    "48129": (34.92, -100.82), "48131": (27.68, -98.52), "48133": (32.33, -98.83), "48135": (31.87, -102.54),
    "48137": (30.01, -100.30), "48139": (32.35, -96.70), "48141": (31.77, -106.31), "48143": (32.23, -98.22),
    "48145": (31.26, -96.93), "48147": (33.59, -96.11), "48149": (29.86, -96.93), "48151": (32.73, -100.53),
    "48153": (34.08, -101.30), "48155": (33.97, -99.78), "48157": (29.53, -95.77), "48159": (33.20, -95.22),
    "48161": (31.70, -96.30), "48163": (28.87, -99.18), "48165": (32.75, -102.63), "48167": (29.38, -94.96),
    "48169": (33.18, -101.30), "48171": (30.31, -98.95), "48173": (31.87, -101.53), "48175": (28.57, -97.39),
    "48177": (29.45, -97.45), "48179": (35.41, -100.81), "48181": (33.62, -96.65), "48183": (32.48, -94.79),
    "48185": (30.55, -95.99), "48187": (29.59, -97.95), "48189": (34.07, -101.76), "48191": (34.53, -100.69),
    "48193": (31.71, -98.11), "48195": (36.27, -101.36), "48197": (34.28, -99.75), "48199": (30.34, -94.40),
    "48201": (29.81, -95.39), "48203": (32.55, -94.37), "48205": (35.84, -102.63), "48207": (33.18, -99.73),
    "48209": (30.06, -97.99), "48211": (35.84, -100.28), "48213": (32.21, -95.85), "48215": (26.31, -98.18),
    "48217": (32.00, -97.13), "48219": (33.61, -102.35), "48221": (32.44, -97.79), "48223": (33.15, -95.56),
    "48225": (31.32, -95.44), "48227": (32.30, -101.44), "48229": (31.45, -105.03), "48231": (33.12, -96.08),
    "48233": (35.83, -101.35), "48235": (31.30, -100.76), "48237": (33.24, -98.14), "48239": (28.95, -96.52),
    "48241": (30.86, -93.99), "48243": (30.72, -104.01), "48245": (29.86, -94.15), "48247": (26.97, -98.68),
    "48249": (27.76, -98.16), "48251": (32.38, -97.37), "48253": (32.74, -99.87), "48255": (28.81, -97.87),
    "48257": (32.60, -96.28), "48259": (29.95, -98.71), "48261": (26.93, -97.70), "48263": (32.99, -100.77),
    "48265": (30.06, -99.24), "48267": (30.49, -99.75), "48269": (33.61, -100.25), "48271": (29.35, -100.41),
    "48273": (27.42, -97.66), "48275": (33.59, -99.74), "48277": (33.67, -95.57), "48279": (34.07, -102.35),
    "48281": (31.07, -98.24), "48283": (28.34, -99.10), "48285": (29.38, -96.93), "48287": (30.29, -97.00),
    "48289": (31.25, -96.00), "48291": (30.06, -94.85), "48293": (31.54, -96.58), "48295": (36.27, -100.27),
    "48297": (28.45, -98.12), "48299": (30.70, -98.52), "48301": (31.85, -103.57), "48303": (33.61, -101.82),
    "48305": (33.18, -101.82), "48307": (31.20, -99.35), "48309": (31.48, -97.30), "48311": (28.46, -98.57),
    "48313": (30.95, -95.95), "48315": (32.79, -94.35), "48317": (32.30, -101.95), "48319": (30.72, -99.23),
    "48321": (28.97, -96.00), "48323": (28.74, -100.32), "48325": (29.35, -99.11), "48327": (30.89, -99.82),
    "48329": (31.87, -102.03), "48331": (30.80, -96.93), "48333": (31.33, -98.59), "48335": (32.30, -100.92),
    "48337": (33.67, -97.72), "48339": (30.34, -95.50), "48341": (35.83, -101.89), "48343": (33.12, -94.73),
    "48345": (34.07, -100.78), "48347": (31.61, -94.61), "48349": (32.04, -96.47), "48351": (30.91, -93.75),
    "48353": (32.41, -100.40), "48355": (27.74, -97.52), "48357": (36.28, -100.82), "48359": (35.40, -102.59),
    "48361": (30.13, -93.86), "48363": (32.92, -98.27), "48365": (32.17, -94.31), "48367": (32.78, -97.80),
    "48369": (34.53, -102.82), "48371": (30.84, -102.53), "48373": (30.68, -94.94), "48375": (35.20, -101.90),
    "48377": (29.81, -104.03), "48379": (32.86, -95.82), "48381": (34.96, -101.89), "48383": (31.36, -101.53),
    "48385": (29.84, -99.81), "48387": (33.62, -95.05), "48389": (31.32, -103.40), "48391": (28.31, -97.16),
    "48393": (35.84, -100.82), "48395": (31.03, -96.51), "48397": (32.89, -96.41), "48399": (31.70, -99.96),
    "48401": (31.98, -94.86), "48403": (31.35, -93.85), "48405": (31.43, -94.18), "48407": (30.64, -95.03),
    "48409": (28.05, -97.52), "48411": (31.16, -98.98), "48413": (30.89, -100.45), "48415": (28.00, -97.16),
    "48417": (32.75, -100.92), "48419": (33.12, -99.20), "48421": (35.21, -101.10), "48423": (32.38, -95.27),
    "48425": (32.18, -97.77), "48427": (26.37, -98.86), "48429": (32.79, -98.31), "48431": (32.99, -100.25),
    "48433": (31.75, -99.04), "48435": (31.83, -101.21), "48437": (33.67, -99.20), "48439": (32.77, -97.29),
    "48441": (32.28, -99.89), "48443": (30.52, -102.04), "48445": (33.91, -99.21), "48447": (33.24, -99.21),
    "48449": (33.22, -94.97), "48451": (31.85, -102.05), "48453": (30.28, -97.75), "48455": (31.09, -95.14),
    "48457": (30.77, -94.38), "48459": (32.73, -94.89), "48461": (31.25, -102.26), "48463": (29.35, -99.77),
    "48465": (29.89, -102.32), "48467": (32.73, -95.39), "48469": (32.55, -95.89), "48471": (31.62, -99.20),
    "48473": (28.47, -99.47), "48475": (34.42, -101.30), "48477": (30.74, -95.57), "48479": (27.76, -99.33),
    "48481": (29.28, -96.22), "48483": (35.40, -100.27), "48485": (31.68, -94.20), "48487": (33.97, -98.52),
    "48489": (26.98, -97.90), "48491": (30.65, -97.60), "48493": (29.17, -98.08), "48495": (31.85, -103.05),
    "48497": (33.22, -97.66), "48499": (32.85, -95.38), "48501": (33.17, -102.83), "48503": (33.17, -97.99),
    "48505": (27.00, -99.18), "48507": (28.68, -99.76)
}

def get_real_world_coords(pat_county_value):
    """
    Standardizes 3-digit PAT_COUNTY FIPS to a 5-digit Texas FIPS 
    and returns (Latitude, Longitude).
    """
    # 1. Ensure it is a string and remove whitespace
    clean_val = str(pat_county_value).strip()
    
    # 2. Pad to 3 digits (e.g., '1' -> '001')
    padded_county = clean_val.zfill(3)
    
    # 3. Create the 5-digit Texas FIPS
    full_fips = "48" + padded_county
    
    # 4. Lookup in your centroids dictionary
    return TX_COUNTY_CENTROIDS.get(full_fips, (0.0, 0.0))

def get_specialty_category(code):
    """Maps ICD-10 code prefix to simulation specialty."""
    s_code = str(code).upper()
    if s_code.startswith(('I', 'i')): return 'Cardiology'
    if s_code.startswith(('G', 'g')): return 'Neurology'
    if s_code.startswith(('M', 'm')): return 'Orthopedics'
    return 'General'

def main():
    print("Loading data files...")
    
    # 1. Load Facility Type Data
    facility_path = os.path.join(RAW_DIR, "Facility_type1q2019_tab.txt")
    if not os.path.exists(facility_path):
        print(f"Error: File not found at {facility_path}")
        return
    
    # Relaxed loading: Only drop if THCIC_ID is missing
    df_facility = pd.read_csv(facility_path, sep='\t', dtype=str)
    initial_fac_len = len(df_facility)
    df_facility = df_facility.dropna(subset=['THCIC_ID'])
    print(f"Facility Data Cleaned: Kept {len(df_facility)}/{initial_fac_len} records.")
    
    # 2. Load Patient Data
    base1_path = os.path.join(RAW_DIR, "PUDF_base1_1q2019_tab.txt")
    if not os.path.exists(base1_path):
        print(f"Error: File not found at {base1_path}")
        return
        
    use_cols = ["RECORD_ID", "THCIC_ID", "PRINC_DIAG_CODE", "PAT_COUNTY", "PAT_STATUS"]
    df_base1 = pd.read_csv(base1_path, sep='\t', dtype=str, usecols=use_cols)

    # --- 2a. RELAXED FILTERING (PATIENT) ---
    initial_pat_len = len(df_base1)
    
    # Only drop if critical linking/grouping columns are missing
    df_base1 = df_base1.dropna(subset=['THCIC_ID', 'PRINC_DIAG_CODE'])
    
    valid_facility_ids = set(df_facility['THCIC_ID'].unique())
    df_base1 = df_base1[df_base1['THCIC_ID'].isin(valid_facility_ids)]
    
    print(f"Patient Data Cleaned: Kept {len(df_base1)}/{initial_pat_len} records.")

    # --- 3. BALANCED SELECTION LOGIC ---
    print("Classifying data to setup Mismatch Scenario...")
    
    df_base1['Sim_Specialty'] = df_base1['PRINC_DIAG_CODE'].apply(get_specialty_category)
    
    selected_facilities = set()
    final_chunks = []
    
    targets = {
        'Cardiology': TARGET_NUM_PER_SPECIALTY,
        'Neurology': TARGET_NUM_PER_SPECIALTY,
        'Orthopedics': TARGET_NUM_PER_SPECIALTY,
        'General': TARGET_NUM_PER_SPECIALTY
    }
    
    def select_hospitals_for_specialty(specialty, count_needed, exclude_ids):
        # Strict filtering for ALL categories to avoid overlap
        subset = df_base1[(df_base1['Sim_Specialty'] == specialty) & (~df_base1['THCIC_ID'].isin(exclude_ids))]
            
        counts = subset['THCIC_ID'].value_counts()
        # Lowered threshold to 1 to ensure we find enough hospitals even with small data
        candidates = counts[counts >= 1].index.tolist()
        
        selected = candidates[:count_needed]
        return selected

    facility_roles = {} 
    
    for spec, count in targets.items():
        chosen_ids = select_hospitals_for_specialty(spec, count, selected_facilities)
        
        print(f"Selected {len(chosen_ids)} hospitals for {spec}...")
        for hid in chosen_ids:
            selected_facilities.add(hid)
            facility_roles[hid] = spec

    # Fill remaining if any
    remaining_needed = TOTAL - len(selected_facilities)
    if remaining_needed > 0:
        print(f"Filling {remaining_needed} remaining slots...")
        all_counts = df_base1['THCIC_ID'].value_counts()
        candidates = all_counts.index.tolist()
        for cand in candidates:
            if cand not in selected_facilities:
                selected_facilities.add(cand)
                facility_roles[cand] = 'General'
                remaining_needed -= 1
                if remaining_needed == 0:
                    break

    # --- 4. CREATE PATIENT DATASET (STRICT MISMATCH) ---
    print("Generating patients with GUARANTEED MISMATCH (Supply vs Demand)...")
    
    # Pre-filter pools to ensure we don't pick empty rows
    specialty_pools = {
        'Cardiology': df_base1[df_base1['Sim_Specialty'] == 'Cardiology'],
        'Neurology': df_base1[df_base1['Sim_Specialty'] == 'Neurology'],
        'Orthopedics': df_base1[df_base1['Sim_Specialty'] == 'Orthopedics'],
        'General': df_base1[df_base1['Sim_Specialty'] == 'General']
    }
    
    for fac_id in selected_facilities:
        role = facility_roles.get(fac_id, 'General')
        
        # Randomize Load
        target_load = random.randint(12, 28)
        
        # STRICT MISMATCH LOGIC:
        # Select from pools that are NOT the facility's assigned role
        available_pools = [k for k in specialty_pools.keys() if k != role]
        
        chunks_to_add = []
        
        for _ in range(target_load):
            chosen_pool_key = random.choice(available_pools)
            pool_df = specialty_pools[chosen_pool_key]
            
            if len(pool_df) > 0:
                patient = pool_df.sample(n=1, random_state=random.randint(0, 100000))
                chunks_to_add.append(patient)
        
        if chunks_to_add:
            fac_chunk = pd.concat(chunks_to_add)
            fac_chunk = fac_chunk.copy()
            # Overwrite THCIC_ID to teleport patient to the mismatched facility
            fac_chunk['THCIC_ID'] = fac_id
            final_chunks.append(fac_chunk)

    if final_chunks:
        df_filtered = pd.concat(final_chunks, ignore_index=True)
    else:
        df_filtered = pd.DataFrame(columns=use_cols + ['Sim_Specialty'])
    
    # Add the "Assigned_Specialty" column for verification
    df_roles = pd.DataFrame(list(facility_roles.items()), columns=['THCIC_ID', 'ASSIGNED_SPECIALTY'])
    
    print(f"Filtered dataset size: {len(df_filtered)} patients across {len(selected_facilities)} facilities.")

    # 5. Merge Data
    print("Merging data...")
    df_merged = pd.merge(df_filtered, df_facility, on="THCIC_ID", how="left")
    df_merged = pd.merge(df_merged, df_roles, on="THCIC_ID", how="left")

    print("Mapping real-world coordinates...")
    # Map the coordinates as a tuple (Lat, Lon)
    coords_series = df_merged['PAT_COUNTY'].apply(get_real_world_coords)
    
    # Split the tuple into separate columns for the environment to read
    df_merged['LATITUDE'] = coords_series.apply(lambda x: x[0])
    df_merged['LONGITUDE'] = coords_series.apply(lambda x: x[1])
    
    # 6. Save to CSV
    print(f"Saving merged data to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Done! Data is ready.")

if __name__ == "__main__":
    main()