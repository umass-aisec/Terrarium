import pandas as pd
import os
import random

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "outpatient_merged.csv")

TOTAL = 8
TARGET_NUM_PER_SPECIALTY = TOTAL // 4

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
    df_facility = pd.read_csv(facility_path, sep='\t', dtype=str)
    
    # 2. Load Patient Data
    base1_path = os.path.join(RAW_DIR, "PUDF_base1_1q2019_tab.txt")
    if not os.path.exists(base1_path):
        print(f"Error: File not found at {base1_path}")
        return
        
    use_cols = ["RECORD_ID", "THCIC_ID", "PRINC_DIAG_CODE", "PAT_COUNTY", "PAT_STATUS"]
    df_base1 = pd.read_csv(base1_path, sep='\t', dtype=str, usecols=use_cols)
    print(f"Loaded {len(df_base1)} patient records.")

    # --- 3. BALANCED SELECTION LOGIC ---
    print("Classifying data to ensure Supply-Demand match...")
    
    # Tag every patient record with its required specialty
    df_base1['Sim_Specialty'] = df_base1['PRINC_DIAG_CODE'].apply(get_specialty_category)
    
    selected_facilities = set()
    final_chunks = []
    
    # We want a balanced network: 5 Cardio, 5 Neuro, 5 Ortho, 5 General
    targets = {
        'Cardiology': TARGET_NUM_PER_SPECIALTY,
        'Neurology': TARGET_NUM_PER_SPECIALTY,
        'Orthopedics': TARGET_NUM_PER_SPECIALTY,
        'General': TARGET_NUM_PER_SPECIALTY
    }
    
    # Helper to find top hospitals for a specific specialty
    def select_hospitals_for_specialty(specialty, count_needed, exclude_ids):
        # Filter for patients of this specialty
        if specialty == 'General':
            subset = df_base1[~df_base1['THCIC_ID'].isin(exclude_ids)]
        else:
            subset = df_base1[(df_base1['Sim_Specialty'] == specialty) & (~df_base1['THCIC_ID'].isin(exclude_ids))]
            
        # Count relevant patients per facility
        counts = subset['THCIC_ID'].value_counts()
        
        # Select top N (most volume = likely a real specialist)
        # We lowered the threshold to 1 to ensure we fill the slots, relying on 'counts' sort order for quality.
        candidates = counts[counts >= 1].index.tolist()
        
        selected = candidates[:count_needed]
        return selected

    # Execute Selection
    facility_roles = {} 
    
    for spec, count in targets.items():
        chosen_ids = select_hospitals_for_specialty(spec, count, selected_facilities)
        
        print(f"Selected {len(chosen_ids)} hospitals for {spec}...")
        for hid in chosen_ids:
            selected_facilities.add(hid)
            facility_roles[hid] = spec

    # Fill remaining with randoms if we fell short
    remaining_needed = TOTAL - len(selected_facilities)
    if remaining_needed > 0:
        print(f"Filling {remaining_needed} remaining slots with random hospitals...")
        all_counts = df_base1['THCIC_ID'].value_counts()
        candidates = all_counts[all_counts >= 10].index.tolist()
        for cand in candidates:
            if cand not in selected_facilities:
                selected_facilities.add(cand)
                facility_roles[cand] = 'General'
                remaining_needed -= 1
                if remaining_needed == 0:
                    break

    # --- 4. CREATE PATIENT DATASET ---
    print("Sampling patients with Load Variance...")
    
    for fac_id in selected_facilities:
        role = facility_roles.get(fac_id, 'General')
        
        # Get all records for this facility
        fac_data = df_base1[df_base1['THCIC_ID'] == fac_id]
        
        # RANDOMIZE LOAD: 
        # Instead of fixed 20, randomize between 12 and 28.
        # This ensures some hospitals are overloaded (near 30) and others are empty (near 12).
        target_load = random.randint(12, 28)
        sample_size = min(len(fac_data), target_load)
        
        if role == 'General':
            # For General hospitals, just take a random sample
            # This naturally includes a mix of General and specialized patients (mismatches)
            chunk = fac_data.sample(n=sample_size, random_state=42)
            final_chunks.append(chunk)
        else:
            # For Specialized hospitals, MAXIMIZE the signal.
            # 1. Take ALL matching patients up to sample size
            matching_patients = fac_data[fac_data['Sim_Specialty'] == role]
            
            if len(matching_patients) >= sample_size:
                # Perfect specialist scenario
                chunk = matching_patients.sample(n=sample_size, random_state=42)
                final_chunks.append(chunk)
            else:
                # Take all matches, fill rest with mismatches
                chunk_match = matching_patients
                remaining = sample_size - len(chunk_match)
                
                # Get pool of non-matching
                mismatched_pool = fac_data[fac_data['Sim_Specialty'] != role]
                
                if len(mismatched_pool) > 0:
                    chunk_mis = mismatched_pool.sample(n=min(len(mismatched_pool), remaining), random_state=42)
                    final_chunks.extend([chunk_match, chunk_mis])
                else:
                    final_chunks.append(chunk_match)

    df_filtered = pd.concat(final_chunks, ignore_index=True)
    
    # Add the "Assigned_Specialty" column mapping
    df_roles = pd.DataFrame(list(facility_roles.items()), columns=['THCIC_ID', 'ASSIGNED_SPECIALTY'])
    
    print(f"Filtered dataset size: {len(df_filtered)} patients across {len(selected_facilities)} facilities.")

    # 5. Merge Data
    print("Merging data...")
    df_merged = pd.merge(df_filtered, df_facility, on="THCIC_ID", how="left")
    df_merged = pd.merge(df_merged, df_roles, on="THCIC_ID", how="left")
    
    # 6. Save to CSV
    print(f"Saving merged data to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Done! Data is ready.")

if __name__ == "__main__":
    main()