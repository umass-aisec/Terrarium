import pandas as pd
import os
import random

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "outpatient_merged.csv")

def main():
    print("Loading data files...")
    
    # 1. Load Facility Type Data (Hospitals)
    facility_path = os.path.join(RAW_DIR, "Facility_type1q2019_tab.txt")
    df_facility = pd.read_csv(facility_path, sep='\t', dtype=str)
    print(f"Loaded {len(df_facility)} facilities.")

    # 2. Load Base1 Data (Patients - Demographics & Diagnosis)
    base1_path = os.path.join(RAW_DIR, "PUDF_base1_1q2019_tab.txt")
    use_cols = [
        "RECORD_ID", "THCIC_ID", "PRINC_DIAG_CODE", 
        "PAT_COUNTY", "PAT_STATUS"
    ]
    df_base1 = pd.read_csv(base1_path, sep='\t', dtype=str, usecols=use_cols)
    print(f"Loaded {len(df_base1)} patient records.")

    # --- NEW FILTERING LOGIC ---
    print("Filtering for 20 facilities with 10-20 patients each...")
    
    # Group by facility to get counts
    facility_counts = df_base1['THCIC_ID'].value_counts()
    
    # Identify facilities that have at least 10 patients
    # (We need at least 10 to meet the lower bound of the requirement)
    eligible_facilities = facility_counts[facility_counts >= 10].index.tolist()
    
    if not eligible_facilities:
        print("Error: No facilities found with at least 10 patients.")
        return

    # Select 20 random facilities (or fewer if we don't have 20 total)
    num_to_select = min(len(eligible_facilities), 20)
    selected_facilities = random.sample(eligible_facilities, num_to_select)
    
    filtered_chunks = []
    for fac_id in selected_facilities:
        # Get all records for this facility
        fac_data = df_base1[df_base1['THCIC_ID'] == fac_id]
        
        # Determine sample size: 
        # Requirement: "10-20 patients each".
        # If they have > 20, take 20. If they have 10-20, take all.
        count = len(fac_data)
        sample_size = min(count, 20)
        
        # Sample the patients
        sampled_data = fac_data.sample(n=sample_size, random_state=42)
        filtered_chunks.append(sampled_data)
    
    # Replace the original large dataframe with the filtered version
    df_base1 = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Filtered dataset size: {len(df_base1)} patients across {len(selected_facilities)} facilities.")
    # ---------------------------

    # 3. Merge Data
    print("Merging data...")
    df_merged = pd.merge(df_base1, df_facility, on="THCIC_ID", how="left")
    
    # 4. Save to CSV
    print(f"Saving merged data to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Done! The environment is ready to use.")

if __name__ == "__main__":
    main()