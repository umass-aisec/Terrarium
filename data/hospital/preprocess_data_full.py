import pandas as pd
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "outpatient_merged.csv")

def main():
    print("Loading data files...")
    
    # 1. Load Facility Type Data (Hospitals)
    # Contains: THCIC_ID, PROVIDER_NAME, FAC_*_IND (Specialties)
    facility_path = os.path.join(RAW_DIR, "Facility_type1q2019_tab.txt")
    df_facility = pd.read_csv(facility_path, sep='\t', dtype=str)
    print(f"Loaded {len(df_facility)} facilities.")

    # 2. Load Base1 Data (Patients - Demographics & Diagnosis)
    # Contains: RECORD_ID, THCIC_ID, PRINC_DIAG_CODE, PAT_COUNTY, PAT_STATUS
    base1_path = os.path.join(RAW_DIR, "PUDF_base1_1q2019_tab.txt")
    # We only read columns we actually need to save memory
    use_cols = [
        "RECORD_ID", "THCIC_ID", "PRINC_DIAG_CODE", 
        "PAT_COUNTY", "PAT_STATUS"
    ]
    df_base1 = pd.read_csv(base1_path, sep='\t', dtype=str, usecols=use_cols)
    print(f"Loaded {len(df_base1)} patient records.")

    # 3. Merge Data
    # Join Patients with their Hospital info on THCIC_ID
    print("Merging data...")
    df_merged = pd.merge(df_base1, df_facility, on="THCIC_ID", how="left")
    
    # 4. Save to CSV
    print(f"Saving merged data to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Done! The environment is ready to use.")

if __name__ == "__main__":
    main()