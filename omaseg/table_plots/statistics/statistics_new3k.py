import pandas as pd
import numpy as np
import ast
import math

def parse_age(age_str):
    """
    '032Y' -> 32
    '075Y' -> 75
    '32Y'  -> 32
    '22M'  -> 1.83 (22/12)
    '32'   -> None
    'NA'   -> None
    """
    if pd.isna(age_str):
        return None
    
    age_str = str(age_str).strip()
    
    if age_str.endswith('Y'):
        try:
            age = int(age_str[:-1])
            return age
        except ValueError:
            return None
    
    elif age_str.endswith('M'):
        try:
            months = int(age_str[:-1])
            return math.ceil(months / 12) 
        except ValueError:
            return None
    return None


def analyze_patient_info(patient_info_list):
    gender_counts = {
        'M': sum(1 for p in patient_info_list if p['Gender'] == 'M'),
        'F': sum(1 for p in patient_info_list if p['Gender'] == 'F')
    }
    
    ages = [p['Age'] for p in patient_info_list if p['Age'] is not None]
    age_stats = {
        'mean': np.mean(ages),
        'std': np.std(ages),
        'median': np.median(ages)
    }
    
    xy_spacings = []
    for p in patient_info_list:
        try:
            xy = ast.literal_eval(p['XYSpacing'])
            xy_spacings.extend(xy)
        except:
            continue
    
    xy_stats = {
        'mean': np.mean(xy_spacings),
        'std': np.std(xy_spacings),
        'median': np.median(xy_spacings)
    }
    
    z_spacings = [p['ZSpacing'] for p in patient_info_list if p['ZSpacing'] is not None]
    z_stats = {
        'mean': np.mean(z_spacings),
        'std': np.std(z_spacings),
        'median': np.median(z_spacings)
    }
    
    print("Gender Distribution:")
    print(f"Male: {gender_counts['M']}")
    print(f"Female: {gender_counts['F']}")
    print(f"M:F ratio = {gender_counts['M']}/{gender_counts['F']} = {gender_counts['M']/gender_counts['F']:.2f}")
    print("\nAge Distribution:")
    print(f"Mean ± Std: {age_stats['mean']:.1f} ± {age_stats['std']:.1f}")
    print(f"Median: {age_stats['median']:.1f}")
    print("\nXY Spacing Distribution (mm):")
    print(f"Mean ± Std: {xy_stats['mean']:.3f} ± {xy_stats['std']:.3f}")
    print(f"Median: {xy_stats['median']:.3f}")
    print("\nZ Spacing Distribution (mm):")
    print(f"Mean ± Std: {z_stats['mean']:.3f} ± {z_stats['std']:.3f}")
    print(f"Median: {z_stats['median']:.3f}")
    
    return {
        'gender': gender_counts,
        'age': age_stats,
        'xy_spacing': xy_stats,
        'z_spacing': z_stats
    }
    

dataset_summary_path = '/mnt/hdda/murong/22k/metadata/dataset_summary.xlsx'  #TODO:
df = pd.read_excel(dataset_summary_path)
all_ids = df['ids'].tolist()

brain_ids = [id for id in all_ids if str(id).lower().startswith('brain')]
other_ids = [id for id in all_ids if str(id).lower().startswith('train')]

csv_path = '/mnt/hdda/murong/22k/metadata/train_metadata.csv'  #TODO:
csv_df = pd.read_csv(csv_path)

patient_info = []
for id in other_ids:
    matching_row = csv_df[csv_df['VolumeName'] == id+'.nii.gz']
    if not matching_row.empty:
        gender = matching_row['PatientSex'].iloc[0]
        age_str = matching_row['PatientAge'].iloc[0]
        age = parse_age(age_str)
        xyspacing = matching_row['XYSpacing'].iloc[0]
        zspacing = matching_row['ZSpacing'].iloc[0]
        patient_info.append({'ID': id, 'Gender': gender, 'Age': age, 'XYSpacing': xyspacing, 'ZSpacing': zspacing})

print('Thoracic CTs')
stats_thoracic = analyze_patient_info(patient_info)


csv_path = '/mnt/hdda/murong/22k/metadata/patient_info_age.csv'  #TODO:
csv_df = pd.read_csv(csv_path)
brain_patient_info = []
for id in brain_ids:
    parts = id.split('_', 3)
    id_name = '_'.join(parts[:3])
    matching_row = csv_df[csv_df['CT_Experiment_Folder'] == id_name]
    if not matching_row.empty:
        age_str = matching_row['Patient_Age'].iloc[0]
        age = parse_age(age_str)
        brain_patient_info.append({'ID': id, 'Age': age})

ages = [p['Age'] for p in brain_patient_info if p['Age'] is not None]
brain_patient_age_stats = {
    'mean': np.mean(ages),
    'std': np.std(ages),
    'median': np.median(ages)
}
