import os
import numpy as np
import pandas as pd
import fnmatch
import re
from collections import defaultdict

from cads.dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps, labelmap_all_structure, labelmap_all_structure_renamed

def get_part_and_range(filename, split):
    match = re.match(rf'{split}_summary_(\d+)_(\d+)_(\d+)\.npz', os.path.basename(filename))
    if match:
        part = int(match.group(1))
        start_idx = int(match.group(2))
        end_idx = int(match.group(3))
        return part, start_idx, end_idx
    return None, None, None

def get_part_column_mapping():
    column_mapping = {}
    for part, labelmap in map_taskid_to_labelmaps.items():
        # Skip background (0) in each part's labelmap
        for orig_idx in range(1, len(labelmap)):
            structure_name = labelmap[orig_idx]
            # Find the corresponding index in labelmap_all_structure
            for all_idx, all_name in labelmap_all_structure.items():
                if all_name == structure_name:
                    # Map to the renamed version
                    column_mapping[(part, orig_idx)] = labelmap_all_structure_renamed[all_idx]
                    break
    
    return column_mapping

def main():
    root_dir = "/mnt/hdda/murong/debug_ctrate/segmentation_statistics"  #TODO:
    output_dir = "/mnt/hdda/murong/debug_ctrate/merged"  #TODO:
    split = 'valid'
    output_filename = f'{split}_label_summary'
    os.makedirs(output_dir, exist_ok=True)

    # Get all npz files
    npzfiles = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, f'{split}_summary_*.npz'):  #TODO:
            npzfiles.append(os.path.join(dirpath, filename))

    # Organize files by part and range
    files_by_part = defaultdict(list)
    for file in npzfiles:
        part, start_idx, end_idx = get_part_and_range(file, split)
        if part:
            files_by_part[part].append((start_idx, end_idx, file))

    # Sort files within each part by start index
    for part in files_by_part:
        files_by_part[part].sort()

    # Get column mapping
    column_mapping = get_part_column_mapping()

    # Initialize result dictionary with patient IDs
    result_dict = defaultdict(lambda: [None] * (len(labelmap_all_structure_renamed) + 1))  # +1 for patient ID

    # Process each part's files
    for part in sorted(files_by_part.keys()):
        for start_idx, end_idx, file in files_by_part[part]:
            print(f"Processing {file}")
            data = np.load(file, allow_pickle=True)['info']
            
            for row in data:
                patient_id = row[0]
                result_dict[patient_id][0] = patient_id  # Set patient ID
                
                # Map values to correct columns
                for i, value in enumerate(row[1:], 1):
                    if (part, i) in column_mapping:
                        col_name = column_mapping[(part, i)]
                        col_idx = list(labelmap_all_structure_renamed.values()).index(col_name) + 1
                        result_dict[patient_id][col_idx] = value

    # Convert to final format
    columns = ['Patient ID'] + list(labelmap_all_structure_renamed.values())
    result = [result_dict[pid] for pid in sorted(result_dict.keys())]
    
    # Save results
    np.savez(os.path.join(output_dir, f'{output_filename}.npz'), info=np.array(result))
    df = pd.DataFrame(result, columns=columns)
    df.to_excel(os.path.join(output_dir, f'{output_filename}.xlsx'), index=False)

if __name__ == '__main__':
    main()