"""
This script is to retrieve images in CTRATE dataset that have non-standard orientation, which might have led to issues in conversion.
"""

import os
import pandas as pd
import numpy as np
import ast
import glob

def analyze_original_affines():
    path_csv = "/mnt/hdda/murong/debug/ctrate/train_metadata.csv"  #TODO:
    image_folder = "/mnt/hddb/murong/CADS-dataset/0041_ctrate/images"  #TODO:
    output_file = '/home/murong/22k/OMASeg_sync/OMASeg/nonstandard_affine_analysis.json'  #TODO:
    
    metadata = pd.read_csv(path_csv)
    
    def compute_original_affine(row):
        try:
            orientation = ast.literal_eval(row['ImageOrientationPatient'])
            position = ast.literal_eval(row['ImagePositionPatient'])
            spacing_z = row['ZSpacing']
            spacing_xy = ast.literal_eval(row['XYSpacing'])
            
            a_L, a_P, a_H = orientation[0:3]
            b_L, b_P, b_H = orientation[3:6]
            p_L, p_P, p_H = position
            
            c = np.cross(np.array(orientation[:3]), np.array(orientation[3:]))
            c_L, c_P, c_H = c
            
            s_x = spacing_xy[0]
            s_y = spacing_xy[1]
            s_z = spacing_z
            
            # construct affine (DICOM LPS to RAS)
            affine = np.array([
                [-a_L * s_x, -b_L*s_y, -c_L*s_z, -p_L],
                [-a_P * s_x, -b_P*s_y, -c_P*s_z, -p_P],
                [a_H * s_x, b_H*s_y, c_H*s_z, p_H],
                [0, 0, 0, 1]
            ])
            
            return affine
            
        except Exception as e:
            return None
    
    def is_standard_orientation(orientation):
        """Check if affine corresponds to standard orientation [1,0,0,0,1,0]"""
        standard = [1,0,0,0,1,0]
        return all(abs(x - y) < 1e-6 for x, y in zip(orientation, standard))
    
    results = []
    nii_files = glob.glob(os.path.join(image_folder, "*.nii.gz"))
    nii_files = [f for f in nii_files if not os.path.basename(f).startswith('brain_')]
    
    print(f"Found {len(nii_files)} cases to analyze")
    
    for nii_file in nii_files:
        base_name = os.path.basename(nii_file)
        case_id = base_name.split('_0000.nii.gz')[0]
        original_name = f"{case_id}"
        
        metadata_row = metadata[metadata['VolumeName'] == original_name]
        
        if len(metadata_row) == 0:
            print(f"Warning: No metadata found for {base_name}")
            continue
            
        metadata_row = metadata_row.iloc[0]
        
        try:
            orientation = ast.literal_eval(metadata_row['ImageOrientationPatient'])
            is_standard = is_standard_orientation(orientation)
            original_affine = compute_original_affine(metadata_row)
            
            if original_affine is not None:
                result = {
                    'filename': base_name,
                    'original_name': original_name,
                    'Manufacturer': metadata_row['Manufacturer'],
                    'ImageOrientationPatient': orientation,
                    'PatientPosition': metadata_row['PatientPosition'],
                    'is_standard_orientation': is_standard,
                    'original_affine': original_affine.tolist(),
                    'spacing_xy': ast.literal_eval(metadata_row['XYSpacing']),
                    'spacing_z': metadata_row['ZSpacing']
                }
                results.append(result)
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
    
    total_cases = len(results)
    non_standard_cases = [case for case in results if not case['is_standard_orientation']]
    
    print("\n=== Analysis Results ===")
    print(f"Total cases analyzed: {total_cases}")
    print(f"Cases with non-standard orientation: {len(non_standard_cases)}")
    
    manufacturer_stats = {}
    for case in results:
        mfr = case['Manufacturer']
        if mfr not in manufacturer_stats:
            manufacturer_stats[mfr] = {
                'total': 0, 
                'non_standard': 0,
                'spacing_stats': {'xy': [], 'z': []}
            }
        manufacturer_stats[mfr]['total'] += 1
        if not case['is_standard_orientation']:
            manufacturer_stats[mfr]['non_standard'] += 1
        manufacturer_stats[mfr]['spacing_stats']['xy'].append(case['spacing_xy'])
        manufacturer_stats[mfr]['spacing_stats']['z'].append(case['spacing_z'])
    
    for mfr in manufacturer_stats:
        xy_spacings = np.array(manufacturer_stats[mfr]['spacing_stats']['xy'])
        z_spacings = np.array(manufacturer_stats[mfr]['spacing_stats']['z'])
        manufacturer_stats[mfr]['spacing_stats'] = {
            'xy_mean': float(np.mean(xy_spacings)),
            'xy_std': float(np.std(xy_spacings)),
            'z_mean': float(np.mean(z_spacings)),
            'z_std': float(np.std(z_spacings))
        }
    
    print("\n=== Manufacturer-wise Statistics ===")
    for mfr, stats in manufacturer_stats.items():
        print(f"\n{mfr}:")
        print(f"Total cases: {stats['total']}")
        print(f"Non-standard orientations: {stats['non_standard']} ({(stats['non_standard']/stats['total']*100):.2f}%)")
        print(f"Average spacing (xy, z): ({stats['spacing_stats']['xy_mean']:.3f}, {stats['spacing_stats']['z_mean']:.3f})")
    
    import json
    output_file = 'original_affine_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_cases': total_cases,
                'non_standard_cases': len(non_standard_cases),
                'manufacturer_stats': manufacturer_stats
            },
            'all_cases': results,
            'non_standard_cases': non_standard_cases
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    print("\n=== Sample Non-standard Cases ===")
    for case in non_standard_cases[:5]:
        print(f"\nFilename: {case['filename']}")
        print(f"Manufacturer: {case['Manufacturer']}")
        print(f"Original orientation: {case['ImageOrientationPatient']}")
        print(f"Original affine matrix:")
        print(np.array(case['original_affine']))
        print(f"Spacing (xy, z): {case['spacing_xy']}, {case['spacing_z']}")

if __name__ == "__main__":
    analyze_original_affines()