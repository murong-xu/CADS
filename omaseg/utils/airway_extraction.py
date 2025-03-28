import numpy as np
import nibabel as nib
import glob
import os
from scipy import ndimage

def erode_3d(mask, distance_voxels):
    # Create a spherical structure
    struct = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_erosion(mask, structure=struct, iterations=distance_voxels)

def main():
    output_dir = '/mnt/hdda/murong/airway/output'
    pred_dir = '/mnt/hdda/murong/airway/seg'
    fm_dir = '/mnt/hdda/murong/airway/vesselFM'
    cases = glob.glob(pred_dir + '/*')
    cases.sort()

    lung_lobe_labels = [13, 14, 15, 16, 17]
    erosion_distance = 5

    for case in cases:
        img_id = case.split('/')[-1]
        file_seg_raw = os.path.join(case, f'{img_id}_part_551.nii.gz')
        file_fm = os.path.join(fm_dir, f'{img_id}_0000_vesselFM.nii.gz')

        seg_raw = nib.load(file_seg_raw)
        seg_raw_data = seg_raw.get_fdata().astype(np.uint8)
        seg_fm = nib.load(file_fm)
        seg_fm_data = seg_fm.get_fdata().astype(np.uint8)

         # Create lung mask by merging all lobes
        lung_mask = np.zeros_like(seg_raw_data)
        for label in lung_lobe_labels:
            lung_mask[seg_raw_data == label] = 1

        # Erode lung mask to exclude some boundary regions
        if erosion_distance > 0:
            eroded_lung_mask = erode_3d(lung_mask, erosion_distance)
        else:
            eroded_lung_mask = lung_mask

        # Extract vessels only within lung mask
        airways = seg_fm_data * eroded_lung_mask

        airways_nii = nib.Nifti1Image(airways, seg_raw.affine)
        nib.save(airways_nii, os.path.join(output_dir, f'{img_id}_airways_eroded_{erosion_distance}.nii.gz'))

if __name__ == "__main__":
    main()
