import os
import numpy as np
import pickle
import glob
import nibabel as nib
import argparse
import shutil

folder_target = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format/0039_han_seg/segmentations_resampled_GT'
folder_gt = '/net/cephfs/shares/menze.dqbm.uzh/fercho/ct_ready/0039_han_seg/labels'
folder_output = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format/0039_han_seg/segmentations'
img_paths = glob.glob(folder_target + '/*')

for img_path in img_paths:
    img_id = img_path.split('/')[-1].split('.nii.gz')[0]
    print(img_id)
    # overwrite
    seg_gt = os.path.join(folder_gt, f"{img_id}.nii.gz")
    seg_target = os.path.join(folder_target, img_id, f"{img_id}_part_558.nii.gz")
    
    gt_nib = nib.load(seg_gt)
    gt_data = gt_nib.get_fdata()
    gt_data[gt_data == 30] = 0  # Remove label 30
    
    target_nib = nib.load(seg_target)
    new_seg = nib.Nifti1Image(gt_data.astype(np.uint8), target_nib.affine)
    
    output_dir = os.path.join(folder_output, img_id)
    os.makedirs(output_dir, exist_ok=True)
    nib.save(new_seg, os.path.join(output_dir, f"{img_id}_part_558.nii.gz"))
    
    # copy other seg files (551-559, except 558)
    source_dir = os.path.join(folder_target, img_id)
    for i in range(551, 560):
        if i != 558:  # Skip 558 as it's already handled
            source_file = os.path.join(source_dir, f"{img_id}_part_{i}.nii.gz")
            dest_file = os.path.join(output_dir, f"{img_id}_part_{i}.nii.gz")
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)