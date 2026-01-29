import glob
import os
import nibabel as nib
import numpy as np

from cads.dataset_utils.preprocessing import restore_seg_in_orig_format

path_pseudo_flavor = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_predictions/small_unet_model_predictions/final_checkpoint/02_pseudo/20000000/0004_lits/0'
path_cads = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/lits_backup_with_testGT'
output_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format/0004_lits'
output_folder_restore_pseudo = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/lits_restore_pseudo_flavor_seg'

pseudo_files = glob.glob(path_pseudo_flavor + "/*/*_part_251.nii.gz")
pseudo_files.sort()
for pseudo_file in pseudo_files:
    img_id = pseudo_file.split('/')[-2]
    cads_file_old = os.path.join(path_cads, img_id, img_id+'_part_551.nii.gz')
    # Load files
    cads_nib = nib.load(cads_file_old)
    cads = cads_nib.get_fdata()

    # Restore to original format
    metadata_orig = {
        'affine': cads_nib.affine,
        'spacing': np.diag(cads_nib.affine, k=0)[:3],
        'x_size': cads_nib.shape[0],
        'y_size': cads_nib.shape[1],
        'z_size': cads_nib.shape[2],
    }
    pseudo_file_restore = os.path.join(output_folder_restore_pseudo, img_id+'_251.nii.gz')
    restore_seg_in_orig_format(pseudo_file, pseudo_file_restore, metadata_orig, num_threads_preprocessing=4)

    pseudo_restore_nib = nib.load(pseudo_file_restore)
    pseudo_restore = pseudo_restore_nib.get_fdata()
    
    # Remove label 5 from CADS file and get label 5 from pseudo file
    cads[cads == 5] = 0
    pseudo_label_5 = (pseudo_restore == 5).astype(np.float32) * 5
    
    # Combine the segmentations
    combined_seg = cads + pseudo_label_5
    
    # Create output directory for this image
    output_img_dir = os.path.join(output_folder, img_id)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Save the new segmentation
    output_file = os.path.join(output_img_dir, img_id+'_part_551.nii.gz')
    seg_restored = nib.Nifti1Image(combined_seg.astype(np.uint8), cads_nib.affine)
    nib.save(seg_restored, output_file)
    print(f'Processed {img_id}: Replaced label 5 and saved to {output_file}')