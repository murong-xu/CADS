import glob
import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import argparse

from cads.dataset_utils.preprocessing import change_spacing, reorient_to

def restore_seg_in_orig_format(file_seg_in, file_seg_out, metadata_orig, num_threads_preprocessing=2):
    print('Restore the segmentation to original format...')

    seg_preprocessed = nib.load(file_seg_in)
    orig_spacing = metadata_orig['spacing']
    orig_affine = metadata_orig['affine']

    # Resample to original spacing (using absolute values for zoom)
    abs_spacing = np.abs(orig_spacing)
    orig_shape = (metadata_orig['x_size'], 
                 metadata_orig['y_size'], 
                 metadata_orig['z_size'])
    seg_resampled = change_spacing(seg_preprocessed, abs_spacing, target_shape=orig_shape, order=0, 
                                 dtype=np.int32, nr_cpus=num_threads_preprocessing)

    # Reorient to original orientation
    orig_orientation = nio.ornt2axcodes(nio.io_orientation(orig_affine))
    seg_reoriented = reorient_to(seg_resampled.get_fdata(), seg_resampled.affine, 
                                orig_orientation, verb=True)
    
    # Restore final segmentation with exactly the original affine matrix (translation etc.)
    seg_restored = nib.Nifti1Image(seg_reoriented.get_fdata().astype(np.uint8), 
                                  orig_affine)
    
    nib.save(seg_restored, file_seg_out)

def main():
    seg_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/best_22k_more_stat_tests/0041_3k'
    original_img_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/metadata/nii_gz_brain_600'
    ribfixed_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/totalseg_gt_correction/rib_fixed_22k/0041_3k'
    output_folder = "/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format/0042_new_brainct_1mm"

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, help="Starting index for processing (optional)")
    parser.add_argument("--end_idx", type=int, help="Ending index for processing (optional)")
    args = parser.parse_args()

    parts= np.arange(551,560)
    
    img_ids = glob.glob(seg_folder + '/*')
    img_ids = [os.path.basename(img_id) for img_id in img_ids]
    img_ids = [img_id for img_id in img_ids if img_id.startswith('brain_')]
    img_ids.sort()

    if args.start_idx is not None and args.end_idx is not None:
        img_ids = img_ids[args.start_idx:args.end_idx]
    
    for i, img_id in enumerate(img_ids):
        print("Restoring file {}/{}   ".format(i+1, len(img_ids)), img_id)
        img_id_short = img_id.split('_', 3)[0] + '_' + img_id.split('_', 3)[1] + '_' + img_id.split('_', 3)[2]
        original_img_file = os.path.join(original_img_folder, 'brain_ct_'+img_id_short, img_id+'.nii.gz')
        original_img = nib.load(original_img_file)
        original_img_data = original_img.get_fdata()
        metadata_orig = {
            'affine': original_img.affine,
            'spacing': np.diag(original_img.affine, k=0)[:3],
            'x_size': original_img_data.shape[0],
            'y_size': original_img_data.shape[1],
            'z_size': original_img_data.shape[2],
        }
        for part in parts:
            if part == 555:
                seg_file_in = os.path.join(ribfixed_folder, img_id, f'{img_id}_part_{part-300}_fixed.nii.gz')
            else:
                seg_file_in = os.path.join(seg_folder, img_id, f'{img_id}_part_{part-300}.nii.gz')
            if not os.path.exists(seg_file_in):
                print(f"Segmentation file {seg_file_in} does not exist.")
                continue
            seg_file_out = os.path.join(output_folder, img_id, f'{img_id}_part_{part}.nii.gz')
            if not os.path.exists(seg_file_out):
                os.makedirs(os.path.dirname(seg_file_out), exist_ok=True)
            restore_seg_in_orig_format(seg_file_in, seg_file_out, metadata_orig, num_threads_preprocessing=4)
    print('end')

if __name__ == "__main__":
    main()
