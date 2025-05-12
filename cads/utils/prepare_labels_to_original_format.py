import glob
import os
import numpy as np
import nibabel as nib
import argparse

from cads.dataset_utils.preprocessing import restore_seg_in_orig_format

# import debugpy
# debugpy.listen(('0.0.0.0', 4444))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('You can debug your script now')

def main():
    best22k_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/best_22k_more_stat_tests'
    ctready_folder = '/net/cephfs/shares/menze.dqbm.uzh/fercho/ct_ready'
    ribfixed_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/totalseg_gt_correction/rib_fixed_22k'
    output_folder = "/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format"

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", '--dataset', type=str, required=True)
    parser.add_argument("--start_idx", type=int, help="Starting index for processing (optional)")
    parser.add_argument("--end_idx", type=int, help="Ending index for processing (optional)")
    args = parser.parse_args()
    dataset = args.dataset

    parts= np.arange(551,560)
    
    seg_folder = os.path.join(best22k_folder, dataset)
    original_img_folder = os.path.join(ctready_folder, dataset)
    img_ids = glob.glob(seg_folder + '/*')
    img_ids = [os.path.basename(img_id) for img_id in img_ids]
    img_ids.sort()

    if args.start_idx is not None and args.end_idx is not None:
        img_ids = img_ids[args.start_idx:args.end_idx]
    
    for i, img_id in enumerate(img_ids):
        print("Restoring file {}/{}   ".format(i+1, len(img_ids)), img_id)
        original_img_file = os.path.join(original_img_folder, 'images', img_id+'_0000.nii.gz')
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
                seg_file_in = os.path.join(ribfixed_folder, dataset, img_id, f'{img_id}_part_{part-300}_fixed.nii.gz')
            else:
                seg_file_in = os.path.join(seg_folder, img_id, f'{img_id}_part_{part-300}.nii.gz')
            if not os.path.exists(seg_file_in):
                print(f"Segmentation file {seg_file_in} does not exist.")
                continue
            seg_file_out = os.path.join(output_folder, dataset, img_id, f'{img_id}_part_{part}.nii.gz')
            if not os.path.exists(seg_file_out):
                os.makedirs(os.path.dirname(seg_file_out), exist_ok=True)
            restore_seg_in_orig_format(seg_file_in, seg_file_out, metadata_orig, num_threads_preprocessing=4)
    print('end')

if __name__ == "__main__":
    main()

