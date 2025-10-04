import glob
import os
import numpy as np
import nibabel as nib
import argparse
import pandas as pd
import ast

from cads.dataset_utils.preprocessing import restore_seg_in_orig_format

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, help="Starting index for processing (optional)")
    parser.add_argument("--end_idx", type=int, help="Ending index for processing (optional)")
    args = parser.parse_args()

    seg_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/best_22k_more_stat_tests/0041_3k'
    path_csv = "/net/cephfs/shares/menze.dqbm.uzh/haozhe/Datasets/CT_RATE2/CT-RATE/dataset/metadata/train_metadata.csv"
    ribfixed_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/totalseg_gt_correction/rib_fixed_22k/0041_3k'
    original_img_folder = "/net/cephfs/shares/menze.dqbm.uzh/haozhe/Datasets/CT_RATE2/CT-RATE/dataset/train"
    output_folder = "/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format/0041_ctrate"

    img_ids = glob.glob(seg_folder + '/*')
    img_ids = [os.path.basename(img_id) for img_id in img_ids]
    img_ids = [img_id for img_id in img_ids if img_id.startswith('train_')]
    img_ids.sort()

    if args.start_idx is not None and args.end_idx is not None:
        img_ids = img_ids[args.start_idx:args.end_idx]

    metadata = pd.read_csv(path_csv)
    hu_min, hu_max = -1000, 1000
    parts= np.arange(551,560)

    for i, img_id in enumerate(img_ids):
        print("Restoring file {}/{}   ".format(i+1, len(img_ids)), img_id)
        # Step 1: Correct CT-RATE images and save to nifti
        img_id_short = img_id.split('_a_2')[0]
        original_img_file = os.path.join(original_img_folder, img_id_short, img_id_short+'_a', img_id_short+'_a_2.nii.gz')

        img = nib.load(original_img_file)
        img_data = img.get_fdata()
        idx = metadata[metadata["VolumeName"]==img_id+".nii.gz"].index.tolist()[0]
        patientPosition = metadata.loc[idx, "PatientPosition"]
        slope = metadata.loc[idx, "RescaleSlope"]
        intercept = metadata.loc[idx, "RescaleIntercept"]
        manufacturer = metadata.loc[idx, "Manufacturer"]
        spacing_z = metadata.loc[idx, "ZSpacing"]
        spacing_xy = ast.literal_eval(metadata.loc[idx, "XYSpacing"])
        imgOrientationPatient = ast.literal_eval(metadata.loc[idx, "ImageOrientationPatient"])
        imgPositionPatient = ast.literal_eval(metadata.loc[idx, "ImagePositionPatient"])
        spacing_xyz = (spacing_xy[0], spacing_xy[1], spacing_z)
        
        img_data = img_data.astype(np.float32)
        img_data = slope * img_data + intercept
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = ((img_data / 1000)).astype(np.float32)
        slices = []
        if manufacturer == 'PNMS':
            for i in reversed(range(img_data.shape[2])):
                img_slice = img_data[:, :, i].astype(np.float32)
                slices.append(img_slice)
        else:
            for i in range(img_data.shape[2]):
                img_slice = img_data[:, :, i].astype(np.float32)
                slices.append(img_slice)

        numpy_image = np.stack(slices, axis=0)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = numpy_image * 1000

        # compute the affine matrix
        a_L = imgOrientationPatient[0]
        a_P = imgOrientationPatient[1]
        a_H = imgOrientationPatient[2]
        b_L = imgOrientationPatient[3]
        b_P = imgOrientationPatient[4]
        b_H = imgOrientationPatient[5]
        p_L = imgPositionPatient[0]
        p_P = imgPositionPatient[1]
        p_H = imgPositionPatient[2]
        c = np.cross(np.asarray(imgOrientationPatient[:3]), np.asarray(imgOrientationPatient[3:]))
        c_L = c[0]
        c_P = c[1]
        c_H = c[2]
        s_x = spacing_xyz[0]
        s_y = spacing_xyz[1]
        s_z = spacing_xyz[2]

        # To: LPS (DICOM)
        affine_original = np.array(
        [
            [-a_L * s_x, -b_L*s_y, -c_L*s_z, -p_L],
            [-a_P * s_x, -b_P*s_y, -c_P*s_z, -p_P],
            [a_H * s_x, b_H*s_y, c_H*s_z, p_H],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        img_corrected = nib.Nifti1Image(numpy_image.astype(np.float32), affine_original)
        output_img_file = os.path.join(output_folder, 'images', f"{img_id}.nii.gz")
        nib.save(img_corrected, output_img_file)
    
        metadata_orig = {
            'affine': img_corrected.affine,
            'spacing': np.diag(img_corrected.affine, k=0)[:3],
            'x_size': numpy_image.shape[0],
            'y_size': numpy_image.shape[1],
            'z_size': numpy_image.shape[2],
        }
        for part in parts:
            if part == 555:
                seg_file_in = os.path.join(ribfixed_folder, img_id, f'{img_id}_part_{part-300}_fixed.nii.gz')
            else:
                seg_file_in = os.path.join(seg_folder, img_id, f'{img_id}_part_{part-300}.nii.gz')
            if not os.path.exists(seg_file_in):
                print(f"Segmentation file {seg_file_in} does not exist.")
                continue
            seg_file_out = os.path.join(output_folder, 'segmentations', img_id, f'{img_id}_part_{part}.nii.gz')
            if not os.path.exists(seg_file_out):
                os.makedirs(os.path.dirname(seg_file_out), exist_ok=True)
            restore_seg_in_orig_format(seg_file_in, seg_file_out, metadata_orig, num_threads_preprocessing=4)
    print('end')

if __name__ == "__main__":
    main()

