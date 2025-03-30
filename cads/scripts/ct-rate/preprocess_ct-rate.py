import argparse
import torch
import os
import numpy as np
import nibabel as nib
import pandas as pd
import ast
import pickle

from cads.dataset_utils.bodyparts_labelmaps import labelmap_all_structure
from cads.dataset_utils.preprocessing import preprocess_nifti_ctrate

def check_input_task(value):
    valid_numbers = {551, 552, 553, 554, 555, 556, 557, 558, 559}
    if value == 'all':
        return list(valid_numbers)
    else:
        try:
            values = list(map(int, value.split(',')))
            if all(v in valid_numbers for v in values):
                return values
            else:
                raise ValueError
        except:
            raise argparse.ArgumentTypeError(
                f"Invalid input: {value}. Expected 'all' or a comma-separated list of valid numbers 551-559.")


def check_input_targets(value):
    valid_structures = list(labelmap_all_structure.values())
    if value.lower() == 'false':
        return False
    elif value == 'all':
        return valid_structures
    else:
        try:
            values = value.split(',')
            if all(v in valid_structures for v in values):
                return values
            else:
                raise ValueError
        except:
            raise argparse.ArgumentTypeError(
                f"Invalid input: {value}. Expected 'False', 'all', or a comma-separated list of valid structures in labelmap_all_structure.")


def extract_middle_path(filepath, split):
    start = filepath.find(f'/{split}/')
    end = filepath.find('.nii.gz')
    if start != -1 and end != -1:
        middle = filepath[start+1:end]
        return '/'.join(middle.split('/')[:-1])
    return None


def main():
    parser = argparse.ArgumentParser(
        description="CADS!!", epilog="Trust the process!!!")

    parser.add_argument("-in", metavar="input_files_directory", dest="input_folder",
                        help="Directory of input CT nifti images", required=True)

    parser.add_argument("-out", metavar="output_files_directory", dest="output_folder",
                        help="Output directory for segmentation masks", required=True)
    
    parser.add_argument("-split", "--split", type=str, required=True)

    parser.add_argument("-start_idx", "--start_idx", type=int, required=True)

    parser.add_argument("-end_idx", "--end_idx", type=int, required=True)

    parser.add_argument("-model", metavar="models_directory",
                        dest="model_folder", help="Directory of nnUNet models", required=True)

    parser.add_argument("-task", dest='task_id', type=check_input_task,
                        help="Input either 'all' or a subset of [551, 552, 553, 554, 555, 556, 557, 558, 559]")
    
    parser.add_argument('--preprocessing', action='store_true',
                        help='Set this flag to enable CADS preprocessing (reorient RAS, resampling 1.5, remove rotation and translation)', default=False)
    
    parser.add_argument('--postprocessing', action='store_true',
                        help='Set this flag to enable CADS postprocessing', default=False)

    parser.add_argument("--save_all_combined_seg", action="store_true",
                        help="Save one multilabel segmentation file for all classes", default=False)

    parser.add_argument("--snapshot", action="store_true",
                        help="Generate snapshot.png overlay of segmentations on input image", default=False)

    parser.add_argument("-save_separate_targets", dest='save_targets', type=check_input_targets,
                        help="Input 'False', 'all', or a subset of valid structures in labelmap_all_structure")

    parser.add_argument("-np", "--nr_thr_preprocess", type=int,
                        help="Nr of threads for preprocessing", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int,
                        help="Nr of threads for saving segmentations", default=6)

    parser.add_argument("--verbose", action="store_true",
                        help="Show more intermediate output", default=False)

    args = parser.parse_args()

    folds = 'all'

    hu_min, hu_max = -1000, 1000
    with open(args.input_folder, 'rb') as f:
        test_images = pickle.load(f)
        f.close()
    test_images.sort()
    start_idx = args.start_idx
    end_idx = args.end_idx
    test_images = test_images[start_idx:end_idx]
    split = args.split
    if split == 'valid':
        path_csv = "/net/cephfs/shares/menze.dqbm.uzh/haozhe/Datasets/CT_RATE2/CT-RATE/dataset/metadata/validation_metadata.csv"
    else:
        path_csv = "/net/cephfs/shares/menze.dqbm.uzh/haozhe/Datasets/CT_RATE2/CT-RATE/dataset/metadata/train_metadata.csv"
    metadata = pd.read_csv(path_csv)

    processed_images = []
    for test_image in test_images:
        img = nib.load(test_image)
        img_data = img.get_fdata()
        img_id = test_image.split("/")[-1].split(".nii.gz")[-2]
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

        output_dir = os.path.join(args.output_folder, extract_middle_path(test_image, split))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        filename_path = os.path.join(output_dir, f"{img_id}.nii.gz")
        preprocess_nifti_ctrate(img_corrected, filename_path, spacing=1.5, num_threads_preprocessing=args.nr_thr_preprocess)
        processed_images.append(filename_path)

    with open(os.path.join(f'/net/cephfs/shares/menze.dqbm.uzh/murong/CT-RATE_segmentations', f'batch_{split}_{start_idx}_{end_idx}.pkl'), 'wb') as f:
        pickle.dump(processed_images, f)
        f.close()

if __name__ == "__main__":
    main()
