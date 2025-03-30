import argparse
import torch
import os
import numpy as np
import nibabel as nib
import pandas as pd
import ast
import pickle

from cads.utils.inference_wwo_restore import predict_ct_rate_without_restore
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

def main():
    parser = argparse.ArgumentParser(
        description="CADS!!", epilog="Trust the process!!!")

    parser.add_argument("-in", metavar="input_files_directory", dest="input_folder",
                        help="Directory of input CT nifti images", required=True)

    parser.add_argument("-out", metavar="output_files_directory", dest="output_folder",
                        help="Output directory for segmentation masks", required=True)
    
    parser.add_argument("-split", "--split", type=str, required=True)

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

    if not torch.cuda.is_available():
        raise ValueError(
            "CADS only works with a NVidia CUDA GPU. CUDA not found.")

    folds = 'all'

    with open(args.input_folder, 'rb') as f:
        test_images = pickle.load(f)
        f.close()
    test_images.sort()

    predict_ct_rate_without_restore(test_images, args.model_folder, args.task_id, folds=folds, 
            run_in_slicer=False, use_cpu=False,
            preprocess_cads=args.preprocessing, postprocess_cads = args.postprocessing, 
            save_all_combined_seg=args.save_all_combined_seg, snapshot=args.snapshot, save_separate_targets=args.save_targets,
            num_threads_preprocessing=args.nr_thr_preprocess, nr_threads_saving=args.nr_thr_saving, verbose=args.verbose)


if __name__ == "__main__":
    main()
