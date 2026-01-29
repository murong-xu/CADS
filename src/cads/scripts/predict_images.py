import argparse
import os

from cads.utils.libs import setup_nnunet_env
setup_nnunet_env()
from cads.utils.inference import predict
from cads.utils.libs import get_model_weights_dir, check_or_download_model_weights

def check_input_task(value):
    valid_numbers = {551, 552, 553, 554, 555, 556, 557, 558, 559}
    if value == 'all':
        return sorted(list(valid_numbers))
    else:
        try:
            values = [int(value)]
            
            if all(v in valid_numbers for v in values):
                return values
            else:
                raise ValueError
        except:
            raise argparse.ArgumentTypeError(
                f"Invalid input: {value}. Expected 'all' or a single valid number from 551-559.")


def main():
    parser = argparse.ArgumentParser(
        description="CADS!!", epilog="Trust the process!!!")

    parser.add_argument("-in", metavar="input_files_directory", dest="input_folder",
                        help="Directory of input CT nifti images", required=True)

    parser.add_argument("-out", metavar="output_files_directory", dest="output_folder",
                        help="Output directory for segmentation masks", required=True)

    parser.add_argument("-task", dest='task_id', type=check_input_task,
                        help="Input either 'all' or one of the task ids[551, 552, 553, 554, 555, 556, 557, 558, 559]")
                        
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='Use CPU for processing (default: False, use GPU)'
    )

    parser.add_argument('--no-preprocessing', action='store_false',
                    dest='preprocessing',
                    help='Disable CADS preprocessing (reorient RAS, resampling 1.5, remove rotation and translation)')

    parser.add_argument('--no-postprocessing', action='store_false',
                    dest='postprocessing',
                    help='Disable CADS postprocessing')

    parser.add_argument("-np", "--nr_thr_preprocess", type=int,
                        help="Nr of threads for preprocessing", default=4)

    parser.add_argument("-ns", "--nr_thr_saving", type=int,
                        help="Nr of threads for saving segmentations", default=6)

    parser.add_argument("--verbose", action="store_true",
                        help="Show more intermediate output", default=False)

    args = parser.parse_args()

    folds = 'all'
    
    if str(args.input_folder).endswith("nii.gz"):
        input_images = [args.input_folder]
    else:
        input_images = [
            os.path.join(root, filename)
            for root, dirnames, filenames in os.walk(args.input_folder)
            for filename in filenames
            if filename.endswith(".nii.gz")
        ]
    input_images.sort()
    output_seg_folder = args.output_folder
    task_ids = args.task_id

    # parepare local model weights
    model_folder = get_model_weights_dir()
    for task_id in task_ids:
        check_or_download_model_weights(task_id)
    if any(task in task_ids for task in [557, 558]) and 553 not in task_ids:
        check_or_download_model_weights(553)
        if 558 in task_ids and 552 not in task_ids:
            check_or_download_model_weights(552)

    task_ids.sort()
    predict(input_images, output_seg_folder, model_folder, task_ids, folds=folds, use_cpu=args.cpu,
            preprocess_cads=args.preprocessing, postprocess_cads=args.postprocessing,
            save_all_combined_seg=False, snapshot=False, save_separate_targets=False,
            num_threads_preprocessing=args.nr_thr_preprocess, nr_threads_saving=args.nr_thr_saving, 
            mode='auto', verbose=args.verbose)


if __name__ == "__main__":
    main()
