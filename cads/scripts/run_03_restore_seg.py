import argparse
import os
import pickle
import time
from joblib import Parallel, delayed
from cads.dataset_utils.preprocessing import restore_seg_in_orig_format


def _restore_single_segmentation(seg_file_path, file_out, metadata_orig, num_threads_resample):
    restore_seg_in_orig_format(
        seg_file_path,
        file_out,
        metadata_orig,
        num_threads_preprocessing=num_threads_resample,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="This is a standalone script for converting CADS segmentations to the original image geometry. "
        "It assumes the segmentations are in the preprocessed space and the metadata is stored in the specified directory. "
        "The resulting segmentations will be saved in the original image geometry."
    )

    parser.add_argument("-in_seg", metavar="input_segmentations_directory", dest="input_seg_folder",
                        help="Directory of input segmentation masks (in preprocessed space)", required=True)

    parser.add_argument("-in_metadata", metavar="input_metadata_directory", dest="input_metadata_folder",
                        help="Directory of input image metadata", required=True)
    
    parser.add_argument("-out_seg", metavar="output_segmentations_directory", dest="output_seg_folder",
                        help="Directory of output segmentation masks (in original image geometry)", required=True)
    
    parser.add_argument("-np", "--nr_thr_process", type=int,
                        help="Number of segmentations to restore in parallel", default=4)

    parser.add_argument("--nr_thr_resample", type=int,
                        help="Threads used inside each restore job (set >1 only when using low -np)",
                        default=1)


    args = parser.parse_args()
    seg_folder = args.input_seg_folder
    metadata_folder = args.input_metadata_folder
    out_seg_folder = args.output_seg_folder
    nr_thr_process = args.nr_thr_process
    nr_thr_resample = args.nr_thr_resample
    
    image_ids = sorted(
        d for d in os.listdir(seg_folder)
        if os.path.isdir(os.path.join(seg_folder, d))
    )

    jobs = []
    for image_id in image_ids:
        seg_dir = os.path.join(seg_folder, image_id)
        metadata_path = os.path.join(metadata_folder, f'{image_id}_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata_orig = pickle.load(f)

        for seg_file in os.listdir(seg_dir):
            seg_file_path = os.path.join(seg_dir, seg_file)
            output_dir = os.path.join(out_seg_folder, image_id)
            os.makedirs(output_dir, exist_ok=True)  
            file_out = os.path.join(output_dir, str(seg_file))
            jobs.append((seg_file_path, file_out, metadata_orig))

    start = time.time()
    Parallel(n_jobs=nr_thr_process, prefer="threads")(
        delayed(_restore_single_segmentation)(
            seg_file_path,
            file_out,
            metadata_orig,
            nr_thr_resample,
        )
        for seg_file_path, file_out, metadata_orig in jobs
    )
    
    print(f"Finished in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
