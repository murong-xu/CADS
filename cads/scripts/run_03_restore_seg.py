import argparse
import os
import pickle
import time
from cads.dataset_utils.preprocessing import restore_seg_in_orig_format


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
                        help="Nr of threads for processing", default=4)


    args = parser.parse_args()
    seg_folder = args.input_seg_folder
    metadata_folder = args.input_metadata_folder
    out_seg_folder = args.output_seg_folder
    nr_thr_process = args.nr_thr_process
    
    image_ids = sorted(
        d for d in os.listdir(seg_folder)
        if os.path.isdir(os.path.join(seg_folder, d))
    )

    start = time.time()
    for image_id in image_ids:
        seg_dir = os.path.join(seg_folder, image_id)
        metadata_path = os.path.join(metadata_folder, f'{image_id}_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata_orig = pickle.load(f)
            f.close()

        for seg_file in os.listdir(seg_dir):
            seg_file_path = os.path.join(seg_dir, seg_file)
            output_dir = os.path.join(out_seg_folder, image_id)
            os.makedirs(output_dir, exist_ok=True)  
            file_out = os.path.join(output_dir, str(seg_file))
            restore_seg_in_orig_format(seg_file_path, file_out, metadata_orig, num_threads_preprocessing=nr_thr_process)
    
    print(f"Finished in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
