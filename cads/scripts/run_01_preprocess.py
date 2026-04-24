import argparse
import os
import time

from cads.dataset_utils.preprocessing import preprocess_nifti_and_save_to_dir


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="This is a standalone script for preprocessing CT images for CADS. "
        "It runs on CPU only, so you can use it as a separate first step outside the main CADS pipeline. "
        "After that, you can skip preprocessing and run run_02_inference.py directly."
    )

    parser.add_argument("-in", metavar="input_files_directory", dest="input_folder",
                        help="Directory of input CT nifti images", required=True)

    parser.add_argument("-out_preprocessed_img", metavar="output_preprocessed_images_directory", dest="output_preprocessed_images_folder",
                        help="Output directory for preprocessed images", required=True)

    parser.add_argument("-out_metadata", metavar="output_metadata_directory", dest="output_metadata_folder",
                        help="Output directory for image metadata", required=True)

    parser.add_argument("-np", "--nr_thr_preprocess", type=int,
                        help="Nr of threads for preprocessing", default=4)

    args = parser.parse_args()
    
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
    
    start = time.time()
    # Loop images
    for i, file_in in enumerate(input_images):
        if os.path.basename(file_in)[-11:] == "0000.nii.gz":
            patient_id = os.path.basename(file_in)[:-12]
        else:
            patient_id = os.path.basename(file_in)[:-7]
        print("Preprocessing file {}/{}   ".format(i+1, len(input_images)), patient_id)
        # Reorient to RAS, resampling to 1.5, remove rotation and translation
        preprocess_nifti_and_save_to_dir(file_in, args.output_preprocessed_images_folder, args.output_metadata_folder, patient_id, spacing=1.5, num_threads_preprocessing=args.nr_thr_preprocess)

    print(f"Finished in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
