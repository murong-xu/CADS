import argparse
import os
import time

from joblib import Parallel, delayed

from cads.dataset_utils.preprocessing import preprocess_nifti_and_save_to_dir


def _derive_patient_id(file_in: str) -> str:
    filename = os.path.basename(file_in)
    if filename.endswith("0000.nii.gz"):
        return filename[:-12]
    return filename[:-7]


def _preprocess_single_image(file_in, output_preprocessed_images_folder, output_metadata_folder, spacing, num_threads_preprocessing, total_images, index):
    patient_id = _derive_patient_id(file_in)
    print(f"Preprocessing file {index + 1}/{total_images} {patient_id}")
    preprocess_nifti_and_save_to_dir(
        file_in,
        output_preprocessed_images_folder,
        output_metadata_folder,
        patient_id,
        spacing=spacing,
        num_threads_preprocessing=num_threads_preprocessing,
    )


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
                        help="Number of images to preprocess in parallel", default=4)


    args = parser.parse_args()

    os.makedirs(args.output_preprocessed_images_folder, exist_ok=True)
    os.makedirs(args.output_metadata_folder, exist_ok=True)

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
    total_images = len(input_images)
    Parallel(n_jobs=args.nr_thr_preprocess, prefer="threads")(
        delayed(_preprocess_single_image)(
            file_in,
            args.output_preprocessed_images_folder,
            args.output_metadata_folder,
            1.5,
            1,
            total_images,
            index,
        )
        for index, file_in in enumerate(input_images)
    )

    print(f"Finished in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
