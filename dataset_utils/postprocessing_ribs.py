import argparse
import os
import glob
import nibabel as nib
import numpy as np
from skimage import morphology
from scipy import ndimage
from collections import Counter
from TPTBox import NII
from TPTBox.core.np_utils import np_connected_components


def merge_close_components(labeled_image, max_distance):
    labels = np.unique(labeled_image)
    labels = labels[labels != 0]  # exclude background

    for label in labels:
        mask = labeled_image == label
        # dilation for each connected component
        dilated = morphology.binary_dilation(mask, morphology.ball(max_distance))
        # find other labels that overlap with the expanded reached-out region
        overlapping = np.unique(labeled_image[dilated])
        # swallow up other labels
        for overlap_label in overlapping:
            if overlap_label != 0 and overlap_label != label:
                voxel_count = np.sum(labeled_image == overlap_label)
                if voxel_count < 1000:  # TODO:
                    labeled_image[labeled_image == overlap_label] = label

    return labeled_image

def assign_rib_classes(final_seg, original_seg):
    # exclude background
    unique_labels = np.unique(final_seg)[1:]
    
    new_seg = np.zeros_like(final_seg)
    for label in unique_labels:
        mask = (final_seg == label)
        
        # find corresponding label in original mask
        original_classes = original_seg[mask]
        
        # majority vote based on non-background classes
        class_votes = Counter(original_classes[original_classes != 0])
        if class_votes:
            majority_class = max(class_votes, key=class_votes.get)
            new_seg[mask] = majority_class
    
    return new_seg

def all_or_names(value):
    if value == 'all':
        return value
    try:
        return [i for i in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Value must be 'all' or a comma-separated list of dataset names") 

def get_dataset_folders(pred_folder, eval_datasets, split):
    datasetfolders = []
    for folder in os.listdir(pred_folder):
        if eval_datasets == 'all' or any(eval_dataset in folder for eval_dataset in eval_datasets):
            for split_folder in os.listdir(os.path.join(pred_folder, folder)):
                if split is None or split_folder == split:
                    current_folder = os.path.join(pred_folder, folder, split_folder)
                    if os.listdir(current_folder):
                        datasetfolders.append(current_folder)
    return sorted(datasetfolders)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--preddir", required=True)
    parser.add_argument("-out", "--outputdir", required=True)
    parser.add_argument("-split", "--split",
                        help="val=2, test=0, train=1", required=False)
    parser.add_argument("-dataset", "--dataset",
                        help="'all' or names (comma-separated)", type=all_or_names, required=True)

    args = parser.parse_args()

    pred_dir = args.preddir
    output_dir = args.outputdir
    eval_datasets = args.dataset
    split = args.split

    datasetfolders = get_dataset_folders(pred_dir, eval_datasets, str(split))

    for datasetfolder in datasetfolders:
        dataset = datasetfolder.split('/')[-2]
        print(dataset)
        input_segs = glob.glob(datasetfolder + '/*/*_part_255.nii.gz')
        output_dataset_folder = output_dir
        os.makedirs(output_dataset_folder, exist_ok=True)

        for input_seg in input_segs:
            img_id = input_seg.split('/')[-1].split('_part_255.nii.gz')[0]
            seg = NII.load(input_seg, seg=True)
            seg_data = seg.get_seg_array()

            output_seg_folder = os.path.join(output_dataset_folder, img_id)
            os.makedirs(output_seg_folder, exist_ok=True)

            # step 1) binarize segmentaion
            binary_seg = np.where(seg_data > 0, 1, 0).astype(np.uint8)

            # step 2) identify connected components (6-connectivity for 3D voxels)
            cc, _ = np_connected_components(binary_seg, connectivity=1, label_ref=1, verbose=False)
            cc = cc[1]  
            cc = np.where(cc > 0, 1, 0).astype(np.uint8)
            labeled_seg, _ = ndimage.label(cc > 0)

            # # step 3) merge components that are close to each other
            classified_seg = assign_rib_classes(labeled_seg, seg_data)

            final_nifti = nib.Nifti1Image(classified_seg.astype(np.uint8), seg.affine, seg.header)
            output_file = os.path.join(output_seg_folder, f"{img_id}_part_255.nii.gz")
            nib.save(final_nifti, output_file)
   

if __name__ == "__main__":
    main()
