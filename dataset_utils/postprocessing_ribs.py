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

def save_nifti(data, affine, header, output_dataset_folder, name):
    final_nifti = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    output_file = os.path.join(output_dataset_folder, name)
    nib.save(final_nifti, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", required=True)
    args = parser.parse_args()

    pred_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/best_22k_more_stat_tests'
    output_dir = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/totalseg_gt_correction/rib_fixed_22k'
    fm_dir = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/totalseg_gt_correction/vesselFM_22k'
    dataset = args.dataset

    datasetfolder = os.path.join(pred_folder, dataset)
    input_segs = glob.glob(datasetfolder + '/*/*_part_255.nii.gz')
    input_segs.sort()
    output_dataset_folder = os.path.join(output_dir, dataset)
    os.makedirs(output_dataset_folder, exist_ok=True)

    for input_seg in input_segs:
        img_id = input_seg.split('/')[-1].split('_part_255.nii.gz')[0]
        seg = NII.load(input_seg, seg=True)
        seg_data = seg.get_seg_array()

        output_seg_folder = os.path.join(output_dataset_folder, img_id)
        os.makedirs(output_seg_folder, exist_ok=True)
        
        ### Stage 1: fix broken ribs
        # step 1) binarize segmentaion
        binary_seg = np.where(seg_data > 0, 1, 0).astype(np.uint8)

        # step 2) identify connected components (6-connectivity for 3D voxels)
        cc, _ = np_connected_components(binary_seg, connectivity=1, label_ref=1, verbose=False)
        cc = cc[1]  
        cc = np.where(cc > 0, 1, 0).astype(np.uint8)
        labeled_seg, _ = ndimage.label(cc > 0)

        # step 3) merge components that are close to each other
        seg_brokenfixed = assign_rib_classes(labeled_seg, seg_data)
        seg_brokenfixed = seg_brokenfixed.astype(np.uint8)
        # save_nifti(seg_brokenfixed, seg.affine, seg.header, output_seg_folder, f"{img_id}_broken-fixed.nii.gz")

        ### Stage 2: Identify costobvertebral jxns
        file_fm = os.path.join(fm_dir, dataset, f'{img_id}_0000_vesselFM.nii.gz')
        file_spine = os.path.join(pred_folder, dataset, img_id, f'{img_id}_part_252.nii.gz')
        seg_fm = nib.load(file_fm)
        seg_fm_data = seg_fm.get_fdata().astype(np.uint8)
        seg_spine= nib.load(file_spine)
        seg_spine_data = seg_spine.get_fdata().astype(np.uint8)

        potential_junctions = seg_fm_data.copy()
    
        # Step 2: Use seg_spine to locate junctions
        # Dilate the spine segmentation to include nearby junctions
        spine_mask = seg_spine_data > 0
        dilated_spine = morphology.binary_dilation(spine_mask, morphology.ball(10))  # TODO: 5 not enough to cover ribs 
        # save_nifti(dilated_spine, seg.affine, seg.header, output_seg_folder, 'dilated_spine.nii.gz')
        
        # Find overlap between potential junctions and dilated spine
        junction_candidates = (potential_junctions & dilated_spine) & ~spine_mask
        # save_nifti(junction_candidates, seg.affine, seg.header, output_seg_folder, 'jxn_candidates.nii.gz')

        # Apply morphological opening to separate weakly connected components
        opened_junction_candidates = morphology.binary_opening(junction_candidates, morphology.ball(2))  # TODO: sometimes 1 cannot be enought to separte
        # Label connected components in the opened junction candidates
        labeled_junctions, num_features = ndimage.label(opened_junction_candidates)
        # save_nifti(labeled_junctions, seg.affine, seg.header, output_seg_folder, 'jxn_candidates_opened.nii.gz')
            
        # Step 3: Filter and merge junctions
        # Calculate the size of each component
        component_sizes = ndimage.sum(opened_junction_candidates, labeled_junctions, range(1, num_features + 1))
        # Sort components by size
        filtered_components = [(index, size) for index, size in enumerate(component_sizes, start=1) if 100 < size < 1500]  # TODO: 
        sorted_components = sorted(enumerate(filtered_components, start=1), key=lambda x: x[1], reverse=True)

        relabeled_components = np.zeros_like(labeled_junctions)
        for new_label, (_, (component_index, _)) in enumerate(sorted_components, start=1):
            relabeled_components[labeled_junctions == component_index] = new_label
        # save_nifti(relabeled_components, seg.affine, seg.header, output_seg_folder, 'jxn_candidates_size_filtered.nii.gz')
        
        # Initialize completed rib segmentation
        completed_rib_seg = seg_brokenfixed.copy()
        
        # Precompute distance transforms for each rib class
        # Create a mask for all relevant components combined
        combined_relevant_mask = np.zeros_like(seg_brokenfixed, dtype=bool)
        for (_, (component_index, _)) in sorted_components:
            combined_relevant_mask |= (labeled_junctions == component_index)

        # Precompute distance transforms for each rib class only for the combined relevant components
        distance_transforms = {}
        for rib_class in range(1, 25):
            rib_mask = (seg_brokenfixed == rib_class)
            
            # Calculate distance transform only for the combined relevant components
            distance_transforms[rib_class] = np.where(
                combined_relevant_mask,
                ndimage.distance_transform_edt(~rib_mask),
                np.inf  # Assign infinity where the distance is not needed
            )
        
        for (_, (component_index, size)) in sorted_components:
            
            junction_component = (labeled_junctions == component_index)
            
            # Determine the closest rib class for the junction using precomputed distance transforms
            closest_rib_class = None
            min_distance = np.inf
            
            for rib_class, distance_transform in distance_transforms.items():
                # Calculate the minimum distance to the current rib class
                distance = np.min(distance_transform[junction_component])  # TODO: min is better than mean -> results in distance 0, distinctive
                
                if distance < min_distance:
                    min_distance = distance
                    closest_rib_class = rib_class
            
            # Merge the junction component into the rib segmentation with the closest rib class
            if closest_rib_class is not None and min_distance < 6:  # TODO: 10 is a bit large
                completed_rib_seg[junction_component] = closest_rib_class
                # print(f'Merge {size} voxels from cc index {component_index} to rib {closest_rib_class}, distance {min_distance}')

        final_nifti = nib.Nifti1Image(completed_rib_seg.astype(np.uint8), seg.affine, seg.header)
        output_file = os.path.join(output_seg_folder, f"{img_id}_part_255_fixed.nii.gz")
        nib.save(final_nifti, output_file)

if __name__ == "__main__":
    main()
