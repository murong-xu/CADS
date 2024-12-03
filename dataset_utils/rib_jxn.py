import os
import glob
import nibabel as nib
import numpy as np
from skimage import morphology
from scipy import ndimage


def save_nifti(data, affine, header, output_dataset_folder, name):
    final_nifti = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    output_file = os.path.join(output_dataset_folder, name)
    nib.save(final_nifti, output_file)

def main():
    output_dir = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/debug/rib_jxn/out/smaller10'
    pred_dir = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/debug/rib_jxn/raw_seg'
    fm_dir = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/debug/rib_jxn/vessels'
    cases = glob.glob(pred_dir + '/*')
    cases.sort()
    # cases = [cases[1]]

    for case in cases:
        img_id = case.split('/')[-1]
        file_seg_raw = os.path.join(case, f'{img_id}_part_255.nii.gz')
        file_fm = os.path.join(fm_dir, f'{img_id}_pred.nii.gz')
        file_spine = os.path.join(case, f'{img_id}_part_252.nii.gz')
        output_dataset_folder = os.path.join(output_dir, img_id)
        os.makedirs(output_dataset_folder, exist_ok=True)

        seg_raw = nib.load(file_seg_raw)
        seg_raw_data = seg_raw.get_fdata().astype(np.uint8)
        seg_fm = nib.load(file_fm)
        seg_fm_data = seg_fm.get_fdata().astype(np.uint8)
        seg_spine= nib.load(file_spine)
        seg_spine_data = seg_spine.get_fdata().astype(np.uint8)

        potential_junctions = seg_fm_data.copy()
    
        # Step 2: Use seg_spine to locate junctions
        # Dilate the spine segmentation to include nearby junctions
        spine_mask = seg_spine_data > 0
        dilated_spine = morphology.binary_dilation(spine_mask, morphology.ball(10))  # TODO: 5 not enough to cover ribs 
        save_nifti(dilated_spine, seg_raw.affine, seg_raw.header, output_dataset_folder, 'dilated_spine.nii.gz')
        
        # Find overlap between potential junctions and dilated spine
        junction_candidates = (potential_junctions & dilated_spine) & ~spine_mask
        save_nifti(junction_candidates, seg_raw.affine, seg_raw.header, output_dataset_folder, 'jxn_candidates.nii.gz')

        # Apply morphological opening to separate weakly connected components
        opened_junction_candidates = morphology.binary_opening(junction_candidates, morphology.ball(2))  # TODO: sometimes 1 cannot be enought to separte
        # Label connected components in the opened junction candidates
        labeled_junctions, num_features = ndimage.label(opened_junction_candidates)

        save_nifti(labeled_junctions, seg_raw.affine, seg_raw.header, output_dataset_folder, 'jxn_candidates_opened.nii.gz')
            
        # Step 3: Filter and merge junctions
        # Calculate the size of each component
        component_sizes = ndimage.sum(opened_junction_candidates, labeled_junctions, range(1, num_features + 1))
        # Sort components by size
        filtered_components = [(index, size) for index, size in enumerate(component_sizes, start=1) if 100 < size < 1500]  # TODO: 
        sorted_components = sorted(enumerate(filtered_components, start=1), key=lambda x: x[1], reverse=True)

        relabeled_components = np.zeros_like(labeled_junctions)
        for new_label, (_, (component_index, _)) in enumerate(sorted_components, start=1):
            relabeled_components[labeled_junctions == component_index] = new_label
        save_nifti(relabeled_components, seg_raw.affine, seg_raw.header, output_dataset_folder, 'jxn_candidates_size_filtered.nii.gz')
        
        # Initialize completed rib segmentation
        completed_rib_seg = seg_raw_data.copy()
        
        # Precompute distance transforms for each rib class
        # Create a mask for all relevant components combined
        combined_relevant_mask = np.zeros_like(seg_raw_data, dtype=bool)
        for (_, (component_index, _)) in sorted_components:
            combined_relevant_mask |= (labeled_junctions == component_index)

        # Precompute distance transforms for each rib class only for the combined relevant components
        distance_transforms = {}
        for rib_class in range(1, 25):
            rib_mask = (seg_raw_data == rib_class)
            
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
                print(f'Merge {size} voxels from cc index {component_index} to rib {closest_rib_class}, distance {min_distance}')

        final_nifti = nib.Nifti1Image(completed_rib_seg.astype(np.uint8), seg_raw.affine, seg_raw.header)
        output_file = os.path.join(output_dataset_folder, f"{img_id}_with_jxn.nii.gz")
        nib.save(final_nifti, output_file)


if __name__ == "__main__":
    main()
