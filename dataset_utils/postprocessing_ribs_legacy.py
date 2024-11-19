import argparse
import os
import glob
import nibabel as nib
import numpy as np
from skimage import measure, morphology
from scipy import ndimage
from collections import Counter

def compute_min_distance(mask1, mask2):
    """min distance between two masks"""
    dist_transform_mask2 = ndimage.distance_transform_edt(~mask2)
    min_dist_1_to_2 = np.min(dist_transform_mask2[mask1])

    dist_transform_mask1 = ndimage.distance_transform_edt(~mask1)
    min_dist_2_to_1 = np.min(dist_transform_mask1[mask2])
    
    # bi-direction
    return min(min_dist_1_to_2, min_dist_2_to_1)

def calculate_angle(vector1, vector2):
    """angle between two vectors (in degree)"""
    # normalize
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    
    dot_product = np.clip(np.dot(vector1, vector2), -1.0, 1.0)
    angle = np.arccos(dot_product) * 180 / np.pi
    
    # return angle in a range of 0-90
    return min(angle, 180 - angle)


def get_component_direction(component):
    # find the main direction in which the component stretches/extends the most
    coords = np.array(np.where(component)).T
    if coords.shape[0] < 2:
        # if only one point/no point
        return np.array([1, 0, 0])
    
    mean_coords = np.mean(coords, axis=0)
    centered_coords = coords - mean_coords
    cov = np.cov(centered_coords.T) + np.eye(3) * 0.01
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvectors[:, np.argmax(eigenvalues)]

def are_components_aligned(comp1, comp2, threshold_angle=10):
    dir1 = get_component_direction(comp1)
    dir2 = get_component_direction(comp2)
    angle = np.arccos(np.clip(np.dot(dir1, dir2), -1, 1)) * 180 / np.pi
    return angle < threshold_angle

def get_common_normal_vector(direction1, direction2):
    """common normal vector between two vectors"""
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)
    
    mean_direction = (direction1 + direction2) / 2
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    
    reference = np.array([1, 0, 0])
    
    normal = np.cross(mean_direction, reference)
    normal = normal / np.linalg.norm(normal)
    
    return normal

def calculate_projected_height(component, normal_vector, reference_point):
    """projection height of a component on a normal vector based on a reference point"""
    coords = np.array(np.where(component)).T
    centered_coords = coords - reference_point
    return np.mean(np.dot(centered_coords, normal_vector))


def get_normal_vector(component):
    coords = np.array(np.where(component)).T
    # avoid too few points
    if coords.shape[0] < 2:
        return np.array([0, 0, 1])
    
    # avoid points are all identical
    if np.all(np.all(coords == coords[0, :], axis=1)):
        return np.array([0, 0, 1])
    
    mean_coords = np.mean(coords, axis=0)
    centered_coords = coords - mean_coords
    cov = np.cov(centered_coords.T) + np.eye(3) * 0.1
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvectors[:, np.argmax(eigenvalues)]

def calculate_plane_height(component, normal_vector):
    coords = np.array(np.where(component)).T
    return np.mean(np.dot(coords, normal_vector))

def get_local_region(component, center_point, radius=5):
    """get a local region around a center point"""
    x, y, z = center_point
    shape = component.shape
    
    # define boundaries
    x_min = max(0, x - radius)
    x_max = min(shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(shape[1], y + radius + 1)
    z_min = max(0, z - radius)
    z_max = min(shape[2], z + radius + 1)
    
    # create a local mask
    local_mask = np.zeros_like(component)
    local_mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1
    
    return component & local_mask

def find_overlap_center(comp1, comp2, dilated):
    """find the center of overlapping area between two components. if multiple overlapping areas, return one of their center"""
    overlap = dilated & comp2
    if not np.any(overlap):
        return None
    labeled_overlap, num_features = ndimage.label(overlap)
    
    if num_features == 0:
        return None
    
    # choose the 1st overlapping area 
    first_overlap = (labeled_overlap == 1)
    overlap_coords = np.array(np.where(first_overlap)).T
    center = np.mean(overlap_coords, axis=0).astype(int)
    return tuple(center)

def are_components_on_same_rib(comp1, comp2, dilate_ball=1, height_threshold=10, local_radius=20):
    # find the center of overlapping area
    dilated_comp1 = morphology.binary_dilation(comp1, morphology.ball(dilate_ball))
    center = find_overlap_center(comp1, comp2, dilated_comp1)
    
    if center is None:
        return False
    
    # get local regions
    local_comp1 = get_local_region(comp1, center, local_radius)
    local_comp2 = get_local_region(comp2, center, local_radius)
    
    # ensure enough points in local regions
    if np.sum(local_comp1) < 10 or np.sum(local_comp2) < 10:
        return False
    
    # calc normal vector of normal regions, and projection height
    normal1 = get_normal_vector(local_comp1)
    normal2 = get_normal_vector(local_comp2)
    common_normal = get_common_normal_vector(normal1, normal2)
    reference_point = np.array(center)
    height1 = calculate_projected_height(local_comp1, common_normal, reference_point)
    height2 = calculate_projected_height(local_comp2, common_normal, reference_point)
    
    # if two components are at similar height
    height_diff = abs(height1 - height2)
    print('height difference: ', height_diff)
    return height_diff < height_threshold

def merge_close_components(labeled_image, max_distance, min_size=10, z_threshold=10):
    labels = np.unique(labeled_image)
    labels = labels[labels != 0]  # exclude background

    for label in labels:
        mask = labeled_image == label
        if np.sum(mask) < min_size:
            continue  # exclude small outliers

        # dilation for each connected component
        dilated = morphology.binary_dilation(mask, morphology.ball(max_distance))
        overlapping = np.unique(labeled_image[dilated & (labeled_image != label)])
        
        # swallow up other labels
        for overlap_label in overlapping:
            if overlap_label != 0 and overlap_label != label:
                overlap_mask = labeled_image == overlap_label
                if np.sum(overlap_mask) < min_size:
                    continue
                print('Check ', label, ' and  ', overlap_label)
                if are_components_on_same_rib(mask, overlap_mask, dilate_ball=max_distance, height_threshold=z_threshold, local_radius=10):
                    print('belong to same rib: ', label, ' and  ', overlap_label)
                    mask_size = np.sum(mask)
                    overlap_size = np.sum(overlap_mask)
                    
                    if mask_size >= overlap_size:
                        labeled_image[overlap_mask] = label
                    else:
                        labeled_image[mask] = overlap_label

    return labeled_image

def assign_rib_classes(merged_seg, original_seg):
    unique_labels = np.unique(merged_seg)[1:]
    new_seg = np.zeros_like(merged_seg)
    
    for label in unique_labels:
        mask = (merged_seg == label)
        original_classes = original_seg[mask]
        
        # majority vote
        class_votes = Counter(original_classes[original_classes != 0])  # TODO: what if equal class number
        
        if class_votes:
            majority_class = max(class_votes, key=class_votes.get)
            new_seg[mask] = majority_class
    
    return new_seg

def separate_touching_components(labeled_image, erosion_size=1, min_size=10):
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]

    new_labeled_image = np.zeros_like(labeled_image)
    next_label = 1

    for label in unique_labels:
        component = (labeled_image == label)
        
        # erode
        eroded = morphology.binary_erosion(component, morphology.ball(erosion_size))
        eroded_labeled, num_eroded_features = ndimage.label(eroded)
        
        if num_eroded_features > 1:
            # if multiple components after erosion, could be potentially wrongly merged ribs
            for i in range(1, num_eroded_features + 1):
                eroded_component = (eroded_labeled == i)
                # use conditional dilation to restore ribs
                reconstructed = morphology.reconstruction(eroded_component, component, method='dilation')  # TODO: component?
                reconstructed = reconstructed.astype(bool)
                if np.sum(reconstructed) >= min_size:
                    new_labeled_image[reconstructed] = next_label
                    next_label += 1
        else:
            new_labeled_image[component] = next_label
            next_label += 1

    return new_labeled_image

# def separate_touching_components(labeled_image, erosion_size=1, min_size=10):
#     unique_labels = np.unique(labeled_image)
#     unique_labels = unique_labels[unique_labels != 0]

#     new_labeled_image = np.zeros_like(labeled_image)
#     next_label = 1

#     for label in unique_labels:
#         component = (labeled_image == label)
        
#         eroded = morphology.binary_erosion(component, morphology.ball(erosion_size))
#         dilated = morphology.binary_dilation(eroded, morphology.ball(erosion_size))
        
#         final_component = dilated & component

#         new_labeled_image[final_component] = next_label
#         next_label += 1
        
#         # if num_eroded_features > 1:
#         #     for i in range(1, num_eroded_features + 1):
#         #         eroded_component = (eroded_labeled == i)
#         #         dilated = morphology.binary_dilation(eroded_component, morphology.ball(erosion_size))
#         #         final_component = dilated & component
#         #         if np.sum(final_component) >= min_size:
#         #             new_labeled_image[final_component] = next_label
#         #             next_label += 1
#         # else:

#         #     new_labeled_image[component] = next_label
#         #     next_label += 1

#     return new_labeled_image

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
        input_segs = glob.glob(datasetfolder + '/*_part_555.nii.gz')  # TODO:
        input_segs = input_segs[1:]
        # input_segs = glob.glob(datasetfolder + '/*/*_part_555.nii.gz')  # TODO:
        input_segs.sort()
        output_dataset_folder = os.path.join(output_dir, dataset, str(split))
        os.makedirs(output_dataset_folder, exist_ok=True)

        for input_seg in input_segs:
            img_id = input_seg.split('/')[-1].split('_part_555.nii.gz')[0]  # TODO:
            print('image: ', img_id)
            seg = nib.load(input_seg)
            seg_data = seg.get_fdata().astype(np.uint8)

            output_seg_folder = os.path.join(output_dataset_folder, img_id)
            os.makedirs(output_seg_folder, exist_ok=True)

            # step 1) binarize segmentaion
            # binary_seg = np.where(seg_data > 0, 1, 0)

            # step 2) identify connected components (6-connectivity for 3D voxels)
            # labeled_seg, num_features = ndimage.label(binary_seg)

            # a = nib.Nifti1Image(labeled_seg.astype(np.uint8), seg.affine, seg.header)
            # output_file = os.path.join(output_seg_folder, f"{img_id}_step2.nii.gz")
            # nib.save(a, output_file)

            # separated_seg = separate_touching_components(labeled_seg, erosion_size=1, min_size=10)
            # d = nib.Nifti1Image(separated_seg.astype(np.uint8), seg.affine, seg.header)
            # output_file = os.path.join(output_seg_folder, f"{img_id}_erosion_1.nii.gz")
            # nib.save(d, output_file)

            # binary_seg = np.where(separated_seg > 0, 1, 0)
            # labeled_seg, num_features = ndimage.label(binary_seg)

            # c = nib.Nifti1Image(labeled_seg.astype(np.uint8), seg.affine, seg.header)
            # output_file = os.path.join(output_seg_folder, f"{img_id}_erosion_2.nii.gz")
            # nib.save(c, output_file)

            # step 3) merge components that are close to each other
            max_distance = 2  # TODO: good?
            merged_seg = merge_close_components(seg_data, max_distance)

            e = nib.Nifti1Image(merged_seg.astype(np.uint8), seg.affine, seg.header)
            output_file = os.path.join(output_seg_folder, f"{img_id}_merged.nii.gz")
            nib.save(e, output_file)

            merged_seg, final_num_features = ndimage.label(merged_seg > 0)

            b = nib.Nifti1Image(merged_seg.astype(np.uint8), seg.affine, seg.header)
            output_file = os.path.join(output_seg_folder, f"{img_id}_step3.nii.gz")
            nib.save(b, output_file)

            # step 4) majority vote on each single rib class
            reordered_seg = assign_rib_classes(merged_seg, seg_data)

            final_nifti = nib.Nifti1Image(reordered_seg.astype(np.uint8), seg.affine, seg.header)
            output_file = os.path.join(output_seg_folder, f"{img_id}_part_555.nii.gz")
            nib.save(final_nifti, output_file)


if __name__ == "__main__":
    main()
