from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from scipy.ndimage.measurements import center_of_mass
from scipy import ndimage
from typing import Sequence, Union, Optional, List

PathLike = Union[str, Path]
LabelType = Union[int, List[int]]

from cads.dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps

_do_outlier_postprocessing_groups = [551, 553, 554, 556, 559]

single_medium_organ = {"typ": "organ", "min": 1, "max": 1, "autofix": 1000}
single_medium_vessel = {"typ": "vessel", "min": 1, "max": 1, "autofix": 1000}
single_small_vessel = {"typ": "vessel", "min": 1, "max": 3, "autofix": 500}
single_small_glands = {"typ": "vessel", "min": 1, "max": 1, "autofix": 10}
single_large_muscle = {"typ": "muscle", "min": 1, "max": 1, "autofix": 1000}
lung = {"typ": "lung", "min": 1, "max": 1, "autofix": 300}
vertebra = {"typ": "lung", "min": 1, "max": 1, "autofix": 100}
heart = vertebra
complex_557 = {"typ": "lung", "min": 1, "max": None, "autofix": 10}

labels_info = {
    # 551
    "spleen": single_medium_organ,
    "kidney_right": single_medium_organ,
    "kidney_left": single_medium_organ,
    "gallbladder": {"typ": "organ", "min": 1, "max": 1, "autofix": 20},
    "liver": {"typ": "organ", "min": 1, "max": 1, "autofix": 2200},
    "stomach": {"typ": "digenstion", "min": 1, "max": 1, "autofix": 1000},
    "aorta": {"typ": "vessel", "min": 1, "max": 1, "autofix": 1000},
    "inferior_vena_cava": {"typ": "vessel", "min": 1, "max": 2, "autofix": 500},
    # "portal_vein_and_splenic_vein": {"typ": "vessel", "min": 1, "max": None, "autofix": 0},  # this is quite fractionated -> do nothing here
    "pancreas": {"typ": "digenstion", "min": 0, "max": 0, "autofix": 1000},
    # "adrenal_gland_right": single_small_glands,
    # "adrenal_gland_left": single_small_glands,
    "lung_upper_lobe_left": lung,
    "lung_lower_lobe_left": lung,
    "lung_upper_lobe_right": lung,
    "lung_middle_lobe_right": lung,
    "lung_lower_lobe_right": lung,
    # 553
    "esophagus": {"typ": "digenstion", "min": 1, "max": 1, "autofix": 1000},
    "trachea": {"typ": "lung", "min": 1, "max": 1, "autofix": 100},
    "heart_myocardium": heart,
    "heart_atrium_left": heart,
    "heart_ventricle_left": heart,
    "heart_atrium_right": heart,
    "heart_ventricle_right": heart,
    # "pulmonary_artery": {"typ": "vessel", "min": 2, "max": 2, "autofix": 50},
    "brain": heart,
    "iliac_artery_left": {"typ": "vessel", "min": 1, "max": 3, "autofix": 10},
    "iliac_artery_right": {"typ": "vessel", "min": 1, "max": 3, "autofix": 10},
    "iliac_vena_left": single_small_vessel,
    "iliac_vena_right": single_small_vessel,
    "small_bowel": {"typ": "organ", "min": 1, "max": None, "autofix": 150},
    # "duodenum": {"typ": "digenstion", "min": 1, "max": 1, "autofix": 800},
    "colon": {"typ": "organ", "min": 1, "max": None, "autofix": 100},
    "urinary_bladder": {"typ": "organ", "min": 1, "max": 1, "autofix": 1500},
    "face": {"typ": "organ", "min": 1, "max": None, "autofix": 100},
    # 554
    "humerus_left": {"typ": "bone", "min": 1, "max": 1, "autofix": 500},
    "humerus_right": {"typ": "bone", "min": 1, "max": 1, "autofix": 500},
    "scapula_left": {"typ": "bone", "min": 1, "max": 1, "autofix": 500},
    "scapula_right": {"typ": "bone", "min": 1, "max": 1, "autofix": 500},
    "clavicula_left": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "clavicula_right": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "femur_left": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "femur_right": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "hip_left": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "hip_right": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "sacrum": {"typ": "bone", "min": 1, "max": 1, "autofix": 1000},
    "gluteus_maximus_left": single_large_muscle,
    "gluteus_maximus_right": single_large_muscle,
    "gluteus_medius_left": single_large_muscle,
    "gluteus_medius_right": single_large_muscle,
    "gluteus_minimus_left": single_large_muscle,
    "gluteus_minimus_right": single_large_muscle,
    "autochthon_left": single_large_muscle,
    "autochthon_right": single_large_muscle,
    "iliopsoas_left": single_large_muscle,
    "iliopsoas_right": single_large_muscle,
    # 556
    "spinal_canal": {"typ": "cns", "min": 1, "max": 1, "autofix": 1000},
    "larynx": {"typ": "organ", "min": 1, "max": 1, "autofix": 1000},
    "heart": {"typ": "organ", "min": 1, "max": 1, "autofix": 1000},
    "bowel_bag": {"typ": "device", "min": 1, "max": 1, "autofix": 1000},
    "sigmoid": {"typ": "organ", "min": 1, "max": 1, "autofix": 1000},
    "rectum": {"typ": "digenstion", "min": 1, "max": 1, "autofix": 1000},
    # "prostate": {"typ": "organ", "min": 1, "max": 1, "autofix": 3600},
    # "seminal_vesicle": {"typ": "digenstion", "min": 2, "max": 2, "autofix": 0},
    "left_mammary_gland": {"typ": "organ", "min": 1, "max": 1, "max_cc": 1, "autofix": 0, "fill": True, "convex_hull": 0},
    "right_mammary_gland": {"typ": "organ", "min": 1, "max": 1, "max_cc": 1, "autofix": 0, "fill": True, "convex_hull": 0},
    "sternum": {"typ": "bone", "min": 1, "max": 1, "max_cc": 1, "autofix": 1500},
    "right psoas major": single_large_muscle,
    "left psoas major": single_large_muscle,
    "right rectus abdominis": single_large_muscle,
    "left rectus abdominis": single_large_muscle,
    # 557
    # "white matter": complex_557,
    # "gray matter": complex_557,
    # "csf": complex_557,  # {"typ": "organ", "min": 1, "max": None, "autofix": 0},
    # "scalp": {"typ": "organ", "min": 1, "max": 1, "autofix": 4000},
    # "eye balls": complex_557,
    # "compact bone": complex_557,
    # "spongy bone": complex_557,
    # "blood": complex_557,
    # "head muscles": complex_557,
    # 558
    # "common_carotid_artery_right": single_medium_vessel,  # do nothing with 558
    # "common_carotid_artery_left": single_medium_vessel,  # do nothing with 558
    # "thyroid_gland": {"typ": "organ", "min": 2, "max": 2, "autofix": 1000},  # TODO: do nothing with 558
    # 559
    "subcutaneous_tissue": {"typ": "rest", "min": 1, "max": 1000, "autofix": 4000},
    "muscle": {"typ": "rest", "min": 1, "max": 1000, "autofix": 4000},
    "abdominal_cavity": {"typ": "rest", "min": 1, "max": 1000, "max_cc": 1, "autofix": 200},
    # "thoracic_cavity": {"typ": "rest", "min": 1, "max": 1000, "autofix": 1500},
    "pericardium": {"typ": "rest", "min": 1, "max": 1, "max_cc": 1, "autofix": 1000},
    "bones": {"typ": "rest", "min": 1, "max": 1000, "autofix": 200},
    "spinal_cord": {"typ": "cns", "min": 1, "max": 1, "autofix": 500},
    # others
    "unused": {"typ": "digenstion", "min": 1, "max": 1, "autofix": 1000},
}

# legacy functions (used to replace TPTBox when python==3.9)
def get_largest_k_connected_components(
    arr: np.ndarray,
    k: Optional[int] = None,
    labels: Optional[Union[int, Sequence[int]]] = None,
    connectivity: int = 3,
    return_original_labels: bool = True,
) -> np.ndarray:
    """finds the largest k connected components in a given array (does NOT work with zero as label!)

    Args:
        arr (np.ndarray): input array
        k (int | None): finds the k-largest components. If k is None, will find all connected components and still sort them by size
        labels (int | list[int] | None): Labels that the algorithm should be applied to. If none, applies on all labels found in arr.
        connectivity: in range [1,3]. For 2D images, 2 and 3 is the same.
        return_original_labels (bool): If set to False, will label the components from 1 to k. Defaults to True

    Returns:
        np.ndarray: array with the largest k connected components
    """
    import cc3d
    # Input validation
    assert k is None or k > 0
    assert 2 <= arr.ndim <= 3, f"expected 2D or 3D, but got {arr.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"

    # Convert connectivity to cc3d format
    if arr.ndim == 2:
        connectivity = min(connectivity * 2, 8)  # 1:4, 2:8, 3:8
    else:
        connectivity = 6 if connectivity == 1 else 18 if connectivity == 2 else 26

    # Copy array and prepare labels
    arr2 = arr.copy()
    
    # Handle labels
    if labels is None:
        unique_labels = np.unique(arr2)
        labels = [l for l in unique_labels if l != 0]
    elif isinstance(labels, (int, np.integer)):
        labels = [labels]
    else:
        labels = list(labels)

    # Set non-target labels to zero
    arr2[~np.isin(arr2, labels)] = 0

    # Find connected components using cc3d
    labels_out = cc3d.connected_components(arr2 > 0, connectivity=connectivity)
    n = labels_out.max()

    if k is None:
        k = n
    k = min(k, n)  # if k > N, will return all N but still sorted

    # Calculate volumes for each component
    volumes = {}
    for i in range(1, n + 1):
        volumes[i] = np.sum(labels_out == i)

    # Sort components by volume
    label_volume_pairs = [(i, ct) for i, ct in volumes.items() if ct > 0]
    label_volume_pairs.sort(key=lambda x: x[1], reverse=True)
    preserve = [x[0] for x in label_volume_pairs[:k]]

    # Create output array
    cc_out = np.zeros(arr.shape, dtype=arr.dtype)
    for i, preserve_label in enumerate(preserve):
        cc_out[labels_out == preserve_label] = i + 1

    if return_original_labels:
        # Multiply with original array to get original labels
        result = arr.copy()
        result *= (cc_out > 0)
        return result
    return cc_out

def np_extract_label(arr: np.ndarray, label: Union[int, Sequence[int]]) -> np.ndarray:
    result = arr.copy()
    if isinstance(label, (list, tuple)):
        mask = np.zeros_like(result, dtype=bool)
        for l in label:
            mask = mask | (result == l)
        result[~mask] = 0
    else:
        result[result != label] = 0
    return result  

def fill(binary_array: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(binary_array)

def np_fill_holes(arr: np.ndarray, 
                 label_ref: Optional[Union[int, Sequence[int]]] = None, 
                 slice_wise_dim: Optional[int] = None) -> np.ndarray:
    assert 2 <= arr.ndim <= 3
    assert arr.ndim == 3 or slice_wise_dim is None, "slice_wise_dim set but array is not 3D"
    
    if label_ref is None:
        labels = list(np.unique(arr))
        if 0 in labels:
            labels.remove(0)
    elif isinstance(label_ref, (int, np.integer)):
        labels = [label_ref]
    else:
        labels = list(label_ref)

    result = arr.copy()
    for l in labels:
        # extract current label
        arr_l = np_extract_label(result, l)
        binary_mask = (arr_l == l)
        
        if slice_wise_dim is None:
            # 3D
            filled = fill(binary_mask)
        else:
            # 2D
            assert 0 <= slice_wise_dim <= arr.ndim - 1
            filled = np.swapaxes(binary_mask, 0, slice_wise_dim)
            filled = np.stack([fill(x) for x in filled])
            filled = np.swapaxes(filled, 0, slice_wise_dim)
        
        # only fill the holes in region within original mask
        holes = filled & ~binary_mask
        result[holes] = l
    
    return result

def fill_holes_3d(img_data: np.ndarray, 
                 label: Optional[Union[int, List[int]]] = None,
                 verbose: bool = True) -> np.ndarray:
    """Fill holes along 3 axes (RAS)"""
    if verbose:
        print("Fill holes")
    
    result = img_data.copy()
    
    # 3D
    result = np_fill_holes(result, label_ref=label, slice_wise_dim=None)
    
    # S
    result = np_fill_holes(result, label_ref=label, slice_wise_dim=2)
    
    # R
    result = np_fill_holes(result, label_ref=label, slice_wise_dim=0)
    
    # A
    result = np_fill_holes(result, label_ref=label, slice_wise_dim=1)
    
    # Again 3D
    result = np_fill_holes(result, label_ref=label, slice_wise_dim=None)
    
    return result

def erode_mask(img_data: np.ndarray, 
               mm: int = 5, 
               labels: Optional[LabelType] = None, 
               connectivity: int = 3) -> np.ndarray:
    if labels is None:
        labels = list(np.unique(img_data))
    if isinstance(labels, int):
        labels = [labels]
    
    result = img_data.copy()
    struct = ndimage.generate_binary_structure(3, connectivity)
    
    for label in labels:
        if label == 0:
            continue
        binary = (result == label)
        eroded = ndimage.binary_erosion(binary, structure=struct, iterations=mm)
        result[binary & ~eroded] = 0
    
    return result

def calc_convex_hull(img_data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    import cv2
    result = img_data.copy()
    binary = result > 0
    
    if axis is not None:
        for i in range(binary.shape[axis]):
            if axis == 0:
                slice_data = binary[i, :, :]
            elif axis == 1:
                slice_data = binary[:, i, :]
            else:
                slice_data = binary[:, :, i]
            
            if np.any(slice_data):
                contours, _ = cv2.findContours(slice_data.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    hull = cv2.convexHull(contours[0])
                    hull_mask = np.zeros_like(slice_data)
                    cv2.fillPoly(hull_mask, [hull], 1)
                    
                    if axis == 0:
                        binary[i, :, :] = hull_mask
                    elif axis == 1:
                        binary[:, i, :] = hull_mask
                    else:
                        binary[:, :, i] = hull_mask
    else:
        for ax in range(3):
            binary = calc_convex_hull(binary, axis=ax)
    
    result[binary] = img_data[binary]
    return result


def filter_connected_components(
    arr: np.ndarray,
    labels: Union[int, List[int], None],
    min_volume: Optional[int] = None,
    max_volume: Optional[int] = None,
    max_count_component: Optional[int] = None,
    connectivity: int = 3,
    removed_to_label: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Replicate TPTBox: filter_connected_components
    """
    cc_arr = get_largest_k_connected_components(
        arr, k=None, labels=labels, 
        connectivity=connectivity, 
        return_original_labels=False
    )
    
    # exclude background 0
    idxs = np.unique(cc_arr)
    idxs = idxs[idxs != 0]
    
    result = arr.copy()
    
    for k, idx in enumerate(idxs, start=1):
        msk = (cc_arr == idx)
        cc_arr[msk] = 0
        
        # calc volume
        s = np.sum(msk)
        
        # check max_count
        if max_count_component is not None and k > max_count_component:
            if verbose:
                print("Remove additional components", "n =",
                      len(idxs) - k, "/", len(idxs))
            result[msk] = removed_to_label
            result[cc_arr != 0] = removed_to_label
            break
            
        # check min volume
        if min_volume is not None and s < min_volume:
            if verbose:
                print(f"Remove components that are too small; n = {len(idxs) - k + 1} with {s} or smaller < {min_volume}")
            result[msk] = removed_to_label
            result[cc_arr != 0] = removed_to_label
            break
            
        # check max volume
        if max_volume is not None and s > max_volume:
            result[msk] = removed_to_label
            
    return result


def postprocess_seg(seg_path: PathLike, 
                   task_id: int, 
                   out_path: Optional[PathLike] = None,
                   aggressiveness: int = 1,
                   verbose: bool = False) -> np.ndarray:
    """
    Let's try not using TPTBox (because we have to run in python 3.9 ToT)
    """
    nii_img = nib.load(seg_path)
    img_data = nii_img.get_fdata()
    
    label_map = map_taskid_to_labelmaps[task_id]
    
    for idx, key in label_map.items():
        if idx == 0:
            continue
        if key not in labels_info:
            print(f"Warning: {key=} is not set up in the info file") if verbose else None
            continue
            
        info = labels_info[key]
        print(key) if verbose else None
        
        if info.get("fill", False):
            print("Fill holes") if verbose else None
            img_data = fill_holes_3d(img_data, idx, verbose=verbose)
        
        img_data = filter_connected_components(
            img_data,
            idx,
            min_volume=info.get("autofix", 0) * aggressiveness,
            max_count_component=info.get("max_cc", None),
            verbose=verbose
        )
        
        if info.get("convex_hull", 0) != 0:
            binary = (img_data == idx)
            eroded = erode_mask(binary, info.get("convex_hull", 0))
            hull = calc_convex_hull(eroded, axis=2)
            img_data[hull > 0] = idx
    
    if out_path is not None:
        if isinstance(out_path, str):
            out_path = Path(out_path)
        nii_out = nib.Nifti1Image(img_data.astype(np.uint8), nii_img.affine, nii_img.header)
        nib.save(nii_out, str(out_path))
    
        print(f'Postprocessed group {task_id}') if verbose else None
    return img_data


def calc_centroids_by_index(msk, label_index, decimals=1, world=False):
    msk_data = np.asanyarray(msk.dataobj, dtype=msk.dataobj.dtype)
    axc = nio.aff2axcodes(msk.affine)
    ctd_list = [axc]
    msk_temp = np.zeros(msk_data.shape, dtype=bool)
    msk_temp[msk_data == label_index] = True
    ctr_mass = center_of_mass(msk_temp)
    if world:
        ctr_mass = msk.affine[:3, :3].dot(ctr_mass) + msk.affine[:3, 3]
        ctr_mass = ctr_mass.tolist()
    ctd_list = [int(x) for x in ctr_mass]
    return ctd_list


def postprocess_head_and_neck(task_id, file_seg_brain_group, file_seg_vertebrae_group, file_out):
    seg_brain_group = nib.load(file_seg_brain_group)
    seg_brain_array = seg_brain_group.get_fdata()
    original_affine = seg_brain_group.affine
    h, w, d = np.shape(seg_brain_array)

    brain_volume_thres = 2000
    brain_volume = np.count_nonzero(seg_brain_array == 9)
    has_brain = brain_volume >= brain_volume_thres

    if has_brain:
        # Case 1: Brain exists - use brain-centered bounding box
        brain_ctd = calc_centroids_by_index(seg_brain_group, label_index=9)
        seg = nib.load(file_out).get_fdata()

        offset_h = 100
        offset_w = 100
        offset_d = 200

        hmin = max(brain_ctd[0] - offset_h, 0)
        hmax = min(brain_ctd[0] + offset_h, h)
        wmin = max(brain_ctd[1] - offset_w, 0)
        wmax = min(brain_ctd[1] + offset_w, w)
        dmin = max(brain_ctd[2] - offset_d, 0)
        dmax = min(brain_ctd[2] + offset_d, d)

        # Apply the bounding box
        seg_cropped = np.zeros((h, w, d), dtype=np.uint8)
        seg_cropped[hmin:hmax, wmin:wmax, dmin:dmax] = seg[hmin:hmax, wmin:wmax, dmin:dmax]
        seg_cropped = nib.Nifti1Image(seg_cropped, original_affine)
        nib.save(seg_cropped, file_out)
        print(f'Postprocessed group {task_id} using brain')
        return True

    # Only load vertebrae if brain check failed and vertebrae file is provided
    if file_seg_vertebrae_group is None:
        return False
        
    # Load cervical vertebrae segmentation (classes 14-24)
    seg_vertebrae_group = nib.load(file_seg_vertebrae_group)
    seg_vertebrae_array = seg_vertebrae_group.get_fdata()
    vertebrae_mask = np.zeros_like(seg_vertebrae_array, dtype=bool)
    for vertebrae_class in range(14, 25):
        vertebrae_mask |= seg_vertebrae_array == vertebrae_class
    has_neck = np.any(vertebrae_mask)

    if has_neck:
        # Case 2: No brain but cervical vertebrae exist - use inferior vertebrae as lower boundary
        seg = nib.load(file_out).get_fdata()
        vertebrae_z_coords = np.where(np.any(vertebrae_mask, axis=(0, 1)))[0]
        inferior_vertebrae_z = vertebrae_z_coords.min()  # Most inferior point of cervical vertebrae

        # Create bounding box from inferior lung boundary to top of image
        hmin, hmax = 0, h
        wmin, wmax = 0, w
        dmin, dmax = inferior_vertebrae_z, d

        # Apply the determined bounding box
        seg_cropped = np.zeros((h, w, d), dtype=np.uint8)
        seg_cropped[hmin:hmax, wmin:wmax, dmin:dmax] = seg[hmin:hmax, wmin:wmax, dmin:dmax]
        seg_cropped = nib.Nifti1Image(seg_cropped, original_affine)
        nib.save(seg_cropped, file_out)
        print(f'Postprocessed group {task_id} using vertebrae')
        return True

    else:
        # Case 3: Neither brain nor neck exist - skip processing
        print('Current file contains neither head nor neck. Skip the neck structures prediction.')
        seg_skipped = np.zeros((h, w, d), dtype=np.uint8)
        seg_skipped = nib.Nifti1Image(seg_skipped, original_affine)
        nib.save(seg_skipped, file_out)
        return True


def postprocess_head(task_id, file_seg_brain_group, file_out):
    seg_brain_group = nib.load(file_seg_brain_group)
    seg_brain_array = seg_brain_group.get_fdata()
    original_affine = seg_brain_group.affine
    h, w, d = np.shape(seg_brain_array)

    brain_volume_thres = 2000
    brain_volume = np.count_nonzero(seg_brain_array == 9)

    if brain_volume < brain_volume_thres:
        print(f'Current file does not contain head part or field out of view. Skip the Brain group prediction.')
        seg_skipped = np.zeros((h, w, d), dtype=np.uint8)
        seg_skipped = nib.Nifti1Image(seg_skipped, original_affine)
        nib.save(seg_skipped, file_out)
    else:
        brain_ctd = calc_centroids_by_index(seg_brain_group, label_index=9)
        seg = nib.load(file_out).get_fdata()

        offset_h = 100
        offset_w = 100
        offset_d = 133

        hmin = max(brain_ctd[0] - offset_h, 0)
        hmax = min(brain_ctd[0] + offset_h, h)
        wmin = max(brain_ctd[1] - offset_w, 0)
        wmax = min(brain_ctd[1] + offset_w, w)
        dmin = max(brain_ctd[2] - offset_d, 0)
        dmax = min(brain_ctd[2] + offset_d, d)

        seg_cropped = np.zeros((h, w, d), dtype=np.uint8)
        seg_cropped[hmin:hmax, wmin:wmax,
                    dmin:dmax] = seg[hmin:hmax, wmin:wmax, dmin:dmax]
        seg_cropped = nib.Nifti1Image(seg_cropped, original_affine)
        nib.save(seg_cropped, file_out)
        print(f'Postprocessed group {task_id}')
