from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from scipy.ndimage.measurements import center_of_mass

from TPTBox import NII, to_nii
from TPTBox.logger import Print_Logger

from dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps

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


def postprocess_seg(seg_path: Path | str | NII, task_id: int, out_path: Path | str | None = None, aggressiveness=1, verbose=False):
    nii = to_nii(seg_path, True)
    logger = Print_Logger()
    for idx, key in map_taskid_to_labelmaps[task_id].items():
        if idx == 0:
            continue
        if key not in labels_info:
            logger.on_warning(f"{key=} is not set up in the info file")
            continue
        info = labels_info[key]
        logger.on_neutral(key)
        if info.get("fill", False):
            logger.on_neutral("Fill holes")
            nii = (
                nii.fill_holes_(idx)
                .fill_holes_(idx, nii.get_axis("S"))
                .fill_holes_(idx, nii.get_axis("R"))
                .fill_holes_(idx, nii.get_axis("A"))
                .fill_holes_(idx)
            )
        nii = filter_connected_components(
            nii, idx, min_volume=info.get("autofix", 0) * aggressiveness, max_count_component=info.get("max_cc", None), verbose=verbose
        )
        if info.get("convex_hull", 0) != 0:
            nii2 = nii.extract_label(idx).erode_msk_(
                info.get("convex_hull", 0)).calc_convex_hull_("S")
            nii[nii2 != 0] = nii2[nii2 != 0] * idx

    if out_path is not None:
        nii.save(out_path)
    print(f'Postprocessed group {task_id}')


def filter_connected_components(
    self: NII,
    labels: int | list[int] | None,
    min_volume: int | None = None,
    max_volume: int | None = None,
    max_count_component=None,
    connectivity: int = 3,
    removed_to_label=0,
    inplace=False,
    verbose=True,
):
    """
    Filter connected components in a segmentation array based on specified volume constraints.

    Parameters:
    labels (int | list[int]): The labels of the components to filter.
    min_volume (int | None): Minimum volume for a component to be retained. Components smaller than this will be removed.
    max_volume (int | None): Maximum volume for a component to be retained. Components larger than this will be removed.
    max_count_component (int | None): Maximum number of components to retain. Once this limit is reached, remaining components will be removed.
    connectivity (int): Connectivity criterion for defining connected components (default is 3).
    removed_to_label (int): Label to assign to removed components (default is 0).

    Returns:
    None
    """
    arr = self.get_seg_array()
    nii = self.get_largest_k_segmentation_connected_components(
        None, labels, connectivity=connectivity, return_original_labels=False)
    idxs = nii.unique()
    for k, idx in enumerate(idxs, start=1):
        msk = nii.extract_label(idx)
        nii *= -msk + 1
        s = msk.sum()
        if max_count_component is not None and k > max_count_component:  # for mammary glands and sternum
            print("Remove additional components", "n =",
                  idxs[-1] - k, "/", idxs[-1]) if verbose else None
            arr[msk.get_array() != 0] = removed_to_label
            arr[nii.get_array() != 0] = removed_to_label  # set all future to 0
            break
        if min_volume is not None and s < min_volume:
            print(
                f"Remove components that are to small; n = {idxs[-1] - k+1} with {s} or smaller < {min_volume=}") if verbose else None
            arr[msk.get_array() != 0] = removed_to_label
            arr[nii.get_array() != 0] = removed_to_label  # set all future to 0
            break
        if max_volume is not None and s > max_volume:
            arr[msk.get_array() == 1] = removed_to_label
    return self.set_array(arr, inplace)


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
