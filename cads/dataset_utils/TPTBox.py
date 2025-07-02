from pathlib import Path
from typing import Union

from cads.dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps
from cads.dataset_utils.postprocessing import labels_info

from TPTBox import NII, to_nii
from TPTBox.logger import Print_Logger


### TPTBOX NOW EXECUTABLE FOR PYTHON>=3.9

def postprocess_seg_TPTBox(seg_path: Union[Path, str, NII], task_id: int, out_path: Union[Path, str, None] = None, aggressiveness=1, verbose=False):
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
        nii = filter_connected_components_TPTBox(
            nii, idx, min_volume=info.get("autofix", 0) * aggressiveness, max_count_component=info.get("max_cc", None), verbose=verbose
        )
        if info.get("convex_hull", 0) != 0:
            nii2 = nii.extract_label(idx).erode_msk_(
                info.get("convex_hull", 0)).calc_convex_hull_("S")
            nii[nii2 != 0] = nii2[nii2 != 0] * idx

    if out_path is not None:
        nii.save(out_path)
    print(f'Postprocessed group {task_id}')


def filter_connected_components_TPTBox(
    self: NII,
    labels: Union[int, list[int], None],
    min_volume: Union[int, None] = None,
    max_volume: Union[int, None] = None,
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
    try:
        nii = self.get_largest_k_segmentation_connected_components(
            None, labels, connectivity=connectivity, return_original_labels=False)
    except (ValueError, AssertionError):  # TPTBox may throw error for empty class (here we skip it!), AssertionError("bbox_nd: img is empty, cannot calculate a bbox")
        return self if inplace else self.copy() 
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
