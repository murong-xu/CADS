import os
import tempfile
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from scipy import ndimage
import psutil
from joblib import Parallel, delayed
import pickle


def get_metadata(img_nib):
    dim= img_nib.header['dim'] [1:4]
    spacing = img_nib.header['pixdim'][1:4]
    meta = {
    'x_spacing': float(spacing[0]),
    'y_spacing': float(spacing[1]),
    'z_spacing': float(spacing[2]),
    'x_size': int(dim[0]),
    'y_size': int(dim[1]),
    'z_size': int(dim[2]),
    'train': int(3),
    'labelmap': {0: 'None'},
    }
    return meta


def remove_rotation_and_translation(affine: np.ndarray) -> np.ndarray:
    """
    For an affine matrix with positive scaling, return a diagonal affine that only contains the scaling.
    """
    assert affine.shape == (4, 4) and np.allclose(affine[3], [0, 0, 0, 1])
    linear = affine[:3, :3]
    # Make sure the scaling was positive to begin with:
    if np.any(np.diag(linear) < 0):
        raise ValueError(f'Image seems to be not in RAS+ format.\n{affine}')
    spacings: list[float] = np.linalg.norm(linear, axis=0).tolist()  # noqa
    new_affine = np.diag(spacings + [1.])
    return new_affine

def reorient_to(arr, aff, axcodes_to=('R', 'A', 'S'), target_affine=None, verb=False):
    """
    Reorient image to target orientation.
    """
    ornt_fr = nio.io_orientation(aff)
    axcodes_fr = nio.ornt2axcodes(ornt_fr)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    if verb:
        print(f"From orientation: {axcodes_fr}")
        print(f"To orientation: {axcodes_to}")
    arr_reoriented = nio.apply_orientation(arr, ornt_trans)
    # If target_affine is given, use it directly. Otherwise, compute new affine.
    if target_affine is not None:
        newaff = target_affine
    else:
        aff_trans = nio.inv_ornt_aff(ornt_trans, arr_reoriented.shape)
        newaff = np.matmul(aff, aff_trans)
    return nib.Nifti1Image(arr_reoriented, newaff)


def resample_img(img, zoom=0.5, order=0, nr_cpus=-1):
    """
    img: [x,y,z,(t)]
    zoom: 0.5 will halfen the image resolution (make image smaller)

    Resize numpy image array to new size.

    Faster than resample_img_nnunet.
    Resample_img_nnunet maybe slighlty better quality on CT (but not sure).

    Works for 2D and 3D and 4D images.
    """

    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    dim = len(img.shape)

    # Add dimesions to make each input 4D
    if dim == 2:
        img = img[..., None, None]
    if dim == 3:
        img = img[..., None]

    nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    img_sm = np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back
    # Remove added dimensions
    # img_sm = img_sm[:,:,:,0] if img_sm.shape[3] == 1 else img_sm  # remove channel dim if only 1 element
    if dim == 3:
        img_sm = img_sm[:, :, :, 0]
    if dim == 2:
        img_sm = img_sm[:, :, 0, 0]
    return img_sm

def resample_img_nnunet(data, mask=None, original_spacing=1.0, target_spacing=2.0):
    """
    Args:
        data: [x,y,z]
        mask: [x,y,z]
        original_spacing:
        target_spacing:

    Zoom = original_spacing / target_spacing
    (1 / 2 will reduce size by 50%)

    Returns:
        [x,y,z], [x,y,z]
    """
    from nnunet.preprocessing.preprocessing import resample_patient

    if type(original_spacing) is float:
        original_spacing = [original_spacing, ] * 3
    original_spacing = np.array(original_spacing)

    if type(target_spacing) is float:
        target_spacing = [target_spacing, ] * 3
    target_spacing = np.array(target_spacing)

    data = data.transpose((2, 0, 1))  # z is in front for nnUnet
    data = data[None, ...]  # [1,z,x,y], nnunet requires a channel dimension
    if mask is not None:
        mask = mask.transpose((2, 0, 1))
        mask = mask[None, ...]

    def move_last_elem_to_front(l):
        return np.array([l[2], l[0], l[1]])

    # if anisotropy too big, then will resample z axis separately with order=0
    original_spacing = move_last_elem_to_front(original_spacing)
    target_spacing = move_last_elem_to_front(target_spacing)
    data_res, mask_res = resample_patient(data, mask, original_spacing, target_spacing, force_separate_z=None)

    data_res = data_res[0, ...]  # remove channel dimension
    data_res = data_res.transpose((1, 2, 0))  # Move z to back
    if mask is not None:
        mask_res = mask_res[0, ...]
        mask_res = mask_res.transpose((1, 2, 0))
    return data_res, mask_res


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                    nnunet_resample=False, dtype=None):
    """
    Resample nifti image to the new spacing (uses resample_img internally).

    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    nnunet_resample: nnunet resampling will use order=0 sampling for z if very anisotropic. Sometimes results
                        in a little bit less blurry results

    Works for 2D and 3D and 4D images.

    If downsampling an image and then upsampling again to original resolution the resulting image can have
    a shape which is +-1 compared to original shape, because of rounding of the shape to int.
    To avoid this the exact output shape can be provided. Then new_spacing will be ignored and the exact
    spacing will be calculated which is needed to get to target_shape.

    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
    """
    data = img_in.get_fdata()
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())

    if type(new_spacing) is float:
        new_spacing = [new_spacing, ] * 3  # for 3D and 4D
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        img_spacing = np.array(list(img_spacing) + [new_spacing[2], ])

    if target_shape is not None:
        # Find the right zoom to exactly reach the target_shape.
        # We also have to adapt the spacing to this new zoom.
        zoom = np.array(target_shape) / old_shape
        new_spacing = img_spacing / zoom
    else:
        zoom = img_spacing / new_spacing

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)

    # This is only correct if all off-diagonal elements are 0
    # new_affine[0, 0] = new_spacing[0] if img_in.affine[0, 0] > 0 else -new_spacing[0]
    # new_affine[1, 1] = new_spacing[1] if img_in.affine[1, 1] > 0 else -new_spacing[1]
    # new_affine[2, 2] = new_spacing[2] if img_in.affine[2, 2] > 0 else -new_spacing[2]

    # This is the proper solution
    # Scale each column vector by the zoom of this dimension
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    # Just for information: How to get spacing from affine with rotation:
    # Calc length of each column vector:
    # vecs = affine[:3, :3]
    # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

    if nnunet_resample:
        new_data, _ = resample_img_nnunet(data, None, img_spacing, new_spacing)
    else:
        new_data = resample_img(data, zoom=zoom, order=order, nr_cpus=nr_cpus)

    if dtype is not None:
        new_data = new_data.astype(dtype)

    return nib.Nifti1Image(new_data, new_affine)

def determine_orientation(affine):
    axis_labels = ['R', 'L', 'A', 'P', 'S', 'I']
    orientation = []

    # iterate over each column, identify the principal direction for each axis
    for i in range(3): 
        axis = affine[:3, i]
        axis_direction = np.argmax(np.abs(axis)) 
        axis_sign = np.sign(axis[axis_direction])  # determine the sign of the direction

        # assign labels based on the axis and direction sign
        if axis_sign == -1:
            label_index = axis_direction * 2 + 1  # label negative directions
        else:
            label_index = axis_direction * 2  # label positive directions

        orientation.append(axis_labels[label_index])

    return ''.join(orientation)

def preprocess_nifti(file_in, spacing=1.5, num_threads_preprocessing=2):
    raw_img = nib.load(file_in)
    raw_img_numpy = raw_img.get_fdata()
    
    original_affine = raw_img.affine
    original_spacing = np.diag(original_affine, k=0)[:3]
    original_orientation = nio.ornt2axcodes(nio.io_orientation(original_affine))
    original_x_size = raw_img_numpy.shape[0]
    original_y_size = raw_img_numpy.shape[1]
    original_z_size = raw_img_numpy.shape[2]

    # If both spacing and orientation are correct, return original image
    if np.all(np.isclose(original_spacing, spacing)) and original_orientation == ('R', 'A', 'S') and np.allclose(original_affine[:, -1], np.array([0, 0, 0, 1])):
        print(f'Image {file_in} already has correct spacing and orientation. Skipping preprocessing.')
        return None, file_in, None, False

    else:
        print(f'Preprocessing image {file_in}')
        # Reorient to RAS
        img_reoriented = reorient_to(raw_img_numpy, original_affine, axcodes_to=('R', 'A', 'S'), verb=True)

        # Resampling to 1.5
        img_resampled = change_spacing(img_reoriented, [spacing, spacing, spacing], order=3, dtype=np.int32, nr_cpus=num_threads_preprocessing)
        
        # Remove rotation & translation
        affine_removed = remove_rotation_and_translation(img_resampled.affine)
        img_removed = nib.Nifti1Image(img_resampled.get_fdata(), affine_removed)

        # Create temp file path and save
        temp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        basename = os.path.basename(file_in).split('.nii.gz')[0]
        temp_subdir = os.path.join(temp_dir, 'cads', 'temp_results', basename)
        os.makedirs(temp_subdir, exist_ok=True)
        temp_path = os.path.join(temp_subdir, f"{basename}_preprocessed.nii.gz")
        nib.save(img_removed, temp_path)

        metadata_orig = {
            'affine': original_affine,
            'spacing': original_spacing,
            'x_size': original_x_size,
            'y_size': original_y_size,
            'z_size': original_z_size,
        }
        return temp_subdir, temp_path, metadata_orig, True

def get_abs_spacing_from_affine(affine):
    """
    Get spacing from affine matrix using vector lengths, this function ensures correct spacing 
    calculation even for non-diagonal affines.
    """
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    return spacing

def is_affine_diagonal(affine, tol=1e-6):
    """Check if the affine matrix is diagonal (no rotation or shearing)."""
    return np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])), atol=tol)

def restore_seg_in_orig_format(file_seg_in, file_seg_out, metadata_orig, num_threads_preprocessing=2):
    print('Restore the segmentation to original format...')

    seg_preprocessed = nib.load(file_seg_in)
    orig_affine = metadata_orig['affine']

    # 1. Get current and target orientations
    current_ornt = nio.io_orientation(seg_preprocessed.affine)
    current_axcodes = nio.ornt2axcodes(current_ornt)
    
    target_ornt = nio.io_orientation(orig_affine)
    target_axcodes = nio.ornt2axcodes(target_ornt)
    
    # 2. Get original spacing and sizes
    spacings = get_abs_spacing_from_affine(orig_affine)
    orig_sizes = [metadata_orig['x_size'], 
                  metadata_orig['y_size'], 
                  metadata_orig['z_size']]
    
    # 3. Determine the correct reading order of spacing and shape from current to target orientation
    reordered_spacings = np.zeros(3)
    reordered_shape = np.zeros(3, dtype=int)
    
    for curr_idx, curr_code in enumerate(current_axcodes):
        if curr_code in target_axcodes:
            tgt_idx = target_axcodes.index(curr_code)
        else:
            opposite_code = {'R':'L', 'L':'R', 'A':'P', 'P':'A', 'S':'I', 'I':'S'}[curr_code]
            tgt_idx = target_axcodes.index(opposite_code)
        reordered_spacings[curr_idx] = spacings[tgt_idx]
        reordered_shape[curr_idx] = orig_sizes[tgt_idx]

    # 4. Resample to original spacing
    seg_resampled = change_spacing(seg_preprocessed, reordered_spacings, 
                                 target_shape=tuple(reordered_shape), 
                                 order=0, dtype=np.int32, 
                                 nr_cpus=num_threads_preprocessing)

    # 5. Reorient to original orientation
    seg_reoriented = reorient_to(seg_resampled.get_fdata(), seg_resampled.affine, 
                                target_axcodes, target_affine=orig_affine, verb=True)

    # 6. Restore final segmentation
    seg_restored = nib.Nifti1Image(seg_reoriented.get_fdata().astype(np.uint8), 
                                  orig_affine)
    
    nib.save(seg_restored, file_seg_out)


def preprocess_nifti_ctrate(raw_img, output_filename, spacing=1.5, num_threads_preprocessing=2):
    raw_img_numpy = raw_img.get_fdata()
    
    original_affine = raw_img.affine
    original_spacing = np.diag(original_affine, k=0)[:3]
    original_x_size = raw_img_numpy.shape[0]
    original_y_size = raw_img_numpy.shape[1]
    original_z_size = raw_img_numpy.shape[2]

    # Reorient to RAS
    img_reoriented = reorient_to(raw_img_numpy, original_affine, axcodes_to=('R', 'A', 'S'), verb=True)

    # Resampling to 1.5
    img_resampled = change_spacing(img_reoriented, [spacing, spacing, spacing], order=3, dtype=np.int32, nr_cpus=num_threads_preprocessing)
    
    # Remove rotation & translation
    affine_removed = remove_rotation_and_translation(img_resampled.affine)
    img_removed = nib.Nifti1Image(img_resampled.get_fdata(), affine_removed)

    # Create temp file path and save
    output_folder = os.path.dirname(output_filename)
    basename = os.path.basename(output_filename).split('.nii.gz')[0]
    nib.save(img_removed, output_filename)

    metadata_orig = {
        'affine': original_affine,
        'spacing': original_spacing,
        'x_size': original_x_size,
        'y_size': original_y_size,
        'z_size': original_z_size,
    }
    metadata_file = os.path.join(output_folder, f'{basename}_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata_orig, f)
        f.close()
    

def restore_seg_in_orig_format_ctrate(file_seg, metadata_orig, num_threads_preprocessing=2):
    affine_identity = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    print('Restore the segmentation to original format...')

    seg_preprocessed = nib.load(file_seg)
    orig_spacing = metadata_orig['spacing']
    orig_affine = metadata_orig['affine']

    # Resample to original spacing (using absolute values for zoom)
    abs_spacing = np.abs(orig_spacing)
    orig_shape = (metadata_orig['x_size'], 
                 metadata_orig['y_size'], 
                 metadata_orig['z_size'])
    seg_resampled = change_spacing(seg_preprocessed, abs_spacing, target_shape=orig_shape, order=0, 
                                 dtype=np.int32, nr_cpus=num_threads_preprocessing)

    # Reorient to original orientation
    orig_orientation = nio.ornt2axcodes(nio.io_orientation(orig_affine))
    seg_reoriented = reorient_to(seg_resampled.get_fdata(), seg_resampled.affine, 
                                orig_orientation, verb=True)
    
    # Restore final segmentation with exactly the original affine matrix (translation etc.)
    seg_restored = nib.Nifti1Image(seg_reoriented.get_fdata().astype(np.uint8), 
                                  affine_identity)

    nib.save(seg_restored, file_seg)
