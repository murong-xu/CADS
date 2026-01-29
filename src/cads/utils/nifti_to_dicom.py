import os
import numpy as np
import pydicom
import nibabel as nib
from tqdm import tqdm


def nifti_to_dicom(path_defaced_nii, path_original_dcm, output_dir):
    """
    Convert a NIfTI file to DICOM series based on original DICOM
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the defaced NIfTI
    nii_img = nib.load(path_defaced_nii)
    nii_data = nii_img.get_fdata()
    # by default DICOM's pixel_array is stored in (row -> P/A, column -> L/R), in order to have LPS it needs to transpose dim 0 and 1
    nii_data = np.transpose(nii_data, (1, 0, 2))

    # load original DICOM files
    dicom_files = [os.path.join(path_original_dcm, f) for f in os.listdir(path_original_dcm)
                   if f.endswith(".dcm")]
    slices = [pydicom.dcmread(dcm_file) for dcm_file in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # sort by z

    # check if number of slices match
    if len(slices) != nii_data.shape[2]:
        raise ValueError(f"Number of slices doesn't match: DICOM has {len(slices)}, "
                         f"NIfTI has {nii_data.shape[2]}")

    first_slice = slices[0]
    # convert NIfTI data to appropriate data type and scale
    if hasattr(first_slice, 'RescaleSlope') and hasattr(first_slice, 'RescaleIntercept'):
        slope = float(first_slice.RescaleSlope)
        intercept = float(first_slice.RescaleIntercept)
        nii_data = (nii_data - intercept) / slope
    original_dtype = first_slice.pixel_array.dtype
    nii_data = nii_data.astype(original_dtype)

    # create new DICOM files
    print("Converting NIfTI to DICOM series...")
    new_series_uid = pydicom.uid.generate_uid()
    new_study_uid = pydicom.uid.generate_uid()
    new_series_description = first_slice.SeriesDescription + \
        '_DEFACED' if hasattr(first_slice, 'SeriesDescription') else 'DEFACED'
    original_study_desc = getattr(first_slice, 'StudyDescription', '')
    new_study_description = original_study_desc + \
        ' DEFACED' if original_study_desc else 'DEFACED Study'

    for idx, original_slice in enumerate(tqdm(slices)):
        new_slice = pydicom.Dataset()
        new_slice = original_slice

        # update study-level info
        new_slice.StudyInstanceUID = new_study_uid
        new_slice.StudyDescription = new_study_description

        # update series-level info
        new_slice.SeriesDescription = new_series_description
        new_slice.SeriesInstanceUID = new_series_uid
        new_slice.ImageComments = "DEFACED: Facial features removed"

        # change patient name
        if hasattr(new_slice, 'PatientName'):
            new_slice.PatientName = str(new_slice.PatientName) + '_DEFACED'

        # change image type
        if hasattr(new_slice, 'ImageType'):
            image_type = list(new_slice.ImageType)
            if 'DEFACED' not in image_type:
                image_type.append('DEFACED')
            new_slice.ImageType = image_type

        # store in slice data
        slice_data = nii_data[:, :, idx].astype(
            original_slice.pixel_array.dtype)
        new_slice.PixelData = slice_data.tobytes()
        new_slice.Rows, new_slice.Columns = slice_data.shape[0], slice_data.shape[1]

        new_slice.ImageOrientationPatient = original_slice.ImageOrientationPatient
        new_slice.ImagePositionPatient = original_slice.ImagePositionPatient
        new_slice.PixelSpacing = original_slice.PixelSpacing
        new_slice.SliceThickness = original_slice.SliceThickness

        # save the new DICOM file
        output_path = os.path.join(output_dir, f'slice_{idx:04d}.dcm')
        new_slice.save_as(output_path)


path_defaced_nii = '/mnt/hdda/murong/debug_conversion/python script_ original dicom and defaced nifit/1.2.40.0.13.1.254279914657978503889172197480895470432_defaced/1303_Schadel_1.0_J70h_2_5_Schadel_1.0_J70h_2_LPS_DFmasked.nii.gz'
path_original_dcm = '/mnt/hdda/murong/debug_conversion/python script_ original dicom and defaced nifit/1.2.40.0.13.1.254279914657978503889172197480895470432'
output_dir = '/mnt/hdda/murong/debug_conversion/python script_ original dicom and defaced nifit/output'
nifti_to_dicom(path_defaced_nii, path_original_dcm, output_dir)
