import os
import numpy as np
import pydicom
import nibabel as nib


def dicom_to_nifti(dicom_folder, savenifti=False, output_file=None):
    # Get a list of DICOM files in the folder
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
    
    if not dicom_files:
        raise ValueError("No DICOM files found in the specified folder.")
    
    # Read the DICOM files
    slices = [pydicom.dcmread(dcm_file) for dcm_file in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by slice position
    
    # Extract the pixel data and stack into a 3D array
    pixel_arrays = [pydicom.pixels.apply_modality_lut(s.pixel_array, s) for s in slices]  # need to convert to HU (otherwise pydicom.dcmread will read intensity from [0, ])
    image_3d = np.stack(pixel_arrays, axis=-1)
    image_3d = np.transpose(image_3d, (1, 0, 2))  # important: by default DICOM's pixel_array is stored in (row -> P/A, column -> L/R), in order to have LPS it needs to transpose dim 0 and 1. 

    # Z spacing: use slices[0].SliceThickness or distance between adjacent slices. Typical CT/MRI slice spacing usually ranges from 0.5mm to 5mm. A resolution of 0.001mm is practically impossible in clinical medical imaging
    if len(slices) > 1:
        # Calculate slice spacing from the position of first two slices
        pos1 = np.array([float(x) for x in slices[0].ImagePositionPatient])
        pos2 = np.array([float(x) for x in slices[1].ImagePositionPatient])
        spacing_z = np.sqrt(np.sum(np.square(pos2 - pos1)))
        
        # Verify spacing is consistent throughout the volume
        if len(slices) > 2:
            spacings = []
            for i in range(len(slices)-1):
                pos1 = np.array([float(x) for x in slices[i].ImagePositionPatient])
                pos2 = np.array([float(x) for x in slices[i+1].ImagePositionPatient])
                spacings.append(np.sqrt(np.sum(np.square(pos2 - pos1))))
            if not np.allclose(spacings, spacing_z, rtol=1e-3):
                print("Warning: Variable slice spacing detected")
    else:
        # Fallback to SliceThickness if only one slice
        spacing_z = float(slices[0].SliceThickness)
    spacingxy = slices[0].PixelSpacing

    a_L = float(slices[0].ImageOrientationPatient[0])
    a_P = float(slices[0].ImageOrientationPatient[1])
    a_H = float(slices[0].ImageOrientationPatient[2])
    b_L = float(slices[0].ImageOrientationPatient[3])
    b_P = float(slices[0].ImageOrientationPatient[4])
    b_H = float(slices[0].ImageOrientationPatient[5])
    p_L = float(slices[0].ImagePositionPatient[0])
    p_P = float(slices[0].ImagePositionPatient[1])
    p_H = float(slices[0].ImagePositionPatient[2])
    c = np.cross(np.asarray(slices[0].ImageOrientationPatient[:3]), np.asarray(slices[0].ImageOrientationPatient[3:]))
    c_L = c[0]
    c_P = c[1]
    c_H = c[2]
    s_x = float(spacingxy[0])
    s_y = float(spacingxy[1])
    s_z = spacing_z

    # Calc transformation matrix (DICOM is LPS)
    affine = np.array(
        [
            [-a_L * s_x, -b_L*s_y, -c_L*s_z, -p_L],
            [-a_P * s_x, -b_P*s_y, -c_P*s_z, -p_P],
            [a_H * s_x, b_H*s_y, c_H*s_z, p_H],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    nifti_image = nib.Nifti1Image(image_3d.astype(np.float32), affine)
    if savenifti:
        nib.save(nifti_image, output_file)
    return nifti_image
