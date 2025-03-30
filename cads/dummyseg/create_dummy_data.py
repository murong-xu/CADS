from pathlib import Path
import nibabel
import numpy as np

def main():
    tgt_dir = Path('/Users/tamaz/Downloads/tmp_slicer/dummy_data')
    affine = np.eye(4)
    affine *= 1.5
    affine[-1, -1] = 1.0
    shape = [128, 128, 128]
    image = np.zeros(shape, dtype=np.float32)
    image[:64, :64, :64] = 0.2
    image[64:, 64:, 64:] = 0.999
    image = nibabel.Nifti1Image(image, affine)
    nibabel.save(image, tgt_dir / 'image.nii.gz')

    # label = np.zeros(shape, dtype=np.int32)
    # label[:64, :64, :64] = 1
    # label[64:, 64:, 64:] = 2
    # label = nibabel.Nifti1Image(label, affine)
    # nibabel.save(label, tgt_dir / 'label.nii.gz')


if __name__ == '__main__':
    main()
