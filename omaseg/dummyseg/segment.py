from pathlib import Path
import numpy as np
import torch


import argparse
import nibabel as nib


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        metavar="filepath",
        dest="input",
        help="CT nifti image.",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    parser.add_argument(
        "-o",
        metavar="filepath",
        dest="output",
        help="Output path for segmentation mask file",
        type=lambda p: Path(p).absolute(),
        required=True,
    )
    args = parser.parse_args()
    img_path = args.input
    tgt_path = args.output

    print(f'dummyseg: loading image from {img_path}')

    # load input image
    img = nib.load(str(img_path))  # type: ignore
    img_data = img.get_fdata()
    affine = img.affine

    # download latest weights

    # create seg
    seg = (img_data > 0).astype(np.uint8) + (img_data >= 0.5).astype(np.uint8)

    seg = nib.Nifti1Image(seg, affine)  # type: ignore
    nib.save(seg, tgt_path)  # type: ignore
    print(f'dummyseg: saved seg to {tgt_path}')

    net = torch.nn.Linear(8, 8)  # make sure pytorch works


if __name__ == '__main__':
    main()
