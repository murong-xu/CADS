#!/usr/bin/env python3
"""Convert Spine-R segmentation label IDs to the CADS label schema."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import nibabel as nib
import numpy as np

labelmap_part_vertebrae = {
    1: ("vertebrae_L5", 24),
    2: ("vertebrae_L4", 23),
    3: ("vertebrae_L3", 22),
    4: ("vertebrae_L2", 21),
    5: ("vertebrae_L1", 20),
    6: ("vertebrae_T12", 19),
    7: ("vertebrae_T11", 18),
    8: ("vertebrae_T10", 17),
    9: ("vertebrae_T9", 16),
    10: ("vertebrae_T8", 15),
    11: ("vertebrae_T7", 14),
    12: ("vertebrae_T6", 13),
    13: ("vertebrae_T5", 12),
    14: ("vertebrae_T4", 11),
    15: ("vertebrae_T3", 10),
    16: ("vertebrae_T2", 9),
    17: ("vertebrae_T1", 8),
    18: ("vertebrae_C7", 7),
    19: ("vertebrae_C6", 6),
    20: ("vertebrae_C5", 5),
    21: ("vertebrae_C4", 4),
    22: ("vertebrae_C3", 3),
    23: ("vertebrae_C2", 2),
    24: ("vertebrae_C1", 1),
}
# 26-32: sacrum (skip)


def get_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """
    Get spacing from affine matrix using vector lengths.
    This ensures correct spacing calculation even for non-diagonal affines.
    """
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return spacing


def build_original_to_target_map() -> Mapping[int, int]:
    """Invert the mapping so keys are original label IDs."""
    return {
        original_label: target_label
        for target_label, (_, original_label) in labelmap_part_vertebrae.items()
    }


def discover_segmentation_files(root: Path) -> Iterable[Path]:
    """Find Spine-R segmentation masks under the provided root."""
    yield from sorted(root.rglob("*_seg-vert_msk.nii.gz"))


def build_output_path(
    seg_path: Path, src_root: Path, dst_root: Path, suffix: str
) -> Path:
    """Determine the destination NIfTI path that mirrors the expected tree."""
    rel_path = seg_path.relative_to(src_root)
    if len(rel_path.parts) < 2:
        raise ValueError(f"unexpected path layout for {seg_path}")
    study_name = seg_path.parent.name
    filename = f"{study_name}{suffix}.nii.gz"
    return dst_root / "segmentations" / study_name / filename


def find_image_for_study(
    study_name: str, dataset_name: str, image_root: Path
) -> Optional[Path]:
    """Find the corresponding image file for a study ID."""
    image_path = image_root / dataset_name / "images" / f"{study_name}_0000.nii.gz"
    return image_path if image_path.exists() else None


def write_segmentation_from_affine(
    data: np.ndarray, affine: np.ndarray, dst_path: Path, overwrite: bool
) -> bool:
    """Save a segmentation with the supplied affine, matching previous pipeline output."""
    if dst_path.exists() and not overwrite:
        logging.info("skip (exists) %s", dst_path)
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    new_img = nib.Nifti1Image(data, affine)
    new_img.header.set_data_dtype(np.uint8)
    spacing = get_spacing_from_affine(affine)
    new_img.header['pixdim'][1:4] = spacing
    new_img.to_filename(dst_path)
    return True


def remap_segmentation(
    src_path: Path,
    dst_path: Path,
    mapping: Mapping[int, int],
    overwrite: bool,
    template_holder: Optional[List[Optional[nib.Nifti1Image]]] = None,
) -> bool:
    """Load one segmentation, remap labels, and save them with the new schema."""
    if dst_path.exists() and not overwrite:
        logging.info("skip (exists) %s -> %s", src_path.name, dst_path)
        return False

    img = nib.load(src_path)
    if template_holder is not None and template_holder[0] is None:
        template_holder[0] = img
    data = np.asarray(img.dataobj, dtype=np.uint8)
    remapped = np.zeros_like(data, dtype=np.uint8)

    for original_value, target_value in mapping.items():
        remapped[data == original_value] = target_value

    success = write_segmentation_from_affine(remapped, img.affine, dst_path, overwrite)
    if success:
        logging.info("converted %s -> %s", src_path.name, dst_path)
    return success


def write_empty_segmentation(
    dst_path: Path, template_img: nib.Nifti1Image, overwrite: bool
) -> bool:
    """Generate an all-zero segmentation following the template image."""
    zeros = np.zeros(template_img.shape, dtype=np.uint8)
    success = write_segmentation_from_affine(zeros, template_img.affine, dst_path, overwrite)
    if success:
        logging.info("created empty segmentation %s", dst_path)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Spine-R vertebra segmentation labels to CADS schema."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Root containing Spine-R derivatives (e.g. /mnt/.../spine-r-processing).",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Destination root for the converted files (e.g. /mnt/.../CADS-dataset-v2/Spine-R).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Root containing original CADS images (e.g. /mnt/.../CADS-dataset).",
    )
    parser.add_argument(
        "--suffix",
        default="_part_552",
        help="Suffix used for the converted filename (default: _part_552).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing remapped files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be done without writing any files.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                        filename="/home/murong/22k/OMASeg_sync/OMASeg/convert_spine_r_seg.log",
                        filemode="a")
    mapping = build_original_to_target_map()

    seg_files = list(discover_segmentation_files(args.src_root))
    
    # Find all study directories and identify which ones lack segmentation files
    seg_dirs = sorted(
        child
        for child in args.src_root.iterdir()
        if child.is_dir() and child.name.startswith("s")
    )
    missing_dirs: List[Path] = []
    for seg_dir in seg_dirs:
        matches = list(seg_dir.glob("*_seg-vert_msk.nii.gz"))
        if not matches:
            missing_dirs.append(seg_dir)

    if not seg_files and not missing_dirs:
        logging.warning("no segmentation files or directories found under %s", args.src_root)
        return

    # Process existing segmentation files
    total, converted = 0, 0
    template_holder: List[Optional[nib.Nifti1Image]] = [None]
    for seg_path in seg_files:
        total += 1
        try:
            dst_path = build_output_path(seg_path, args.src_root, args.dst_root, args.suffix)
        except ValueError as exc:
            logging.warning("skip %s (%s)", seg_path, exc)
            continue

        if args.dry_run:
            logging.info("dry run: %s -> %s", seg_path, dst_path)
            continue

        if remap_segmentation(
            seg_path,
            dst_path,
            mapping,
            args.overwrite,
            template_holder=template_holder,
        ):
            converted += 1

    # Process missing directories by creating empty segmentations
    empty_created = 0
    empty_failed = 0
    if missing_dirs:
        logging.info(
            "directories without seg file: %d (e.g. %s)",
            len(missing_dirs),
            ", ".join(dir.name for dir in missing_dirs[:5]),
        )
        
        for seg_dir in missing_dirs:
            study_name = seg_dir.name
            dataset_name = str(args.src_root).split('/')[-2]
            
            # Find corresponding image file to use as template
            image_path = find_image_for_study(study_name, dataset_name, args.image_root)
            if image_path is None:
                logging.warning(
                    "skip %s: no image found at %s",
                    study_name,
                    args.image_root / dataset_name / "images" / f"{study_name}_0000.nii.gz",
                )
                empty_failed += 1
                continue
            
            # Build output path
            fake_seg_path = seg_dir / f"{study_name}_seg-vert_msk.nii.gz"
            try:
                dst_path = build_output_path(
                    fake_seg_path, args.src_root, args.dst_root, args.suffix
                )
            except ValueError as exc:
                logging.warning("skip %s (%s)", study_name, exc)
                empty_failed += 1
                continue
            
            if args.dry_run:
                logging.info("dry run (empty): %s (template: %s)", dst_path, image_path.name)
                empty_created += 1
                continue
            
            # Load image as template and create empty segmentation
            try:
                template_img = nib.load(image_path)
                if write_empty_segmentation(dst_path, template_img, args.overwrite):
                    empty_created += 1
            except Exception as exc:
                logging.error("failed to create empty seg for %s: %s", study_name, exc)
                empty_failed += 1

    logging.info(
        "finished (%d found, %d converted, %d missing, %d empties created, %d empty failed)",
        total,
        converted,
        len(missing_dirs),
        empty_created,
        empty_failed,
    )


if __name__ == "__main__":
    main()
