"""
Create absolute symbolic links for CADS dataset images
into a target structure suitable for automated processing.

Source:
  /mnt/hddb/murong/CADS-dataset/<dataset>/images/*.nii.gz

Target:
  <dst_root>/<dataset>/rawdata/<id>/<id>_ct.nii.gz -> absolute symlink to source
"""

import os
from pathlib import Path

def main():
    src_root = Path("/mnt/hddb/murong/CADS-dataset")
    dst_root = Path("/mnt/hddb/murong/spine-r-processing")
    overwrite = True

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    created_links = 0
    skipped = 0

    for dataset_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
        dataset_name = dataset_dir.name
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            print(f"[WARN] Skip dataset {dataset_name}: no 'images' directory found.")
            continue

        dst_rawdata = dst_root / dataset_name / "rawdata"
        dst_rawdata.mkdir(parents=True, exist_ok=True)

        image_files = sorted(images_dir.glob("*.nii.gz"))
        if not image_files:
            print(f"[WARN] No .nii.gz files found in {images_dir}")
            continue

        for img_path in image_files:
            if not img_path.is_file():
                continue
            
            name = img_path.name
            # Remove suffix according to pattern
            if name.endswith("_0000.nii.gz"):
                img_id = name[:-len("_0000.nii.gz")]
            elif name.endswith(".nii.gz"):
                img_id = name[:-len(".nii.gz")]
            else:
                img_id = Path(name).stem # remove .nii.gz
            dst_img_dir = dst_rawdata / img_id
            dst_img_dir.mkdir(parents=True, exist_ok=True)

            dst_link = dst_img_dir / f"{img_id}_ct.nii.gz"
            if dst_link.exists():
                if overwrite:
                    dst_link.unlink()
                else:
                    skipped += 1
                    continue

            dst_link.symlink_to(img_path.resolve())
            created_links += 1

    print(f"Done. Created {created_links} symlinks, skipped {skipped} existing entries.")
    print(f"Output root: {dst_root}")

if __name__ == "__main__":
    main()