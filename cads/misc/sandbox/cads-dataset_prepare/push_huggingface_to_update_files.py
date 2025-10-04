"""
This script replaces specific files in a HuggingFace Dataset repository.
File update on HuggingFace can be done by directly uploading new files to the same paths.
"""

from huggingface_hub import HfApi
import argparse
import os
from pathlib import Path

os.environ["HF_HOME"] = "/scratch/muxu/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='Replace specific files in HuggingFace Dataset')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., 0008_ctorg)')
    parser.add_argument('--local_dir', type=str, required=True,
                      help='Local directory containing new files')
    parser.add_argument('--folder_in_repo', type=str, default='images',
                      help='Folder in the repository to replace files in (default: images)')
    return parser.parse_args()

def get_local_files(local_path: Path, folder_type: str):
    if folder_type == 'images':
        # image folder
        files = [f.name for f in local_path.glob("*.nii.gz")]
        return [(f, f) for f in files]
    else:
        # segmentations folder
        files = []
        for img_folder in local_path.glob("*"):
            if img_folder.is_dir():
                for f in img_folder.glob("*.nii.gz"):
                    # return (filename, relative path), e.g.: ("liver.nii.gz", "sub-verse123/liver.nii.gz")
                    rel_path = f.relative_to(local_path)
                    files.append((f.name, str(rel_path)))
        return files

def replace_files(args):
    api = HfApi()
    repo_id = 'mrmrx/CADS-dataset'
    
    local_path = Path(args.local_dir)
    local_files = get_local_files(local_path, args.folder_in_repo)
    
    print(f"Found {len(local_files)} files locally to process:")
    for _, rel_path in local_files:
        print(f"- {rel_path}")
    
    if args.folder_in_repo == 'images':
        # if for images, only match files in the specified folder
        allow_patterns = [
            f"{args.dataset}/{args.folder_in_repo}/*.nii.gz"
        ]
    else:
        # if for segmentations, match all files in subfolders
        allow_patterns = [
            f"{args.dataset}/{args.folder_in_repo}/**/*.nii.gz"
        ]
    
    print(f"\nUploading new files...")
    try:
        api.upload_large_folder(
            folder_path=str(local_path.parent.parent),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns, # only upload matched files (updated version)
        )
        print("\nUpload completed successfully!")
    except Exception as e:
        print(f"\nError during upload: {e}")

def main():
    args = parse_args()
    replace_files(args)
    print("\nReplacement completed!")

if __name__ == "__main__":
    main()