from huggingface_hub import HfApi
import argparse
import os
os.environ["HF_HOME"] = "/scratch/muxu/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='Upload dataset to HuggingFace Hub')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., 0008_ctorg)')
    parser.add_argument('--upload_seg', action='store_true',
                      help='Upload segmentations')
    parser.add_argument('--upload_img', action='store_true',
                      help='Upload images')
    return parser.parse_args()

def main():
    args = parse_args()
    api = HfApi()
    
    source_root = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/original_format'
    if args.upload_seg:
        print(f"Uploading segmentations...")
        api.upload_large_folder(
            folder_path=str(source_root),
            repo_id="mrmrx/CADS-dataset",
            repo_type="dataset",
            ignore_patterns=["*.json"],
            allow_patterns=[f"{args.dataset}/segmentations/**"],
        )
        print("Finished uploading segmentations")

    if args.upload_img:
        print(f"Uploading images...")
        api.upload_large_folder(
            folder_path=str(source_root),
            repo_id="mrmrx/CADS-dataset",
            repo_type="dataset",
            ignore_patterns=["*.json"],
            allow_patterns=[f"{args.dataset}/images/**"],
        )
        print("Finished uploading images")

if __name__ == "__main__":
    main()