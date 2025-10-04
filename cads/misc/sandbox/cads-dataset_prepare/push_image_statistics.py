from huggingface_hub import HfApi
import os

os.environ["HF_HOME"] = "/scratch/muxu/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def upload_csv_files(csv_folder):
    api = HfApi()
    
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    csv_files.sort()
    csv_files = csv_files[2:]
    
    for csv_file in csv_files:
        csv_file_path = os.path.join(csv_folder, csv_file)
        print(f"Uploading {csv_file}...")
        dataset_name = csv_file.replace('.csv', '')

        api.upload_file(
            path_or_fileobj=csv_file_path,
            repo_id="mrmrx/CADS-dataset",
            path_in_repo=f"{dataset_name}/{csv_file}",
            repo_type="dataset"
        )

if __name__ == "__main__":
    csv_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/CADS-dataset/statistics/csv'  #TODO:
    upload_csv_files(csv_folder)