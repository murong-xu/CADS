"""
Giving all my best to download large datasets from HuggingFace without being rate-limited.
"""

import os
import time
import random
import requests
from huggingface_hub import snapshot_download, login, HfApi

# TODO: input param
repo_id = "mrmrx/CADS-dataset"
download_folder = "/mnt/hddb/murong/CADS-dataset"
folder_type = "segmentations"  # "images" or "segmentations"
dataset_names = [
    "0001_visceral_gc",
    "0002_visceral_sc",
    "0003_kits21",
    "0004_lits",
    "0005_bcv_abdomen",
    "0006_bcv_cervix",
    "0007_chaos",
    "0008_ctorg",
    "0009_abdomenct1k",
    "0010_verse",
    "0011_exact",
    "0012_cad_pe",
    "0013_ribfrac",
    "0014_learn2reg",
    "0015_lndb",
    "0016_lidc",
    "0017_lola11",
    "0018_sliver07",
    "0019_tcia_ct_lymph_nodes",
    "0020_tcia_cptac_ccrcc",
    "0021_tcia_cptac_luad",
    "0022_tcia_ct_images_covid19",
    "0023_tcia_nsclc_radiomics",
    "0024_pancreas_ct",
    "0025_pancreatic_ct_cbct_seg",
    "0026_rider_lung_ct",
    "0027_tcia_tcga_kich",
    "0028_tcia_tcga_kirc",
    "0029_tcia_tcga_kirp",
    "0030_tcia_tcga_lihc",
    "0032_stoic2021",
    "0033_tcia_nlst",
    "0034_empire",
    "0037_totalsegmentator",
    "0038_amos",
    "0039_han_seg",
    "0040_saros",
    "0041_ctrate",
    "0042_new_brainct_1mm",
    "0043_new_ct_tri",
]

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    login()
except Exception:
    pass

api = HfApi()
rev = api.repo_info(repo_id, repo_type="dataset").sha

os.makedirs(download_folder, exist_ok=True)

def download_one(name: str, max_tries: int = 6) -> None:
    delay = 4
    for attempt in range(1, max_tries + 1):
        try:
            print(f"==> Downloading {name}/{folder_type}  [try {attempt}/{max_tries}]")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=rev,
                allow_patterns=[f"{name}/{folder_type}/**"],
                local_dir=download_folder,
                max_workers=2,
                tqdm_class=None,
            )
            print(f"✓ Done: {name}/{folder_type}")
            return
        except requests.exceptions.HTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429 and attempt < max_tries:
                sleep_s = delay + random.uniform(0, 3)
                print(f"429 Too Many Requests → sleep {sleep_s:.1f}s then retry…")
                time.sleep(sleep_s)
                delay = min(delay * 2, 60)
                continue
            raise

if __name__ == "__main__":
    for name in dataset_names:
        download_one(name)
        time.sleep(60 + random.random() * 2)

    print("All done.")