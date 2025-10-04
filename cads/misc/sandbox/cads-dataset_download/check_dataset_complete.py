"""
This script check the downloaded dataset to see if the number of image files and segmentation folders match.
"""
import os
import logging
import sys

# TODO: input param
dir_dataset = '/mnt/hddb/murong/CADS-dataset'
dir_check_number = '/home/murong/22k/OMASeg_sync/OMASeg/check_conversion_number'

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
os.makedirs(dir_check_number, exist_ok=True)

for dataset in dataset_names:
    print(f'\nProcessing Dataset: {dataset}')
    logfile = os.path.join(dir_check_number, f'{dataset}.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(logfile, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    img_folder = os.path.join(dir_dataset, dataset, 'images')
    seg_folder = os.path.join(dir_dataset, dataset, 'segmentations')
    
    if not os.path.exists(img_folder) or not os.path.exists(seg_folder):
        logging.error(f"Dataset {dataset}: Missing folder - images: {os.path.exists(img_folder)}, segmentations: {os.path.exists(seg_folder)}")
        continue
    
    img_files = set()
    if os.path.exists(img_folder):
        if dataset not in ['0041_ctrate', '0042_new_brainct_1mm']:
            img_files = {f.split('_0000.nii.gz')[0] for f in os.listdir(img_folder) 
                        if f.endswith('_0000.nii.gz')}
        else:
            img_files = {f.split('.nii.gz')[0] for f in os.listdir(img_folder) 
                        if f.endswith('.nii.gz')}
    
    seg_files = set()
    if os.path.exists(seg_folder):
        seg_files = {d for d in os.listdir(seg_folder) 
                    if os.path.isdir(os.path.join(seg_folder, d))}
    
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Total image files: {len(img_files)}")
    logging.info(f"Total segmentation folders: {len(seg_files)}")
    
    imgs_without_segs = img_files - seg_files
    segs_without_imgs = seg_files - img_files
    
    if imgs_without_segs:
        logging.warning(f"Images without segmentations ({len(imgs_without_segs)}):")
        for img_id in sorted(imgs_without_segs):
            logging.warning(f"  - {img_id}")
    
    if segs_without_imgs:
        logging.warning(f"Segmentations without images ({len(segs_without_imgs)}):")
        for seg_id in sorted(segs_without_imgs):
            logging.warning(f"  - {seg_id}")
    
    for seg_id in sorted(seg_files):
        seg_dir = os.path.join(seg_folder, seg_id)
        missing_parts = []
        
        for part in range(551, 560):
            seg_file = os.path.join(seg_dir, f"{seg_id}_part_{part}.nii.gz")
            if not os.path.exists(seg_file):
                missing_parts.append(part)
        
        if missing_parts:
            logging.warning(f"Segmentation {seg_id} missing parts: {missing_parts}")
    
    logging.info("-" * 80)

logging.info("All datasets processed!")