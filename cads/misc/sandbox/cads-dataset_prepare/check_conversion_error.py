"""
This script checks for conversion errors in datasets by verifying shape mismatches.                                                  
"""

import os
import nibabel as nib
import pickle
import logging
import sys

from cads.dataset_utils.preprocessing import is_affine_diagonal

# TODO: input param
dir_dataset = '/mnt/hddb/murong/CADS-dataset'
dir_output_check_error = '/home/murong/22k/OMASeg_sync/OMASeg/check_conversion_error'
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

for dataset in dataset_names:
    print('Dataset: ', dataset)
    logfile = os.path.join(dir_output_check_error, f'{dataset}.log')
    
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

    file_pkl_shape_mismatch = os.path.join(dir_output_check_error, f'{dataset}_shape_mismatch.pkl')
    list_shape_mismatch = []
    file_pkl_nondiagnoal = os.path.join(dir_output_check_error, f'{dataset}_nondiagnoal.pkl')
    list_nondiagnoal = []

    if not os.path.exists(seg_folder):
        print(f"Error: {seg_folder} does not exist!")
        continue
    files = os.listdir(seg_folder)
    files.sort()
    if len(files) == 0:
        print(f"Error: {seg_folder} is empty!")
    else:
        print(f"OK: {seg_folder} has {len(files)} files.")

    for img_id in files:
        if dataset in ['0041_ctrate', '0042_new_brainct_1mm']:
            img_path = os.path.join(img_folder, f"{img_id}.nii.gz")
        else:
            img_path = os.path.join(img_folder, f"{img_id}_0000.nii.gz")
        seg_path = os.path.join(seg_folder, img_id, f"{img_id}_part_552.nii.gz")  # pick one part to check
        
        img_file = nib.load(img_path)
        seg_file = nib.load(seg_path)

        # check shape mismatch
        if img_file.shape != seg_file.shape:
            logging.info(f"Shape mismatch in {dataset} for {img_id}: img shape {img_file.shape}, seg shape {seg_file.shape}")
            list_shape_mismatch.append(img_id)
        
        # check if non-diagnoal affine
        if not is_affine_diagonal(img_file.affine):
            logging.info(f"Non-diagonal affine in {dataset} for {img_id}")
            list_nondiagnoal.append(img_id)

    # save lists to pkl
    with open(file_pkl_shape_mismatch, 'wb') as f:
        pickle.dump(list_shape_mismatch, f)
        f.close()

    with open(file_pkl_nondiagnoal, 'wb') as f:
        pickle.dump(list_nondiagnoal, f)
        f.close()
