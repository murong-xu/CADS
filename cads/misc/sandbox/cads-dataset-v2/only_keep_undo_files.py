"""
This script check the downloaded dataset to see if the number of image files and segmentation folders match.
"""
import os
import logging
import sys
import shutil

# TODO: input param
dir_dataset = '/mnt/hddb/murong/spine-r-processing'
# Configure logging
logging.basicConfig(level=logging.INFO)

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
    # "0033_tcia_nlst",
    "0034_empire",
    "0037_totalsegmentator",
    "0038_amos",
    "0039_han_seg",
    "0040_saros",
    "0041_ctrate",
    "0042_new_brainct_1mm",
    "0043_new_ct_tri",
]


def get_expected_seg_path(img_id, dataset_folder):
    """Get the expected segmentation file path based on image ID."""
    return os.path.join(dataset_folder, 'derivatives_spine_r', img_id, f'{img_id}_seg-vert_msk.nii.gz')

def check_missing_segmentations_and_cleanup(dataset):
    """Check missing segmentations and remove image directories that already have segmentations."""
    img_folder = os.path.join(dir_dataset, dataset, 'rawdata')
    seg_base_folder = os.path.join(dir_dataset, dataset, 'derivatives_spine_r')
    
    if not os.path.exists(img_folder):
        logging.error(f"Dataset {dataset}: Missing image folder at {img_folder}")
        return []
        
    missing_segs = []
    dirs_to_remove = []
    
    # Get all items in rawdata
    img_items = [f for f in os.listdir(img_folder)]
    
    for img_item in img_items:
        # Use the directory name as the image ID
        img_id = img_item  # No need for splitext since it's a directory name
        img_dir_path = os.path.join(img_folder, img_item)
        
        # Skip if not a directory
        if not os.path.isdir(img_dir_path):
            continue
        
        # Get expected segmentation file path
        expected_seg_path = get_expected_seg_path(img_id, os.path.join(dir_dataset, dataset))
        
        # Check if segmentation file exists
        if os.path.isfile(expected_seg_path):
            # If segmentation exists, add image directory to removal list
            dirs_to_remove.append(img_dir_path)
        else:
            missing_segs.append({
                'dataset': dataset,
                'image_id': img_id,
                'expected_seg_path': expected_seg_path
            })
    
    # Remove directories that have corresponding segmentations
    # for dir_path in dirs_to_remove:
    #     try:
    #         shutil.rmtree(dir_path)
    #         logging.info(f"Removed directory {dir_path} (segmentation exists)")
    #     except Exception as e:
    #         logging.error(f"Failed to remove directory {dir_path}: {str(e)}")
    
    # # Log summary for this dataset
    # if dirs_to_remove:
    #     logging.info(f"Dataset {dataset}: Removed {len(dirs_to_remove)} image directories that have segmentations")
    
    return missing_segs

# Process all datasets
all_missing_segs = []
for dataset in dataset_names:
    logging.info(f"\nProcessing dataset: {dataset}")
    missing = check_missing_segmentations_and_cleanup(dataset)
    
    if missing:
        logging.warning(f"Dataset {dataset}: Found {len(missing)} missing segmentations")
        all_missing_segs.extend(missing)
    else:
        logging.info(f"Dataset {dataset}: All segmentations present")

# Print final summary
logging.info(f"\nFinal Summary:")
logging.info(f"Total missing segmentations: {len(all_missing_segs)}")
for missing in all_missing_segs:
    logging.info(f"Missing: {missing['dataset']} - {missing['image_id']}")