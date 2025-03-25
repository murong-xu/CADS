import glob
import argparse
import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import pickle
from omaseg.dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps

def process_single_case(subfolder, part, labelmap):
    img_id = os.path.basename(subfolder)
    segfile = os.path.join(subfolder, f'{img_id}_part_{part}.nii.gz')
    
    n_classes = max(labelmap.keys())
    labels_info = [None] * n_classes
    
    if os.path.exists(segfile):
        seg = nib.load(segfile).get_fdata()
        unique = np.unique(seg)
        
        # Count non-background labels (skip 0)
        if len(unique) > 1:
            for u in unique[1:]:
                label_idx = int(u)
                labels_info[label_idx-1] = True
    
    return [img_id] + labels_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-part", '--part', type=int, required=True)
    args = parser.parse_args()

    part = args.part
    file_unique_folders = '/net/cephfs/shares/menze.dqbm.uzh/murong/CT-RATE_segmentations/seg_unique_train.pkl'  #TODO:
    with open(file_unique_folders, 'rb') as f:
        unique_folders = pickle.load(f)
        f.close()
    unique_folders.sort()

    outputfolder = "/net/cephfs/shares/menze.dqbm.uzh/murong/CT-RATE_segmentations/segmentation_statistics"  #TODO:
    output_filename = f'train_summary_{part}'  #TODO:

    labelmap = map_taskid_to_labelmaps[part]
    
    columns = ['patientid']
    names = [labelmap[i] for i in range(1, max(labelmap.keys()) + 1)]
    columns.extend(names)
    info = []
    
    total = len(unique_folders)
    count = 1
    for folder in tqdm(unique_folders):
        print(f'Processing {count}/{total}...')
        subfolder = sorted(glob.glob(folder + '/*'))[0]
        case_info = process_single_case(subfolder, part, labelmap)
        info.append(case_info)
        count += 1

    info = np.asarray(info)
    df = pd.DataFrame(info, columns=columns)
    
    df.to_excel(os.path.join(outputfolder, output_filename + '.xlsx'), index=False)
    np.savez(os.path.join(outputfolder, output_filename + '.npz'), info=info)

if __name__ == '__main__':
    main()