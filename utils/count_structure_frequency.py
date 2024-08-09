import numpy as np
import os
import pickle
from dataset_utils.bodyparts_labelmaps import labelmap_all_structure, map_taskid_to_labelmaps

def main():
    path_merged_label_summary = '/mnt/hdda/murong/22k/22k_label_summary.npz'  # TODO: only works for the best 22k summary file (in a specific format)

    count_structures = {}
    data = np.load(path_merged_label_summary, allow_pickle=True)
    data = data['info']

    # only scans for training or without original labels
    traindata = data[data[:, 2] == 1, :]  # train
    valdata = data[data[:, 2] == 2, :]  # val
    psudodata = data[data[:, 2] == 3, :]  # no labels
    psudodata = np.concatenate([traindata, valdata, psudodata], axis=0)
    
    for ind, name in labelmap_all_structure.items():
        psudodata_converted = psudodata[psudodata[:, ind+11] != None, :]
        psudodata_filtered = np.where(psudodata_converted == None, False, psudodata_converted).astype(bool)
        counts = np.sum(psudodata_filtered[:, ind+11])
        count_structures[name] = counts

    count_structures_all = {}
    for part in range(551, 560):
        count_structures_all[part] = {}
        structures_in_group = map_taskid_to_labelmaps[part]
        for idx, structure in structures_in_group.items():
            if idx != 0:
                count_structures_all[part][structure] = int(count_structures[structure])
        max_voxel_count = max(count_structures_all[part].values())
        for idx, structure in structures_in_group.items():
            if idx != 0:
                count_structures_all[part][structure] = count_structures_all[part][structure]/max_voxel_count
    print('a')

if __name__ == "__main__":
    main()