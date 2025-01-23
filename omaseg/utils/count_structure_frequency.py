import numpy as np
import os
import pickle
from omaseg.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, map_taskid_to_labelmaps

def main():
    path_merged_label_summary = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/label_count_summary/merged/22k_label_summary.npz'  # TODO: 
    pseudolabel_path = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/final_nnUNets/nnUNet_raw/Dataset552_Totalseg252/labelsTr' # TODO: 
    output_path = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/oversampling' # TODO: 
    group = 552
    threshold = 0.16 # For filtering 552 C1-C5: 0.16
    ind_start = 18 # 552: 18
    ind_end = 41 # 552: 41
    minority_class_indices_in_group = [20, 21, 22, 23, 24] # 552: [20, 21, 22, 23, 24]

    # Part: calculate structure frequency in 22k
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

    # Part: determine oversampling probabilities for a target group
    pattern = 'nii.gz'
    pseudolabels = []
    for path, subdirs, files in os.walk(pseudolabel_path):
        for name in files:
            if name.endswith(pattern):
                pseudolabels.append(os.path.join(path, name))
    pseudolabels.sort()
    pseudolabels = [i.split('/')[-1].split('.nii.gz')[0] for i in pseudolabels]

    class_weights = {}
    for cls, prob in count_structures_all[group].items():
        if prob < threshold:
            class_weights[cls] = round(1 / prob) # Rare classes
        else:
            class_weights[cls] = 1 # Common classes

    sampling_probabilities = []
    for pseudolabel in pseudolabels:
        data_slice = psudodata[psudodata[:, 1] == pseudolabel][:, ind_start+11:ind_end+11+1].squeeze()
        if group == 552:
            # C1-C5
            if data_slice[-3]:
                prob = class_weights['vertebrae_C3']
            elif data_slice[-4]:
                prob = class_weights['vertebrae_C4']
            elif data_slice[-1]:
                prob = class_weights['vertebrae_C1']
            elif data_slice[-2]:
                prob = class_weights['vertebrae_C2']
            elif data_slice[-5]:
                prob = class_weights['vertebrae_C5']
            else:
                prob = 1
        sampling_probabilities.append(prob)
    total = sum(sampling_probabilities)
    normalized_sampling_probabilities = [x / total for x in sampling_probabilities]

    summary = {
        'probabilities': normalized_sampling_probabilities,
        'minority_class_indices': minority_class_indices_in_group,
    }
    with open(os.path.join(output_path, f'oversampling_{group}.pkl'), 'wb') as f:
        pickle.dump(summary, f)
        f.close()

if __name__ == "__main__":
    main()