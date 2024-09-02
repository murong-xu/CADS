import os
import pickle
import pandas as pd
import glob

from dataset_utils.bodyparts_labelmaps import labelmap_all_structure
from dataset_utils.mappings import replacements

input_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm'
output_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results'
splits = ['test', 'train']
path_count_volume = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/monai_models/avg_organ_volume/organ_volume.csv'

datasets = glob.glob(input_folder + '/*')
datasets = [d + '/dataset_summary.xlsx' for d in datasets]
datasets.append(
    '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/gt_data/257_han_seg/dataset_summary.xlsx')
datasets.sort()


def get_values_from_excel(file_path: str, start_column_index: int = 10, split: str = 'train') -> dict:
    df = pd.read_excel(file_path)
    if split == 'train':
        filtered_df = df[(df['split'] == 1) | (
            df['split'] == 2) | (df['split'] == 3)]
    elif split == 'test':
        filtered_df = df[(df['split'] == 0)]
    result = {}

    for col in filtered_df.columns[start_column_index-1:]:
        if col == 'organs_sum':
            break
        result[col] = filtered_df[col].sum()
    return result


def replace_labelmap(labelmap, replacements):
    updated_labelmap = {}
    for key, value in labelmap.items():
        if key in replacements:
            new_key = replacements[key]
        else:
            new_key = key
        updated_labelmap[new_key] = value

    return updated_labelmap


print('Number count now')
counts_number_splits = {}
counts_number_from_totalseg_splits = {}
counts_ratio_from_totalseg_splits = {}
for split in splits:
    count_number_all = {v: 0 for v in labelmap_all_structure.values()}
    count_number_all_from_totalseg = {
        v: 0 for v in labelmap_all_structure.values()}
    count_ratio_all_from_totalseg = {
        v: 0 for v in labelmap_all_structure.values()}
    for summary_file in datasets:
        count_group = {}
        counts = get_values_from_excel(summary_file, 10, split)
        counts = replace_labelmap(counts, replacements)

        for structure, count in counts.items():
            dataset = summary_file.split('/')[-2]
            if structure in count_number_all:
                count_number_all[structure] += count
                if dataset == '0037_totalsegmentator':
                    count_number_all_from_totalseg[structure] += count
            else:
                print(f"{dataset} '{structure}' not found")
    counts_number_splits[split] = count_number_all
    counts_number_from_totalseg_splits[split] = count_number_all_from_totalseg

    for structure, count in count_number_all_from_totalseg.items():
        if count != 0:
            count_ratio_all_from_totalseg[structure] = round(
                counts_number_from_totalseg_splits[split][structure] / counts_number_splits[split][structure], 2)
    counts_ratio_from_totalseg_splits[split] = count_ratio_all_from_totalseg

with open(os.path.join(output_folder, 'per_structure_GT_image_counts.pkl'), 'wb') as f:
    pickle.dump(counts_number_splits, f)
    f.close()

with open(os.path.join(output_folder, 'per_structure_GT_image_from_totalseg_counts.pkl'), 'wb') as f:
    pickle.dump(counts_ratio_from_totalseg_splits, f)
    f.close()


print('Volume count now')
df_organ_volume = pd.read_csv(path_count_volume)
key_column = 'organ'
value_column = 'avg_clipped'
organ_volume_dict = dict(
    zip(df_organ_volume[key_column], df_organ_volume[value_column]))
count_volume_all = {v: 0 for v in labelmap_all_structure.values()}
for structure, count in organ_volume_dict.items():
    if structure in count_volume_all:
        count_volume_all[structure] = count
    else:
        print(f"'{structure}' not found")

with open(os.path.join(output_folder, 'per_structure_GT_volume_clipped_avg_counts.pkl'), 'wb') as f:
    pickle.dump(count_volume_all, f)
    f.close()
