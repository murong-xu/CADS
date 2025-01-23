import glob
import os
import pickle
import numpy as np
import nibabel as nib
import argparse
import pandas as pd
import matplotlib.pyplot as plt 

from dataset_utils.bodyparts_labelmaps import labelmap_all_structure, map_taskid_to_labelmaps
from dataset_utils.mappings import map_labels

# import debugpy
# debugpy.listen(('0.0.0.0', 4444))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('You can debug your script now')

def main():
    outputfolder = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/manuscript/best22k_labelsummary"
    summary_files = glob.glob(outputfolder + '/*.pickle')

    summary_all = {key: [] for key in labelmap_all_structure.values()}
    for summary_file in summary_files:
        with open(summary_file, 'rb') as f:
            data = pickle.load(f)
            f.close()

        for class_idx, items in data.items():
            summary_all[labelmap_all_structure[class_idx]].extend(items)
        
        # num_structures = len(labelmap_all_structure)
        # cols = 4  # Number of columns for subplots
        # rows = (num_structures + cols - 1) // cols
        # plt.figure(figsize=(20, 5 * rows))

        # for idx, (class_idx, items) in enumerate(summary_all.items()):
        #     plt.subplot(rows, cols, idx + 1)
        #     plt.hist(items, bins=30, alpha=0.7, color='blue')
        #     plt.title(class_idx)  # Set title for each subplot
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(outputfolder, 'histogram', 'all_structures_histograms.png'))
        # plt.close() 
    
    for class_idx, items in summary_all.items():
        count = len(items)
        mean = np.mean(items)
        std = np.std(items)
        median = np.median(items)
        max_ = np.max(items)
        min_ = np.min(items)
        Q1 = np.percentile(items, 25)
        Q3 = np.percentile(items, 75)
        IQR = Q3 - Q1
        new_list = [count, median, IQR, mean, std, max_, min_, Q1, Q3]
        summary_all[class_idx] = new_list

    summary_df = pd.DataFrame.from_dict(summary_all, orient='index', columns=['Count', 'Median', 'IQR', 'Mean', 'Std', 'Max', 'Min', 'Q1', 'Q3'])
    summary_df.index.name = 'Structure' 
    summary_df.to_csv(os.path.join(outputfolder, 'overall', 'summary_all.csv')) 
    with open(os.path.join(outputfolder, 'overall', 'summary_all.pkl'), 'wb') as f:
        pickle.dump(summary_all, f)
        f.close()

    # list_0 = ["0001_visceral_gc", "0002_visceral_sc", "0003_kits21", "0004_lits", "0005_bcv_abdomen", "0013_ribfrac"]
    # list_1 = ["0006_bcv_cervix", "0007_chaos", "0008_ctorg"]
    # list_2 = ["0009_abdomenct1k"]
    # list_3 = ["0010_verse", "0011_exact", "0012_cad_pe"]
    # list_4 = ["0014_learn2reg", "0015_lndb", "0016_lidc"]
    # list_5 = ["0017_lola11", "0018_sliver07", "0019_tcia_ct_lymph_nodes", "0020_tcia_cptac_ccrcc", "0021_tcia_cptac_luad", "0022_tcia_ct_images_covid19"]
    # list_6 = ["0023_tcia_nsclc_radiomics", "0024_pancreas_ct", "0025_pancreatic_ct_cbct_seg", "0026_rider_lung_ct", "0027_tcia_tcga_kich", "0028_tcia_tcga_kirc"]
    # list_7 = ["0029_tcia_tcga_kirp", "0030_tcia_tcga_lihc", "0034_empire", "0035_ct_tri"]
    # list_8 = ["0032_stoic2021"]
    # list_9 = ["0033_tcia_nlst"]
    # list_10 = ["0037_totalsegmentator"]
    # list_11 = ["0038_amos", "0039_han_seg", "0040_saros"]
    # list_12 = ["0041_3k"]
    
    # predfolder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/best_22k/best_22k_more_stat_tests'  # TODO:
    # outputfolder = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/manuscript"  # TODO:

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-nr", '--list_nr', type=int, required=True)
    # args = parser.parse_args()
    # list_nr = args.list_nr
    # if list_nr == 0:
    #     list_datasetname = list_0
    # elif list_nr == 1:
    #     list_datasetname = list_1
    # elif list_nr == 2:
    #     list_datasetname = list_2
    # elif list_nr == 3:
    #     list_datasetname = list_3
    # elif list_nr == 4:
    #     list_datasetname = list_4
    # elif list_nr == 5:
    #     list_datasetname = list_5
    # elif list_nr == 6:
    #     list_datasetname = list_6
    # elif list_nr == 7:
    #     list_datasetname = list_7
    # elif list_nr == 8:
    #     list_datasetname = list_8
    # elif list_nr == 9:
    #     list_datasetname = list_9
    # elif list_nr == 10:
    #     list_datasetname = list_10
    # elif list_nr == 11:
    #     list_datasetname = list_11
    # elif list_nr == 12:
    #     list_datasetname = list_12
    # else:
    #     print("wrong nr")

    # parts= np.arange(551,560)
    # for dataset in list_datasetname:
    #     print('dataset: ', dataset)
    #     summary = {key: [] for key in labelmap_all_structure.keys()}
        
    #     seg_folder = os.path.join(predfolder, dataset)
    #     seg_files = glob.glob(seg_folder + '/*')
    #     output_file = os.path.join(outputfolder, 'best22k_labelsummary', dataset+'.pickle')

    #     for part in parts:
    #         print('part: ', part)
    #         part_labelmap = map_taskid_to_labelmaps[part]
    #         label_mapping = map_labels(part_labelmap, labelmap_all_structure, check_identical=True)

    #         for i, seg_file in enumerate(seg_files):
    #             print('{}/{}'.format(i+1, len(seg_files)), flush=True)
    #             seg_id = seg_file.split('/')[-1]
    #             seg_data = os.path.join(predfolder, dataset, seg_id, seg_id+'_part_{}.nii.gz'.format(part-300))
    #             seg = nib.load(seg_data).get_fdata()
    #             unique, counts = np.unique(seg[seg > 0], return_counts=True)
    #             unique = unique.tolist()
    #             counts = counts.tolist() 
    #             for j, class_id in enumerate(unique):
    #                 summary[label_mapping[class_id]].append(counts[j])

    #     with open(output_file, 'wb') as f:
    #         pickle.dump(summary, f)
    #         f.close()

if __name__ == "__main__":
    main()

