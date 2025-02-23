import pandas as pd
import glob
import os

from omaseg.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, labelmap_all_structure_renamed
from omaseg.table_plots.utils.utils import filter_rows, transitional_ids, amos_uterus_ids, compare_models_stat_test
from omaseg.table_plots.plots.plot_functions import generate_boxplot_comparison

datasets_eval = [
    '0001_visceral_gc',
    "0001_visceral_gc_new",
    '0002_visceral_sc',
    '0003_kits21',
    '0004_lits',
    '0005_bcv_abdomen',
    '0006_bcv_cervix',
    '0007_chaos',
    '0008_ctorg',
    '0009_abdomenct1k',
    '0010_verse',
    '0014_learn2reg',
    '0018_sliver07',
    '0034_empire',
    '0037_totalsegmentator',
    '0038_amos',
    '0039_han_seg',
    '0039_han_seg_reg',
    '0040_saros',
    # Extra test sets
    # '0080_SegTHOR',
    # '0081_ribseg',
    # '0082_word',
    # '0083_word_lits',
    # '0084_BTCV_VNet',
    # '0086_Private_CTPelvic1K',
    # '0087_COVID19_CTSpine1K',
    # '0088_MSD_CTSpine1K',
]

def collect_scores(dataset, analysis_name, prefix):
    if analysis_name == 'filtered_unreliable_and_limited_fov':
        experiment_results_path = {
            'omaseg': '/mnt/hdda/murong/22k/ct_predictions/final_models/scores_labelata_confirmed_reliable_GT/test_0',
            'totalsegmentator': '/mnt/hdda/murong/22k/ct_predictions/baselines/totalseg/metrics_labelata_confirmed_reliable_GT/test_0',
        }
    if analysis_name == 'scores_final':
        experiment_results_path = {
            'omaseg': '/mnt/hdda/murong/22k/ct_predictions/final_models/scores_final/test_0',
            'totalsegmentator': '/mnt/hdda/murong/22k/ct_predictions/baselines/totalseg/metrics_roirobust_new/test_0',
        }
    if analysis_name == 'filtered_unreliable':
        experiment_results_path = {
            'omaseg': '/mnt/hdda/murong/22k/ct_predictions/final_models/scores_labelata_confirmed_reliable_GT_notdo_FOV/test_0',
            'totalsegmentator': '/mnt/hdda/murong/22k/ct_predictions/baselines/totalseg/metrics_labelata_confirmed_reliable_GT_notdo_FOV/test_0',
        }
    if analysis_name == 'original_GT_but_remove_limited_fov':
        experiment_results_path = {
            'omaseg': '/mnt/hdda/murong/22k/ct_predictions/final_models/scores_remove_limited_fov/test_0',
            'totalsegmentator': '/mnt/hdda/murong/22k/ct_predictions/baselines/totalseg/metrics_remove_limited_fov/test_0',
        }

    experiment_to_name_dict = {
        'omaseg': 'OMASeg',
        'totalsegmentator': 'TotalSeg',
    }

    splits = ['test']
    filter_transitional_in_verse = True
    significance_level = 0.05  #TODO: test more values
    do_benjamini_hochberg = False

    if prefix in ['dice', 'normalized_distance']:
        higher_better = True
    else:
        higher_better = False
        
    original_to_index = {name: idx for idx, name in labelmap_all_structure.items()}
    results_models = {}
    # Collect scores from models
    for experiment in experiment_to_name_dict.keys():
        challenge_results = {}
        aux = os.path.join(experiment_results_path[experiment], dataset)
        if dataset != '0037_totalsegmentator':
            xlsx_file = glob.glob(aux+f'/{prefix}*.xlsx')[0]
            df = pd.read_excel(xlsx_file)
            if df.shape[0] <= 1:
                continue
            if dataset == '0010_verse' and filter_transitional_in_verse:
                df = df[~df["ids"].isin(transitional_ids)]
            df = filter_rows(df, splits=splits)

            for column in df.columns:
                if column in original_to_index:
                    index = original_to_index[column]
                    column_renamed = labelmap_all_structure_renamed[index]
                else:
                    words = column.split()
                    words[0] = words[0].capitalize()
                    column_renamed = ' '.join(words)
                if dataset == '0038_amos' and column == 'prostate':
                    df_tmp = pd.read_excel(xlsx_file)
                    df_tmp = df_tmp[~df_tmp["ids"].isin(amos_uterus_ids)]
                    df_tmp = filter_rows(df_tmp, splits=splits)
                    challenge_results[column_renamed] = df_tmp[column].to_list()
                # Not drop NA here
                else:
                    challenge_results[column_renamed] = df[column].tolist()
            results_models[experiment] = challenge_results
        else:
            xlsx_files = glob.glob(aux+f'/{prefix}*.xlsx')
            for part in [551, 552, 553, 554, 555]:
                xlsx_file = [f for f in xlsx_files if str(part) in f][0]
                df = pd.read_excel(xlsx_file)
                if df.shape[0] <= 1:
                    continue
                df = filter_rows(df, splits=splits)

                for column in df.columns:
                    if column in original_to_index:
                        index = original_to_index[column]
                        column_renamed = labelmap_all_structure_renamed[index]
                    else:
                        words = column.split()
                        words[0] = words[0].capitalize()
                        column_renamed = ' '.join(words)
                    # Not drop NA here
                    challenge_results[column_renamed] = df[column].tolist()
                results_models[experiment] = challenge_results

    # Compre models
    combined_results_df, aligned_omaseg, aligned_totalseg = compare_models_stat_test(
        results_models['omaseg'], results_models['totalsegmentator'], alpha=significance_level, 
        higher_better=higher_better, do_benjamini_hochberg=do_benjamini_hochberg)
    
    stat_results = {}
    all_organs = combined_results_df['Organ'].unique()
    for organ in all_organs:
        row = combined_results_df[combined_results_df['Organ'] == organ]
        if not row.empty and row['Better Model'].iloc[0]:
            stat_results[organ] = {
                'Better Model': row['Better Model'].iloc[0],
                'p': row['p-value'].iloc[0]
            }
    
    return aligned_omaseg, aligned_totalseg, stat_results


if __name__ == "__main__":
    # analysis_name = 'scores_final'
    analysis_name = 'filtered_unreliable_and_limited_fov'
    # analysis_name = 'filtered_unreliable'
    # analysis_name = 'original_GT_but_remove_limited_fov'
    prefix = 'dice'  # TODO:

    plot_metric_name = 'Dice'  # TODO:
    plot_output_path = "/mnt/hdda/murong/22k/plots/per-challenge/per-challenge_boxplot_compare_dice"  # TODO:

    for dataset in datasets_eval:
        aligned_omaseg, aligned_totalseg, stat_results = collect_scores(dataset, analysis_name, prefix)

        generate_boxplot_comparison(
            model1_scores=aligned_totalseg,
            model2_scores=aligned_omaseg,
            model1_name='TotalSeg',
            model2_name='OMASeg',
            stat_results=stat_results,
            output_path=plot_output_path,
            metric_name=plot_metric_name,
            datasetname=dataset
        )