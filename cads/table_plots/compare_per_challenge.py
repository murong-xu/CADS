import pandas as pd
import glob
import os

from cads.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, labelmap_all_structure_renamed
from cads.table_plots.utils.utils import filter_rows, transitional_ids, amos_uterus_ids, compare_models_stat_test, ambigious_gt_structures_to_skip

# TODO: param
output_folder = '/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005'

# analysis_name = 'scores_final'
analysis_name = 'filtered_unreliable_and_limited_fov'
# analysis_name = 'filtered_unreliable'
# analysis_name = 'original_GT_but_remove_limited_fov'
# analysis_name = 'single_source_baseline'

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
if analysis_name == 'single_source_baseline':
    experiment_results_path = {
        'omaseg': '/mnt/hdda/murong/22k/ct_predictions/final_models/scores_labelata_confirmed_reliable_GT_notdo_FOV/test_0',
        'totalsegmentator': '/mnt/hdda/murong/22k/ct_predictions/baselines/single-source_scores/test_0',
    }    

experiment_to_name_dict = {
    'omaseg': 'OMASeg',
    'totalsegmentator': 'TotalSeg',
}

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
splits = ['test']
prefixes = ['dice', 'hd95', 'hd', 'normalized_distance']
filter_transitional_in_verse = True
significance_level = 0.05  #TODO: test more values
do_benjamini_hochberg = False
skip_ambigious_gt_eval = True

for prefix in prefixes:
    if prefix in ['dice', 'normalized_distance']:
        higher_better = True
    else:
        higher_better = False

    original_to_index = {name: idx for idx, name in labelmap_all_structure.items()}    
    for dataset in datasets_eval:
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
                if skip_ambigious_gt_eval:
                    for key, value in ambigious_gt_structures_to_skip.items():
                        if key in dataset:
                            if value in df.columns:
                                df = df.drop(value, axis=1)
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
        combined_results_df, _, _ = compare_models_stat_test(
            results_models['omaseg'], results_models['totalsegmentator'], alpha=significance_level, 
            higher_better=higher_better, do_benjamini_hochberg=do_benjamini_hochberg)

        # Highlight best scores
        output_compare_folder = os.path.join(output_folder, 'per-challenge', analysis_name)
        if not os.path.exists(output_compare_folder):
            os.makedirs(output_compare_folder)
        filename = os.path.join(output_compare_folder, f'{prefix}_{dataset}.xlsx')
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            combined_results_df.to_excel(
                writer, index=False, sheet_name='Summary')
            workbook = writer.book
            worksheet = writer.sheets['Summary']
            format_mean_std = workbook.add_format({'bg_color': '#f2db96'})
            format_median = workbook.add_format({'bg_color': '#bfd0e1'})
            format_totalsegmentator = workbook.add_format(
                {'bg_color': '#fc9272'})
            format_omaseg = workbook.add_format({'bg_color': '#a1d99b'})

            for row in range(combined_results_df.shape[0]):
                mean_stds = [combined_results_df.iloc[row][f'{model} mean±std'] for model in experiment_to_name_dict.values()]
                
                # if there is None value (model doesn't have this target) -> highlight other model's results
                if any(x is None for x in mean_stds):
                    valid_indices = [i for i, x in enumerate(mean_stds) if x is not None]
                    rowindex = row + 1
                    
                    for item in valid_indices:
                        colname = list(experiment_to_name_dict.values())[item] + " mean±std"
                        colindex_mean = combined_results_df.columns.get_loc(colname)
                        value_mean_std = combined_results_df[colname].iloc[row]
                        worksheet.write(rowindex, colindex_mean, value_mean_std, format_mean_std)
                    continue
                
                # both models' results available
                means = [float(x.split('±')[0]) for x in mean_stds]
                if all(x == means[0] for x in means):
                    continue
                if prefix in ['dice', 'normalized_distance']:
                    max_mean_col = [i for i, x in enumerate(means) if x == max(means)]
                else:
                    max_mean_col = [i for i, x in enumerate(means) if x == min(means)]
                rowindex = row + 1
                for item in max_mean_col:
                    colname = list(experiment_to_name_dict.values())[item] + " mean±std"
                    colindex_mean = combined_results_df.columns.get_loc(colname)
                    value_mean_std = combined_results_df[colname].iloc[row]
                    worksheet.write(rowindex, colindex_mean, value_mean_std, format_mean_std)

            for row in range(combined_results_df.shape[0]):
                medians = [combined_results_df.iloc[row][f'{model} median'] 
                          for model in experiment_to_name_dict.values()]
                
                if any(x is None for x in medians):
                    valid_indices = [i for i, x in enumerate(medians) if x is not None]
                    rowindex = row + 1
                    
                    for item in valid_indices:
                        colname = list(experiment_to_name_dict.values())[item] + " median"
                        colindex_median = combined_results_df.columns.get_loc(colname)
                        value_median = combined_results_df[colname].iloc[row]
                        worksheet.write(rowindex, colindex_median, value_median, format_median)
                    continue
                
                medians = [float(x) for x in medians]
                if all(x == medians[0] for x in medians):
                    continue
                if prefix in ['dice', 'normalized_distance']:
                    max_median_col = [i for i, x in enumerate(medians) if x == max(medians)]
                else:
                    max_median_col = [i for i, x in enumerate(medians) if x == min(medians)]
                rowindex = row + 1
                for item in max_median_col:
                    colname = list(experiment_to_name_dict.values())[item] + " median"
                    colindex_median = combined_results_df.columns.get_loc(colname)
                    value_median = combined_results_df[colname].iloc[row]
                    worksheet.write(rowindex, colindex_median,
                                 value_median, format_median)
                    
            better_model_col = combined_results_df.columns.get_loc('Better Model')
            for row in range(combined_results_df.shape[0]):
                better_model = combined_results_df['Better Model'].iloc[row]
                is_significant = combined_results_df['Significant After Correction'].iloc[row]
                if do_benjamini_hochberg:
                    if is_significant: # only highlight if the difference is significant after correction
                        if better_model == 'TotalSeg':
                            worksheet.write(row + 1, better_model_col,
                                            better_model, format_totalsegmentator)
                        elif better_model == 'OMASeg':
                            worksheet.write(row + 1, better_model_col,
                                            better_model, format_omaseg)
                else:
                    if better_model == 'TotalSeg':
                            worksheet.write(row + 1, better_model_col, better_model, format_totalsegmentator)
                    elif better_model == 'OMASeg':
                        worksheet.write(row + 1, better_model_col, better_model, format_omaseg)
