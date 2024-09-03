import pandas as pd
import glob
import os

from table_plots.utils.utils import filter_rows, transitional_ids, bootstrap_ci, compare_models

# TODO: param
root_directory = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/test_set_scores'
output_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/figures_tables/excel/per-challenge'

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
    '0080_SegTHOR',
    '0081_ribseg',
    '0082_word',
    '0083_word_lits',
    '0084_BTCV_VNet',
    '0086_Private_CTPelvic1K',
    '0087_COVID19_CTSpine1K',
    '0088_MSD_CTSpine1K',
]
splits = ['test']
prefixes = ['dice']
filter_transitional_in_verse = True
significance_level = 0.05
stat_test_method = "wilcoxon_test_median"

experiment_to_name_dict = {
    'omaseg': 'OMASeg',
    'totalsegmentator': 'TotalSeg',
}
for prefix in prefixes:
    for dataset in datasets_eval:
        results_models = {}

        # Collect scores from models
        for experiment in experiment_to_name_dict.keys():
            challenge_results = {}
            aux = os.path.join(root_directory, experiment, dataset)
            if dataset != '0037_totalsegmentator':
                xlsx_file = glob.glob(aux+f'/{prefix}*.xlsx')[0]
                df = pd.read_excel(xlsx_file)
                if df.shape[0] <= 1:
                    continue
                if dataset == '0010_verse' and filter_transitional_in_verse:
                    df = df[~df["ids"].isin(transitional_ids)]
                df = filter_rows(df, splits=splits)

                for column in df.columns:
                    # Not drop NA here
                    challenge_results[column] = df[column].tolist()
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
                        # Not drop NA here
                        challenge_results[column] = df[column].tolist()
                    results_models[experiment] = challenge_results

        # Compre models
        combined_results_df = compare_models(results_models['omaseg'], results_models['totalsegmentator'],
                                             experiment_to_name_dict['omaseg'], experiment_to_name_dict['totalsegmentator'], stat_test_method=stat_test_method)

        # Highlight best scores
        filename = os.path.join(output_folder, f'{prefix}_{dataset}.xlsx')
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            combined_results_df.to_excel(
                writer, index=False, sheet_name='Summary')
            workbook = writer.book
            worksheet = writer.sheets['Summary']
            format_mean_std = workbook.add_format({'bg_color': '#bfd0e1'})
            format_median = workbook.add_format({'bg_color': '#f2db96'})
            format_totalsegmentator = workbook.add_format(
                {'bg_color': '#fc9272'})
            format_omaseg = workbook.add_format({'bg_color': '#a1d99b'})

            for row in range(combined_results_df.shape[0]):
                means = [float(combined_results_df.iloc[row][f'{model} mean±std'].split(
                    '±')[0]) for model in experiment_to_name_dict.values()]
                if all(x == means[0] for x in means):
                    continue
                if prefix in ['dice', 'normalized_distance']:
                    max_mean_col = [i for i, x in enumerate(
                        means) if x == max(means)]
                else:
                    max_mean_col = [i for i, x in enumerate(
                        means) if x == min(means)]
                rowindex = row + 1
                for item in max_mean_col:
                    colname = list(experiment_to_name_dict.values())[
                        item] + " mean±std"
                    colindex_mean = combined_results_df.columns.get_loc(
                        colname)
                    value_mean_std = combined_results_df[colname].iloc[row]
                    worksheet.write(rowindex, colindex_mean,
                                    value_mean_std, format_mean_std)

            for row in range(combined_results_df.shape[0]):
                medians = [float(combined_results_df.iloc[row][f'{model} median'])
                           for model in experiment_to_name_dict.values()]
                if all(x == medians[0] for x in medians):
                    continue
                if prefix in ['dice', 'normalized_distance']:
                    max_median_col = [i for i, x in enumerate(
                        medians) if x == max(medians)]
                else:
                    max_median_col = [i for i, x in enumerate(
                        medians) if x == min(medians)]
                rowindex = row + 1
                for item in max_median_col:
                    colname = list(experiment_to_name_dict.values())[
                        item] + " median"
                    colindex_median = combined_results_df.columns.get_loc(
                        colname)
                    value_median = combined_results_df[colname].iloc[row]
                    worksheet.write(rowindex, colindex_median,
                                    value_median, format_median)

            better_model_col = combined_results_df.columns.get_loc(
                'better_model')
            for row in range(combined_results_df.shape[0]):
                better_model = combined_results_df['better_model'].iloc[row]
                if better_model == 'TotalSeg':
                    worksheet.write(row + 1, better_model_col,
                                    better_model, format_totalsegmentator)
                elif better_model == 'OMASeg':
                    worksheet.write(row + 1, better_model_col,
                                    better_model, format_omaseg)
