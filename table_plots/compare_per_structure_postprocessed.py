import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict

from table_plots.utils.utils import filter_rows, align_and_filter_scores, list_specific_files, transitional_ids, amos_uterus_ids, bootstrap_ci, wilcoxon_test, wilcoxon_test_median, paired_t_test
from dataset_utils.bodyparts_labelmaps import labelmap_all_structure, labelmap_all_structure_renamed


# TODO: param
output_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/compare_omaseg'
analysis_name = 'raw_vs_postprocessed_refined'
stat_name = 'paired_t_test'

experiment_results_path = {
    'raw': '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_predictions/final_models/scores_dependency/test_0',
    'postprocess': '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/debug/postprocess_score_refined/test_0',
}

prefixes = ['dice', 'hd95', 'normalized_distance']
distributions = ['all']
splits = ['test']
significance_level = 0.05
stat_test_method = paired_t_test
filter_transitional_in_verse = True
exclude_face_from_overall_score = True

path_count_GT_number = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/per_structure_GT_image_counts.pkl'
path_count_GT_from_totalseg_number = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/per_structure_GT_image_from_totalseg_counts.pkl'
path_count_GT_volume = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/per_structure_GT_volume_clipped_avg_counts.pkl'

experiment_to_name_dict = {
    'raw': 'raw',
    'postprocess': 'postprocess',
}

totalseg_exclude_to_compare = []

highlight_diff_threshold = {
    'dice': 0.02,
    'hd95': 0.1,
    'hd': 0.1, 
    'normalized_distance':  0.02,
}

with open(path_count_GT_number, 'rb') as f:
    count_GT_number = pickle.load(f)
    f.close()

with open(path_count_GT_from_totalseg_number, 'rb') as f:
    count_GT_number_from_totalseg = pickle.load(f)
    f.close()

with open(path_count_GT_volume, 'rb') as f:
    count_GT_volume = pickle.load(f)
    f.close()

labelmap_all_structure_inv = {v: k for k, v in labelmap_all_structure.items()}
count_GT_number_renamed = {
    'train': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number['train'].items() if key in labelmap_all_structure_inv},
    'test': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number['test'].items() if key in labelmap_all_structure_inv}
}
count_GT_number_from_totalseg_renamed = {
    'train': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number_from_totalseg['train'].items() if key in labelmap_all_structure_inv},
    'test': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number_from_totalseg['test'].items() if key in labelmap_all_structure_inv}
}
count_GT_volume_renamed = {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_volume.items() if key in labelmap_all_structure_inv}

table_names = list(labelmap_all_structure_renamed.values())
structuresids = list(labelmap_all_structure.values())

if len(table_names) != len(structuresids):
    print("Error, the number of names and structures ids is not the same")
    exit()

for prefix in prefixes:
    if prefix in ['dice', 'normalized_distance']:
        higher_better = True
    else:
        higher_better = False
    experiments_dicts = {}
    for experiment in experiment_to_name_dict.keys():
        structure_values = {distribution: {table_name: []
                                           for table_name in table_names} for distribution in distributions}
        excelfiles = list_specific_files(experiment_results_path[experiment], prefix=prefix, suffix='.xlsx')
        for file in excelfiles:
            base = os.path.basename(file)
            df = pd.read_excel(file)
            if df.shape[0] <= 1:
                continue
            if '0010_verse' in base and filter_transitional_in_verse:
                df = df[~df["ids"].isin(transitional_ids)]
            df = filter_rows(df, splits=splits)
            column_names = df.columns

            for j, s in enumerate(structuresids):
                matching_columns = [
                    column for column in column_names if s == column]
                if len(matching_columns) > 0:
                    # Decide the current structure belongs to totalsegmentator or not
                    structure_is_in_dist = True  # Totalseg data
                    datasetname = file.split('/')[-2]

                    if '0037_totalsegmentator' != datasetname:
                        structure_is_in_dist = False

                    for column in matching_columns:
                        values = df[column].to_list()  # Drop NA later
                        if datasetname == '0038_amos' and column == 'prostate':
                            df_tmp = pd.read_excel(file)
                            df_tmp = df_tmp[~df_tmp["ids"].isin(amos_uterus_ids)]
                            df_tmp = filter_rows(df_tmp, splits=splits)
                            values = df_tmp[column].to_list()
                        if structure_is_in_dist and 'in' in structure_values:
                            structure_values['in'][table_names[j]].extend(
                                values)
                        elif not structure_is_in_dist and 'out' in structure_values:
                            structure_values['out'][table_names[j]].extend(
                                values)
                        if 'all' in structure_values:
                            structure_values['all'][table_names[j]].extend(
                                values)
        experiments_dicts[experiment_to_name_dict[experiment]
                          ] = structure_values

    d = {}
    d['Organ'] = table_names
    all_scores = []
    sample_size_mismatch = defaultdict(lambda: defaultdict(int))

    # Align scores
    removed_data_points = {distribution: {
        structure: 0 for structure in table_names} for distribution in distributions}
    valid_test_data_points = {distribution: {
        structure: 0 for structure in table_names} for distribution in distributions}
    for distribution in distributions:
        for structure in table_names:
            omaseg_scores = experiments_dicts['raw'][distribution][structure]
            totalseg_scores = experiments_dicts['postprocess'][distribution][structure]

            original_length = len(omaseg_scores)
            aligned_omaseg, aligned_totalseg = align_and_filter_scores(
                omaseg_scores, totalseg_scores)
            new_length = len(aligned_omaseg)

            removed_points = original_length - new_length
            removed_data_points[distribution][structure] += removed_points

            valid_test_data_points[distribution][structure] = new_length

            experiments_dicts['raw'][distribution][structure] = aligned_omaseg
            experiments_dicts['postprocess'][distribution][structure] = aligned_totalseg

    category_means = {} 
    for distribution in distributions:
        omaseg_scores = experiments_dicts['raw'][distribution]
        totalseg_scores = experiments_dicts['postprocess'][distribution]
        stat_results = []
        for structure in table_names:
            if structure in totalseg_exclude_to_compare:
                # skip stat. tests
                stat_results.append({
                'Organ': structure,
                'Statistic': None,
                'p-value': None,
                'Positive Differences': None,
                'Negative Differences': None,
                'Better Model': None
            })
                continue
            else:
                omaseg_structure_scores = omaseg_scores.get(structure, [])
                totalseg_structure_scores = totalseg_scores.get(structure, [])

                # Check if lengths are same
                if len(omaseg_structure_scores) != len(totalseg_structure_scores):
                    sample_size_mismatch[distribution][structure] += 1
                    print(
                        f"Sample size mismatch for {structure} in {distribution} distribution:")
                    print(
                        f"  raw: {len(omaseg_structure_scores)}, postprocess: {len(totalseg_structure_scores)}")

                stat, p, pos_diff, neg_diff, better_model = stat_test_method(
                    omaseg_structure_scores, totalseg_structure_scores, 'raw', 'postprocess', p_value=significance_level, higher_better=higher_better)

                stat_results.append({
                    'Organ': structure,
                    'Statistic': stat,
                    'p-value': p,
                    'Positive Differences': pos_diff,
                    'Negative Differences': neg_diff,
                    'Better Model': better_model
                })
        stat_df = pd.DataFrame(stat_results)
        d[f'{distribution} Statistic'] = stat_df['Statistic']
        d[f'{distribution} p-value'] = stat_df['p-value']
        d[f'{distribution} Positive Differences'] = stat_df['Positive Differences']
        d[f'{distribution} Negative Differences'] = stat_df['Negative Differences']
        d[f'{distribution} Better Model'] = stat_df['Better Model']

        categories = {
            'all': list(labelmap_all_structure_renamed.values()),
        }

        # Exclude 'Face' from categories if exclude_face_from_overall_score is True
        if exclude_face_from_overall_score:
            for key in categories.keys():
                categories[key] = [structure for structure in categories[key] if structure != 'Face']

        category_means[distribution] = {}
        for experiment, data in experiments_dicts.items():
            stats = {'mean±std': [], '95% CI': [], 'IQR': []}
            category_stats = {cat: {'score': [], '95% CI': [], 'IQR': []} for cat in categories.keys()}
            category_means[distribution][experiment] = {}
            for structure, values in data[distribution].items():
                if structure in totalseg_exclude_to_compare:
                    stats['mean±std'].append(None)
                    stats['95% CI'].append(None)
                    stats['IQR'].append(None)
                elif not values:
                    stats['mean±std'].append(None)
                    stats['95% CI'].append(None)
                    stats['IQR'].append(None)
                else:
                    mean, std = np.mean(values), np.std(values)
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    stats['IQR'].append(IQR)
                    if mean == 0 and std == 0:
                        # Changed from "0±0" to None
                        stats['mean±std'].append(None)
                        stats['95% CI'].append(None)
                    else:
                        stats['mean±std'].append(f"{mean:.4f}±{std:.4f}")
                        lower, upper = bootstrap_ci(values)
                        stats['95% CI'].append(f"({lower:.4f}, {upper:.4f})")
                    all_scores.extend(values)

                for cat, list_structures in categories.items():
                    if structure in list_structures:
                        category_stats[cat]['score'].append(stats['mean±std'][-1])
                        category_stats[cat]['95% CI'].append(stats['95% CI'][-1])
                        category_stats[cat]['IQR'].append(stats['IQR'][-1])

            d[f'{experiment} {distribution} mean'] = stats['mean±std']
            d[f'{experiment} {distribution} 95% CI'] = stats['95% CI']
            d[f'{experiment} {distribution} IQR'] = stats['IQR']

            for cat, metrics in category_stats.items():
                category_means[distribution][experiment][cat] = {
                    'overall score': np.mean([float(m.split('±')[0]) for m in metrics['score'] if m is not None]),
                    'overall 95% CI': (
                        np.mean([float(m.split('(')[1].split(',')[0])
                                for m in metrics['95% CI'] if m is not None]),
                        np.mean([float(m.split('(')[1].split(',')[1][:-1])
                                for m in metrics['95% CI'] if m is not None])
                    ),
                    'overall IQR': np.mean([i for i in metrics['IQR'] if i is not None]),
                }
                category_means[distribution][experiment][cat][
                    'overall 95% CI'] = f"({category_means[distribution][experiment][cat]['overall 95% CI'][0]:.4f}, {category_means[distribution][experiment][cat]['overall 95% CI'][1]:.4f})"

    output_df = pd.DataFrame(d)
    output_df['Num Train'] = output_df['Organ'].map(count_GT_number_renamed['train'])
    output_df['Num Test'] = output_df['Organ'].map(count_GT_number_renamed['test'])

    output_df['Num Valid Test'] = output_df['Organ'].map(valid_test_data_points['all'])
    output_df['Proportion Train Totalseg'] = output_df['Organ'].map(
        count_GT_number_from_totalseg_renamed['train'])
    output_df['Proportion Test Totalseg'] = output_df['Organ'].map(
        count_GT_number_from_totalseg_renamed['test'])
    output_df['Avg Volume'] = output_df['Organ'].map(count_GT_volume_renamed)

    output_df['Num Train'] = output_df['Num Train'].fillna(0)
    output_df['Num Test'] = output_df['Num Test'].fillna(0)
    output_df['Num Valid Test'] = output_df['Num Valid Test'].fillna(0)
    output_df['Proportion Train Totalseg'] = output_df['Proportion Train Totalseg'].fillna(0)
    output_df['Proportion Test Totalseg'] = output_df['Proportion Test Totalseg'].fillna(0)
    output_df['Avg Volume'] = output_df['Avg Volume'].fillna(0)

    columns_extra_info = ['Organ', 'Num Train', 'Proportion Train Totalseg', 'Num Test', 'Num Valid Test', 'Proportion Test Totalseg', 'Avg Volume']
    columns = columns_extra_info + [col for col in output_df.columns if col not in columns_extra_info]
    output_df = output_df[columns]

    output_compare_folder = os.path.join(output_folder, analysis_name, stat_name)
    if not os.path.exists(output_compare_folder):
        os.makedirs(output_compare_folder)
    filename = os.path.join(output_compare_folder, f'{prefix}_compare_table.xlsx')

    # Highlight best scores
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name='Summary')
        workbook = writer.book
        worksheet = writer.sheets['Summary']
        format_significant = workbook.add_format({'bg_color': '#2ca25f'})
        format_small_difference = workbook.add_format({'bg_color': '#99d8c9'})
        format_only_value = workbook.add_format({'bg_color': '#e5f5f9'})
        format_totalsegmentator = workbook.add_format({'bg_color': '#fc9272'})
        format_omaseg = workbook.add_format({'bg_color': '#a1d99b'})

        for distribution in distributions:
            for row in range(output_df.shape[0]):
                for result_type in ['mean', 'IQR']:
                    values = [output_df.iloc[row][f'{model} {distribution} {result_type}']
                            for model in experiment_to_name_dict.values()]
                    if all(pd.isna(x) for x in values):
                        continue
                    if result_type == 'mean':
                        means = []
                        for value in values:
                            if pd.isna(value):
                                means.append(None)
                            else:
                                means.append(float(value.split('±')[0]))
                        rowindex = row + 1

                        if None in means:
                            non_none_index = 0 if means[0] is not None else 1
                            colname = f"{list(experiment_to_name_dict.values())[non_none_index]} {distribution} {result_type}"
                            colindex = output_df.columns.get_loc(colname)
                            value = output_df[colname].iloc[row]
                            worksheet.write(rowindex, colindex,
                                            value, format_only_value)
                        else:
                            if prefix in ['dice', 'normalized_distance']:
                                better_index = 0 if means[0] > means[1] else (
                                    1 if means[1] > means[0] else None)
                            else:
                                better_index = 0 if means[0] < means[1] else (
                                    1 if means[1] < means[0] else None)

                            if better_index is not None:
                                colname = f"{list(experiment_to_name_dict.values())[better_index]} {distribution} {result_type}"
                                colindex = output_df.columns.get_loc(colname)
                                value = output_df[colname].iloc[row]

                                if abs(means[0] - means[1]) > highlight_diff_threshold[prefix]: 
                                    worksheet.write(rowindex, colindex,
                                                    value, format_significant)
                                else:
                                    worksheet.write(rowindex, colindex,
                                                    value, format_small_difference)
                    else:
                        iqrs = []
                        for value in values:
                            if pd.isna(value):
                                iqrs.append(None)
                            else:
                                iqrs.append(float(value))
                        rowindex = row + 1

                        if None in iqrs:
                            non_none_index = 0 if iqrs[0] is not None else 1
                            colname = f"{list(experiment_to_name_dict.values())[non_none_index]} {distribution} {result_type}"
                            colindex = output_df.columns.get_loc(colname)
                            value = output_df[colname].iloc[row]
                            worksheet.write(rowindex, colindex,
                                            value, format_only_value)
                        else:
                            better_index = 0 if iqrs[0] < iqrs[1] else (1 if iqrs[1] < iqrs[0] else None)

                            if better_index is not None:
                                colname = f"{list(experiment_to_name_dict.values())[better_index]} {distribution} {result_type}"
                                colindex = output_df.columns.get_loc(colname)
                                value = output_df[colname].iloc[row]

                                if abs(iqrs[0] - iqrs[1]) > 0.02:
                                    worksheet.write(rowindex, colindex,
                                                    value, format_significant)
                                else:
                                    worksheet.write(rowindex, colindex,
                                                    value, format_small_difference)
                            
            better_model_col = output_df.columns.get_loc(f'{distribution} Better Model')
            for row in range(output_df.shape[0]):
                better_model = output_df[f'{distribution} Better Model'].iloc[row]
                if better_model == 'postprocess':
                    worksheet.write(row + 1, better_model_col, better_model, format_totalsegmentator)
                elif better_model == 'raw':
                    worksheet.write(row + 1, better_model_col, better_model, format_omaseg)

    # Store means
    category_results = []
    for distribution, experiments in category_means.items():
        for experiment, categories in experiments.items():
            for cat, metrics in categories.items():
                category_results.append({
                    'Distribution': distribution,
                    'Experiment': experiment,
                    'Category': cat,
                    'Overall Score': metrics['overall score'],
                    'Overall 95% CI': metrics['overall 95% CI'],
                    'Overall IQR': metrics['overall IQR'],
                })
    category_df = pd.DataFrame(category_results)
    category_df.to_csv(os.path.join(output_compare_folder, f'{prefix}_means_summary.csv'), index=False)