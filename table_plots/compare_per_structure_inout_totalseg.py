import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict

from table_plots.utils.utils import filter_rows, align_and_filter_scores, list_specific_files, transitional_ids, bootstrap_ci, wilcoxon_test, wilcoxon_test_median, paired_t_test
from dataset_utils.bodyparts_labelmaps import labelmap_all_structure, labelmap_all_structure_renamed, totalseg_exclude_to_compare


# TODO: param
output_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/compare_totalseg_omaseg'
analysis_name = '1000_vs_roirobust'

experiment_results_path = {
    'omaseg': '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_predictions/final_models/scores_dependency/test_0',
    'totalsegmentator': '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_predictions/baselines/totalseg/metrics_roirobust_new/test_0',
}

prefixes = ['dice', 'hd95', 'hd', 'normalized_distance']
distributions = ['in', 'out', 'all']
splits = ['test']
significance_level = 0.05
stat_test_method = paired_t_test
filter_transitional_in_verse = True
exclude_face_from_overall_score = True

path_count_GT_number = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/per_structure_GT_image_counts.pkl'
path_count_GT_from_totalseg_number = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/per_structure_GT_image_from_totalseg_counts.pkl'
path_count_GT_volume = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/per_structure_GT_volume_clipped_avg_counts.pkl'

experiment_to_name_dict = {
    'omaseg': 'OMASeg',
    'totalsegmentator': 'TotalSeg',
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
    'train': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number['train'].items()},
    'test': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number['test'].items()}
}
count_GT_number_from_totalseg_renamed = {
    'train': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number_from_totalseg['train'].items()},
    'test': {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_number_from_totalseg['test'].items()}
}
count_GT_volume_renamed = {labelmap_all_structure_renamed[labelmap_all_structure_inv[key]]: value for key, value in count_GT_volume.items()}

table_names = list(labelmap_all_structure_renamed.values())
structuresids = list(labelmap_all_structure.values())

if len(table_names) != len(structuresids):
    print("Error, the number of names and structures ids is not the same")
    exit()

for prefix in prefixes:
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
            omaseg_scores = experiments_dicts['OMASeg'][distribution][structure]
            totalseg_scores = experiments_dicts['TotalSeg'][distribution][structure]

            original_length = len(omaseg_scores)
            aligned_omaseg, aligned_totalseg = align_and_filter_scores(
                omaseg_scores, totalseg_scores)
            new_length = len(aligned_omaseg)

            removed_points = original_length - new_length
            removed_data_points[distribution][structure] += removed_points

            valid_test_data_points[distribution][structure] = new_length

            experiments_dicts['OMASeg'][distribution][structure] = aligned_omaseg
            experiments_dicts['TotalSeg'][distribution][structure] = aligned_totalseg

    category_means = {} 
    for distribution in distributions:
        omaseg_scores = experiments_dicts['OMASeg'][distribution]
        totalseg_scores = experiments_dicts['TotalSeg'][distribution]
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
                        f"  OMASeg: {len(omaseg_structure_scores)}, TotalSeg: {len(totalseg_structure_scores)}")

                stat, p, pos_diff, neg_diff, better_model = stat_test_method(
                    omaseg_structure_scores, totalseg_structure_scores, p_value=significance_level)

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
            'totalseg_v1': list(labelmap_all_structure_renamed.values())[:104],
            'overlapping': list(set(list(labelmap_all_structure_renamed.values())) - set(totalseg_exclude_to_compare)),
            'all': list(labelmap_all_structure_renamed.values()),
        }

        # Exclude 'Face' from categories if exclude_face_from_overall_score is True
        if exclude_face_from_overall_score:
            for key in categories.keys():
                categories[key] = [structure for structure in categories[key] if structure != 'Face']

        category_stats = {cat: {'score': [], '95% CI': [], 'IQR': []} for cat in categories.keys()}
        category_means[distribution] = {}
        for experiment, data in experiments_dicts.items():
            stats = {'mean±std': [], '95% CI': [], 'IQR': []}
            category_means[distribution][experiment] = {}
            for structure, values in data[distribution].items():
                if experiment == 'TotalSeg' and structure in totalseg_exclude_to_compare:
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
                        stats['mean±std'].append(f"{mean:.2f}±{std:.2f}")
                        lower, upper = bootstrap_ci(values)
                        stats['95% CI'].append(f"({lower:.2f}, {upper:.2f})")
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
                    'overall 95% CI'] = f"({category_means[distribution][experiment][cat]['overall 95% CI'][0]:.3f}, {category_means[distribution][experiment][cat]['overall 95% CI'][1]:.3f})"

    output_df = pd.DataFrame(d)
    columns = ['Organ']
    for distribution in distributions:
        for experiment in experiment_to_name_dict.values():
            columns.extend([
                f'{experiment} {distribution} mean±std',
                f'{experiment} {distribution} 95% CI'
                f'{experiment} {distribution} IQR'
            ])
        columns.extend([
            f'{distribution} Statistic',
            f'{distribution} p-value',
            f'{distribution} Positive Differences',
            f'{distribution} Negative Differences',
            f'{distribution} Better Model'
        ])
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
    output_df['Proportion Train Totalseg'] = output_df['Proportion Train Totalseg'].fillna(
        0)
    output_df['Proportion Test Totalseg'] = output_df['Proportion Test Totalseg'].fillna(
        0)
    output_df['Avg Volume'] = output_df['Avg Volume'].fillna(0)

    columns = ['Organ', 'Num Train', 'Proportion Train Totalseg', 'Num Test', 'Num Valid Test', 'Proportion Test Totalseg', 'Avg Volume'] + \
        [col for col in output_df.columns if col not in ['Organ', 'Num Train',
                                                         'Proportion Train Totalseg', 'Num Test', 'Num Valid Test', 'Proportion Test Totalseg', 'Avg Volume']]
    output_df = output_df[columns]

    output_compare_folder = os.path.join(output_folder, analysis_name)
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

                                if abs(means[0] - means[1]) > 0.02:
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
                if better_model == 'TotalSeg':
                    worksheet.write(row + 1, better_model_col, better_model, format_totalsegmentator)
                elif better_model == 'OMASeg':
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