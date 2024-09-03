import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict

from table_plots.utils.utils import filter_rows, align_and_filter_scores, list_specific_files, transitional_ids, bootstrap_ci, wilcoxon_test, wilcoxon_test_median, paired_t_test
from dataset_utils.bodyparts_labelmaps import labelmap_all_structure

# TODO: param
root_directory = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/test_set_scores'
output_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/results/figures_tables'

prefixes = ['dice']
distributions = ['in']
splits = ['test']
significance_level = 0.05
stat_test_method = wilcoxon_test_median
filter_transitional_in_verse = True

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

table_names = list(labelmap_all_structure.values())
structuresids = list(labelmap_all_structure.values())

if len(table_names) != len(structuresids):
    print("Error, the number of names and structures ids is not the same")
    exit()

for prefix in prefixes:
    experiments_dicts = {}
    for experiment in experiment_to_name_dict.keys():
        structure_values = {distribution: {table_name: []
                                           for table_name in table_names} for distribution in distributions}
        excelfiles = list_specific_files(os.path.join(
            root_directory, experiment), prefix=prefix, suffix='.xlsx')
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

            experiments_dicts['OMASeg'][distribution][structure] = aligned_omaseg
            experiments_dicts['TotalSeg'][distribution][structure] = aligned_totalseg

    for distribution in distributions:
        omaseg_scores = experiments_dicts['OMASeg'][distribution]
        totalseg_scores = experiments_dicts['TotalSeg'][distribution]
        stat_results = []
        for structure in table_names:
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

        for experiment, data in experiments_dicts.items():
            stats = {'mean±std': [], '95% CI': []}
            for structure, values in data[distribution].items():
                if not values:
                    stats['mean±std'].append(None)
                    stats['95% CI'].append(None)
                else:
                    mean, std = np.mean(values), np.std(values)
                    if mean == 0 and std == 0:
                        # Changed from "0±0" to None
                        stats['mean±std'].append(None)
                        stats['95% CI'].append(None)
                    else:
                        stats['mean±std'].append(f"{mean:.2f}±{std:.2f}")
                        lower, upper = bootstrap_ci(values)
                        stats['95% CI'].append(f"({lower:.2f}, {upper:.2f})")
                    all_scores.extend(values)

            d[f'{experiment} {distribution}'] = stats['mean±std']
            d[f'{experiment} {distribution} 95% CI'] = stats['95% CI']

    output_df = pd.DataFrame(d)
    columns = ['Organ']
    for distribution in distributions:
        for experiment in experiment_to_name_dict.values():
            columns.extend([
                f'{experiment} {distribution} mean±std',
                f'{experiment} {distribution} 95% CI'
            ])
        columns.extend([
            f'{distribution} Statistic',
            f'{distribution} p-value',
            f'{distribution} Positive Differences',
            f'{distribution} Negative Differences',
            f'{distribution} Better Model'
        ])
    output_df['Num Train'] = output_df['Organ'].map(count_GT_number['train'])
    output_df['Num Test'] = output_df['Organ'].map(count_GT_number['test'])
    output_df['Proportion Train Totalseg'] = output_df['Organ'].map(
        count_GT_number_from_totalseg['train'])
    output_df['Proportion Test Totalseg'] = output_df['Organ'].map(
        count_GT_number_from_totalseg['test'])
    output_df['Avg Volume'] = output_df['Organ'].map(count_GT_volume)

    output_df['Num Train'] = output_df['Num Train'].fillna(0)
    output_df['Num Test'] = output_df['Num Test'].fillna(0)
    output_df['Proportion Train Totalseg'] = output_df['Proportion Train Totalseg'].fillna(
        0)
    output_df['Proportion Test Totalseg'] = output_df['Proportion Test Totalseg'].fillna(
        0)
    output_df['Avg Volume'] = output_df['Avg Volume'].fillna(0)

    columns = ['Organ', 'Num Train', 'Proportion Train Totalseg', 'Num Test', 'Proportion Test Totalseg', 'Avg Volume'] + \
        [col for col in output_df.columns if col not in ['Organ', 'Num Train',
                                                         'Proportion Train Totalseg', 'Num Test', 'Proportion Test Totalseg', 'Avg Volume']]
    output_df = output_df[columns]

    excel_folder = os.path.join(output_folder, 'excel')
    if not os.path.exists(excel_folder):
        os.makedirs(excel_folder)
    filename = os.path.join(excel_folder, f'{prefix}_table.xlsx')

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
                values = [output_df.iloc[row][f'{model} {distribution}']
                          for model in experiment_to_name_dict.values()]
                if all(pd.isna(x) for x in values):
                    continue
                means = []
                for value in values:
                    if pd.isna(value):
                        means.append(None)
                    else:
                        means.append(float(value.split('±')[0]))
                rowindex = row + 1

                if None in means:
                    non_none_index = 0 if means[0] is not None else 1
                    colname = f"{list(experiment_to_name_dict.values())[non_none_index]} {distribution}"
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
                        colname = f"{list(experiment_to_name_dict.values())[better_index]} {distribution}"
                        colindex = output_df.columns.get_loc(colname)
                        value = output_df[colname].iloc[row]

                        if abs(means[0] - means[1]) > 0.02:
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