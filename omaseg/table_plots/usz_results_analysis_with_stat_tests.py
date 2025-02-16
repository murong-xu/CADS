import numpy as np
import pandas as pd
import os
from collections import defaultdict
import json

from omaseg.table_plots.utils.utils import align_and_filter_scores, bootstrap_ci, check_distribution_perform_stat_test, benjamini_hochberg_correction

jsonfile_totalseg = '/mnt/hdda/murong/22k/results/usz/STAT/updated_TS.json'
jsonfile_omaseg = '/mnt/hdda/murong/22k/results/usz/STAT/OMA.json'

output_folder = '/mnt/hdda/murong/22k/results/usz/analysis_p005'  # TODO
analysis_name = 'compare_omaseg_totalseg'  # TODO

prefixes = ['Dice', 'normalized_surface_dice', 'hausdorff_dist', 'hausdorff_dist_95', 'TPR', 'FPR', 'vol_error']
score_higher_is_better_metrics = ['Dice', 'normalized_surface_dice', 'TPR']
significance_level = 0.05
do_benjamini_hochberg = False

highlight_diff_threshold = {
    'Dice': 0.02,
    'hausdorff_dist_95': 0.1,
    'hausdorff_dist': 0.1, 
    'normalized_surface_dice':  0.02,
    'TPR': 0.02,
    'FPR': 0.02,
    'vol_error': 0.01
}
table_names = ['Brainstem', 'Eye_L', 'Eye_R', 'Larynx', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R', 
               'SubmandibularGland_L', 'SubmandibularGland_R', 'Aorta', 'Bladder', 'Brain', 'Esophagus',
               'Humerus_L', 'Humerus_R', 'Kidney_L', 'Kidney_R', 'Liver', 'Lung_L', 'Lung_R', 'Prostate',
               'SpinalCord', 'Spleen', 'Stomach', 'Thyroid', 'Trachea', 'V_CavaInferior', 'Heart',
               'Chiasm', 'Glottis', 'LacrimalGland_L', 'LacrimalGland_R', 'Mandible', 'OralCavity', 'Pituitary', 'Rectum', 'SeminalVesicle']
totalseg_exclude_to_compare = ['Chiasm', 'Glottis', 'LacrimalGland_L', 'LacrimalGland_R', 'Mandible', 'OralCavity', 'Pituitary', 'Rectum', 'SeminalVesicle']
experiment_to_name_dict = {
    'omaseg': 'OMASeg',
    'totalsegmentator': 'TotalSeg',
}
        
for prefix in prefixes:
    if prefix in score_higher_is_better_metrics:
        higher_better = True
    else:
        higher_better = False

    # load results
    file = open(jsonfile_totalseg)
    totalseg_scores = json.load(file)
    file.close()
    file = open(jsonfile_omaseg)
    omaseg_scores = json.load(file)
    file.close()

    experiment_results = {
        'OMASeg': omaseg_scores,
        'TotalSeg': totalseg_scores,
    }
    # Collect available test images used for metric calculation
    omaseg_image_paths = set(path.rsplit(':', 1)[0] for path in omaseg_scores.keys())
    totalseg_image_paths = set(path.rsplit(':', 1)[0] for path in totalseg_scores.keys())
    if len(omaseg_image_paths) != len(totalseg_image_paths):
        print("Error, the number of subjects in TotalSeg and OMASeg are not the same")
        exit()
    organ_names_in_results_dict = set(key.rsplit(':', 1)[1] for key in omaseg_scores.keys())
    table_names_set = set(table_names)
    if organ_names_in_results_dict != table_names_set:
        # Find differences
        missing_in_dict = table_names_set - organ_names_in_results_dict
        extra_in_dict = organ_names_in_results_dict - table_names_set
        
        if missing_in_dict:
            print(f"Organs in table_names but not in results dictionary: {sorted(missing_in_dict)}")
        if extra_in_dict:
            print(f"Organs in resuylts dictionary but not in table_names: {sorted(extra_in_dict)}")
        exit()

    image_paths = omaseg_image_paths

    # replace not-existing targets in TotalSeg scores from None to 0 (s.t. I can still keep OMASeg's scores)
    for key in totalseg_scores.keys():
        # Split the key to get the structure name
        structure_name = key.rsplit(':', 1)[1]
        
        # Check if this structure is in the exclude list
        if structure_name in totalseg_exclude_to_compare:
            # Replace None with 0 in all metrics for this structure
            for metric in totalseg_scores[key]:
                if totalseg_scores[key][metric] is None:
                    totalseg_scores[key][metric] = 0

    experiments_dicts = {
        'OMASeg': {},
        'TotalSeg': {}
        }
    for organ in table_names:
        experiments_dicts['OMASeg'][organ] = []
        experiments_dicts['TotalSeg'][organ] = []
        for image_path in image_paths:
            full_path = f"{image_path}:{organ}"

            # Get scores for both models if they exist
            omaseg_score = omaseg_scores.get(full_path, {}).get(prefix)
            totalseg_score = totalseg_scores.get(full_path, {}).get(prefix)

            omaseg_score = np.nan if omaseg_score is None else omaseg_score
            totalseg_score = np.nan if totalseg_score is None else totalseg_score
            
            experiments_dicts['OMASeg'][organ].append(omaseg_score)
            experiments_dicts['TotalSeg'][organ].append(totalseg_score)

    d = {}
    d['Organ'] = table_names
    all_scores = []
    sample_size_mismatch = defaultdict(lambda: defaultdict(int))

    # Align scores
    removed_data_points =  {structure: 0 for structure in table_names}
    valid_test_data_points = {structure: 0 for structure in table_names}
    for structure in table_names:
        omaseg_scores = experiments_dicts['OMASeg'][structure]
        totalseg_scores = experiments_dicts['TotalSeg'][structure]

        original_length = len(omaseg_scores)
        aligned_omaseg, aligned_totalseg = align_and_filter_scores(
            omaseg_scores, totalseg_scores) 
        new_length = len(aligned_omaseg)

        removed_points = original_length - new_length
        removed_data_points[structure] += removed_points

        valid_test_data_points[structure] = new_length

        experiments_dicts['OMASeg'][structure] = aligned_omaseg
        experiments_dicts['TotalSeg'][structure] = aligned_totalseg

    category_means = {} 
    omaseg_scores = experiments_dicts['OMASeg']
    totalseg_scores = experiments_dicts['TotalSeg']

    # Collect all p-values
    p_values = []
    p_value_to_structure = {}

    stat_results = {structure: {
        'Organ': structure,
        'Test Type': None,
        'Statistic': None,
        'p-value': None,
        'Effect Size': None,
        'Effect Size Type': None,
        'Is Normal': None,
        'Normality p-value': None,
        'Mean Difference': None,
        'Positive Differences': None,
        'Negative Differences': None,
        'Better Model': None,
        'Significant After Correction': None
    } for structure in table_names}

    for structure in table_names:
        if structure in totalseg_exclude_to_compare:
            # skip stat. tests
            continue
        omaseg_structure_scores = omaseg_scores.get(structure, [])
        totalseg_structure_scores = totalseg_scores.get(structure, [])

        # Check if lengths are same
        if len(omaseg_structure_scores) != len(totalseg_structure_scores):
            sample_size_mismatch[structure] += 1
            print(f"Sample size mismatch for {structure}")
            print(f"OMASeg: {len(omaseg_structure_scores)}, TotalSeg: {len(totalseg_structure_scores)}")
            continue

        result = check_distribution_perform_stat_test(
            omaseg_structure_scores, 
            totalseg_structure_scores, 
            alpha=significance_level,
            higher_better=higher_better
        )
        
        p_values.append(result['p_value'])
        p_value_to_structure[len(p_values)-1] = structure
        stat_results[structure].update({
            'Test Type': result['test_type'],
            'Statistic': result['statistic'],
            'p-value': result['p_value'],
            'Effect Size': result['effect_size'],
            'Effect Size Type': result['effect_size_type'],
            'Is Normal': result['is_normal'],
            'Normality p-value': result['normality_p'],
            'Mean Difference': result['mean_difference'],
            'Positive Differences': result['pos_diff'],
            'Negative Differences': result['neg_diff'],
            'Better Model': result['better_model']
        })

    # Apply Benjamini-Hochberg correction
    if p_values and do_benjamini_hochberg:
        rejections = benjamini_hochberg_correction(p_values, significance_level)
        
        for idx, is_significant in enumerate(rejections):
            structure = p_value_to_structure[idx]
            stat_results[structure]['Significant After Correction'] = is_significant
            if not is_significant:
                stat_results[structure]['Better Model'] = None

    stat_df = pd.DataFrame(list(stat_results.values()))
    d[f'Statistic'] = stat_df['Statistic']
    d[f'p-value'] = stat_df['p-value']
    d[f'Positive Differences'] = stat_df['Positive Differences']
    d[f'Negative Differences'] = stat_df['Negative Differences']
    d[f'Better Model'] = stat_df['Better Model']
    d[f'Test Type'] = stat_df['Test Type']
    d[f'Effect Size'] = stat_df['Effect Size']
    d[f'Significant After Correction'] = stat_df['Significant After Correction']

    for experiment, data in experiments_dicts.items():
        stats = {'mean±std': [], 'median': [], '95% CI': [], 'IQR': []}
        category_stats = {'score': [], '95% CI': [], 'IQR': []}
        category_means[experiment] = {}
        for structure, values in data.items():
            if experiment == 'TotalSeg' and structure in totalseg_exclude_to_compare:
                stats['mean±std'].append(None)
                stats['median'].append(None)
                stats['95% CI'].append(None)
                stats['IQR'].append(None)
            elif not values:
                stats['mean±std'].append(None)
                stats['median'].append(None)
                stats['95% CI'].append(None)
                stats['IQR'].append(None)
            else:
                mean, std = np.mean(values), np.std(values)
                median = np.median(values)
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                stats['IQR'].append(IQR)
                if mean == 0 and std == 0:
                    # Changed from "0±0" to None
                    stats['mean±std'].append(None)
                    stats['median'].append(None)
                    stats['95% CI'].append(None)
                else:
                    if prefix == 'FPR':
                        # need more digits
                        stats['mean±std'].append(f"{mean:.8f}±{std:.8f}")
                        stats['median'].append(f"{median:.8f}")
                        lower, upper = bootstrap_ci(values)
                        stats['95% CI'].append(f"({lower:.8f}, {upper:.8f})")
                    else:
                        stats['mean±std'].append(f"{mean:.4f}±{std:.4f}")
                        stats['median'].append(f"{median:.4f}")
                        lower, upper = bootstrap_ci(values)
                        stats['95% CI'].append(f"({lower:.4f}, {upper:.4f})")
                all_scores.extend(values)

            category_stats['score'].append(stats['mean±std'][-1])
            category_stats['95% CI'].append(stats['95% CI'][-1])
            category_stats['IQR'].append(stats['IQR'][-1])

        d[f'{experiment} mean'] = stats['mean±std']
        d[f'{experiment} median'] = stats['median']
        d[f'{experiment} 95% CI'] = stats['95% CI']
        d[f'{experiment} IQR'] = stats['IQR']

        category_means[experiment] = {
            'overall score': np.mean([float(m.split('±')[0]) for m in category_stats['score'] if m is not None]),
            'overall 95% CI': (
                np.mean([float(m.split('(')[1].split(',')[0])
                        for m in category_stats['95% CI'] if m is not None]),
                np.mean([float(m.split('(')[1].split(',')[1][:-1])
                        for m in category_stats['95% CI'] if m is not None])
            ),
            'overall IQR': np.mean([i for i in category_stats['IQR'] if i is not None]),
        }
        category_means[experiment][
            'overall 95% CI'] = f"({category_means[experiment]['overall 95% CI'][0]:.4f}, {category_means[experiment]['overall 95% CI'][1]:.4f})"

    output_df = pd.DataFrame(d)
    columns_extra_info = ['Organ']
    columns = columns_extra_info + [col for col in output_df.columns if col not in columns_extra_info]
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

        for row in range(output_df.shape[0]):
            for result_type in ['mean', 'median', 'IQR']:
                values = [output_df.iloc[row][f'{model} {result_type}']
                        for model in experiment_to_name_dict.values()]
                if all(pd.isna(x) for x in values):
                    continue
                if result_type in ['mean', 'median']:
                    means_or_medians = []
                    if result_type == 'mean':
                        for value in values:
                            if pd.isna(value):
                                means_or_medians.append(None)
                            else:
                                means_or_medians.append(float(value.split('±')[0]))
                    else:
                        for value in values:
                            if pd.isna(value):
                                means_or_medians.append(None)
                            else:
                                means_or_medians.append(float(value))
                    rowindex = row + 1

                    if None in means_or_medians:
                        non_none_index = 0 if means_or_medians[0] is not None else 1
                        colname = f"{list(experiment_to_name_dict.values())[non_none_index]} {result_type}"
                        colindex = output_df.columns.get_loc(colname)
                        value = output_df[colname].iloc[row]
                        worksheet.write(rowindex, colindex,
                                        value, format_only_value)
                    else:
                        if prefix in score_higher_is_better_metrics:
                            better_index = 0 if means_or_medians[0] > means_or_medians[1] else (
                                1 if means_or_medians[1] > means_or_medians[0] else None)
                        else:
                            better_index = 0 if means_or_medians[0] < means_or_medians[1] else (
                                1 if means_or_medians[1] < means_or_medians[0] else None)

                        if better_index is not None:
                            colname = f"{list(experiment_to_name_dict.values())[better_index]} {result_type}"
                            colindex = output_df.columns.get_loc(colname)
                            value = output_df[colname].iloc[row]

                            if abs(means_or_medians[0] - means_or_medians[1]) > highlight_diff_threshold[prefix]: 
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
                        colname = f"{list(experiment_to_name_dict.values())[non_none_index]} {result_type}"
                        colindex = output_df.columns.get_loc(colname)
                        value = output_df[colname].iloc[row]
                        worksheet.write(rowindex, colindex,
                                        value, format_only_value)
                    else:
                        better_index = 0 if iqrs[0] < iqrs[1] else (1 if iqrs[1] < iqrs[0] else None)

                        if better_index is not None:
                            colname = f"{list(experiment_to_name_dict.values())[better_index]} {result_type}"
                            colindex = output_df.columns.get_loc(colname)
                            value = output_df[colname].iloc[row]

                            if abs(iqrs[0] - iqrs[1]) > 0.02:
                                worksheet.write(rowindex, colindex,
                                                value, format_significant)
                            else:
                                worksheet.write(rowindex, colindex,
                                                value, format_small_difference)
                        
        better_model_col = output_df.columns.get_loc(f'Better Model')
        for row in range(output_df.shape[0]):
            better_model = output_df[f'Better Model'].iloc[row]
            is_significant = output_df[f'Significant After Correction'].iloc[row]
            if do_benjamini_hochberg:
                if is_significant: # only highlight if the difference is significant after correction
                    if better_model == 'TotalSeg':
                        worksheet.write(row + 1, better_model_col, better_model, format_totalsegmentator)
                    elif better_model == 'OMASeg':
                        worksheet.write(row + 1, better_model_col, better_model, format_omaseg)
            else:
                if better_model == 'TotalSeg':
                        worksheet.write(row + 1, better_model_col, better_model, format_totalsegmentator)
                elif better_model == 'OMASeg':
                    worksheet.write(row + 1, better_model_col, better_model, format_omaseg)

    # Store means
    category_results = []
    for experiment, metrics in category_means.items():
        category_results.append({
            'Experiment': experiment,
            'Overall Score': metrics['overall score'],
            'Overall 95% CI': metrics['overall 95% CI'],
            'Overall IQR': metrics['overall IQR'],
        })
    category_df = pd.DataFrame(category_results)
    category_df.to_csv(os.path.join(output_compare_folder, f'{prefix}_means_summary.csv'), index=False)