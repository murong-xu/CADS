import pandas as pd
import os

from cads.misc.sandbox.table_plots.utils.utils import filter_rows, list_specific_files, transitional_ids, amos_uterus_ids, compare_models_stat_test, ambigious_gt_structures_to_skip
from cads.dataset_utils.bodyparts_labelmaps import anatomical_systems, labelmap_all_structure, labelmap_all_structure_renamed, structure_to_in_dist_training_dataset
from cads.misc.sandbox.table_plots.plots.plot_functions import generate_histogram_plot


def collect_scores(analysis_name, grouping_in_out_dist, prefix, distribution):
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
    
    splits = ['test']
    filter_transitional_in_verse = True
    significance_level = 0.05  #TODO: test more values
    do_benjamini_hochberg = False
    skip_ambigious_gt_eval = True

    if prefix in ['dice', 'normalized_distance']:
        higher_better = True
    else:
        higher_better = False

    experiment_to_name_dict = {
        'omaseg': 'OMASeg',
        'totalsegmentator': 'TotalSeg',
    }

    table_names = list(labelmap_all_structure_renamed.values())
    structuresids = list(labelmap_all_structure.values())

    if len(table_names) != len(structuresids):
        print("Error, the number of names and structures ids is not the same")
        exit()

    experiments_dicts = {}
    test_datasets_sources_dict = {}
    for experiment in experiment_to_name_dict.keys():
        structure_values = {distribution: {table_name: [] for table_name in table_names}}
        test_datasets_sources = {distribution: {table_name: [] for table_name in table_names}}
        excelfiles = list_specific_files(experiment_results_path[experiment], prefix=prefix, suffix='.xlsx')
        for file in excelfiles:
            base = os.path.basename(file)
            df = pd.read_excel(file)
            if df.shape[0] <= 1:
                continue
            if '0010_verse' in base and filter_transitional_in_verse:
                df = df[~df["ids"].isin(transitional_ids)]
            if skip_ambigious_gt_eval:
                for key, value in ambigious_gt_structures_to_skip.items():
                    if key in base:
                        if value in df.columns:
                            df = df.drop(value, axis=1)
            df = filter_rows(df, splits=splits)
            column_names = df.columns

            for j, s in enumerate(structuresids):
                matching_columns = [
                    column for column in column_names if s == column]
                if len(matching_columns) > 0:
                    datasetname = file.split('/')[-2]
                    if grouping_in_out_dist == 'group_by_omaseg_inout':
                        # decide the current structure belongs to its training group or not 
                        structure_id = str([k for k, v in labelmap_all_structure.items() if v == s][0])
                        structure_is_in_dist = (datasetname == structure_to_in_dist_training_dataset[structure_id])
                    elif grouping_in_out_dist == 'group_by_totalseg_dataset':
                        # decide the current structure belongs to totalseg dataset or not 
                        if '0037_totalsegmentator' == datasetname:
                            structure_is_in_dist = True
                        else:
                            structure_is_in_dist = False
                    else:
                        print("Error, select one of the following: group_by_omaseg_inout / group_by_totalseg_dataset")
                        exit()

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
                            test_datasets_sources['in'][table_names[j]].append(datasetname)
                        elif not structure_is_in_dist and 'out' in structure_values:
                            structure_values['out'][table_names[j]].extend(
                                values)
                            test_datasets_sources['out'][table_names[j]].append(datasetname)
                        if 'all' in structure_values:
                            structure_values['all'][table_names[j]].extend(
                                values)
                            test_datasets_sources['all'][table_names[j]].append(datasetname)
        experiments_dicts[experiment_to_name_dict[experiment]
                        ] = structure_values
        test_datasets_sources_dict[experiment_to_name_dict[experiment]] = test_datasets_sources
    
    # Align scores
    combined_results_df, aligned_omaseg, aligned_totalseg = compare_models_stat_test(
        experiments_dicts['OMASeg'][distribution], experiments_dicts['TotalSeg'][distribution], alpha=significance_level,
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
    result_types = [
        'filtered_unreliable_and_limited_fov',
        'filtered_unreliable',
        'original_GT_but_remove_limited_fov',
        'scores_final', 
        ]
    metrics = [
        'dice',
        'hd',
        'hd95',
        'normalized_distance'
    ]

    grouping_in_out_dist = 'group_by_omaseg_inout'  # 'group_by_omaseg_inout'/'group_by_totalseg_dataset'
    plot_dist = 'all'
    
    for result_type in result_types:
        for metric in metrics:
            # Step 1) collect scores
            aligned_omaseg, aligned_totalseg, stat_results = collect_scores(result_type, grouping_in_out_dist, metric, plot_dist)

            # Step 2) generate plot
            plot_output_path = f"/mnt/hdda/murong/22k/plots/{result_type}/per_structure/per-system_histogram_compare_{metric}"
            list_anatomical_systems = list(anatomical_systems.keys())
            for anatomical_system in list_anatomical_systems:
                generate_histogram_plot(
                    model1_scores=aligned_totalseg,
                    model2_scores=aligned_omaseg,
                    model1_name='TotalSeg',
                    model2_name='CADS',
                    stat_results=stat_results,
                    output_path=plot_output_path,
                    metric_name=metric.capitalize().replace('_', ' '),
                    system_group=anatomical_system,
                    anatomical_systems=anatomical_systems
                )
