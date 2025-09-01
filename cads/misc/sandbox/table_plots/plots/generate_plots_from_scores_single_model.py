import pandas as pd
import os
import numpy as np

from cads.misc.sandbox.table_plots.utils.utils import filter_rows, align_and_filter_scores, list_specific_files, transitional_ids, amos_uterus_ids, ambigious_gt_structures_to_skip
from cads.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, labelmap_all_structure_renamed, structure_to_in_dist_training_dataset, anatomical_systems
from cads.misc.sandbox.table_plots.plots.plot_functions import generate_box_plot, generate_box_plot_with_testdata_sources
from cads.dataset_utils.datasets_labelmap import dataset_renamed


def collect_scores(analysis_name, grouping_in_out_dist, prefix):
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

    distributions = ['in', 'out', 'all']
    splits = ['test']
    filter_transitional_in_verse = True
    skip_ambigious_gt_eval = True

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
        structure_values = {distribution: {table_name: []
                                        for table_name in table_names} for distribution in distributions}
        test_datasets_sources = {distribution: {table_name: []
                                        for table_name in table_names} for distribution in distributions}
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
                        
                        datasetname = dataset_renamed[datasetname]

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

    # Align scores: only remove NaNs
    for distribution in distributions:
        for structure in table_names:
            omaseg_scores = experiments_dicts['OMASeg'][distribution][structure]
            scores_1 = np.array(omaseg_scores)
            scores_1 = scores_1[~np.isnan(scores_1)]
            experiments_dicts['OMASeg'][distribution][structure] = scores_1

    # # Align scores: considering baseline
    # for distribution in distributions:
    #     for structure in table_names:
    #         omaseg_scores = experiments_dicts['OMASeg'][distribution][structure]
    #         totalseg_scores = experiments_dicts['TotalSeg'][distribution][structure]

    #         aligned_omaseg, aligned_totalseg = align_and_filter_scores(
    #             omaseg_scores, totalseg_scores)

    #         experiments_dicts['OMASeg'][distribution][structure] = aligned_omaseg
    #         experiments_dicts['TotalSeg'][distribution][structure] = aligned_totalseg

    return experiments_dicts, test_datasets_sources_dict


if __name__ == "__main__":
    result_types = [
    # 'filtered_unreliable_and_limited_fov',
    'filtered_unreliable',
    # 'original_GT_but_remove_limited_fov',
    # 'scores_final', 
    ]
    metrics = {
        'dice': {'is_normalized': True},
        'hd': {'is_normalized': False},
        'hd95': {'is_normalized': False},
        'normalized_distance': {'is_normalized': True},
    }

    grouping_in_out_dist = 'group_by_omaseg_inout'  # 'group_by_omaseg_inout'/'group_by_totalseg_dataset'
    plot_dist = 'all'

    for result_type in result_types:
        for metric in list(metrics.keys()):
            # Step 1) collect sc777ores
            experiments_dicts, test_datasets_sources_dict = collect_scores(result_type, grouping_in_out_dist, metric)

            # Step 2) generate plot
            if metric == 'dice':
                metric_name = 'Dice'
            elif metric == 'hd':
                metric_name = 'Hausdorff Distance'
            elif metric == 'hd95':
                metric_name = 'Hausdorff Distance 95th Percentile'
            elif metric == 'normalized_distance':
                metric_name = 'Normalized Surface Dice'

            # plot_output_path=f'/mnt/hdda/murong/22k/plots/{result_type}/per_structure/omaseg-only/omaseg-only_box_plot_{metric}.pdf'
            # generate_box_plot(experiments_dicts['OMASeg'][plot_dist], 
            #                 metric_name=metric_name, 
            #                 output_path=plot_output_path,
            #                 title=f'Structure-wise {metric_name} Score Distribution',
            #                 anatomical_systems=anatomical_systems,
            #                 is_normalized=metrics[metric]['is_normalized']
            #                 )
            
            plot_output_path=f'/mnt/hdda/murong/22k/plots/{result_type}/per_structure/omaseg-only/omaseg-only_box_plot_{metric}_with_test_sources.pdf'
            generate_box_plot_with_testdata_sources(experiments_dicts['OMASeg'][plot_dist],
                                                    test_datasets_sources_dict['OMASeg'][plot_dist],
                                                    metric_name=metric_name,
                                                    output_path=plot_output_path,
                                                    title=f'Structure-wise {metric_name} Score Distribution',
                                                    anatomical_systems=anatomical_systems,
                                                    is_normalized=metrics[metric]['is_normalized']
                                                    )
