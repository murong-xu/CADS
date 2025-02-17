import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from omaseg.table_plots.utils.utils import filter_rows, align_and_filter_scores, list_specific_files, transitional_ids, amos_uterus_ids
from omaseg.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, labelmap_all_structure_renamed, structure_to_in_dist_training_dataset, anatomical_systems
from omaseg.table_plots.plots.utils import anatomical_system_colors


def generate_box_plot(results_dict, metric_name='DSC', output_path=None, title="Structure-wise Performance Distribution",
                     anatomical_systems=None):
    """
    Generate a box plot grouped by anatomical systems.
    Args:
        results_dict: Dictionary of structure-wise scores
        anatomical_systems: Dictionary of anatomical system groupings
    """
    data_list = []
    
    # In each anatomical system: sort by median
    structure_medians = {organ: np.median(scores) for organ, scores in results_dict.items()}
    for system, structures in anatomical_systems.items():
        valid_structures = [s for s in structures if s in results_dict]
        sorted_structures = sorted(valid_structures, 
                                 key=lambda x: structure_medians[x],
                                 reverse=True)
        for organ in sorted_structures:
            scores = results_dict[organ]
            n_samples = len(scores)
            for score in scores:
                data_list.append({
                    'Organ': f"{organ} (n={n_samples})",
                    metric_name: score,
                    'System': system
                })
            
    df = pd.DataFrame(data_list)
    plt.figure(figsize=(12, len(results_dict) * 0.3))
    
    ax = sns.boxplot(data=df, 
                    y='Organ', 
                    x=metric_name,
                    hue='System',
                    palette=anatomical_system_colors,
                    whis=1.5,
                    showfliers=True,
                    flierprops={'marker': 'o',
                               'markerfacecolor': 'black',
                               'markeredgecolor': 'black',
                               'markersize': 2,
                               'alpha': 0.3})
    
    # add scatters (original data)
    sns.stripplot(data=df,
                 y='Organ',
                 x=metric_name,
                 color='black',
                 size=2,
                 alpha=0.3,
                 jitter=0.2)
    
    # calculate ststistics (median, mean)
    ordered_organs = df['Organ'].unique()
    stats = df.groupby('Organ').agg({
        metric_name: ['median', 'mean', 'std']
    })[metric_name].reindex(ordered_organs)
    
    right_edge = ax.get_xlim()[1]
    ax.text(right_edge, -0.8,
            'median',
            va='center', ha='left', fontsize=10,
            fontweight='bold', color='black')
    ax.text(right_edge + 0.12, -0.8,
            'mean±std',
            va='center', ha='left', fontsize=10,
            fontweight='bold', color='black')
    
    for i, (organ, row) in enumerate(stats.iterrows()):
        median_str = f'{row["median"]:.3f}'
        ax.text(right_edge, i,
                median_str,
                va='center',
                ha='left',
                fontsize=8,
                color='black',
                alpha=0.7)
        mean_std_str = f'{row["mean"]:.3f}±{row["std"]:.3f}'
        ax.text(right_edge + 0.12, i,
                mean_std_str,
                va='center',
                ha='left',
                fontsize=8,
                color='black',
                alpha=0.7)
        
    prev_system = None
    y_coords = []
    for i, (idx, row) in enumerate(df.groupby('Organ').first().iterrows()):
        if prev_system and row['System'] != prev_system:
            plt.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
        prev_system = row['System']
        y_coords.append(i)
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Organ', fontsize=12)
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    ax.set_xlim(0, right_edge + 0.3)  
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    
    plt.legend(title='Anatomical Systems', 
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=10,
              title_fontsize=12)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return ax

if __name__ == "__main__":
    plot_dist = 'all'  # TODO:
    plot_metric_name = 'Dice'  # TODO:
    plot_output_path='/mnt/hdda/murong/22k/plots/per-structure/box_plot_all_dice.png'  # TODO:
    plot_title='Structure-wise Dice Score Distribution'  # TODO:

    grouping_in_out_dist = 'group_by_omaseg_inout'  # 'group_by_omaseg_inout'/'group_by_totalseg_dataset'
    # analysis_name = 'scores_final'
    analysis_name = 'filtered_unreliable_and_limited_fov'
    # analysis_name = 'filtered_unreliable'
    # analysis_name = 'original_GT_but_remove_limited_fov'

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

    prefix = 'dice'  # TODO:
    distributions = ['in', 'out', 'all']
    splits = ['test']
    filter_transitional_in_verse = True
    exclude_face_from_overall_score = True

    experiment_to_name_dict = {
        'omaseg': 'OMASeg',
        'totalsegmentator': 'TotalSeg',
    }

    labelmap_all_structure_inv = {v: k for k, v in labelmap_all_structure.items()}
    table_names = list(labelmap_all_structure_renamed.values())
    structuresids = list(labelmap_all_structure.values())

    if len(table_names) != len(structuresids):
        print("Error, the number of names and structures ids is not the same")
        exit()

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
                        elif not structure_is_in_dist and 'out' in structure_values:
                            structure_values['out'][table_names[j]].extend(
                                values)
                        if 'all' in structure_values:
                            structure_values['all'][table_names[j]].extend(
                                values)
        experiments_dicts[experiment_to_name_dict[experiment]
                        ] = structure_values

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

    #         original_length = len(omaseg_scores)
    #         aligned_omaseg, aligned_totalseg = align_and_filter_scores(
    #             omaseg_scores, totalseg_scores)
    #         new_length = len(aligned_omaseg)

    #         removed_points = original_length - new_length
    #         removed_data_points[distribution][structure] += removed_points

    #         valid_test_data_points[distribution][structure] = new_length

    #         experiments_dicts['OMASeg'][distribution][structure] = aligned_omaseg
    #         experiments_dicts['TotalSeg'][distribution][structure] = aligned_totalseg


    generate_box_plot(experiments_dicts['OMASeg'][plot_dist], 
                      metric_name=plot_metric_name, 
                      output_path=plot_output_path,
                      title=plot_title,
                      anatomical_systems=anatomical_systems) 