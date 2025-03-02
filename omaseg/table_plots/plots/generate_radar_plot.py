from omaseg.table_plots.plots.utils import read_results_table
from omaseg.dataset_utils.bodyparts_labelmaps import anatomical_systems
from omaseg.table_plots.plots.plot_functions import generate_radar_plot_normalized_metrics, generate_radar_plot_distance_metrics


PLOT_CONFIG = {
    'dice': {
        'highlight_high_scores': False,
        'focus_point': None,
        'power': None,
    },
    'hd': {},
    'hd95': {},
    'normalized_distance': {
        'highlight_high_scores': True,
        'focus_point': 0.90,
        'power': 3,
    },
}

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
    metric_avg_types = [
        'median',
        'mean',
    ]
    for result_type in result_types:
        for metric in metrics:
            for metric_avg_type in metric_avg_types:
                table_path = f'/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/{result_type}/{metric}_compare_table.xlsx'
                column_name = f'all {metric_avg_type}'
                omaseg_scores, totalseg_scores = read_results_table(table_path, column_name) 
                
                if metric in ['dice', 'normalized_distance']:
                    generate_radar_plot_normalized_metrics(
                        model1_scores=totalseg_scores,
                        model2_scores=omaseg_scores,
                        model1_name='TotalSeg',
                        model2_name='OMASeg',
                        output_path=f"/mnt/hdda/murong/22k/plots/{result_type}/per_structure/radar_plot_all_{metric_avg_type}_{metric}.png",
                        title=f"Structure-wise {metric_avg_type.capitalize()} {metric.capitalize()} Score Comparison (Full Dataset)",
                        highlight_high_scores=PLOT_CONFIG[metric]['highlight_high_scores'],
                        focus_point=PLOT_CONFIG[metric]['focus_point'],
                        power=PLOT_CONFIG[metric]['power'],
                        system_groups=anatomical_systems
                    )
                else:
                    generate_radar_plot_distance_metrics(
                        model1_scores=totalseg_scores,
                        model2_scores=omaseg_scores,
                        model1_name='TotalSeg',
                        model2_name='OMASeg',
                        output_path=f"/mnt/hdda/murong/22k/plots/{result_type}/per_structure/radar_plot_all_{metric_avg_type}_{metric}.png",
                        title=f"Structure-wise {metric_avg_type.capitalize()} {metric.capitalize()} Score Comparison (Full Dataset)",
                        system_groups=anatomical_systems
                    )
