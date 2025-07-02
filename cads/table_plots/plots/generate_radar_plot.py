from cads.table_plots.plots.utils import read_results_table
from cads.dataset_utils.bodyparts_labelmaps import anatomical_systems
from cads.table_plots.plots.plot_functions import generate_radar_plot_normalized_metrics, generate_radar_plot_distance_metrics


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
        # 'filtered_unreliable_and_limited_fov',
        'filtered_unreliable',
        # 'original_GT_but_remove_limited_fov',
        # 'scores_final', 
        ]
    metrics = [
        'dice',
        # 'hd',
        # 'hd95',
        # 'normalized_distance'
    ]
    metric_avg_types = [
        # 'median',
        'mean',
    ]
    baseline_scores = {
    'VISCERAL Gold Corpus': {'Liver': 0.9, 'Spleen': 0.802, "Pancreas": 0.465, "Gallbladder": 0.334, "Urinary bladder": 0.676, "Aorta": 0.785, "Trachea": 0.847, "Sternum": 0.648, "Thyroid": 0.469, "Kidney R": 0.877, "Kidney L": 0.903, "Adrenal gland R": 0.138, "Adrenal gland L": 0.165, "Psoas major muscle R": 0.771, "Psoas major muscle L": 0.772},  # Best results from Ga1 w/wo contrast, skip rectus abdominis and lungs
    'VISCERAL Silver Corpus': {'Liver': 0.94, 'Spleen': 0.89, "Pancreas": 0.47, "Gallbladder": 0.54, "Urinary bladder": 0.86, "Aorta": 0.82, "Trachea": 0.93, "Sternum": 0.83, "Thyroid": 0.57, "Kidney R": 0.94, "Kidney L": 0.93, "Adrenal gland R": 0.35, "Adrenal gland L": 0.36, "Psoas major muscle R": 0.86, "Psoas major muscle L": 0.85, "Rectus abdominis muscle R": 0.69, "Rectus abdominis muscle L": 0.64}, # Best results from w/wo contrast, skip lungs
    'KiTS (avg)': {"Kidney R": 0.912}, # kidney composite score
    'LiTS': {"Liver": 0.963},
    'BTCV-Abdomen (avg)': {'Stomach': 0.9213}, # avg of multiple organs mostly from digestive system
    'BTCV-Cervix': {'Urinary bladder': 0.8303, 'Rectum': 0.64722, 'Small intestine': 0.35239}, # free competition
    'CHAOS': {'Liver': 0.9779}, # PKDIA
    'CT-ORG': {'Liver': 0.952, 'Urinary bladder': 0.777, },  # NN 3d UNet, skip lungs kidneys
    'AbdomenCT-1K': {'Liver': 0.962, 'Spleen': 0.949, 'Pancreas': 0.829},  # semi-supervised subtask2, skip kidney
    'VerSe (avg)': {'Vertebra L1': 0.9172}, # avg of multiple vertebrae
    'SLIVER07': {'Liver': 0.917}, # grand challenge leaderboard
    # re-train single-source baselines
    # 'VISCERAL Gold Corpus-Extra': {
    #     "Brainstem": 0.7655,
    #     "Spinal canal": 0.9104,
    #     "Parotid gland L": 0.7775,
    #     "Parotid gland R": 0.7429,
    #     "Submandibular gland L": 0.7846,
    #     "Submandibular gland R": 0.7839,
    #     "Larynx": 0.7220,
    #     "Heart": 0.8954,
    #     "Esophagus": 0.7606,
    #     "Stomach": 0.8063,
    #     "Colostomy bag": 0.9118,
    #     "Sigmoid colon": 0.6802,
    #     "Rectum": 0.7716,
    #     "Prostate": 0.7842,
    #     "Seminal vesicle": 0.8004,
    #     "Mammary gland L": 0.8778,
    #     "Mammary gland R": 0.8892,
    #     "Sternum": 0.8779,
    #     "Psoas major muscle R": 0.8690,
    #     "Psoas major muscle L": 0.8586,
    #     "Rectus abdominis muscle R": 0.7678,
    #     "Rectus abdominis muscle L": 0.7566,
    # },
    
} 
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
                        model1_name='TotalSegmentator',
                        model2_name='CADS',
                        baseline_scores=baseline_scores,
                        output_path=f"/mnt/hdda/murong/22k/plots/{result_type}/per_structure/radar_plot_all_{metric_avg_type}_{metric}.pdf",
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
                        model1_name='TotalSegmentator',
                        model2_name='CADS',
                        output_path=f"/mnt/hdda/murong/22k/plots/{result_type}/per_structure/radar_plot_all_{metric_avg_type}_{metric}.pdf",
                        title=f"Structure-wise {metric_avg_type.capitalize()} {metric.capitalize()} Score Comparison (Full Dataset)",
                        system_groups=anatomical_systems
                    )
