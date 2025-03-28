import numpy as np
import json

from omaseg.table_plots.utils.utils import compare_models_stat_test
from omaseg.table_plots.plots.plot_functions import generate_boxplot_comparison

STRUCTURE_NAME_MAPPING = {
    'Brainstem': 'Brainstem',
    'Eye_L': 'Eye L',
    'Eye_R': 'Eye R',
    'Larynx': 'Larynx',
    'OpticNerve_L': 'Optic nerve L',
    'OpticNerve_R': 'Optic nerve R',
    'Parotid_L': 'Parotid gland L',
    'Parotid_R': 'Parotid gland R',
    'SubmandibularGland_L': 'Submandibular gland L',
    'SubmandibularGland_R': 'Submandibular gland R',
    'Aorta': 'Aorta',
    'Bladder': 'Urinary bladder',
    'Brain': 'Brain',
    'Esophagus': 'Esophagus',
    'Humerus_L': 'Humerus L',
    'Humerus_R': 'Humerus R',
    'Kidney_L': 'Kidney L',
    'Kidney_R': 'Kidney R',
    'Liver': 'Liver',
    'Lung_L': 'Lung L',
    'Lung_R': 'Lung R',
    'Prostate': 'Prostate',
    'SpinalCord': 'Spinal cord',
    'Spleen': 'Spleen',
    'Stomach': 'Stomach',
    'Thyroid': 'Thyroid',
    'Trachea': 'Trachea',
    'V_CavaInferior': 'Inferior vena cava',
    'Heart': 'Heart',
    'Chiasm': 'Optic chiasm',
    'Glottis': 'Glottis',
    'LacrimalGland_L': 'Lacrimal gland L',
    'LacrimalGland_R': 'Lacrimal gland R',
    'Mandible': 'Mandible',
    'OralCavity': 'Oral cavity',
    'Pituitary': 'Pituitary gland',
    'Rectum': 'Rectum',
    'SeminalVesicle': 'Seminal vesicle'
}

def collect_scores(prefix):
    jsonfile_totalseg = '/mnt/hdda/murong/22k/results/usz/STAT_new/TS_updated.json'
    jsonfile_omaseg = '/mnt/hdda/murong/22k/results/usz/STAT_new/OMA_updated.json'

    score_higher_is_better_metrics = ['Dice', 'normalized_surface_dice', 'TPR']
    significance_level = 0.05
    do_benjamini_hochberg = False

    table_names = list(STRUCTURE_NAME_MAPPING.keys())
    totalseg_exclude_to_compare = ['Optic chiasm', 'Glottis', 'Lacrimal gland L', 'Lacrimal gland R', 'Mandible', 'Oral cavity', 'Pituitary gland', 'Rectum', 'Seminal vesicle']
            
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

    experiments_dicts = {
        'OMASeg': {},
        'TotalSeg': {}
        }
    for organ in table_names:
        experiments_dicts['OMASeg'][STRUCTURE_NAME_MAPPING[organ]] = []
        experiments_dicts['TotalSeg'][STRUCTURE_NAME_MAPPING[organ]] = []
        for image_path in image_paths:
            full_path = f"{image_path}:{organ}"

            # Get scores for both models if they exist
            omaseg_score = omaseg_scores.get(full_path, {}).get(prefix)
            totalseg_score = totalseg_scores.get(full_path, {}).get(prefix)

            omaseg_score = np.nan if omaseg_score is None else omaseg_score
            totalseg_score = np.nan if totalseg_score is None else totalseg_score
            
            experiments_dicts['OMASeg'][STRUCTURE_NAME_MAPPING[organ]].append(omaseg_score)
            experiments_dicts['TotalSeg'][STRUCTURE_NAME_MAPPING[organ]].append(totalseg_score)

    # Compre models
    combined_results_df, aligned_omaseg, aligned_totalseg = compare_models_stat_test(
        experiments_dicts['OMASeg'], experiments_dicts['TotalSeg'], alpha=significance_level, 
        higher_better=higher_better, do_benjamini_hochberg=do_benjamini_hochberg, totalseg_exclude_to_compare=totalseg_exclude_to_compare)
    
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
    metrics = [
        'Dice',
        'hausdorff_dist',
        'hausdorff_dist_95',
        'normalized_surface_dice',
        'TPR', 
        'FPR', 
        'vol_error'
    ]
    
    for metric in metrics:
        plot_output_path = f"/mnt/hdda/murong/22k/plots/usz/boxplot_compare_{metric}"

        # Step 1) collect scores
        aligned_omaseg, aligned_totalseg, stat_results = collect_scores(metric)

        # Step 2) generate plot
        generate_boxplot_comparison(
            model1_scores=aligned_totalseg,
            model2_scores=aligned_omaseg,
            model1_name='TotalSeg',
            model2_name='OMASeg',
            stat_results=stat_results,
            output_path=plot_output_path,
            metric_name=metric.capitalize().replace('_', ' '),
            datasetname='USZ In-house Data'
        )
 