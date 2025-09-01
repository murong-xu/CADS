import pandas as pd
import os
import numpy as np

# TODO: param
output_folder = '/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005'

# analysis_name = 'scores_final'
# analysis_name = 'filtered_unreliable_and_limited_fov'
analysis_name = 'filtered_unreliable'
# analysis_name = 'original_GT_but_remove_limited_fov'

datasets_eval = [
    '0001_visceral_gc',
    "0001_visceral_gc_new",
    '0002_visceral_sc',
    '0003_kits21',
    '0004_lits',
    '0005_bcv_abdomen',
    '0006_bcv_cervix',
    '0007_chaos',
    '0008_ctorg',
    '0009_abdomenct1k',
    '0010_verse',
    '0014_learn2reg',
    '0018_sliver07',
    '0034_empire',
    '0037_totalsegmentator',
    '0038_amos',
    '0039_han_seg',
    '0039_han_seg_reg',
    '0040_saros',
    # Extra test sets
    # '0080_SegTHOR',
    # '0081_ribseg',
    # '0082_word',
    # '0083_word_lits',
    # '0084_BTCV_VNet',
    # '0086_Private_CTPelvic1K',
    # '0087_COVID19_CTSpine1K',
    # '0088_MSD_CTSpine1K',
]
prefixes = ['dice', 'hd95', 'hd', 'normalized_distance']

for prefix in prefixes:
    summary_results = {prefix: {'overlapping': {}, 'omaseg_all': {}} for prefix in prefixes}
    
    for prefix in prefixes:
        for dataset in datasets_eval:
            xlsx_path = os.path.join(output_folder, 'per-challenge', analysis_name, f'{prefix}_{dataset}.xlsx')
            if not os.path.exists(xlsx_path):
                continue
                
            df = pd.read_excel(xlsx_path)
            
            def extract_mean_from_str(mean_std_str):
                if mean_std_str is None or pd.isna(mean_std_str):
                    return None
                return float(mean_std_str.split('±')[0])

            omaseg_means = [extract_mean_from_str(x) for x in df['OMASeg mean±std']]
            totalseg_means = [extract_mean_from_str(x) for x in df['TotalSeg mean±std']]
            omaseg_medians = df['OMASeg median'].tolist()
            totalseg_medians = df['TotalSeg median'].tolist()
            
            # overlapping structures
            valid_pairs_mean = [(om, tm) for om, tm in zip(omaseg_means, totalseg_means) 
                          if om is not None and tm is not None]
            valid_pairs_median = [(om, tm) for om, tm in zip(omaseg_medians, totalseg_medians) 
                          if not np.isnan(om) and not np.isnan(tm)]
            if valid_pairs_mean:
                omaseg_overlapping_means = [pair[0] for pair in valid_pairs_mean]
                totalseg_overlapping_means = [pair[1] for pair in valid_pairs_mean]
                omaseg_overlapping_medians = [pair[0] for pair in valid_pairs_median]
                totalseg_overlapping_medians = [pair[1] for pair in valid_pairs_median]
                
                summary_results[prefix]['overlapping'][dataset] = {
                    'OMASeg_overall_mean': np.mean(omaseg_overlapping_means),
                    'TotalSeg_overall_mean': np.mean(totalseg_overlapping_means),
                    'OMASeg_overall_median': np.mean(omaseg_overlapping_medians),
                    'TotalSeg_overall_median': np.mean(totalseg_overlapping_medians),
                    'num_structures': len(valid_pairs_mean)
                }
            
            # OMASeg all structures
            valid_omaseg_mean = [x for x in omaseg_means if x is not None]
            valid_omaseg_median = [x for x in omaseg_medians if x is not None]
            if valid_omaseg_mean:
                summary_results[prefix]['omaseg_all'][dataset] = {
                    'OMASeg_overall_mean': np.mean(valid_omaseg_mean),
                    'OMASeg_overall_median': np.mean(valid_omaseg_median),
                    'num_structures': len(valid_omaseg_mean)
                }
    
    # write to a file
    summary_output_path = os.path.join(output_folder, 'per-challenge', analysis_name, 'summary_all_datasets.xlsx')
    with pd.ExcelWriter(summary_output_path, engine='xlsxwriter') as writer:
        for prefix in prefixes:
            overlapping_data = []
            for dataset in datasets_eval:
                if dataset in summary_results[prefix]['overlapping']:
                    result = summary_results[prefix]['overlapping'][dataset]
                    overlapping_data.append({
                        'Dataset': dataset,
                        'OMASeg Mean': result['OMASeg_overall_mean'],
                        'OMASeg Median': result['OMASeg_overall_median'],
                        'TotalSeg Mean': result['TotalSeg_overall_mean'],
                        'TotalSeg Median': result['TotalSeg_overall_median'],
                        'Number of Structures': result['num_structures']
                    })
            
            omaseg_all_data = []
            for dataset in datasets_eval:
                if dataset in summary_results[prefix]['omaseg_all']:
                    result = summary_results[prefix]['omaseg_all'][dataset]
                    omaseg_all_data.append({
                        'Dataset': dataset,
                        'OMASeg Mean': result['OMASeg_overall_mean'],
                        'OMASeg Median': result['OMASeg_overall_median'],
                        'Number of Structures': result['num_structures']
                    })
            
            if overlapping_data:
                df_overlapping = pd.DataFrame(overlapping_data)
                df_overlapping.to_excel(writer, sheet_name=f'{prefix}_overlapping', index=False)
            
            if omaseg_all_data:
                df_omaseg_all = pd.DataFrame(omaseg_all_data)
                df_omaseg_all.to_excel(writer, sheet_name=f'{prefix}_omaseg_all', index=False)
            
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:E', 15)
                worksheet.set_column('F:F', 15)