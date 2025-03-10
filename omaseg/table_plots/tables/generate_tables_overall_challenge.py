import pandas as pd
import numpy as np

from omaseg.dataset_utils.datasets_labelmap import dataset_renamed

METRIC_NAME_MAPPING = {
    'dice': 'Dice',
    'hd95': 'HD95',
    'normalized_distance': 'NSD',
    'hd': 'HD',
}

def read_summary_data(xlsx_path, metric='dice', use_median=False):
    sheet_name = f'{metric}_overlapping'
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    
    dataset_names = df['Dataset'].tolist()
    
    if use_median:
        omaseg_values = df['OMASeg Median'].tolist()
        totalseg_values = df['TotalSeg Median'].tolist()
    else:
        omaseg_values = df['OMASeg Mean'].tolist()
        totalseg_values = df['TotalSeg Mean'].tolist()
    
    return dataset_names, totalseg_values, omaseg_values


def generate_grouped_latex_table(dataset_names, totalseg_values, omaseg_values, datasets_per_row=9, metric_name='dice'):
    """
    Args:
        datasets_per_row: number of datasets per row
    """
    n_groups = (len(dataset_names) + datasets_per_row - 1) // datasets_per_row    
    latex_code = [
        r"\begin{table}[htbp]",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l *{" + str(datasets_per_row) + r"}{>{\centering\arraybackslash}p{0.09\textwidth}}}"
    ]

    if metric_name in ['dice', 'normalized_distance']:
        show_percentage = True
    else:
        show_percentage = False

    def format_value(value, is_diff=False, show_percentage=True, higher_better=True):
        if show_percentage:
            value = value * 100
        
        if is_diff:
            use_green = (value > 0 and higher_better) or (value < 0 and not higher_better)
            
            if value > 0:
                if use_green:
                    return rf"\textcolor{{MyGreen}}{{$\mathbf{{\uparrow{value:.2f}}}$}}"
                else:
                    return rf"$\uparrow{value:.2f}$"
            else:
                if use_green:
                    return rf"\textcolor{{MyGreen}}{{$\mathbf{{\downarrow{abs(value):.2f}}}$}}"
                else:
                    return rf"$\downarrow{abs(value):.2f}$"
        
        return f"{value:.2f}"

    for i in range(n_groups):
        start_idx = i * datasets_per_row
        end_idx = min((i + 1) * datasets_per_row, len(dataset_names))
        all_datasets = dataset_names[start_idx:end_idx]
        current_datasets = [dataset_renamed[i] for i in all_datasets]
        current_totalseg = totalseg_values[start_idx:end_idx]
        current_omaseg = omaseg_values[start_idx:end_idx]
        
        padding_needed = datasets_per_row - len(current_datasets)
        if padding_needed > 0:
            current_datasets.extend([''] * padding_needed)
            current_totalseg.extend([float('nan')] * padding_needed)
            current_omaseg.extend([float('nan')] * padding_needed)
        
        if i == 0:
            latex_code.append(r"\toprule")
        else:
            latex_code.append(r"\midrule[\heavyrulewidth]")
            
        latex_code.append(r"\multirow{2}{*}{\textbf{Model}} & " + 
                 " & ".join([fr"\tiny\textbf{{{name}}}" if name else "" 
                           for name in current_datasets]) + r"\\")
        latex_code.append(r"\midrule")
        
        # TotalSeg
        totalseg_row = "TotalSeg & " + " & ".join([f"{format_value(v, show_percentage=show_percentage)}" if not np.isnan(v) else "" 
                                                  for v in current_totalseg]) + r"\\"
        latex_code.append(totalseg_row)
        
        # OMASeg
        latex_code.append(r"\cmidrule{1-" + str(len(current_datasets) + 1) + "}")
        omaseg_row = "OMASeg & " + " & ".join([f"{format_value(v, show_percentage=show_percentage)}" if not np.isnan(v) else "" 
                                              for v in current_omaseg]) + r"\\"
        latex_code.append(omaseg_row)
        
        # Diff
        latex_code.append(r"\cmidrule{1-" + str(len(current_datasets) + 1) + "}")
        diff_values = [o - t for o, t in zip(current_omaseg, current_totalseg)]
        diff_row = r"\multicolumn{1}{l}{Diff} & " + " & ".join([
            format_value(v, is_diff=True, show_percentage=show_percentage, higher_better=(metric_name in ['dice', 'normalized_distance']))
            if not np.isnan(v) else "" for v in diff_values]) + r"\\"
        latex_code.append(diff_row)
    
    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{" + f"{METRIC_NAME_MAPPING[metric_name]}" + " comparison across different challenges/datasets}",
        r"\label{tab:dataset_comparison}",
        r"\end{table}"
    ])
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_code))

if __name__ == "__main__":
    metric_dice = 'dice'
    metric_hd95 = 'hd95'
    metric_nsd = 'normalized_distance'
    metric_hd = 'hd'

    metric = metric_dice
    use_median = False
    result_type = 'filtered_unreliable'

    xlsx_path = f'/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/per-challenge/{result_type}/summary_all_datasets.xlsx'
    output_path = "/mnt/hdda/murong/22k/plots/latex_tables/kk.txt"    

    dataset_names, totalseg_values, omaseg_values = read_summary_data(
        xlsx_path,
        metric=metric,
        use_median=use_median
    )
    generate_grouped_latex_table(dataset_names, totalseg_values, omaseg_values, datasets_per_row=9, metric_name=metric)