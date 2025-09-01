import pandas as pd

STRUCTURE_NAME_MAPPING = {
    'Brainstem': 'Brainstem',
    'Eye_L': 'Eye L',
    'Eye_R': 'Eye R',
    'Larynx': 'Larynx',
    'OpticNerve_L': 'Optic nerve L',
    'OpticNerve_R': 'Optic nerve R',
    'Parotid_L': 'Parotid gland L',
    'Parotid_R': 'Parotid gland R',
    'SubmandibularGland_L': 'Subman-dibular gland L',
    'SubmandibularGland_R': 'Subman-dibular gland R',
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
METRIC_NAME_MAPPING = {
    'Dice': 'Dice',
    'hausdorff_dist_95': 'HD95',
    'normalized_surface_dice': 'NSD',
    'hausdorff_dist': 'HD',
    'FPR': 'FPR',
    'TPR': 'TPR',
    'vol_error': 'Volume error'
}

def read_summary_data(xlsx_path, use_median=False):
    df = pd.read_excel(xlsx_path)
    organ_names = df['Organ'].tolist()
    
    if use_median:
        omaseg_values = df['OMASeg median'].tolist()
        totalseg_values = df['TotalSeg median'].tolist()
    else:
        omaseg_values = df['OMASeg mean'].tolist()
        totalseg_values = df['TotalSeg mean'].tolist()
    
    return organ_names, totalseg_values, omaseg_values

def extract_mean(value_str):
    if isinstance(value_str, str) and '±' in value_str:
        try:
            mean_str = value_str.split('±')[0]
            return float(mean_str)
        except (ValueError, TypeError):
            return None
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return None
    
def format_number(value, as_percentage=True):
    if isinstance(value, str) and '±' in value:
        try:
            mean_str, std_str = value.split('±')
            mean = float(mean_str)
            std = float(std_str)
            if as_percentage:
                return f"{(mean*100):.2f}±{(std*100):.2f}"
            return f"{mean:.2f}±{std:.2f}"
        except (ValueError, TypeError):
            return value
    
    try:
        num = float(value)
        if as_percentage:
            return f"{(num*100):.2f}"
        return f"{num:.2f}"
    except ValueError:
        return value

def format_value(value, is_diff=False, show_percentage=True, higher_better=True):
    if isinstance(value, str) and '±' in value:
        return format_number(value, as_percentage=show_percentage)
    
    if pd.isna(value) or value == '' or value == 'nan':
        return '-'
    
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    value = format_number(value, as_percentage=show_percentage)

    if is_diff:
        value = float(value)
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
    
    return value

def generate_latex_table(organs, totalseg_values, omaseg_values, output_path, 
                        datasets_per_row=10, metric_name='Dice'):
    latex_code = [
        r"\definecolor{MyGreen}{rgb}{0.133, 0.545, 0.133}",
        r"\begin{table}[htbp]",
        r"\tiny", 
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l *{10}{>{\centering\arraybackslash}p{0.09\textwidth}}}",
        r"\toprule"
    ]

    if metric_name in ['Dice', 'normalized_surface_dice', 'FPR', 'TPR']:
        show_percentage = True
    else:
        show_percentage = False

    if metric_name in ['Dice', 'normalized_surface_dice', 'TPR']:
        higher_better = True
    else:
        higher_better = False

    n_groups = (len(organs) + datasets_per_row - 1) // datasets_per_row
    for i in range(n_groups):
        start_idx = i * datasets_per_row
        end_idx = min((i + 1) * datasets_per_row, len(organs))
        
        current_organs = organs[start_idx:end_idx]
        current_totalseg = totalseg_values[start_idx:end_idx]
        current_omaseg = omaseg_values[start_idx:end_idx]

        if i > 0:
            latex_code.append(r"\midrule[\heavyrulewidth]")
        
        organ_names = [STRUCTURE_NAME_MAPPING[name] for name in current_organs]
        header_row = r"\multirow{2}{*}{\textbf{Model}} & " + " & ".join(
            [fr"\tiny\textbf{{{name}}}" for name in organ_names]
        )

        # fill in empty spaces
        if len(current_organs) < datasets_per_row:
            header_row += " &" * (datasets_per_row - len(current_organs))
        header_row += r"\\"
        latex_code.append(header_row)
        latex_code.append(r"\midrule")

        # TotalSeg, OMASeg
        for name, values in [("TotalSeg", current_totalseg), ("OMASeg", current_omaseg)]:
            if name != "TotalSeg":
                latex_code.append(r"\cmidrule{1-" + str(len(current_organs) + 1) + "}")
            
            formatted_values = []
            for val in values[:len(current_organs)]:
                formatted_values.append(format_value(val, show_percentage=show_percentage, higher_better=higher_better))
            
            row = f"{name} & " + " & ".join(formatted_values)
            # fill in empty spaces
            if len(current_organs) < datasets_per_row:
                row += " &" * (datasets_per_row - len(current_organs))
            row += r"\\"
            latex_code.append(row)

        # Diff
        latex_code.append(r"\cmidrule{1-" + str(len(current_organs) + 1) + "}")
        diff_values = []
        for j in range(len(current_organs)):
            o_mean = extract_mean(current_omaseg[j])
            t_mean = extract_mean(current_totalseg[j])
            if o_mean is not None and t_mean is not None:
                diff = o_mean - t_mean
                diff_values.append(format_value(diff, is_diff=True, show_percentage=show_percentage, higher_better=higher_better))
            else:
                diff_values.append('-')

        diff_row = r"\multicolumn{1}{l}{Diff} & " + " & ".join(diff_values)

        # fill in empty spaces
        if len(current_organs) < datasets_per_row:
            diff_row += " &" * (datasets_per_row - len(current_organs))
        diff_row += r"\\"
        latex_code.append(diff_row)

    # 表格结束
    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        fr"\caption{{{METRIC_NAME_MAPPING[metric_name]} (median) comparison on unseen private hospital data}}",
        r"\label{tab:usz_comparison}",
        r"\end{table}"
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_code))

if __name__ == "__main__":
    metric_dice = 'Dice'
    metric_hd95 = 'hausdorff_dist_95'
    metric_nsd = 'normalized_surface_dice'
    metric_hd = 'hausdorff_dist'
    metric_fpr = 'FPR'
    metric_tpr = 'TPR'
    metric_volerror = 'vol_error'

    metric = metric_dice
    use_median = True  # for this large cohort 

    xlsx_path = f'/mnt/hdda/murong/22k/results/usz/analysis_p005_new/compare_omaseg_totalseg/{metric}_compare_table.xlsx'
    output_path = "/mnt/hdda/murong/22k/plots/latex_tables/kk.txt"    

    organ_names, totalseg_values, omaseg_values = read_summary_data(
        xlsx_path,
        use_median=use_median
    )
    generate_latex_table(organ_names, totalseg_values, omaseg_values, output_path, datasets_per_row=10, metric_name=metric)