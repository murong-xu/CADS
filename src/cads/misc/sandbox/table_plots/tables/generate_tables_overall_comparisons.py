import pandas as pd

def read_metric_csv(csv_path, metric_type='Mean'):
    df = pd.read_csv(csv_path)
    
    result = {}
    for dist in ['in', 'out', 'all']:
        for exp in ['OMASeg', 'TotalSeg']:
            for cat in ['overlapping', 'all']:
                mask = (df['Distribution'] == dist) & \
                       (df['Experiment'] == exp) & \
                       (df['Category'] == cat)
                if not mask.any():
                    continue
                    
                value = df[mask].iloc[0][f'Overall {metric_type}']
                key = (dist, exp, cat)
                result[key] = value
                
    return result

def generate_latex_table(metric_data_1_dict, metric_data_2_dict, metric_type, output_path):
    metric_1_data = read_metric_csv(metric_data_1_dict['path'], metric_type=metric_type)
    metric_2_data = read_metric_csv(metric_data_2_dict['path'], metric_type=metric_type)

    def calc_diff(omaseg, totalseg):
        return omaseg - totalseg
    
    def format_value(value, is_diff=False, show_percentage=True):
        if show_percentage:
            value = value * 100
        
        if is_diff:
            if value > 0:
                return f"{{\\textuparrow{value:.2f}}}"
            else:
                return f"{{\\textdownarrow{abs(value):.2f}}}"
        return f"{value:.2f}"
    
    mutual_rows = []
    # row TotalSeg
    totalseg_values = [
        metric_1_data[('in', 'TotalSeg', 'overlapping')],
        metric_2_data[('in', 'TotalSeg', 'overlapping')],
        metric_1_data[('out', 'TotalSeg', 'overlapping')],
        metric_2_data[('out', 'TotalSeg', 'overlapping')],
        metric_1_data[('all', 'TotalSeg', 'overlapping')],
        metric_2_data[('all', 'TotalSeg', 'overlapping')]
    ]
    show_percentages = [
        metric_data_1_dict['shown_in_percentage'],
        metric_data_2_dict['shown_in_percentage'],
        metric_data_1_dict['shown_in_percentage'],
        metric_data_2_dict['shown_in_percentage'],
        metric_data_1_dict['shown_in_percentage'],
        metric_data_2_dict['shown_in_percentage']
    ]
    mutual_rows.append("& TotalSeg & " + " & ".join(format_value(v, show_percentage=p) for v, p in zip(totalseg_values, show_percentages)) + " \\\\")
    
    # row OMASeg
    omaseg_values = [
        metric_1_data[('in', 'CADS', 'overlapping')],
        metric_2_data[('in', 'CADS', 'overlapping')],
        metric_1_data[('out', 'CADS', 'overlapping')],
        metric_2_data[('out', 'CADS', 'overlapping')],
        metric_1_data[('all', 'CADS', 'overlapping')],
        metric_2_data[('all', 'CADS', 'overlapping')]
    ]
    mutual_rows.append("& CADS & " + " & ".join(format_value(v, show_percentage=p) for v, p in zip(omaseg_values, show_percentages)) + " \\\\")
    
    # row Diff
    diff_values = [calc_diff(o, t) for o, t in zip(omaseg_values, totalseg_values)]
    mutual_rows.append("& {Diff} & " + " & ".join(f"{{\\textbf{{\\textcolor{{MyGreen}}{{{format_value(v, True, show_percentage=p)}}}}}}}" for v, p in zip(diff_values, show_percentages)) + " \\\\")    
    # row All targets
    all_values = [
        metric_1_data[('in', 'CADS', 'all')],
        metric_2_data[('in', 'CADS', 'all')],
        metric_1_data[('out', 'CADS', 'all')],
        metric_2_data[('out', 'CADS', 'all')],
        metric_1_data[('all', 'CADS', 'all')],
        metric_2_data[('all', 'CADS', 'all')]
    ]
    all_row = "& CADS & " + " & ".join(format_value(v, show_percentage=p) for v, p in zip(all_values, show_percentages)) + " \\\\[1ex]"
    
    latex_code = [
        r"\definecolor{MyGreen}{rgb}{0.133, 0.545, 0.133}",
        r"\begin{table}[htbp]",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.2cm} l S[table-format=+2.2] S[table-format=+2.2] S[table-format=+2.2] S[table-format=+2.2] S[table-format=+2.2] S[table-format=+2.2]}",
        r"\toprule",
                r"\multicolumn{2}{c}{\multirow{2}{*}{\textbf{Category}}} & \multicolumn{2}{c}{\textbf{\shortstack{Complete Annotation \\Test Set (Primary)}}} & \multicolumn{2}{c}{\textbf{\shortstack{Partial Annotation \\Test Set (Secondary)}}} & \multicolumn{2}{c}{\textbf{Full test data}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        f"& & {{\\textbf{{{metric_data_1_dict['name']} (\%)}}}} & {{\\textbf{{{metric_data_2_dict['name']}}}}} & {{\\textbf{{{metric_data_1_dict['name']} (\%)}}}} & {{\\textbf{{{metric_data_2_dict['name']}}}}} & {{\\textbf{{{metric_data_1_dict['name']} (\%)}}}} & {{\\textbf{{{metric_data_2_dict['name']}}}}} \\\\",
        r"\midrule",
        r"\multirow{3}{=}{\begin{tabular}[c]{@{}l@{}}Mutual targets\\(119 structures)\end{tabular}}",
        mutual_rows[0],  # TotalSeg row
        r"\cmidrule{2-8}",
        mutual_rows[1],  # CADS row
        r"\cmidrule{2-8}",
        mutual_rows[2],  # Diff row
        r"\midrule",
        r"\multirow{2}{=}{\begin{tabular}[c]{@{}l@{}}All targets\\(167 structures)\end{tabular}}",
        all_row,         # All targets row
        r"& & & & & & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Comparison of metrics across in-/out-distribution test data}",
        r"\label{tab:metrics_comparison}",
        r"\end{table}"
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_code))


if __name__ == "__main__":
    metric_type = 'Mean'
    metric_data_1_dict = {
        'name': 'Dice',
        'path': "/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable/dice_means_summary.csv",
        'shown_in_percentage': True
    }
    metric_data_2_dict = {
        'name': 'HD95',
        'path': "/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable/hd95_means_summary.csv",
        'shown_in_percentage': False
    }
    metric_data_3_dict = {
        'name': 'NSD',
        'path': "/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable/normalized_distance_means_summary.csv",
        'shown_in_percentage': True
    }
    metric_data_4_dict = {
        'name': 'HD',
        'path': "/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable/hd_means_summary.csv",
        'shown_in_percentage': False
    }
    output_path = "/mnt/hdda/murong/22k/plots/latex_tables/kk.txt"    
    
    latex_table = generate_latex_table(metric_data_1_dict, metric_data_2_dict, metric_type, output_path)
