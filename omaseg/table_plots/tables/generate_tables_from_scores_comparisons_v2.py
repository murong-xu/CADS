import pandas as pd
from omaseg.dataset_utils.bodyparts_labelmaps import anatomical_systems


def generate_metric_cell(row, model_name, show_percentage, better_model=None):
    def format_number(value, as_percentage=False):
        if pd.isna(value) or value == '' or value == 'nan':
            return '-'
        
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

    # get mean, median, 95% CI
    mean = format_number(row[f'{model_name} all mean'], show_percentage)
    median = format_number(row[f'{model_name} all median'], show_percentage)
    ci = row[f'{model_name} all 95% CI']

    if pd.isna(ci) or ci == '' or ci == 'nan':
        ci_formatted = '-'
    else:
        try:
            ci = ci.strip('()').split(',')
            if show_percentage:
                ci_formatted = f"({float(ci[0])*100:.2f}, {float(ci[1])*100:.2f})"
            else:
                ci_formatted = f"({float(ci[0]):.2f}, {float(ci[1]):.2f})"
        except:
            ci_formatted = '-'

    # if results not exist
    if mean == '-' and median == '-' and ci_formatted == '-':
        return ['--', '--', '--']

    # Format the mean with ± and add star if better
    if mean != '-':
        if '±' in mean:
            mean = f"${mean}$"
        if better_model == model_name:
            mean = f"{mean}\\textsuperscript{{\\textcolor{{MyGreen}}{{*}}}}"

    return [mean, median, ci_formatted]


def generate_organ_row(organ, metric_1_row, metric_2_row, metric_data_1_dict, metric_data_2_dict):
    show_percentage_1 = metric_data_1_dict.get('shown_in_percentage', False)
    show_percentage_2 = metric_data_2_dict.get('shown_in_percentage', False)

    better_model_1 = metric_1_row.get('all Better Model', None)
    better_model_2 = metric_2_row.get('all Better Model', None)

    # Generate cells for each model and metric
    omaseg_1_values = generate_metric_cell(metric_1_row, 'OMASeg', show_percentage_1, better_model_1)
    totalseg_1_values = generate_metric_cell(metric_1_row, 'TotalSeg', show_percentage_1, better_model_1)
    omaseg_2_values = generate_metric_cell(metric_2_row, 'OMASeg', show_percentage_2, better_model_2)
    totalseg_2_values = generate_metric_cell(metric_2_row, 'TotalSeg', show_percentage_2, better_model_2)

    # Generate the three rows
    rows = [
        "\\nopagebreak[4]",
        f"\\multirow{{3}}{{=}}{{{organ}}} & "
        f"{omaseg_1_values[0]} & {totalseg_1_values[0]} & "
        f"{omaseg_2_values[0]} & {totalseg_2_values[0]} \\\\",
        "\\nopagebreak[4]",
        f"& {omaseg_1_values[1]} & {totalseg_1_values[1]} & "
        f"{omaseg_2_values[1]} & {totalseg_2_values[1]} \\\\",
        "\\nopagebreak[4]",
        f"& {omaseg_1_values[2]} & {totalseg_1_values[2]} & "
        f"{omaseg_2_values[2]} & {totalseg_2_values[2]} \\\\"
    ]

    return rows


def generate_latex_table(metric_data_1_dict, metric_data_2_dict, output_path, system_order=None):
    # read csv
    metric_name_1 = metric_data_1_dict['name']
    xlsx_path_1 = metric_data_1_dict['path']
    metric_name_2 = metric_data_2_dict['name']
    xlsx_path_2 = metric_data_2_dict['path']

    df_1 = pd.read_excel(xlsx_path_1)
    df_2 = pd.read_excel(xlsx_path_2)

    # LaTeX header
    latex_code = [
        "\\definecolor{MyGreen}{rgb}{0.133, 0.545, 0.133}",
        "{\\small",
        "\\begin{longtable}{>{\centering\\arraybackslash}p{2.3cm} c c c c}",
        "\\caption{Comparison of OMASeg and TotalSeg performance}",
        "\\label{tab:model_comparison} \\\\",
        "\\toprule",
        
        "\\multirow{2}{*}{\\textbf{Organ}} & \n"
        "\\multicolumn{1}{c}{\\textbf{OMASeg}} & \n"
        "\\multicolumn{1}{c}{\\textbf{TotalSeg}} & \n"
        "\\multicolumn{1}{c}{\\textbf{OMASeg}} & \n"
        "\\multicolumn{1}{c}{\\textbf{TotalSeg}} \\\\",
        
        "& \\multicolumn{1}{c}{\\textbf{(" + metric_name_1 + ")}} & \n"
        "\\multicolumn{1}{c}{\\textbf{(" + metric_name_1 + ")}} & \n"
        "\\multicolumn{1}{c}{\\textbf{(" + metric_name_2 + ")}} & \n"
        "\\multicolumn{1}{c}{\\textbf{(" + metric_name_2 + ")}} \\\\",
        "\\midrule",
        "\\endfirsthead",
        
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Organ}} & \n"
        "\\multicolumn{1}{c}{\\textbf{OMASeg}} & \n"
        "\\multicolumn{1}{c}{\\textbf{TotalSeg}} & \n"
        "\\multicolumn{1}{c}{\\textbf{OMASeg}} & \n"
        "\\multicolumn{1}{c}{\\textbf{TotalSeg}} \\\\",
        
        "& \\multicolumn{1}{c}{\\textbf{(" + metric_name_1 + ")}} & \n"
        "\\multicolumn{1}{c}{\\textbf{(" + metric_name_1 + ")}} & \n"
        "\\multicolumn{1}{c}{\\textbf{(" + metric_name_2 + ")}} & \n"
        "\\multicolumn{1}{c}{\\textbf{(" + metric_name_2 + ")}} \\\\",
        "\\midrule",
        "\\endhead",
        
        "\\midrule",
        "\\multicolumn{5}{r}{\\textit{Continued on next page}} \\\\",
        "\\endfoot",
        
        "\\bottomrule",
        "\\endlastfoot"
    ]

    if system_order:
        # create table by system
        organ_to_system = {}
        for system, organs in anatomical_systems.items():
            for organ in organs:
                organ_to_system[organ] = system

        system_grouped_data = {system: [] for system in system_order}

        for organ in df_1['Organ']:
            system = organ_to_system.get(organ)
            if system in system_order:
                system_grouped_data[system].append(organ)

        for system in system_order:
            if system_grouped_data[system]:
                # append system name
                latex_code.append(
                    "\\midrule"
                )
                latex_code.append(
                    f"\\multicolumn{{5}}{{c}}{{\\rule{{0pt}}{{1ex}}\\textbf{{{system}}}}} \\\\"
                )
                latex_code.append(
                    "\\addlinespace[0.1ex]"
                )
                latex_code.append(
                    "\\midrule"
                )

                for organ in system_grouped_data[system]:
                    metric_1_row = df_1[df_1['Organ'] == organ].iloc[0]
                    metric_2_row = df_2[df_2['Organ'] == organ].iloc[0]

                    rows = generate_organ_row(organ, metric_1_row, metric_2_row,
                                           metric_data_1_dict, metric_data_2_dict)
                    latex_code.extend(rows)
                    latex_code.append("\\midrule")

    else:
        # plain order
        for organ in df_1['Organ']:
            metric_1_row = df_1[df_1['Organ'] == organ].iloc[0]
            metric_2_row = df_2[df_2['Organ'] == organ].iloc[0]

            rows = generate_organ_row(organ, metric_1_row, metric_2_row,
                                           metric_data_1_dict, metric_data_2_dict)
            latex_code.extend(rows)
            latex_code.append("\\midrule")

    # end lines
    latex_code.extend([
        "\\end{longtable}",
        "}"
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_code))


if __name__ == "__main__":
    metric_data_1_dict = {
        'name': 'Dice',
        'path': "/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable/dice_compare_table.xlsx",
        'shown_in_percentage': True
    }
    metric_data_2_dict = {
        'name': 'HD95',
        'path': "/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable/hd95_compare_table.xlsx",
        'shown_in_percentage': False
    }
    output_path = "/mnt/hdda/murong/22k/plots/latex_tables/kk.txt"

    system_order = [
        "Digestive System",
        "Reproductive System",
        "Nervous System",
        "Cardiovascular System",
        "Respiratory System",
        "Lymphatic and Immune System",
        "Urinary System",
        "Skeletal System",
        "Endocrine System",
        "Muscular System",
        "Sensory Organs",
        "Body Cavities",
        "Glandular System",
        "Other Structures"
    ]

    generate_latex_table(metric_data_1_dict,
                         metric_data_2_dict, output_path, system_order)
