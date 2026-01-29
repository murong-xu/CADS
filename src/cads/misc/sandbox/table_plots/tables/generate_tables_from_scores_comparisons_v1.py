import pandas as pd
from cads.dataset_utils.bodyparts_labelmaps import anatomical_systems


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
        return '--'

    if show_percentage and mean != '-':
        mean = f"{mean}"
    if show_percentage and median != '-':
        median = f"{median}"

    if better_model == model_name:
        return (
            f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}"
            f"\\begin{{minipage}}[c][4em]{{2.3cm}}\\raggedleft"
            f"\\hfill\\raisebox{{-1ex}}{{\\textcolor{{MyGreen}}{{\\textbf{{\\normalsize *}}}}}}\\\\[-2ex]"
            f"\\centering {mean} \\\\ {median} \\\\ {ci_formatted} \\vspace{{0.2ex}}"
            f"\\end{{minipage}}"
            f"\\end{{tabular}}"
        )
    else:
        return (
            f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}"
            f"\\begin{{minipage}}[c][4em]{{2.3cm}}\\centering"
            f"{mean} \\\\ {median} \\\\ {ci_formatted} \\vspace{{0.2ex}}"
            f"\\end{{minipage}}"
            f"\\end{{tabular}}"
        )


def generate_organ_row(organ, metric_1_row, metric_2_row, metric_data_1_dict, metric_data_2_dict):
    show_percentage_1 = metric_data_1_dict.get('shown_in_percentage', False)
    show_percentage_2 = metric_data_2_dict.get('shown_in_percentage', False)

    better_model_1 = metric_1_row.get('all Better Model', None)
    better_model_2 = metric_2_row.get('all Better Model', None)

    metric_1_omaseg_cell = generate_metric_cell(
        metric_1_row, 'CADS', show_percentage_1, better_model_1)
    metric_1_totalseg_cell = generate_metric_cell(
        metric_1_row, 'TotalSeg', show_percentage_1, better_model_1)
    metric_2_omaseg_cell = generate_metric_cell(
        metric_2_row, 'CADS', show_percentage_2, better_model_2)
    metric_2_totalseg_cell = generate_metric_cell(
        metric_2_row, 'TotalSeg', show_percentage_2, better_model_2)

    return (
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\parbox[c]{{1.8cm}}{{\\centering {organ}}} \\end{{tabular}} & "
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} {metric_1_omaseg_cell} \\end{{tabular}} & "
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} {metric_1_totalseg_cell} \\end{{tabular}} & "
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} {metric_2_omaseg_cell} \\end{{tabular}} & "
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} {metric_2_totalseg_cell} \\end{{tabular}} \\\\"
    )


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
        "\definecolor{MyGreen}{rgb}{0.133, 0.545, 0.133}",
        "{\\small",
        "\\begin{longtable}{|c|c|c|c|c|}",
        "\\caption{Comparison of CADS and TotalSeg performance}",
        "\label{tab:model_comparison} \\\\",
        "\\hline",
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\parbox[c]{{2cm}}{{\\centering \\textbf{{Organ}}}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{CADS}} \\\\ \\textbf{{({metric_name_1})}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{TotalSeg}} \\\\ \\textbf{{({metric_name_1})}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{CADS}} \\\\ \\textbf{{({metric_name_2})}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{TotalSeg}} \\\\ \\textbf{{({metric_name_2})}} \\end{{tabular}} \\\\",
        "\\hline",
        "\\endfirsthead",
        "\\hline",
        f"\\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\parbox[c]{{2cm}}{{\\centering \\textbf{{Organ}}}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{CADS}} \\\\ \\textbf{{({metric_name_1})}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{TotalSeg}} \\\\ \\textbf{{({metric_name_1})}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{CADS}} \\\\ \\textbf{{({metric_name_2})}} \\end{{tabular}} & \\begin{{tabular}}[m]{{@{{}}c@{{}}}} \\textbf{{TotalSeg}} \\\\ \\textbf{{({metric_name_2})}} \\end{{tabular}} \\\\",
        "\\hline",
        "\\endhead",
        "\\hline",
        "\\endfoot",
        "\\hline",
        "\\endlastfoot",
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
                    f"\\multicolumn{{5}}{{|c|}}{{\\rule{{0pt}}{{2.5ex}}\\textbf{{{system}}}}} \\\\")
                latex_code.append("\\hline")

                for organ in system_grouped_data[system]:
                    metric_1_row = df_1[df_1['Organ'] == organ].iloc[0]
                    metric_2_row = df_2[df_2['Organ'] == organ].iloc[0]

                    latex_row = generate_organ_row(organ, metric_1_row, metric_2_row,
                                                   metric_data_1_dict, metric_data_2_dict)
                    latex_code.append(latex_row)
                    latex_code.append("\\hline")

    else:
        # plain order
        for organ in df_1['Organ']:
            metric_1_row = df_1[df_1['Organ'] == organ].iloc[0]
            metric_2_row = df_2[df_2['Organ'] == organ].iloc[0]

            latex_row = generate_organ_row(organ, metric_1_row, metric_2_row,
                                           metric_data_1_dict, metric_data_2_dict)
            latex_code.append(latex_row)
            latex_code.append("\\hline")

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
