import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
from scikit_posthocs import posthoc_dunn

from dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps


def filter_rows(dataframe, splits=['test']):
    split_dict = {'test': 0, 'train': 1, 'val': 2}
    splits = [split_dict.get(split, 0) for split in splits]
    dataframe = dataframe.drop(dataframe.index[-1])  # drop the last avg row
    mask = (dataframe != -1) & (dataframe != -2) & (dataframe != -3)
    dataframe = dataframe.where(mask, np.nan)

    # Replace numeric values greater than 1000 with NaN, BUT can't do this, we have FN penalizations
    # numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    # dataframe[numeric_cols] = dataframe[numeric_cols].where(dataframe[numeric_cols] <= 1000, np.nan)

    test_samples = dataframe[dataframe['split'].isin(splits)]
    test_samples = test_samples.dropna(axis=1, how='all')
    test_samples = test_samples.drop(columns=['ids', 'split'], axis=1)
    return test_samples

def align_and_filter_scores(scores_1, scores_2):
    """
    Ensure that the two score lists have the same length by simultaneously removing corresponding NaN values from both lists.
    """
    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)
    
    nan_1 = np.isnan(scores_1)
    nan_2 = np.isnan(scores_2)
    
    valid_indices = (~nan_1) & (~nan_2)
    
    return scores_1[valid_indices].tolist(), scores_2[valid_indices].tolist()

def list_specific_files(directory, prefix='dice', suffix='.xlsx'):
    specific_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.startswith(prefix) and f.endswith(suffix)]:
            specific_files.append(os.path.join(dirpath, filename))
    specific_files.sort()
    return specific_files


def filter_files(paths, keyword, include: bool):
    if include:
        return [path for path in paths if keyword in path]
    else:
        return [path for path in paths if keyword not in path]


def get_task_id(structure):
    for task_id, labelmap in map_taskid_to_labelmaps.items():
        if structure in labelmap.values():
            return task_id
    return None 


def compare_models(model1_results, model2_results, model1_name, model2_name, stat_test_method='wilcoxon_test_median'):
    combined_results = []

    for organ, scores in model1_results.items():
        model1_scores = model1_results[organ]
        model2_scores = model2_results[organ]

        original_length = len(model1_scores)
        aligned_model1, aligned_model2 = align_and_filter_scores(model1_scores, model2_scores)
        new_length = len(aligned_model1)

        removed_points = original_length - new_length

        if stat_test_method == 'wilcoxon_test_median':
            stat, p, pos_diff, neg_diff, better_model = wilcoxon_test_median(aligned_model1, aligned_model2)
        elif stat_test_method == 'wilcoxon_test':
            stat, p, pos_diff, neg_diff, better_model = wilcoxon_test(aligned_model1, aligned_model2)
        elif stat_test_method == 'paired_t_test':
            stat, p, pos_diff, neg_diff, better_model = paired_t_test(aligned_model1, aligned_model2)
        else:
            raise ValueError(f"Unsupported statistical test method: {stat_test_method}")

        combined_results.append({
            'Organ': organ,
            f'{model1_name} mean±std': f"{np.mean(aligned_model1):.2f}±{np.std(aligned_model1):.2f}",
            f'{model1_name} median': np.median(aligned_model1),
            f'{model2_name} mean±std': f"{np.mean(aligned_model2):.2f}±{np.std(aligned_model2):.2f}",
            f'{model2_name} median': np.median(aligned_model2),
            'p_value': p,
            'Positive Differences': pos_diff,
            'Negative Differences': neg_diff,
            'better_model': better_model,
        })
    
    combined_results_df = pd.DataFrame(combined_results)
    return combined_results_df


def bootstrap_ci(data, n_iterations=10000, ci=0.95):
    n_samples = len(data)
    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound


def wilcoxon_test(scores_1, scores_2, p_value=0.05):
    if len(scores_1) != len(scores_2):
        return None, None, None, None, "Unequal sample sizes"
    
    if len(scores_1) == 0:
        return None, None, None, None, "No data"
    
    differences = np.array(scores_1) - np.array(scores_2)
    positive_differences = np.sum(differences > 0)
    negative_differences = np.sum(differences < 0)
    
    try:
        stat, p = stats.wilcoxon(scores_1, scores_2)
        N = len(scores_1)
        Z = (stat - (N * (N + 1) / 4)) / np.sqrt((N * (N + 1) * (2 * N + 1)) / 24)
        r = Z / np.sqrt(N)

    except ValueError:
        return None, None, None, None, "Wilcoxon test failed"

    if p < p_value:
        
        if positive_differences > negative_differences:
            better_model = "OMASeg"
        elif positive_differences < negative_differences:
            better_model = "TotalSeg"
        else:
            better_model = "Unknown"
    else:
        better_model = None

    return stat, p, positive_differences, negative_differences, better_model

def wilcoxon_test_median(scores_1, scores_2, p_value=0.05):
    if len(scores_1) != len(scores_2):
        return None, None, None, None, "Unequal sample sizes"
    
    if len(scores_1) == 0:
        return None, None, None, None, "No data"
        
    try:
        stat, p = stats.wilcoxon(scores_1, scores_2)
        N = len(scores_1)
        Z = (stat - (N * (N + 1) / 4)) / np.sqrt((N * (N + 1) * (2 * N + 1)) / 24)
        r = Z / np.sqrt(N)

    except ValueError:
        return None, None, None, None, "Wilcoxon test failed"
    
    if p < p_value:
        if np.median(scores_1) > np.median(scores_2):
            better_model = "OMASeg"
        elif np.median(scores_1) < np.median(scores_2):
            better_model = "TotalSeg"
        else:
            better_model = "Unknown"
    else:
        better_model = None
    positive_differences = np.sum(np.array(scores_1) > np.array(scores_2))
    negative_differences = np.sum(np.array(scores_1) < np.array(scores_2))

    return stat, p, positive_differences, negative_differences, better_model

def paired_t_test(scores_1, scores_2,  p_value=0.05):
    if len(scores_1) != len(scores_2):
        return None, None, None, None, "Unequal sample sizes"
    
    if len(scores_1) == 0:
        return None, None, None, None, "No data"
    
    try:
        t_statistic, p = stats.ttest_rel(scores_1, scores_2)
    except ValueError:
        return None, None, None, None, "Paired T-test failed"

    if p < p_value:
        if t_statistic > 0:
            better_model = "OMASeg"
        elif t_statistic < 0:
            better_model = "TotalSeg"
        else:
            better_model = "Unknown"
    else:
        better_model = None
        t_statistic = None
    positive_differences = np.sum(np.array(scores_1) > np.array(scores_2))
    negative_differences = np.sum(np.array(scores_1) < np.array(scores_2))
    
    return t_statistic, p, positive_differences, negative_differences, better_model


def statistical_test(data, column_names, csv_filename, significance_level=0.05):  # TODO: 
    """
    Perform paired 2-sided T-test on 3x models' evaluation scores, save results to a CSV file.

    Parameters:
    data: List of dataframes, list of dictionaries, or dict of dictionaries each containing scores for different models.
    column_names: List of class names to perform tests on.
    csv_filename: The path to the CSV file where results are saved.
    significance_level: The significance level for statistical tests. Default is 0.05.

    Returns:
    - DataFrame with significant comparisons and p-values.
    """
    model_mapping = {1: 'GT', 2: 'Pseudo', 3: 'Shape'}

    csv = pd.read_excel(csv_filename)
    if 'Significant Comparisons' not in csv.columns:
        csv['Significant Comparisons'] = pd.NA
        csv['P-Values'] = pd.NA

    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        combined_data = pd.DataFrame()
        for model_idx, (model_name, model_data) in enumerate(data.items()):
            df = pd.DataFrame.from_dict(model_data, orient='index').transpose()
            df.columns = pd.MultiIndex.from_product(
                [[model_mapping[model_idx + 1]], df.columns])
            combined_data = pd.concat([combined_data, df], axis=1)
    else:
        combined_data = pd.concat(
            data, axis=1, keys=[model_mapping[i + 1] for i in range(len(data))])

    for i, col in enumerate(column_names):
        scores_dict = {}
        columns_to_check = [(model_mapping[i+1], col)
                            for i in range(len(data))]
        clean_data = combined_data.dropna(subset=columns_to_check)
        for idx in range(len(data)):
            model_name = model_mapping[idx+1]
            scores_dict[model_name] = clean_data[model_name].to_numpy()[:, i]

        df = pd.DataFrame(scores_dict)

        significant_pairs = set()
        model_pairs = [('GT', 'Pseudo'), ('GT', 'Shape'), ('Pseudo', 'Shape')]
        for model1, model2 in model_pairs:
            t_stat, p_value = stats.ttest_rel(df[model1], df[model2])
            if p_value < significance_level:
                significant_pairs.add(((model1, model2), p_value))

        if significant_pairs:
            significant_comparisons = [
                f"{pair[0][0]} vs {pair[0][1]}" for pair in significant_pairs]
            p_values = [pair[1] for pair in significant_pairs]
            csv.at[i, 'Significant Comparisons'] = significant_comparisons
            csv.at[i, 'P-Values'] = p_values

    return csv[['Significant Comparisons', 'P-Values']]


def statistical_test_upgrade(data, column_names, csv_filename, significance_level=0.05): # TODO: 
    """
    Perform statistical tests on 3x models' evaluation scores, save results to a CSV file.

    Parameters:
    data: List of dataframes, list of dictionaries, or dict of dictionaries each containing scores for different models.
    column_names: List of class names to perform tests on.
    csv_filename: The path to the CSV file where results are saved.
    significance_level: The significance level for statistical tests. Default is 0.05.

    Description:
    - The function performs the Shapiro-Wilk test for normality and Levene's test for equality of variances.
    - Depending on the normality and variance results, it chooses between ANOVA, Welch's ANOVA, and Kruskal-Wallis test.
    - For ANOVA and Welch's ANOVA, it uses Tukey's HSD for post-hoc comparisons.
    - For the Kruskal-Wallis test, it uses Dunn's test for post-hoc comparisons.
    - Significant comparisons and their p-values are saved to the specified CSV file.

    Returns:
    - DataFrame with significant comparisons and p-values.
    """
    model_mapping = {1: 'GT', 2: 'Pseudo', 3: 'Shape'}

    csv = pd.read_excel(csv_filename)
    if 'Significant Comparisons' not in csv.columns:
        csv['Significant Comparisons'] = pd.NA
        csv['P-Values'] = pd.NA
    if 'Statistical Test Used' not in csv.columns:
        csv['Statistical Test Used'] = pd.NA

    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        combined_data = pd.DataFrame()
        for model_idx, (model_name, model_data) in enumerate(data.items()):
            df = pd.DataFrame.from_dict(model_data, orient='index').transpose()
            df.columns = pd.MultiIndex.from_product(
                [[model_mapping[model_idx + 1]], df.columns])
            combined_data = pd.concat([combined_data, df], axis=1)
    else:
        combined_data = pd.concat(
            data, axis=1, keys=[model_mapping[i + 1] for i in range(len(data))])

    for i, col in enumerate(column_names):
        scores_dict = {}
        columns_to_check = [(model_mapping[i+1], col)
                            for i in range(len(data))]
        clean_data = combined_data.dropna(subset=columns_to_check)
        for idx in range(len(data)):
            model_name = model_mapping[idx+1]
            scores_dict[model_name] = clean_data[model_name].to_numpy()[:, i]

        df = pd.DataFrame(scores_dict)
        df_melted = df.melt(var_name='Model', value_name='Score')

        # Shapiro-Wilk Test for Normality
        normality = True
        for model, subset in df_melted.groupby('Model'):
            _, p = stats.shapiro(subset['Score'])
            if p < significance_level:
                normality = False  # if any model's results not passed noramlity test -> the entire results not sufficient in a valid normality

        # Levene test for equality of variances
        _, p_value_levene = stats.levene(
            *[df[model] for model in scores_dict.keys()])

        significant_pairs = set()
        # Select an approriate statistical test
        if normality:
            if p_value_levene > significance_level:
                # Use ANOVA
                model = ols('Score ~ C(Model)', data=df_melted).fit()
                anova_results = sm.stats.anova_lm(model, typ=2)
                if anova_results['PR(>F)'].iloc[0] < significance_level:
                    tukey_results = pairwise_tukeyhsd(
                        endog=df_melted['Score'], groups=df_melted['Model'], alpha=significance_level)
                    significant_pairs_tukey = {(tuple(sorted([pair[0], pair[1]])), pair[3])
                                               for pair in tukey_results.summary().data[1:] if pair[-1]}
                    significant_pairs.update(significant_pairs_tukey)
                    csv.at[i, 'Statistical Test Used'] = 'ANOVA'
            else:
                # Use Welch's ANOVA
                welch_results = stats.f_oneway(
                    df['GT'], df['Pseudo'], df['Shape'])
                if welch_results.pvalue < significance_level:
                    tukey_results = pairwise_tukeyhsd(
                        endog=df_melted['Score'], groups=df_melted['Model'], alpha=significance_level)
                    significant_pairs_tukey = {(tuple(sorted([pair[0], pair[1]])), pair[3])
                                               for pair in tukey_results.summary().data[1:] if pair[-1]}
                    significant_pairs.update(significant_pairs_tukey)
                    csv.at[i, 'Statistical Test Used'] = 'Welch ANOVA'
        else:
            # Use Kruskal-Wallis test
            kruskal_result = stats.kruskal(df['GT'], df['Pseudo'], df['Shape'])
            if kruskal_result.pvalue < significance_level:
                dunn_results = posthoc_dunn(
                    [df['GT'], df['Pseudo'], df['Shape']], p_adjust='bonferroni')
                significant_pairs_dunn = {(tuple(sorted([model_mapping[row], model_mapping[col]])), dunn_results.iloc[i, j])
                                          for i, row in enumerate(dunn_results.index)
                                          for j, col in enumerate(dunn_results.columns)
                                          if dunn_results.iloc[i, j] < significance_level}
                significant_pairs.update(significant_pairs_dunn)
                csv.at[i, 'Statistical Test Used'] = 'Kruskal-Wallis'

        if significant_pairs:
            significant_comparisons = [
                f"{pair[0][0]} vs {pair[0][1]}" for pair in significant_pairs]
            p_values = [pair[1] for pair in significant_pairs]
            csv.at[i, 'Significant Comparisons'] = significant_comparisons
            csv.at[i, 'P-Values'] = p_values

    return csv[['Significant Comparisons', 'P-Values', 'Statistical Test Used']]

def generate_boxplot(dataset, prefix, postfix, data, all_scores, experiment_to_name_dict, columns_names, output_directory):
    folder_plots = os.path.join(output_directory, 'plots')
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots, exist_ok=True)

    colors = ['red', 'blue', 'green'] #TODO: 
    # plot param (manual, optional)
    limit_low_dice, limit_high_dice = 0, 1
    limit_low_NSD, limit_high_NSD = 0, 17777777
    limit_low_HD, limit_high_HD = 0, 1000
    limit_low_HD95, limit_high_HD95 = 0, 1000

    # Determine y-axis limits dynamically
    all_scores = np.array(all_scores, dtype=float)
    all_scores = all_scores[~np.isnan(all_scores)]
    if isinstance(all_scores, np.ndarray) and all_scores.size == 0:
        return
    else:
        y_min = np.percentile(all_scores, 1)
        y_max = np.percentile(all_scores, 99)
        y_margin = (y_max - y_min) * 0.1
        y_min -= y_margin
        y_max += y_margin

        # Determine figure size dynamically
        num_columns = len(columns_names)
        fig_width = max(10, num_columns * 0.5)
        fig_height = 10  # Fixed height

        plt.cla()
        plt.clf()
        plt.close()
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        counter = 0
        offset = 0.2

        for i, col in enumerate(columns_names):
            for j in range(len(data)):
                aux = data[j].to_numpy()
                score = aux[:, i]
                score = np.asarray(score, dtype=float)
                # after removing NaNs: all results of a particular structure under a specific model
                score = score[~np.isnan(score)]

                color = colors[j]
                # change the color of the line around the box
                boxprops = dict(linewidth=1, color='black')

                boxplot = ax.boxplot(score, notch=False, positions=[
                    counter], widths=offset/2.0, patch_artist=True, boxprops=boxprops, vert=True, showfliers=False)
                boxplot['medians'][0].set_color('black')
                for patch in boxplot['boxes']:
                    patch.set_facecolor('None')

                x = np.random.normal(counter, 0.02, size=len(score))
                plt.scatter(x, score, c=color, alpha=0.5, s=10)

                counter += offset
            counter += 2*offset

        ax.set_xlim(-offset/2.0, counter-offset/2.0)
        ax.set_ylim(y_min, y_max)
        if prefix == 'dice':
            plt.ylabel('Dice')
            # ax.set_ylim(limit_low_dice, limit_high_dice)
        elif prefix == 'hd95':
            plt.ylabel('HD95')
            # ax.set_ylim(limit_low_HD95, limit_high_HD95)
        elif prefix == 'hd_score':
            plt.ylabel('HD')
            # ax.set_ylim(limit_low_HD, limit_high_HD)
        elif prefix == 'normalized_distance':
            plt.ylabel('Normalized Surface Distance')
            # ax.set_ylim(limit_low_NSD, limit_high_NSD)

        x_tick_locations = np.arange(0, counter, counter/len(columns_names))
        x_tick_labels = columns_names

        ax.set_xticks(x_tick_locations)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=45, ha='right')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color)
                          for color in colors]
        legend_labels = [str(val) for val in experiment_to_name_dict.values()]
        plt.legend(legend_handles, legend_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

        plt.title(f"{dataset} ({postfix}):  {prefix}", fontsize=12)
        plt.grid(True, linestyle='dashed', linewidth=1.5, color='lightgray')
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        filename = os.path.join(
            folder_plots, f'boxplot_{prefix}_{dataset}_{postfix}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=200)


def generate_violinplot(dataset, prefix, postfix, data, all_scores, experiment_to_name_dict, columns_names, output_directory):
    folder_plots = os.path.join(output_directory, 'plots')
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots, exist_ok=True)

    colors = ['red', 'blue', 'green'] #TODO: 
    # plot param (manual, optional)
    limit_low_dice, limit_high_dice = 0, 1
    limit_low_NSD, limit_high_NSD = 0, 17777777
    limit_low_HD, limit_high_HD = 0, 1000
    limit_low_HD95, limit_high_HD95 = 0, 1000

    # Determine y-axis limits dynamically
    all_scores = np.array(all_scores, dtype=float)
    all_scores = all_scores[~np.isnan(all_scores)]
    if isinstance(all_scores, np.ndarray) and all_scores.size == 0:
        return
    else:
        y_min = np.percentile(all_scores, 1)
        y_max = np.percentile(all_scores, 99)
        y_margin = (y_max - y_min) * 0.1
        y_min -= y_margin
        y_max += y_margin

        # Determine figure size dynamically
        num_columns = len(columns_names)
        fig_width = max(10, num_columns * 0.5)
        fig_height = 10  # Fixed height

        plt.cla()
        plt.clf()
        plt.close()
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        counter = 0
        offset = 0.2

        for i, col in enumerate(columns_names):
            for j in range(len(data)):
                aux = data[j].to_numpy()
                score = aux[:, i]
                score = np.asarray(score, dtype=float)
                # after removing NaNs: all results of a particular structure under a specific model
                score = score[~np.isnan(score)]

                color = colors[j]

                if isinstance(score, np.ndarray) and score.size == 0:
                    counter += offset
                    continue
                else:
                    violin_parts = ax.violinplot(score, showmedians=True, widths=0.3, positions=[
                        counter], showmeans=True, showextrema=True)
                    # Customize characteristics for each group
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor(color)
                        pc.set_alpha(0.4)
                        pc.set_linewidth(1)
                        # pc.set_edgecolor('black')

                    violin_parts['cmedians'].set_edgecolor('black')
                    violin_parts['cmeans'].set_edgecolor('red')
                    violin_parts['cmedians'].set_linewidth(1)
                    violin_parts['cmeans'].set_linewidth(1)

                    for partname in ('cbars', 'cmins', 'cmaxes'):
                        vp = violin_parts[partname]
                        vp.set_edgecolor("black")
                        vp.set_linewidth(1)

                    counter += offset
            counter += 2*offset

        ax.set_xlim(-offset/2.0, counter-offset/2.0)
        ax.set_ylim(y_min, y_max)
        if prefix == 'dice':
            plt.ylabel('Dice')
            # ax.set_ylim(limit_low_dice, limit_high_dice)
        elif prefix == 'hd95':
            plt.ylabel('HD95')
            # ax.set_ylim(limit_low_HD95, limit_high_HD95)
        elif prefix == 'hd_score':
            plt.ylabel('HD')
            # ax.set_ylim(limit_low_HD, limit_high_HD)
        elif prefix == 'normalized_distance':
            plt.ylabel('Normalized Surface Distance')
            # ax.set_ylim(limit_low_NSD, limit_high_NSD)

        x_tick_locations = np.arange(0, counter, counter/len(columns_names))
        x_tick_labels = columns_names

        ax.set_xticks(x_tick_locations)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=45, ha='right')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color)
                          for color in colors]
        legend_labels = [str(val) for val in experiment_to_name_dict.values()]
        plt.legend(legend_handles, legend_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

        plt.title(f"{dataset} ({postfix}):  {prefix}", fontsize=12)
        plt.grid(True, linestyle='dashed', linewidth=1.5, color='lightgray')
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        filename = os.path.join(
            folder_plots, f'violinplot_{prefix}_{dataset}_{postfix}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=200)


def generate_combined_plot(dataset, prefix, postfix, data, all_scores, experiment_to_name_dict, columns_names, output_directory):
    folder_plots = os.path.join(output_directory, 'plots')
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots, exist_ok=True)

    colors = ['red', 'blue', 'green'] #TODO: 
    # plot param (manual, optional)
    limit_low_dice, limit_high_dice = 0, 1
    limit_low_NSD, limit_high_NSD = 0, 17777777
    limit_low_HD, limit_high_HD = 0, 1000
    limit_low_HD95, limit_high_HD95 = 0, 1000

    # Determine y-axis limits dynamically
    all_scores = np.array(all_scores, dtype=float)
    all_scores = all_scores[~np.isnan(all_scores)]
    if isinstance(all_scores, np.ndarray) and all_scores.size == 0:
        return
    else:
        y_min = np.percentile(all_scores, 1)
        y_max = np.percentile(all_scores, 99)
        y_margin = (y_max - y_min) * 0.1
        y_min -= y_margin
        y_max += y_margin

        # Determine figure size dynamically
        num_columns = len(columns_names)
        fig_width = max(10, num_columns * 0.5)
        fig_height = 10  # Fixed height

        plt.cla()
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height))
        offset = 0.2

        # Boxplot
        ax = axs[0]
        counter = 0
        for i, col in enumerate(columns_names):
            for j in range(len(data)):
                aux = data[j].to_numpy()
                score = aux[:, i]
                score = np.asarray(score, dtype=float)
                # after removing NaNs: all results of a particular structure under a specific model
                score = score[~np.isnan(score)]

                color = colors[j]
                # change the color of the line around the box
                boxprops = dict(linewidth=1, color='black')

                boxplot = ax.boxplot(score, notch=False, positions=[
                    counter], widths=offset/2.0, patch_artist=True, boxprops=boxprops, vert=True, showfliers=False)
                boxplot['medians'][0].set_color('black')
                for patch in boxplot['boxes']:
                    patch.set_facecolor('None')

                x = np.random.normal(counter, 0.02, size=len(score))
                ax.scatter(x, score, c=color, alpha=0.5, s=10)

                counter += offset
            counter += 2*offset

        ax.set_xlim(-offset/2.0, counter-offset/2.0)
        ax.set_ylim(y_min, y_max)
        if prefix == 'dice':
            ax.set_ylabel('Dice')
            # ax.set_ylim(limit_low_dice, limit_high_dice)
        elif prefix == 'hd95':
            ax.set_ylabel('HD95')
            # ax.set_ylim(limit_low_HD95, limit_high_HD95)
        elif prefix == 'hd_score':
            ax.set_ylabel('HD')
            # ax.set_ylim(limit_low_HD, limit_high_HD)
        elif prefix == 'normalized_distance':
            ax.set_ylabel('Normalized Surface Distance')
            # ax.set_ylim(limit_low_NSD, limit_high_NSD)

        x_tick_locations = np.arange(0, counter, counter/len(columns_names))
        x_tick_labels = columns_names
        ax.set_xticks(x_tick_locations)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=45, ha='right')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('Box plot')
        ax.grid(True, linestyle='dashed', linewidth=1.5, color='lightgray')

        # Violinplot
        ax = axs[1]
        counter = 0
        for i, col in enumerate(columns_names):
            for j in range(len(data)):
                aux = data[j].to_numpy()
                score = aux[:, i]
                score = np.asarray(score, dtype=float)
                # after removing NaNs: all results of a particular structure under a specific model
                score = score[~np.isnan(score)]

                color = colors[j]

                if isinstance(score, np.ndarray) and score.size == 0:
                    counter += offset
                    continue
                else:
                    violin_parts = ax.violinplot(score, showmedians=True, widths=0.3, positions=[
                        counter], showmeans=True, showextrema=True)
                    # Customize characteristics for each group
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor(color)
                        pc.set_alpha(0.4)
                        pc.set_linewidth(1)
                        # pc.set_edgecolor('black')

                    violin_parts['cmedians'].set_edgecolor('black')
                    violin_parts['cmeans'].set_edgecolor('red')
                    violin_parts['cmedians'].set_linewidth(1)
                    violin_parts['cmeans'].set_linewidth(1)

                    for partname in ('cbars', 'cmins', 'cmaxes'):
                        vp = violin_parts[partname]
                        vp.set_edgecolor("black")
                        vp.set_linewidth(1)

                    counter += offset
            counter += 2*offset

        ax.set_xlim(-offset/2.0, counter-offset/2.0)
        ax.set_ylim(y_min, y_max)
        if prefix == 'dice':
            ax.set_ylabel('Dice')
            # ax.set_ylim(limit_low_dice, limit_high_dice)
        elif prefix == 'hd95':
            ax.set_ylabel('HD95')
            # ax.set_ylim(limit_low_HD95, limit_high_HD95)
        elif prefix == 'hd_score':
            ax.set_ylabel('HD')
            # ax.set_ylim(limit_low_HD, limit_high_HD)
        elif prefix == 'normalized_distance':
            ax.set_ylabel('Normalized Surface Distance')
            # ax.set_ylim(limit_low_NSD, limit_high_NSD)

        x_tick_locations = np.arange(0, counter, counter/len(columns_names))
        x_tick_labels = columns_names
        ax.set_xticks(x_tick_locations)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=45, ha='right')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('Violin plot')
        ax.grid(True, linestyle='dashed', linewidth=1.5, color='lightgray')

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color)
                          for color in colors]
        legend_labels = [str(val) for val in experiment_to_name_dict.values()]
        axs[0].legend(legend_handles, legend_labels, loc='upper center',
                      bbox_to_anchor=(0.9, 1.15), ncol=3, fontsize=8)
        axs[1].legend(legend_handles, legend_labels, loc='upper center',
                      bbox_to_anchor=(0.9, 1.15), ncol=3, fontsize=8)

        plt.suptitle(f"{dataset} ({postfix}):  {prefix}", fontsize=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.96])
        filename = os.path.join(
            folder_plots, f'combined_plot_{prefix}_{dataset}_{postfix}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        plt.close()


# Exclude results containing transitional vertebrae
transitional_ids = ["sub-verse131_ct", "sub-verse530_ct", "sub-verse542_dir-iso_ct", "sub-verse565_dir-ax_ct", "sub-verse576_ct",
                    "sub-verse577_dir-ax_ct", "sub-verse588_ct", "sub-verse604_dir-iso_ct", "sub-verse607_dir-sag_ct", "sub-verse612_ct", "sub-verse810_dir-iso_ct"]

totalseg_out_dist_structures = ['heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 'heart_atrium_right', 'heart_ventricle_right', 'pulmonary_artery', 'face', 'larynx', 'OAR_Brainstem', 'OAR_Glnd_Submand_L', 'OAR_Glnd_Submand_R', 'OAR_OpticNrv_L', 'OAR_OpticNrv_R', 'OAR_Parotid_L', 'OAR_Parotid_R']