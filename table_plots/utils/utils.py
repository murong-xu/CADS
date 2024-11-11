import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
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


def compare_models_stat_test(model1_results, model2_results, model1_name, model2_name, stat_test_method='paired_t_test', p_value=0.05, higher_better=True):
    combined_results = []

    for organ, scores in model1_results.items():
        model1_scores = model1_results[organ]
        model2_scores = model2_results[organ]

        original_length = len(model1_scores)
        aligned_model1, aligned_model2 = align_and_filter_scores(model1_scores, model2_scores)
        new_length = len(aligned_model1)

        removed_points = original_length - new_length

        if stat_test_method == 'wilcoxon_test_median':
            stat, p, pos_diff, neg_diff, better_model = wilcoxon_test_median(aligned_model1, aligned_model2, model1_name, model2_name, p_value=p_value, higher_better=higher_better)
        elif stat_test_method == 'wilcoxon_test':
            stat, p, pos_diff, neg_diff, better_model = wilcoxon_test(aligned_model1, aligned_model2, model1_name, model2_name, p_value=p_value, higher_better=higher_better)
        elif stat_test_method == 'paired_t_test':
            stat, p, pos_diff, neg_diff, better_model = paired_t_test(aligned_model1, aligned_model2, model1_name, model2_name, p_value=p_value, higher_better=higher_better)
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


def wilcoxon_test(scores_1, scores_2, model_1, model_2, p_value=0.05, higher_better=True):
    """
    Performs Wilcoxon signed-rank test to compare two models' scores
    
    Args:
        scores_1: Scores from model_1 
        scores_2: Scores from model_2
        model_1: Name of the first model
        model_2: Name of the second model
        p_value: Significance threshold (default: 0.05)
        higher_better: Whether higher scores are better (True for metrics like Dice, False for HD) (default: True)
        
    Returns:
        stat: The test statistic
        p: The p-value from the test
        positive_differences: Number of cases where model_1 > model_2
        negative_differences: Number of cases where model_1 < model_2
        better_model: Name of significantly better model (or None if no significant difference)
    """
    if len(scores_1) != len(scores_2):
        return None, None, None, None, "Unequal sample sizes"
    
    if len(scores_1) == 0:
        return None, None, None, None, "No data"
    
    # Calculate differences
    differences = np.array(scores_1) - np.array(scores_2)
    if not higher_better:
        # For metrics where lower is better, invert the differences
        differences = -differences
    
    # Count simple differences (for additional information)
    positive_differences = np.sum(differences > 0)
    negative_differences = np.sum(differences < 0)

    try:
        # The test statistic is based on the ranks of the absolute differences
        stat, p = stats.wilcoxon(scores_1, scores_2, alternative='two-sided')

        # Determine better model based on the sign of the differences and their ranks
        if p < p_value:
            # Calculate rank sums for positive and negative differences
            abs_diff = np.abs(differences)
            ranks = stats.rankdata(abs_diff)
            signed_ranks = ranks * np.sign(differences)
            
            positive_rank_sum = np.sum(signed_ranks[signed_ranks > 0])
            negative_rank_sum = abs(np.sum(signed_ranks[signed_ranks < 0]))
            
            if positive_rank_sum > negative_rank_sum:
                better_model = model_1
            elif positive_rank_sum < negative_rank_sum:
                better_model = model_2
            else:
                better_model = "Unknown"
        else:
            better_model = None
    except ValueError:
        return None, None, None, None, "Wilcoxon test failed"

    return stat, p, positive_differences, negative_differences, better_model

def wilcoxon_test_median(scores_1, scores_2, model_1, model_2, p_value=0.05, higher_better=True):
    """
    Performs Wilcoxon signed-rank test with median comparison
    
    Args:
        scores_1: Scores from model_1 (main model)
        scores_2: Scores from model_2 (baseline model)
        model_1: Name of the first model
        model_2: Name of the second model
        p_value: Significance threshold (default: 0.05)
        higher_better: Whether higher scores are better (default: True)
        
    Returns:
        stat: The Wilcoxon test statistic
        p: The p-value from the test
        positive_differences: Number of cases where model_1 > model_2
        negative_differences: Number of cases where model_1 < model_2
        better_model: Name of significantly better model (or None if no significant difference)
    """
    if len(scores_1) != len(scores_2):
        return None, None, None, None, "Unequal sample sizes"
    
    if len(scores_1) == 0:
        return None, None, None, None, "No data"
        
    try:
        stat, p = stats.wilcoxon(scores_1, scores_2, alternative='two-sided')
    except ValueError:
        return None, None, None, None, "Wilcoxon test failed"
    
    median_1 = np.median(scores_1)
    median_2 = np.median(scores_2)
    if not higher_better:
        median_1, median_2 = -median_1, -median_2
    
    differences = np.array(scores_1) - np.array(scores_2)
    if not higher_better:
        differences = -differences
    positive_differences = np.sum(differences > 0)
    negative_differences = np.sum(differences < 0)
    
    if p < p_value:
        if median_1 > median_2:
            better_model = model_1
        elif median_1 < median_2:
            better_model = model_2
        else:
            better_model = "Unknown"
    else:
        better_model = None

    return stat, p, positive_differences, negative_differences, better_model

def paired_t_test(scores_1, scores_2, model_1, model_2, p_value=0.05, higher_better=True):
    """
    Performs paired t-test to compare two models' scores
    
    Args:
        scores_1: Scores from model_1 
        scores_2: Scores from model_2
        model_1: Name of the first model
        model_2: Name of the second model
        p_value: Significance threshold (default: 0.05)
        higher_better: Whether higher scores are better (True for metrics like Dice, False for HD) (default: True)
        
    Returns:
        t_statistic: The t-statistic from the test
        p: The p-value from the test
        positive_differences: Number of cases where model_1 > model_2
        negative_differences: Number of cases where model_1 < model_2
        better_model: Name of significantly better model (or None if no significant difference)
    """
    if len(scores_1) != len(scores_2):
        return None, None, None, None, "Unequal sample sizes"
    
    if len(scores_1) == 0:
        return None, None, None, None, "No data"
    
    try:
        t_statistic, p = stats.ttest_rel(scores_1, scores_2)
    except ValueError:
        return None, None, None, None, "Paired T-test failed"

    differences = np.array(scores_1) - np.array(scores_2)
    # For metrics where lower is better (like HD), invert the t_statistic
    if not higher_better:
        t_statistic = -t_statistic
        differences = -differences

    # Calculate differences regardless of significance
    positive_differences = np.sum(differences > 0)
    negative_differences = np.sum(differences < 0)
    
    # Determine better model only if difference is significant
    if p < p_value:
        if t_statistic > 0:
            better_model = model_1
        elif t_statistic < 0:
            better_model = model_2
        else:
            better_model = "Unknown"
    else:
        better_model = None  # No significant difference
    
    return t_statistic, p, positive_differences, negative_differences, better_model

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