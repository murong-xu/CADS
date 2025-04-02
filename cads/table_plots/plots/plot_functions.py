import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cads.table_plots.plots.utils import anatomical_system_colors

# dark blue / orange
# MODEL1_COLOR = "#115d76" 
# MODEL2_COLOR = "#ffa505"
# purple / orange
MODEL1_COLOR = "#871282"
MODEL2_COLOR = "#ffa505"
# purple / green
# MODEL1_COLOR = "#765898"
# MODEL2_COLOR = "#52d053"

BASELINE_COLORS = {
    'baseline1': '#FF1E1E',  # light red
    'baseline2': '#1E90FF',  # light blue
    'baseline3': '#FFD700',  # gold
    'baseline4': '#32CD32',  # light green
    'baseline5': '#FF1493',  # dark pink
    'baseline6': '#00FFFF',  # blue
    'baseline7': '#FF8C00',  # dark orange
    'baseline8': '#8A2BE2',  # purple blue
    'baseline9': '#00FF7F',  # spring green
    'baseline10': '#FF69B4', # pink
    }


def generate_histogram_plot(model1_scores, model2_scores, model1_name, model2_name, 
                          stat_results, output_path, metric_name, system_group, anatomical_systems):
    """
    Generate histogram plot comparing two models' performance for a specific anatomical system.
    NaNs should already be removed from model scroes!!
    """    
    organs = anatomical_systems[system_group]
    system_model1_scores = {}
    system_model2_scores = {}
    
    for organ in organs:
        if organ in model1_scores and organ in model2_scores:
            system_model1_scores[organ] = model1_scores[organ]
            system_model2_scores[organ] = model2_scores[organ]
    
    if not system_model1_scores:
        print(f"No matching organs found for system: {system_group}")
        return
    
    # calc mean and std
    organs = list(system_model2_scores.keys())
    model1_means = []
    model1_stds = []
    model2_means = []
    model2_stds = []
    
    for organ in organs:
        scores1 = np.array(system_model1_scores[organ])
        model1_means.append(np.mean(scores1))
        model1_stds.append(np.std(scores1))
        
        scores2 = np.array(system_model2_scores[organ])
        model2_means.append(np.mean(scores2))
        model2_stds.append(np.std(scores2))
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(organs))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, model1_means, width, yerr=model1_stds,
                    label=model1_name, color=MODEL1_COLOR, alpha=0.3, capsize=5)
    rects2 = plt.bar(x + width/2, model2_means, width, yerr=model2_stds,
                    label=model2_name, color=MODEL2_COLOR, alpha=0.5, capsize=5)
    
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel('Organs', fontsize=12)
    plt.title(f"{system_group}", fontsize=14, pad=20)
    plt.xticks(x, organs, rotation=45, ha='right')
    plt.legend(fontsize=10)
    
    # acquire actual score limits
    all_means = []
    all_stds = []
    
    for organ in organs:
        scores1 = np.array(model1_scores[organ])
        all_means.append(np.mean(scores1))
        all_stds.append(np.std(scores1))
        
        scores2 = np.array(model2_scores[organ])
        all_means.append(np.mean(scores2))
        all_stds.append(np.std(scores2))
    
    max_mean = max(all_means)
    max_std = max(all_stds)
    y_max = max_mean + max_std
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, y_max * 1.1)
    plt.tight_layout()
    y_range = y_max

    # add significance indicators
    for i, organ in enumerate(organs):
        if organ in stat_results:
            result = stat_results[organ]
            if result.get('p') is not None and isinstance(result['p'], (int, float)):
                x = i
                current_max = max(
                    model1_means[i] + model1_stds[i],
                    model2_means[i] + model2_stds[i]
                )
                
                fig = plt.gcf()
                bracket_width = width/2 
                bracket_height = y_range * 0.01
                bracket_color = 'gray'
                line_width = 1
                
                p_text = '**'
                if result['Better Model'] == 'TotalSeg':
                    significance_color = MODEL1_COLOR
                else:
                    significance_color = MODEL2_COLOR
                
                plt.plot([x - bracket_width, x - bracket_width], 
                        [current_max, current_max + bracket_height], 
                        color=bracket_color, lw=line_width, alpha=0.7)
                plt.plot([x + bracket_width, x + bracket_width], 
                        [current_max, current_max + bracket_height], 
                        color=bracket_color, lw=line_width, alpha=0.7)
                plt.plot([x - bracket_width, x + bracket_width], 
                        [current_max + bracket_height, current_max + bracket_height], 
                        color=bracket_color, lw=line_width, alpha=0.7)
                
                plt.text(x, current_max + bracket_height * 1.05, p_text,
                        ha='center', va='bottom', 
                        fontsize=12, 
                        weight='bold', 
                        color=significance_color)
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{system_group.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_boxplot_comparison(model1_scores, model2_scores, model1_name, model2_name, 
                              stat_results, output_path, metric_name="Dice", datasetname=None):
    """
    Generate comparison boxplots for two models with significance annotations.
    """
    data_list = []
    common_organs = sorted(set(model1_scores.keys()) & set(model2_scores.keys()))
    
    # skip all-zero cases (unique targets)
    for organ in common_organs:
        scores1 = model1_scores[organ]
        if np.all(np.array(scores1) == 0):
            scores1 = [None] * len(scores1)
        for score in scores1:
            if score is not None:
                data_list.append({
                    'Organ': organ,
                    metric_name: score,
                    'Model': model1_name
                })
        scores2 = model2_scores[organ]
        if np.all(np.array(scores2) == 0):
            scores2 = [None] * len(scores2)
        for score in scores2:
            if score is not None:
                data_list.append({
                    'Organ': organ,
                    metric_name: score,
                    'Model': model2_name
                })
    df = pd.DataFrame(data_list)
    
    min_width = 4
    width_per_organ = 0.5
    fig_width = max(min_width, len(common_organs) * width_per_organ)
    box_width = min(0.9, 0.9 / min_width * len(common_organs))  # if only a few organs - smaller box
    plt.figure(figsize=(fig_width, 8))
    
    color_dict = {
        model1_name: MODEL1_COLOR,
        model2_name: MODEL2_COLOR
    }
    ax = sns.boxplot(data=df, y=metric_name, x='Organ', hue='Model',
                    palette=color_dict,
                    width=box_width,
                    showfliers=False, 
                    boxprops=dict(alpha=0.6), 
                    medianprops=dict(color="black"), 
                    whiskerprops=dict(alpha=0.6), 
                    capprops=dict(alpha=0.6),)
    num_models = len(df['Model'].unique())
    offset = box_width / num_models
    jitter_width = 0.02
    for i, model in enumerate(df['Model'].unique()):
        model_data = df[df['Model'] == model].copy()
        if i == 0:
            shift = -offset/2
        else:
            shift = offset/2
        base_positions = pd.Categorical(model_data['Organ']).codes
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(model_data))
        model_data['x_position'] = base_positions + shift + jitter
        sns.scatterplot(data=model_data, 
                       y=metric_name, 
                       x='x_position',
                       color=color_dict[model],
                       s=30,
                       alpha=0.4,
                       legend=False)
    
    # decide the ylim
    whisker_mins = []
    whisker_maxs = []
    whisker_dict = {}
    for (organ, model), group in df.groupby(['Organ', 'Model']):
        if organ not in whisker_dict:
            whisker_dict[organ] = {}
        q1 = group[metric_name].quantile(0.25)
        q3 = group[metric_name].quantile(0.75)
        iqr = q3 - q1

        lower_whisker = group[metric_name][group[metric_name] >= q1 - 1.5 * iqr].min()
        upper_whisker = group[metric_name][group[metric_name] <= q3 + 1.5 * iqr].max()
        whisker_mins.append(lower_whisker)
        whisker_maxs.append(upper_whisker)
        whisker_dict[organ][model] = upper_whisker
    
    y_min = min(whisker_mins) if whisker_mins else df[metric_name].min()
    y_max = max(whisker_maxs) if whisker_maxs else df[metric_name].max()
    y_range = y_max - y_min
    
    plt.ylim(y_min - 0.02 * y_range, y_max + 0.05 * y_range)

    # add significance indicators
    for i, organ in enumerate(common_organs):
        if organ in stat_results:
            result = stat_results[organ]
            if result.get('p') is not None and isinstance(result['p'], (int, float)):
                x = i
                current_whisker_max = max(
                    whisker_dict[organ][model1_name],
                    whisker_dict[organ][model2_name]
                )
                fig = plt.gcf()
                fig_width, fig_height = fig.get_size_inches()
                bracket_width = box_width/2
                bracket_height = y_range * 0.01
                bracket_color = 'gray'
                line_width = 1
                
                p_text = '**'
                if result['Better Model'] == 'TotalSeg':
                    significance_color = MODEL1_COLOR
                else:
                    significance_color = MODEL2_COLOR
                plt.plot([x - bracket_width, x - bracket_width], 
                        [current_whisker_max, current_whisker_max + bracket_height], 
                        color=bracket_color, lw=line_width, alpha=0.7)
                plt.plot([x + bracket_width, x + bracket_width], 
                        [current_whisker_max, current_whisker_max + bracket_height], 
                        color=bracket_color, lw=line_width, alpha=0.7)
                plt.plot([x - bracket_width, x + bracket_width], 
                        [current_whisker_max + bracket_height, current_whisker_max + bracket_height], 
                        color=bracket_color, lw=line_width, alpha=0.7)
                plt.text(x, current_whisker_max + bracket_height * 1.05, p_text,
                        ha='center', va='bottom', 
                        fontsize=12, 
                        weight='bold', 
                        color=significance_color)
    
    plt.title(f'{metric_name} Comparison: {datasetname}' if datasetname else f'{metric_name} Comparison',
             pad=20, fontsize=10, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)  
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{datasetname}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

def generate_box_plot(results_dict, metric_name='DSC', output_path=None, title="Structure-wise Performance Distribution",
                     anatomical_systems=None, is_normalized=True):
    """
    Generate a box plot grouped by anatomical systems.
    Args:
        results_dict: Dictionary of structure-wise scores
        anatomical_systems: Dictionary of anatomical system groupings
        is_normalized: whether the input metric is normalized in range [0, 1]
    """
    data_list = []
    stats_dict = {}  # store original scores
    for organ, scores in results_dict.items():
        scores_array = np.array(scores)
        stats_dict[organ] = {
            'median': np.median(scores_array),
            'mean': np.mean(scores_array),
            'std': np.std(scores_array)
        }

    if not is_normalized:
        # better display of distance metrics
        def log_transform(x, min_val=0.1):
            if np.isscalar(x):
                if x == 0:
                    return 0
                elif x < 0:
                    return np.log10(min_val) 
                return np.log10(x + min_val)
            return np.array([log_transform(val, min_val) for val in x])
        
        transformed_dict = {}
        for organ, scores in results_dict.items():
            transformed_dict[organ] = [log_transform(score) for score in scores]
        plot_dict = transformed_dict
    else:
        plot_dict = results_dict
    
    # In each anatomical system: sort by median
    structure_medians = {organ: stats_dict[organ]['median'] for organ, scores in results_dict.items()}
    reverse_sort = is_normalized  # normalized metrics (high->low), distance metrics (low->high)

    for system, structures in anatomical_systems.items():
        valid_structures = [s for s in structures if s in plot_dict]
        sorted_structures = sorted(valid_structures, 
                                 key=lambda x: structure_medians[x],
                                 reverse=reverse_sort)
        for organ in sorted_structures:
            scores = plot_dict[organ]
            n_samples = len(scores)
            for score in scores:
                data_list.append({
                    'Organ': f"{organ} (n={n_samples})",
                    metric_name: score,
                    'System': system
                })
            
    df = pd.DataFrame(data_list)
    plt.figure(figsize=(12, len(results_dict) * 0.3))

    # determine xlim
    whisker_mins = []
    whisker_maxs = []
    for organ, group in df.groupby('Organ'):
        q1 = group[metric_name].quantile(0.25)
        q3 = group[metric_name].quantile(0.75)
        iqr = q3 - q1
        
        lower_whisker = group[metric_name][group[metric_name] >= q1 - 1.5 * iqr].min()
        upper_whisker = group[metric_name][group[metric_name] <= q3 + 1.5 * iqr].max()
        whisker_mins.append(lower_whisker)
        whisker_maxs.append(upper_whisker)
    
    x_min = min(whisker_mins) if whisker_mins else df[metric_name].min()
    x_max = max(whisker_maxs) if whisker_maxs else df[metric_name].max()
    x_range = x_max - x_min
    
    ax = sns.boxplot(data=df, 
                    y='Organ', 
                    x=metric_name,
                    hue='System',
                    palette=anatomical_system_colors,
                    whis=1.5,
                    boxprops=dict(alpha=0.6), 
                    medianprops=dict(color="black"), 
                    whiskerprops=dict(alpha=0.6), 
                    capprops=dict(alpha=0.6),
                    showfliers=False,
                    )
    
    # add scatters (original data)
    sns.stripplot(data=df,
                 y='Organ',
                 x=metric_name,
                 color='black',
                 size=2,
                 alpha=0.3,
                 jitter=0.2)

    # calculate ststistics (median, mean shown in the columns)
    ordered_organs = df['Organ'].unique() 
    stats_data = []
    for organ in ordered_organs:
        organ_name = organ.split(" (n=")[0]
        stats_data.append({
            'Organ': organ,
            'median': stats_dict[organ_name]['median'],
            'mean': stats_dict[organ_name]['mean'],
            'std': stats_dict[organ_name]['std']
        })
    stats = pd.DataFrame(stats_data).set_index('Organ')
    
    right_edge = x_max    
    format_str = '{:.3f}'
    if is_normalized:
        # normalized metrics (0-1)
        text_spacing = 0.12
        final_margin = 0.15
        text_begin = 0.1

        x_min = 0
        x_max = min(1.1, x_max + 0.05 * x_range)
        x_ticks = np.arange(0, 1.2, 0.2)
        ax.set_xticks(x_ticks)

        ax.text(right_edge + text_begin, -0.8,
            'median',
            va='center', ha='center', fontsize=10,
            fontweight='bold', color='black')
        ax.text(right_edge + text_begin + text_spacing, -0.8,
                'mean±std',
                va='center', ha='center', fontsize=10,
                fontweight='bold', color='black')
        
        for i, (organ, _) in enumerate(stats.iterrows()):
            organ_name = organ.split(" (n=")[0]
            stats_data = stats_dict[organ_name]
            
            if np.isnan(stats_data['mean']) or np.isnan(stats_data['std']):
                median_str = "N/A"
                mean_std_str = "N/A"
            else:
                median_str = format_str.format(stats_data['median'])
                mean_std_str = f'{format_str.format(stats_data["mean"])}±{format_str.format(stats_data["std"])}'
            
            ax.text(right_edge + text_begin, i,
                    median_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
            ax.text(right_edge + text_begin + text_spacing, i,
                    mean_std_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
    else:
        # distance metrics
        text_spacing = 0.12 * x_range
        final_margin = 0.15 * x_range
        text_begin = 0.01 * x_range

        def inverse_log_transform(x, min_val=0.1):
            if np.isscalar(x):
                if x == 0:
                    return 0
                elif x == np.log10(min_val):
                    return 0
                return 10**x - min_val
            return np.array([inverse_log_transform(val, min_val) for val in x])
        
        target_values = [0, 1.5, 5, 10, 20, 50, 100]
        ticks = []
        labels = []
        
        for val in target_values:
            tick = log_transform(val)
            if tick <= x_max:
                ticks.append(tick)
                if val < 1:
                    labels.append("0")
                elif val == 1.5:
                    labels.append("1.5")
                else:
                    labels.append(f"{int(val)}")
        
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        ax.text(right_edge + text_begin, -0.8,
            'median',
            va='center', ha='center', fontsize=10,
            fontweight='bold', color='black')
        ax.text(right_edge + text_begin + text_spacing, -0.8,
                'mean±std',
                va='center', ha='center', fontsize=10,
                fontweight='bold', color='black')
        
        for i, (organ, _) in enumerate(stats.iterrows()):
            organ_name = organ.split(" (n=")[0]
            stats_data = stats_dict[organ_name]
            
            if np.isnan(stats_data['mean']) or np.isnan(stats_data['std']):
                median_str = "N/A"
                mean_std_str = "N/A"
            else:
                median_str = format_str.format(stats_data['median'])
                mean_std_str = f'{format_str.format(stats_data["mean"])}±{format_str.format(stats_data["std"])}'
            
            ax.text(right_edge + text_begin, i,
                    median_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
            ax.text(right_edge + text_begin + text_spacing, i,
                    mean_std_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
        
    prev_system = None
    y_coords = []
    for i, (idx, row) in enumerate(df.groupby('Organ').first().iterrows()):
        if prev_system and row['System'] != prev_system:
            plt.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
        prev_system = row['System']
        y_coords.append(i)
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Organ', fontsize=12)
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.legend(title='Anatomical Systems', 
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=10,
              title_fontsize=12)
    
    ax.set_xlim(x_min, right_edge + text_begin + text_spacing + final_margin)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return ax


def generate_box_plot_with_testdata_sources(results_dict, test_datasets_sources_dict, metric_name='DSC', output_path=None, final_margin=0.15,
                                            title="Structure-wise Performance Distribution", anatomical_systems=None, is_normalized=True):
    """
    Generate a box plot with dataset source indicators.
    Args:
        results_dict: Dictionary of structure-wise scores
        test_datasets_sources_dict: Dictionary mapping each structure to its test datasets
        is_normalized: whether the input metric is normalized in range [0, 1]
    """    
    data_list = []
    stats_dict = {}  # store original scores
    for organ, scores in results_dict.items():
        scores_array = np.array(scores)
        stats_dict[organ] = {
            'median': np.median(scores_array),
            'mean': np.mean(scores_array),
            'std': np.std(scores_array)
        }

    if not is_normalized:
        # better display of distance metrics
        def log_transform(x, min_val=0.1):
            if np.isscalar(x):
                if x == 0:
                    return 0
                elif x < 0:
                    return np.log10(min_val) 
                return np.log10(x + min_val)
            return np.array([log_transform(val, min_val) for val in x])
        
        transformed_dict = {}
        for organ, scores in results_dict.items():
            transformed_dict[organ] = [log_transform(score) for score in scores]
        plot_dict = transformed_dict
    else:
        plot_dict = results_dict    
    
    # in each anatomical system: sort by median
    structure_medians = {organ: stats_dict[organ]['median'] for organ, scores in results_dict.items()}
    reverse_sort = is_normalized  # normalized metrics (high->low), distance metrics (low->high)

    for system, structures in anatomical_systems.items():
        valid_structures = [s for s in structures if s in plot_dict]
        sorted_structures = sorted(valid_structures, 
                                 key=lambda x: structure_medians[x],
                                 reverse=reverse_sort)
        for organ in sorted_structures:
            scores = plot_dict[organ]
            n_samples = len(scores)
            for score in scores:
                data_list.append({
                    'Organ': f"{organ} (n={n_samples})",
                    metric_name: score,
                    'System': system
                })
    
    df = pd.DataFrame(data_list)
    
    fig = plt.figure(figsize=(20, len(results_dict) * 0.3))

    # determine xlim
    whisker_mins = []
    whisker_maxs = []
    for organ, group in df.groupby('Organ'):
        q1 = group[metric_name].quantile(0.25)
        q3 = group[metric_name].quantile(0.75)
        iqr = q3 - q1
        
        lower_whisker = group[metric_name][group[metric_name] >= q1 - 1.5 * iqr].min()
        upper_whisker = group[metric_name][group[metric_name] <= q3 + 1.5 * iqr].max()
        whisker_mins.append(lower_whisker)
        whisker_maxs.append(upper_whisker)
    
    x_min = min(whisker_mins) if whisker_mins else df[metric_name].min()
    x_max = max(whisker_maxs) if whisker_maxs else df[metric_name].max()
    x_range = x_max - x_min

    # 3 major parts: 1) box-plot+stat, 2) dataset indicator, 3) legend
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1, 0.3], wspace=0.1)
    
    ax_main = fig.add_subplot(gs[0])      # box plot + statistics
    ax_sets = fig.add_subplot(gs[1])      # datasets
    ax_legend = fig.add_subplot(gs[2])    # legend
    ax_legend.axis('off')
    
    box_plot = sns.boxplot(data=df, 
                y='Organ', 
                x=metric_name,
                hue='System',
                palette=anatomical_system_colors,
                boxprops=dict(alpha=0.6), 
                medianprops=dict(color="black"), 
                whiskerprops=dict(alpha=0.6), 
                capprops=dict(alpha=0.6),
                showfliers=False,
                ax=ax_main)
    sns.stripplot(data=df,
                 y='Organ',
                 x=metric_name,
                 color='black',
                 size=2,
                 alpha=0.3,
                 jitter=0.2,
                 ax=ax_main)
    
    handles, labels = ax_main.get_legend_handles_labels()
    ax_main.get_legend().remove()
    ax_legend.legend(handles, labels,
                    title='Anatomical Systems',
                    loc='upper left',
                    fontsize=10,
                    title_fontsize=12)
    
    # calculate ststistics (median, mean shown in the columns)
    ordered_organs = df['Organ'].unique() 
    stats_data = []
    for organ in ordered_organs:
        organ_name = organ.split(" (n=")[0]
        stats_data.append({
            'Organ': organ,
            'median': stats_dict[organ_name]['median'],
            'mean': stats_dict[organ_name]['mean'],
            'std': stats_dict[organ_name]['std']
        })
    stats = pd.DataFrame(stats_data).set_index('Organ')
    
    ax_main.set_yticks(range(len(ordered_organs)))
    ax_main.set_yticklabels(ordered_organs, fontsize=10, ha='right')
    ax_main.tick_params(axis='y', pad=5)
    
    right_edge = x_max
    format_str = '{:.3f}'

    if is_normalized:
        # normalized metrics (0-1)
        text_spacing = 0.12
        final_margin = 0.15
        text_begin = 0.1

        x_min = 0
        x_max = min(1.1, x_max + 0.05 * x_range)
        x_ticks = np.arange(0, 1.2, 0.2)
        ax_main.set_xticks(x_ticks)

        ax_main.text(right_edge + text_begin, -0.8,
            'median',
            va='center', ha='center', fontsize=10,
            fontweight='bold', color='black')
        ax_main.text(right_edge + text_begin + text_spacing, -0.8,
                'mean±std',
                va='center', ha='center', fontsize=10,
                fontweight='bold', color='black')
        
        for i, (organ, _) in enumerate(stats.iterrows()):
            organ_name = organ.split(" (n=")[0]
            stats_data = stats_dict[organ_name]
            
            if np.isnan(stats_data['mean']) or np.isnan(stats_data['std']):
                median_str = "N/A"
                mean_std_str = "N/A"
            else:
                median_str = format_str.format(stats_data['median'])
                mean_std_str = f'{format_str.format(stats_data["mean"])}±{format_str.format(stats_data["std"])}'
            
            ax_main.text(right_edge + text_begin, i,
                    median_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
            ax_main.text(right_edge + text_begin + text_spacing, i,
                    mean_std_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
    else:
        # distance metrics
        text_begin = 0.01 * x_range
        text_spacing = 0.12 * x_range
        final_margin = 0.15 * x_range

        def inverse_log_transform(x, min_val=0.1):
            if np.isscalar(x):
                if x == 0:
                    return 0
                elif x == np.log10(min_val):
                    return 0
                return 10**x - min_val
            return np.array([inverse_log_transform(val, min_val) for val in x])
        
        target_values = [0, 1.5, 5, 10, 20, 50, 100]
        ticks = []
        labels = []
        
        for val in target_values:
            tick = log_transform(val)
            if tick <= x_max:
                ticks.append(tick)
                if val < 1:
                    labels.append("0")
                elif val == 1.5:
                    labels.append("1.5")
                else:
                    labels.append(f"{int(val)}")
        
        ax_main.set_xticks(ticks)
        ax_main.set_xticklabels(labels)

        ax_main.text(right_edge + text_begin, -0.8,
            'median',
            va='center', ha='center', fontsize=10,
            fontweight='bold', color='black')
        ax_main.text(right_edge + text_begin + text_spacing, -0.8,
                'mean±std',
                va='center', ha='center', fontsize=10,
                fontweight='bold', color='black')
        
        for i, (organ, _) in enumerate(stats.iterrows()):
            organ_name = organ.split(" (n=")[0]
            stats_data = stats_dict[organ_name]
            
            if np.isnan(stats_data['mean']) or np.isnan(stats_data['std']):
                median_str = "N/A"
                mean_std_str = "N/A"
            else:
                median_str = format_str.format(stats_data['median'])
                mean_std_str = f'{format_str.format(stats_data["mean"])}±{format_str.format(stats_data["std"])}'
            
            ax_main.text(right_edge + text_begin, i,
                    median_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
            ax_main.text(right_edge + text_begin + text_spacing, i,
                    mean_std_str,
                    va='center', ha='center',
                    fontsize=8, color='black', alpha=0.7)
    
    prev_system = None
    for i, (idx, row) in enumerate(df.groupby('Organ').first().iterrows()):
        if prev_system and row['System'] != prev_system:
            ax_main.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
            ax_sets.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
        prev_system = row['System']
    
    ax_main.set_xlabel(metric_name, fontsize=12)
    ax_main.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax_main.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax_main.set_xlim(x_min, right_edge + text_begin + text_spacing + final_margin)
    
    all_datasets = sorted(list(set(
        dataset for datasets in test_datasets_sources_dict.values() 
        for dataset in datasets
    )))
    
    for i, organ in enumerate(ordered_organs):
        organ_name = organ.split(" (n=")[0]
        if organ_name in test_datasets_sources_dict:
            datasets = test_datasets_sources_dict[organ_name]
            dataset_indices = [j for j, d in enumerate(all_datasets) if d in datasets]
            if len(dataset_indices) > 1:
                ax_sets.plot(dataset_indices, [i]*len(dataset_indices), 
                           color='black', linewidth=1.5, alpha=0.5)
            for j, dataset in enumerate(all_datasets):
                if dataset in datasets:
                    ax_sets.plot([j], [i], 'o', color='black', markersize=8)
                else:
                    ax_sets.plot([j], [i], 'o', color='lightgray', markersize=8)
    
    ax_sets.set_xticks(range(len(all_datasets)))
    ax_sets.set_xticklabels(all_datasets, rotation=45, ha='right', fontsize=8)
    ax_sets.set_title('Test Datasets', pad=5, fontsize=14, fontweight='bold')
    ax_sets.set_yticks([])
    
    ylim = ax_main.get_ylim()
    ax_sets.set_ylim(ylim)
    
    # add frame
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    for spine in ax_sets.spines.values():
        spine.set_visible(False)
        
    ax_main.spines['left'].set_visible(True)
    ax_main.spines['bottom'].set_visible(True)
    ax_main.spines['right'].set_visible(True)
    ax_main.spines['top'].set_visible(True)
    
    for spine in ax_main.spines.values():
        if spine.get_visible():
            spine.set_linewidth(1.0)
    
    ax_sets.spines['left'].set_visible(True)
    ax_sets.spines['right'].set_visible(True)
    ax_sets.spines['top'].set_visible(True)
    ax_sets.spines['bottom'].set_visible(True)
    
    for spine in ax_sets.spines.values():
        if spine.get_visible():
            spine.set_linewidth(1.0)
    
    ylim = ax_main.get_ylim()
    ax_sets.set_ylim(ylim)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return ax_main, ax_sets, ax_legend

def generate_radar_plot_normalized_metrics(model1_scores, model2_scores, model1_name, model2_name, baseline_scores, output_path, title="Radar Plot", 
                       circle_positions=None, system_groups=None, highlight_high_scores=False, focus_point=0.9, power=3):
    """
    Generate radar plot for normalized metrics (0-1 range, with 0 the worst and 1 the best).
    
    Args:
        circle_positions: custom circle positions
        system_groups: anatomical system groups
        highlight_high_scores: whether to use non-linear transformation to highlight high scores
        focus_point: score value where non-linear transformation begins (default 0.9)
        power: controls the strength of the non-linear transformation (default 3)
    """
    def highlight_transform(x, focus_point=0.9, power=2):
        """
        non-linear transformations
        - parts lower than focus_point: remains linear display
        - parts higher than focus_point: use an exponential function to make differences closer to 1 more significant
        """
        if np.isscalar(x):
            if x < focus_point:
                return 0.5 * x / focus_point
            else:
                normalized = (x - focus_point) / (1 - focus_point)
                enhanced = np.exp(power * normalized) - 1 
                max_enhanced = np.exp(power) - 1
                return 0.5 + 0.5 * (enhanced / max_enhanced)
        return np.array([highlight_transform(val, focus_point, power) for val in x])

    if system_groups:
        system_colors = anatomical_system_colors
        ordered_labels = []
        label_colors = []
        for system, structures in system_groups.items():
            valid_structures = [s for s in structures if s in model2_scores]
            ordered_labels.extend(valid_structures)
            label_colors.extend([system_colors[system]] * len(valid_structures))
        
        all_structures = set(model2_scores.keys())
        grouped_structures = set(ordered_labels)
        ungrouped = all_structures - grouped_structures
        if ungrouped:
            ordered_labels.extend(sorted(ungrouped))
            label_colors.extend([[0.7, 0.7, 0.7, 1.0]] * len(ungrouped))
    else:
        ordered_labels = list(model2_scores.keys())
        label_colors = [[0, 0, 0, 1.0]] * len(ordered_labels)

    # find unique structures
    unique_structures = [struct for struct in model2_scores.keys() 
                        if struct not in model1_scores or np.isnan(model1_scores[struct])]

    # prepare scores
    scores1 = []
    scores2 = []
    for label in ordered_labels:
        scores2.append(model2_scores[label])
        if label in unique_structures:
            scores1.append(0.0)  # unique structures set to 0
        else:
            scores1.append(model1_scores[label])

    if highlight_high_scores:
        if circle_positions is None:
            circle_positions = [
                0.2, 0.4, 0.6, 0.8,  # sparse scales in low scoring areas
                0.90, 0.92, 0.94, 0.96, 0.98, 1.0  # dense scales in high scoring areas
            ]
        transformed_scores1 = highlight_transform(scores1, focus_point, power)
        transformed_scores2 = highlight_transform(scores2, focus_point, power)
        transformed_circles = highlight_transform(circle_positions, focus_point, power)
    else:
        if circle_positions is None:
            circle_positions = [0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]
        transformed_scores1 = scores1
        transformed_scores2 = scores2
        transformed_circles = circle_positions

    fig = plt.figure(figsize=(30, 30))
    fig.suptitle(title, fontsize=50, y=0.95)
    ax = fig.add_subplot(111, projection='polar')
    ax.spines['polar'].set_visible(False)

    angles = np.linspace(0, 2*np.pi, len(ordered_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    transformed_scores1 = np.concatenate((transformed_scores1, [transformed_scores1[0]]))
    transformed_scores2 = np.concatenate((transformed_scores2, [transformed_scores2[0]]))
    label_colors.append(label_colors[0])

    ax.fill(angles, transformed_scores1, color=MODEL1_COLOR, alpha=0.08, zorder=0)
    ax.fill(angles, transformed_scores2, color=MODEL2_COLOR, alpha=0.16, zorder=1)

    ax.plot(angles, transformed_scores1, 'o-', color=MODEL1_COLOR, linewidth=1, 
            label=model1_name, markersize=8, markerfacecolor=MODEL1_COLOR, 
            markeredgecolor=MODEL1_COLOR, zorder=2)
    ax.plot(angles, transformed_scores2, 'o-', color=MODEL2_COLOR, linewidth=3, 
            label=model2_name, markersize=8, markerfacecolor=MODEL2_COLOR, 
            markeredgecolor=MODEL2_COLOR, zorder=3)

    for angle, score, label in zip(angles[:-1], transformed_scores2[:-1], ordered_labels):
        if label in unique_structures:
            ax.plot(angle, score, 'X', color='#03c03c', markersize=12, 
                   markerfacecolor='#03c03c', markeredgecolor='#03c03c', zorder=4)

    unique_marker = plt.Line2D([], [], marker='X', color='#03c03c',
                              markerfacecolor='#03c03c', markeredgecolor='#03c03c',
                              markersize=8, linestyle='None', label='CADS Unique')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(unique_marker)

    ax.set_rlabel_position(0)
    plt.yticks(transformed_circles, 
               [f"{pos:.2f}" for pos in circle_positions],
               color="grey", size=10)

    for circle in transformed_circles:
        ax.plot(angles, [circle]*len(angles), '--', color='grey', alpha=0.2)

    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1.1], 
                '--', color='grey', alpha=0.3, linewidth=0.5)

    ax.set_xticks([])

    max_r = 1
    label_radius = max_r * 1.05
    ax.set_rmax(max_r * 1.2)

    for i, (angle, label, color) in enumerate(zip(angles[:-1], ordered_labels, label_colors[:-1])):
        angle_deg = angle * 180 / np.pi
        rotation = angle_deg
        if 90 < angle_deg <= 270:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.text(angle, label_radius, label,
                rotation=rotation,
                ha=ha,
                va='center',
                rotation_mode='anchor',
                fontsize=22,
                color=color)

    if baseline_scores:
        for i, (baseline_name, scores) in enumerate(baseline_scores.items()):
            color = BASELINE_COLORS[f'baseline{i+1}'] 
            marker = 'P'
            # create handle for legends
            baseline_handle = plt.Line2D([], [], marker=marker, color=color,
                                       markerfacecolor=color, markeredgecolor='black',
                                       markersize=10, linestyle='None',
                                       label=baseline_name)
            handles.append(baseline_handle)
            
            # scatter plot baseline results
            for label in ordered_labels:
                if label in scores:
                    score = scores[label]
                    if highlight_high_scores:
                        score = highlight_transform(score, focus_point, power)
                    angle = angles[ordered_labels.index(label)]
                    ax.plot(angle, score, marker, color=color, 
                           markersize=15, markerfacecolor=color,
                           markeredgecolor='black', zorder=5)
    # model legend
    model_legend = ax.legend(handles=handles,
                            loc='center left', 
                            bbox_to_anchor=(1.1, 0.95),
                            fontsize=20,
                            title="Models",
                            title_fontsize=25)
    ax.add_artist(model_legend)

    if system_groups:
        system_patches = [plt.Rectangle((0, 0), 1, 1, fc=system_colors[system]) 
                         for system in system_groups.keys()]
        system_legend = ax.legend(system_patches, 
                                system_groups.keys(), 
                                loc='center left', 
                                bbox_to_anchor=(1.1, 0.62),
                                title="Anatomical Systems", 
                                fontsize=20, 
                                title_fontsize=25)
        ax.add_artist(model_legend)

    if highlight_high_scores:
        plt.figtext(0.95, 0.15, 
                   f"* Non-linear scale enhancement above {focus_point:.2f}\n" +
                   "* For unique targets in CADS, the other model's score is set to 0",
                   fontsize=12, ha='right', style='italic')
    else:
        plt.figtext(0.95, 0.15, 
                   "* For unique targets in CADS, the other model's score is set to 0",
                   fontsize=12, ha='right', style='italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def generate_radar_plot_distance_metrics(model1_scores, model2_scores, model1_name, model2_name, output_path, title="Radar Plot", 
                       circle_positions=None, system_groups=None):
    """
    Generate radar plot with logarithmic scale. For metrics where lower is better (e.g., HD95),
    unique targets in model1 are set to 1.2 * max_score to indicate non-existing performance.
    """
    def log_transform(x, min_val=0.1):
        if np.isscalar(x):
            if x <= 0:
                return np.log10(min_val)
            return np.log10(x + min_val)
        return np.array([log_transform(val, min_val) for val in x])

    if system_groups:
        system_colors = anatomical_system_colors
        ordered_labels = []
        label_colors = []
        for system, structures in system_groups.items():
            valid_structures = [s for s in structures if s in model2_scores]
            ordered_labels.extend(valid_structures)
            label_colors.extend([system_colors[system]] * len(valid_structures))
        
        all_structures = set(model2_scores.keys())
        grouped_structures = set(ordered_labels)
        ungrouped = all_structures - grouped_structures
        if ungrouped:
            ordered_labels.extend(sorted(ungrouped))
            label_colors.extend([[0.7, 0.7, 0.7, 1.0]] * len(ungrouped))
    else:
        ordered_labels = list(model2_scores.keys())
        label_colors = [[0, 0, 0, 1.0]] * len(ordered_labels)

    # unique structures
    unique_structures = [struct for struct in model2_scores.keys() 
                        if struct not in model1_scores or np.isnan(model1_scores[struct])]

    # find the min/max value of scores
    all_valid_scores = []
    all_valid_scores.extend([s for s in model1_scores.values() if not np.isnan(s)])
    all_valid_scores.extend([s for s in model2_scores.values() if not np.isnan(s)])
    min_score = min(all_valid_scores)
    max_score = max(all_valid_scores)
    unique_target_score = max_score * 1.2  # unique targets

    # circle positions
    if circle_positions is None:
        magnitude = int(np.log10(max_score))
        circle_positions = []
        for exp in range(-1, magnitude + 1):
            # add more scales to each order of magnitude
            circle_positions.extend([
                1 * 10**exp,  # 1, 10, 100, ...
                2 * 10**exp,  # 2, 20, 200, ...
                3 * 10**exp,  # 3, 30, 300, ...
                5 * 10**exp,  # 5, 50, 500, ...
                8 * 10**exp   # 8, 80, 800, ...
            ])        
        circle_positions = sorted([pos for pos in circle_positions 
                                 if min_score <= pos <= unique_target_score])

    # prepare scores
    scores1 = []
    scores2 = []
    for label in ordered_labels:
        scores2.append(model2_scores[label])
        if label in unique_structures:
            scores1.append(unique_target_score)
        else:
            scores1.append(model1_scores[label])

    # log transform
    log_scores1 = log_transform(scores1)
    log_scores2 = log_transform(scores2)
    log_circle_positions = log_transform(circle_positions)

    fig = plt.figure(figsize=(30, 30))
    fig.suptitle(title, fontsize=50, y=0.95)
    ax = fig.add_subplot(111, projection='polar')
    ax.spines['polar'].set_visible(False)

    angles = np.linspace(0, 2*np.pi, len(ordered_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    log_scores1 = np.concatenate((log_scores1, [log_scores1[0]]))
    log_scores2 = np.concatenate((log_scores2, [log_scores2[0]]))
    label_colors.append(label_colors[0])

    ax.fill(angles, log_scores1, color=MODEL1_COLOR, alpha=0.08, zorder=0)
    ax.fill(angles, log_scores2, color=MODEL2_COLOR, alpha=0.06, zorder=1)

    ax.plot(angles, log_scores1, 'o-', color=MODEL1_COLOR, linewidth=1, 
            label=model1_name, markersize=8, markerfacecolor=MODEL1_COLOR, 
            markeredgecolor=MODEL1_COLOR, zorder=2)
    ax.plot(angles, log_scores2, 'o-', color=MODEL2_COLOR, linewidth=3, 
            label=model2_name, markersize=8, markerfacecolor=MODEL2_COLOR, 
            markeredgecolor=MODEL2_COLOR, zorder=3)

    for angle, score, label in zip(angles[:-1], log_scores2[:-1], ordered_labels):
        if label in unique_structures:
            ax.plot(angle, score, 'X', color='#03c03c', markersize=12, 
                   markerfacecolor='#03c03c', markeredgecolor='#03c03c', zorder=4)

    unique_marker = plt.Line2D([], [], marker='X', color='#03c03c',
                              markerfacecolor='#03c03c', markeredgecolor='#03c03c',
                              markersize=8, linestyle='None', label='CADS Unique')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(unique_marker)

    ax.set_rlabel_position(0)
    plt.yticks(log_circle_positions, 
               [f"{pos:.1f}" for pos in circle_positions],
               color="grey", size=10)

    for log_circle in log_circle_positions:
        ax.plot(angles, [log_circle]*len(angles), '--', color='grey', alpha=0.2)

    for angle in angles[:-1]:
        ax.plot([angle, angle], [log_transform(min_score), log_transform(unique_target_score)], 
                '--', color='grey', alpha=0.3, linewidth=0.5)

    ax.set_xticks([])

    max_r = log_transform(unique_target_score)
    label_radius = max_r * 1.05
    ax.set_rmax(max_r * 1.2)

    for i, (angle, label, color) in enumerate(zip(angles[:-1], ordered_labels, label_colors[:-1])):
        angle_deg = angle * 180 / np.pi
        rotation = angle_deg
        if 90 < angle_deg <= 270:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.text(angle, label_radius, label,
                rotation=rotation,
                ha=ha,
                va='center',
                rotation_mode='anchor',
                fontsize=22,
                color=color)

    model_legend = ax.legend(handles=handles,
                            loc='center left', 
                            bbox_to_anchor=(1.05, 0.95),
                            fontsize=20,
                            title="Models",
                            title_fontsize=25)
    ax.add_artist(model_legend)

    if system_groups:
        system_patches = [plt.Rectangle((0, 0), 1, 1, fc=system_colors[system]) 
                         for system in system_groups.keys()]
        system_legend = ax.legend(system_patches, 
                                system_groups.keys(), 
                                loc='center left', 
                                bbox_to_anchor=(1.05, 0.7),
                                title="Anatomical Systems", 
                                fontsize=20, 
                                title_fontsize=25)
        ax.add_artist(model_legend)

    plt.figtext(0.95, 0.15, 
                "* For unique targets in CADS, the other model's score is set to 1.2 × maximum score for visualization",
                fontsize=12, ha='right', style='italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
