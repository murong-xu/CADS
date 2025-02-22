import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from omaseg.table_plots.plots.utils import anatomical_system_colors


def generate_histogram_plot(model1_scores, model2_scores, model1_name, model2_name, 
                          output_path, metric_name, system_group, anatomical_systems):
    """
    Generate histogram plot comparing two models' performance for a specific anatomical system.
    NaNs should already be removed from model scroes!!
    """
    model1_color = "#0072BD"
    model2_color = "#FF0000"
    
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
                    label=model1_name, color=model1_color, alpha=0.5, capsize=5)
    rects2 = plt.bar(x + width/2, model2_means, width, yerr=model2_stds,
                    label=model2_name, color=model2_color, alpha=0.5, capsize=5)
    
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel('Organs', fontsize=12)
    plt.title(f"{system_group}", fontsize=14, pad=20)
    plt.xticks(x, organs, rotation=45, ha='right')
    plt.legend(fontsize=10)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{system_group.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_box_plot(results_dict, metric_name='DSC', output_path=None, title="Structure-wise Performance Distribution",
                     anatomical_systems=None):
    """
    Generate a box plot grouped by anatomical systems.
    Args:
        results_dict: Dictionary of structure-wise scores
        anatomical_systems: Dictionary of anatomical system groupings
    """
    data_list = []
    
    # In each anatomical system: sort by median
    structure_medians = {organ: np.median(scores) for organ, scores in results_dict.items()}
    for system, structures in anatomical_systems.items():
        valid_structures = [s for s in structures if s in results_dict]
        sorted_structures = sorted(valid_structures, 
                                 key=lambda x: structure_medians[x],
                                 reverse=True)
        for organ in sorted_structures:
            scores = results_dict[organ]
            n_samples = len(scores)
            for score in scores:
                data_list.append({
                    'Organ': f"{organ} (n={n_samples})",
                    metric_name: score,
                    'System': system
                })
            
    df = pd.DataFrame(data_list)
    plt.figure(figsize=(12, len(results_dict) * 0.3))
    
    ax = sns.boxplot(data=df, 
                    y='Organ', 
                    x=metric_name,
                    hue='System',
                    palette=anatomical_system_colors,
                    whis=1.5,
                    showfliers=True,
                    flierprops={'marker': 'o',
                               'markerfacecolor': 'black',
                               'markeredgecolor': 'black',
                               'markersize': 2,
                               'alpha': 0.3})
    
    # add scatters (original data)
    sns.stripplot(data=df,
                 y='Organ',
                 x=metric_name,
                 color='black',
                 size=2,
                 alpha=0.3,
                 jitter=0.2)
    
    # calculate ststistics (median, mean)
    ordered_organs = df['Organ'].unique()
    stats = df.groupby('Organ').agg({
        metric_name: ['median', 'mean', 'std']
    })[metric_name].reindex(ordered_organs)
    
    right_edge = ax.get_xlim()[1]
    ax.text(right_edge, -0.8,
            'median',
            va='center', ha='left', fontsize=10,
            fontweight='bold', color='black')
    ax.text(right_edge + 0.12, -0.8,
            'mean±std',
            va='center', ha='left', fontsize=10,
            fontweight='bold', color='black')
    
    for i, (organ, row) in enumerate(stats.iterrows()):
        median_str = f'{row["median"]:.3f}'
        ax.text(right_edge, i,
                median_str,
                va='center',
                ha='left',
                fontsize=8,
                color='black',
                alpha=0.7)
        mean_std_str = f'{row["mean"]:.3f}±{row["std"]:.3f}'
        ax.text(right_edge + 0.12, i,
                mean_std_str,
                va='center',
                ha='left',
                fontsize=8,
                color='black',
                alpha=0.7)
        
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
    
    ax.set_xlim(0, right_edge + 0.3)  
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    
    plt.legend(title='Anatomical Systems', 
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=10,
              title_fontsize=12)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return ax


def generate_box_plot_with_testdata_sources(results_dict, test_datasets_sources_dict, metric_name='DSC', output_path=None,
                                            title="Structure-wise Performance Distribution", anatomical_systems=None):
    """
    Generate a box plot with dataset source indicators.
    Args:
        results_dict: Dictionary of structure-wise scores
        test_datasets_sources_dict: Dictionary mapping each structure to its test datasets
    """    
    data_list = []
    
    # in each anatomical system: sort by median
    structure_medians = {organ: np.median(scores) for organ, scores in results_dict.items()}
    for system, structures in anatomical_systems.items():
        valid_structures = [s for s in structures if s in results_dict]
        sorted_structures = sorted(valid_structures, 
                                 key=lambda x: structure_medians[x],
                                 reverse=True)
        for organ in sorted_structures:
            scores = results_dict[organ]
            n_samples = len(scores)
            for score in scores:
                data_list.append({
                    'Organ': f"{organ} (n={n_samples})",
                    metric_name: score,
                    'System': system
                })
    
    df = pd.DataFrame(data_list)
    
    fig = plt.figure(figsize=(20, len(results_dict) * 0.3))
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
                whis=1.5,
                showfliers=True,
                flierprops={'marker': 'o',
                           'markerfacecolor': 'black',
                           'markeredgecolor': 'black',
                           'markersize': 2,
                           'alpha': 0.3},
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
    
    # calc stat.
    ordered_organs = df['Organ'].unique()
    stats = df.groupby('Organ').agg({
        metric_name: ['median', 'mean', 'std']
    })[metric_name].reindex(ordered_organs)
    
    ax_main.set_yticks(range(len(ordered_organs)))
    ax_main.set_yticklabels(ordered_organs, fontsize=10, ha='right')
    ax_main.tick_params(axis='y', pad=5)
    
    right_edge = ax_main.get_xlim()[1]
    ax_main.text(right_edge, -0.8,
                'median',
                va='bottom', ha='left', fontsize=10,
                fontweight='bold', color='black')
    ax_main.text(right_edge + 0.12, -0.8,
                'mean±std',
                va='bottom', ha='left', fontsize=10,
                fontweight='bold', color='black')
    
    for i, (organ, row) in enumerate(stats.iterrows()):
        median_str = f'{row["median"]:.3f}'
        ax_main.text(right_edge, i,
                    median_str,
                    va='center',
                    ha='left',
                    fontsize=8,
                    color='black',
                    alpha=0.7)
        mean_std_str = f'{row["mean"]:.3f}±{row["std"]:.3f}'
        ax_main.text(right_edge + 0.12, i,
                    mean_std_str,
                    va='center',
                    ha='left',
                    fontsize=8,
                    color='black',
                    alpha=0.7)
    
    prev_system = None
    for i, (idx, row) in enumerate(df.groupby('Organ').first().iterrows()):
        if prev_system and row['System'] != prev_system:
            ax_main.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
            ax_sets.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
        prev_system = row['System']
    
    ax_main.set_xlabel(metric_name, fontsize=12)
    ax_main.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax_main.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax_main.set_xlim(0, right_edge + 0.3)
    ax_main.set_xticks(np.arange(0, 1.2, 0.2))
    
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

def generate_radar_plot(model1_scores, model2_scores, model1_name, model2_name, output_path, title="Radar Plot", 
                       circle_positions=None, system_groups=None):
    """
    Generate radar plot with structures grouped by anatomical systems.
    Args:
        model1_scores: baseline
        model2_scores: our model
    """

    model1_color = "#0072BD"
    model2_color = "#FF0000"
    # Set default circle positions if none provided
    if circle_positions is None:
        circle_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Reorder labels based on system groups if provided
    if system_groups:
        # Generate a color map for systems
        # n_systems = len(system_groups)
        # colors = plt.cm.tab20(np.linspace(0, 1, n_systems))
        system_colors = anatomical_system_colors
        
        # Create ordered list of labels and their colors
        ordered_labels = []
        label_colors = []
        for system, structures in system_groups.items():
            valid_structures = [s for s in structures if s in model2_scores]
            ordered_labels.extend(valid_structures)
            label_colors.extend([system_colors[system]] * len(valid_structures))
        
        # Check if all structures are included
        all_structures = set(model2_scores.keys())
        grouped_structures = set(ordered_labels)
        ungrouped = all_structures - grouped_structures
        if ungrouped:
            ordered_labels.extend(sorted(ungrouped))
            label_colors.extend([[0.7, 0.7, 0.7, 1.0]] * len(ungrouped))
    else:
        ordered_labels = list(model2_scores.keys())
        label_colors = [[0, 0, 0, 1.0]] * len(ordered_labels)
    
    # Find unique structures
    unique_structures = [struct for struct in model2_scores.keys() 
                        if struct not in model1_scores or np.isnan(model1_scores[struct])]
    
    scores1 = []
    scores2 = []
    for label in ordered_labels:
        scores2.append(model2_scores[label])
        if label in unique_structures:
            scores1.append(0.0)  # unique structures in baseline: set to 0
        else:
            scores1.append(model1_scores[label])
    
    angles = np.linspace(0, 2*np.pi, len(ordered_labels), endpoint=False)
    
    angles = np.concatenate((angles, [angles[0]]))
    scores1 = np.concatenate((scores1, [scores1[0]]))
    scores2 = np.concatenate((scores2, [scores2[0]]))
    label_colors.append(label_colors[0])
    
    fig = plt.figure(figsize=(30, 30))
    fig.suptitle(title, fontsize=50, y=0.95)
    
    ax = fig.add_subplot(111, projection='polar')
    ax.spines['polar'].set_visible(False)
    
    # Fill in covered area
    ax.fill(angles, scores1, color=model1_color, alpha=0.08, zorder=0)  # Baseline
    ax.fill(angles, scores2, color=model2_color, alpha=0.06, zorder=1)  # OMASeg
       
    # Plot data with markers
    ax.plot(angles, scores1, 'o-', color=model1_color, linewidth=1, label=model1_name,
            markersize=8, markerfacecolor=model1_color, markeredgecolor=model1_color, zorder=2)
    ax.plot(angles, scores2, 'o-', color=model2_color, linewidth=3, label=model2_name,
            markersize=8, markerfacecolor=model2_color, markeredgecolor=model2_color, zorder=3)
    
    # Unique targets
    for angle, score, label in zip(angles[:-1], scores2[:-1], ordered_labels):
        if label in unique_structures:
            ax.plot(angle, score, 'X', color='#03c03c', markersize=12, 
                   markerfacecolor='#03c03c', markeredgecolor='#03c03c', zorder=4)
            
    unique_marker = plt.Line2D([], [], 
                              marker='X',
                              color='#03c03c',
                              markerfacecolor='#03c03c',
                              markeredgecolor='#03c03c',
                              markersize=8,
                              linestyle='None',
                              label='OMASeg Unique')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(unique_marker)
    model_legend = ax.legend(handles=handles,
                            loc='center left', 
                            bbox_to_anchor=(1.05, 0.95),
                            fontsize=20,
                            title="Models",
                            title_fontsize=25)
    ax.add_artist(model_legend)
    
    # Add background circles
    ax.set_rlabel_position(0)
    plt.yticks(circle_positions, 
               [f"{pos:.1f}" for pos in circle_positions],
               color="grey", size=10)
    
    # Add dotted circles
    for circle in circle_positions:
        ax.plot(angles, [circle]*len(angles), '--', color='grey', alpha=0.2)
    
    # Add radial grid lines
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], '--', color='grey', alpha=0.3, linewidth=0.5)
    
    # Remove default axis labels
    ax.set_xticks([])
    
    # Add colored radial labels
    label_position = 1.05
    for i, (angle, label, color) in enumerate(zip(angles[:-1], ordered_labels, label_colors[:-1])):
        angle_deg = angle * 180 / np.pi
        
        # Rotate text outward
        rotation = angle_deg
        if 90 < angle_deg <= 270:
            rotation += 180
            
        # Determine text alignment based on rotated angle
        if 90 < angle_deg <= 270:
            ha = 'right'
        else:
            ha = 'left'
            
        ax.text(angle, label_position, label,
                rotation=rotation,
                ha=ha,
                va='center',
                rotation_mode='anchor',
                fontsize=22,
                color=color)
        
    if system_groups:
        # First add model legend
        # model_legend = ax.legend(loc='center left', 
        #                        bbox_to_anchor=(1.05, 0.8),
        #                        fontsize=20,
        #                        title="Models",
        #                        title_fontsize=25)
        
        # Add system legend below the model legend
        system_patches = [plt.Rectangle((0, 0), 1, 1, fc=system_colors[system]) 
                         for system in system_groups.keys()]
        system_legend = ax.legend(system_patches, 
                                system_groups.keys(), 
                                loc='center left', 
                                bbox_to_anchor=(1.05, 0.7),
                                title="Anatomical Systems", 
                                fontsize=20, 
                                title_fontsize=25)
        
        # Make sure both legends are visible
        ax.add_artist(model_legend)
    # else:
    #     # Only add model legend if no system groups
    #     ax.legend(loc='center left', 
    #              bbox_to_anchor=(1.05, 0.8), 
    #              fontsize=25,
    #              title="Models",
    #             title_fontsize=30)
    
    ax.set_rmax(1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()