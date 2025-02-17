import numpy as np
import matplotlib.pyplot as plt
from math import pi

from omaseg.table_plots.plots.utils import read_results_table, anatomical_system_colors
from omaseg.dataset_utils.bodyparts_labelmaps import anatomical_systems

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
    
    angles = np.linspace(0, 2*pi, len(ordered_labels), endpoint=False)
    
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
        angle_deg = angle * 180 / pi
        
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


if __name__ == "__main__":
    table_path = '/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable_and_limited_fov/dice_compare_table.xlsx'  #TODO: 
    column_name = 'all median'  #TODO: 
    omaseg_dice, totalseg_dice = read_results_table(table_path, column_name)
        
    # 生成并保存图表
    generate_radar_plot(
        model1_scores=totalseg_dice,
        model2_scores=omaseg_dice,
        model1_name='TotalSeg',
        model2_name='OMASeg',
        output_path = "/mnt/hdda/murong/22k/plots/per-structure/radar_plot_unreliable_limitedFOV_all_median_dice.png",  #TODO: 
        title="Structure-wise Median Dice Score Comparison (Full Dataset)",  #TODO: 
        circle_positions=[0.5, 0.6, 0.8, 0.9, 1.0],  #TODO: 
        system_groups=anatomical_systems
    )
