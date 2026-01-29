metrics_data = {
    'Mutual targets (119 structures)': {
        'TotalSeg': {
            'Primary': {'Dice': 89.22, 'HD95': 5.63},
            'Secondary': {'Dice': 81.78, 'HD95': 10.70},
            'Full test data': {'Dice': 88.12, 'HD95': 7.38}
        },
        'OMASeg': {
            'Primary': {'Dice': 91.51, 'HD95': 3.21},
            'Secondary': {'Dice': 83.90, 'HD95': 7.60},
            'Full test data': {'Dice': 90.52, 'HD95': 4.36}
        },
        'Diff': {
            'Primary': {'Dice': '+2.29', 'HD95': '-2.42'},
            'Secondary': {'Dice': '+2.11', 'HD95': '-3.10'},
            'Full test data': {'Dice': '+2.40', 'HD95': '-3.02'}
        }
    },
    'All targets (167 structures)': {
        'OMASeg': {
            'Primary': {'Dice': 86.66, 'HD95': 4.36},
            'Secondary': {'Dice': 83.03, 'HD95': 8.40},
            'Full test data': {'Dice': 85.87, 'HD95': 5.23}
        }
    }
}

COLOR_CADS = '#ffa505'
COLOR_TOTALSEG = '#9c9c9c'
COLOR_CADS_POINT = '#c74000'
COLOR_TOTALSEG_POINT = '#412f57'
COLOR_BETTER = '#02b836'
COLOR_WORSE = '#d14536'

import matplotlib.pyplot as plt
import numpy as np

categories = ['Primary\nTest Data', 'Secondary\nTest Data', 'Full\nTest Data', 'All\n167 Targets']
bar_width = 0.35

# Dice scores
totalseg_dice = [
    metrics_data['Mutual targets (119 structures)']['TotalSeg']['Primary']['Dice'],
    metrics_data['Mutual targets (119 structures)']['TotalSeg']['Secondary']['Dice'],
    metrics_data['Mutual targets (119 structures)']['TotalSeg']['Full test data']['Dice'],
    None  # All Targets no TotalSeg scores
]

omaseg_dice = [
    metrics_data['Mutual targets (119 structures)']['OMASeg']['Primary']['Dice'],
    metrics_data['Mutual targets (119 structures)']['OMASeg']['Secondary']['Dice'],
    metrics_data['Mutual targets (119 structures)']['OMASeg']['Full test data']['Dice'],
    metrics_data['All targets (167 structures)']['OMASeg']['Full test data']['Dice']
]

# HD95 values
totalseg_hd95 = [
    metrics_data['Mutual targets (119 structures)']['TotalSeg']['Primary']['HD95'],
    metrics_data['Mutual targets (119 structures)']['TotalSeg']['Secondary']['HD95'],
    metrics_data['Mutual targets (119 structures)']['TotalSeg']['Full test data']['HD95'],
    None  # All Targets no TotalSeg scores
]

omaseg_hd95 = [
    metrics_data['Mutual targets (119 structures)']['OMASeg']['Primary']['HD95'],
    metrics_data['Mutual targets (119 structures)']['OMASeg']['Secondary']['HD95'],
    metrics_data['Mutual targets (119 structures)']['OMASeg']['Full test data']['HD95'],
    metrics_data['All targets (167 structures)']['OMASeg']['Full test data']['HD95']
]

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

x = np.arange(len(categories))

# Dice bars
# TotalSeg bars
totalseg_bars = ax1.bar(x[:3] - bar_width/2, totalseg_dice[:3], bar_width,
                       label='TotalSeg Dice', color=COLOR_TOTALSEG, alpha=1)

# CADS bars
omaseg_bars_first = ax1.bar(x[:3] + bar_width/2, omaseg_dice[:3], bar_width,
                           label='CADS Dice', color=COLOR_CADS, alpha=1)
omaseg_bars_last = ax1.bar(x[3], omaseg_dice[3], bar_width,
                          color=COLOR_CADS, alpha=1)
# hd95
# TotalSeg
totalseg_hd95_x = x[:3] - bar_width/2
ax2.plot(totalseg_hd95_x, totalseg_hd95[:3], marker='o', linestyle='None', color=COLOR_TOTALSEG_POINT,
         label='TotalSeg HD95', alpha=0.7, markersize=20)

# CADS
# Adjust x-coordinates to center markers over CADS bars
omaseg_hd95_x = np.array([x[i] + bar_width/2 for i in range(3)] + [x[3]])
ax2.plot(omaseg_hd95_x, omaseg_hd95, marker='X', linestyle='None', color=COLOR_CADS_POINT,
         label='CADS HD95', alpha=0.7, markersize=20)

# add text labels
# TotalSeg bars
for i in range(3):
    if totalseg_dice[i] is not None:
        ax1.text(x[i] - bar_width/2, totalseg_dice[i], f'{totalseg_dice[i]:.1f}',
                ha='center', va='bottom', fontsize=16)

# CADS bars
for i in range(3):
    ax1.text(x[i] + bar_width/2, omaseg_dice[i], f'{omaseg_dice[i]:.1f}',
            ha='center', va='bottom', fontsize=16, color=COLOR_BETTER, fontweight='bold')

# CADS last bar (All Targets)
ax1.text(x[3], omaseg_dice[3], f'{omaseg_dice[3]:.1f}',
        ha='center', va='bottom', fontsize=16, color=COLOR_BETTER, fontweight='bold')

ax1.set_ylabel('Dice Score (%)', fontsize=18)
ax2.set_ylabel('HD95 (mm)', fontsize=18)
ax1.set_ylim(75, 95)
ax2.set_ylim(0, 12)

ax1.set_xticks(x) # Ensure the tick positions are set
ax1.set_xticklabels(categories) # Ensure the labels are set
ax1.tick_params(axis='x', which='major', labelsize=16) # Set the font size for major x-axis tick labels

ax1.grid(True, linestyle='--', alpha=0.3)
ax1.grid(True, linestyle='--', alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
new_lines = [lines1[1], lines1[0], lines2[1], lines2[0]]
new_labels = [labels1[1], labels1[0], labels2[1], labels2[0]]

ax1.legend(new_lines, new_labels, loc='upper right', markerscale=0.5)

# plt.title('Part B (Enhanced): Overall Performance with Heatmap & Extra Targets', pad=20)

plt.tight_layout()

plt.savefig('/mnt/hdda/murong/22k/plots/result_b_overall_comparison.eps', bbox_inches='tight')
plt.show()