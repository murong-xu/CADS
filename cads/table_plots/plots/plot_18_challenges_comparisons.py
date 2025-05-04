dice = {
    'VISCERAL Gold Corpus': {'OMASeg': 83.66, 'TotalSeg': 83.39},
    'VISCERAL Gold Corpus-Extra': {'OMASeg': 75.95, 'TotalSeg': 66.94},
    'VISCERAL Silver Corpus': {'OMASeg': 72.10, 'TotalSeg': 70.94},
    'KiTS': {'OMASeg': 91.31, 'TotalSeg': 88.36},
    'LiTS': {'OMASeg': 96.08, 'TotalSeg': 95.89},
    'BTCV-Abdomen': {'OMASeg': 83.84, 'TotalSeg': 85.15},
    'BTCV-Cervix': {'OMASeg': 65.81, 'TotalSeg': 47.18},
    'CHAOS': {'OMASeg': 96.90, 'TotalSeg': 96.84},
    'CT-ORG': {'OMASeg': 93.03, 'TotalSeg': 93.27},
    'AbdomenCT-1K': {'OMASeg': 92.93, 'TotalSeg': 92.15},
    'VerSe': {'OMASeg': 91.79, 'TotalSeg': 91.57},
    'Learn2reg': {'OMASeg': 91.40, 'TotalSeg': 91.25},
    'SLIVER07': {'OMASeg': 96.87, 'TotalSeg': 96.65},
    'EMPIRE10': {'OMASeg': 91.18, 'TotalSeg': 97.67},
    'Total-Segmentator': {'OMASeg': 93.15, 'TotalSeg': 91.49},
    'AMOS': {'OMASeg': 82.57, 'TotalSeg': 82.39},
    'HaN-Seg': {'OMASeg': 72.98, 'TotalSeg': 60.80},
    'SAROS': {'OMASeg': 91.64, 'TotalSeg': 87.97}
}

hd95 = {
    'VISCERAL Gold Corpus': {'OMASeg': 7.98, 'TotalSeg': 8.12},
    'VISCERAL Gold Corpus-Extra': {'OMASeg': 8.68, 'TotalSeg': 18.43},
    'VISCERAL Silver Corpus': {'OMASeg': 29.65, 'TotalSeg': 30.12},
    'KiTS': {'OMASeg': 7.36, 'TotalSeg': 7.65},
    'LiTS': {'OMASeg': 4.05, 'TotalSeg': 4.46},
    'BTCV-Abdomen': {'OMASeg': 4.06, 'TotalSeg': 5.55},
    'BTCV-Cervix': {'OMASeg': 40.99, 'TotalSeg': 83.20},
    'CHAOS': {'OMASeg': 3.21, 'TotalSeg': 3.33},
    'CT-ORG': {'OMASeg': 9.29, 'TotalSeg': 9.05},
    'AbdomenCT-1K': {'OMASeg': 3.54, 'TotalSeg': 4.07},
    'VerSe': {'OMASeg': 1.92, 'TotalSeg': 2.05},
    'Learn2reg': {'OMASeg': 4.87, 'TotalSeg': 4.99},
    'SLIVER07': {'OMASeg': 3.53, 'TotalSeg': 3.70},
    'EMPIRE10': {'OMASeg': 8.89, 'TotalSeg': 2.23},
    'Total-Segmentator': {'OMASeg': 2.40, 'TotalSeg': 3.86},
    'AMOS': {'OMASeg': 6.15, 'TotalSeg': 10.85},
    'HaN-Seg': {'OMASeg': 18.71, 'TotalSeg': 32.89},
    'SAROS': {'OMASeg': 3.00, 'TotalSeg': 4.97}
}

COLOR_CADS = '#ffa505'
COLOR_TOTALSEG = '#9c9c9c'
COLOR_CADS_POINT = '#c74000'
COLOR_TOTALSEG_POINT = '#412f57'
COLOR_BETTER = '#02b836'
COLOR_WORSE = '#d14536'

import matplotlib.pyplot as plt
import numpy as np

# Caloculate the difference between OMASeg and TotalSeg for each dataset
dice_improvements = {k: dice[k]['OMASeg'] - dice[k]['TotalSeg'] for k in dice.keys()}
sorted_datasets = sorted(dice_improvements.items(), key=lambda x: x[1])
dataset_order = [x[0] for x in sorted_datasets]

datasets = dataset_order
omaseg_dice = [dice[d]['OMASeg'] for d in datasets]
totalseg_dice = [dice[d]['TotalSeg'] for d in datasets]
omaseg_hd95 = [hd95[d]['OMASeg'] for d in datasets]
totalseg_hd95 = [hd95[d]['TotalSeg'] for d in datasets]

fig, ax1 = plt.subplots(figsize=(15, 10))

y_pos = np.arange(len(datasets))
bar_width = 0.35

ax1.barh(y_pos + bar_width/2, omaseg_dice, bar_width, label='CADS Dice', color=COLOR_CADS, alpha=1)
ax1.barh(y_pos - bar_width/2, totalseg_dice, bar_width, label='TotalSeg Dice', color=COLOR_TOTALSEG, alpha=1)

for idx, y in enumerate(y_pos):
    diff = omaseg_dice[idx] - totalseg_dice[idx]
    if diff > 0:
        color = COLOR_BETTER  # green for better
        sign = '+'
    else:
        color = COLOR_WORSE  # red for worse
        sign = ''
    
    # add difference text info
    ax1.text(max(omaseg_dice[idx], totalseg_dice[idx]) + 0.5, y, 
            f'{sign}{diff:.2f}%', 
            color=color,
            va='center',
            fontsize=10,
            fontweight='bold')
    
    ax1.plot([totalseg_dice[idx], omaseg_dice[idx]], 
            [y, y],
            color=color,
            alpha=0.5,
            linestyle='--',
            linewidth=1)

# hd95
ax2 = ax1.twiny()

ax2.plot(omaseg_hd95, y_pos, 'o-', color=COLOR_CADS_POINT, label='CADS HD95', markersize=8)
ax2.plot(totalseg_hd95, y_pos, 'o--', color=COLOR_TOTALSEG_POINT, label='TotalSeg HD95', markersize=8)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(datasets)

ax1.set_xlim(40, 102)
ax1.set_xlabel('Dice Score (%)', fontsize=14)
ax2.set_xlim(0, 100)
ax2.set_xlabel('HD95 (mm)', fontsize=14)

ax1.grid(True, linestyle='--', alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()

plt.savefig('/mnt/hdda/murong/22k/plots/result_a_18_challenges_comparison.png', dpi=300, bbox_inches='tight')
plt.show()