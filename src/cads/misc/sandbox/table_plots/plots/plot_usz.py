dice_data = {
    'Brainstem': {'TotalSeg': 60.11, 'OMASeg': 83.19},
    'Eye L': {'TotalSeg': 90.90, 'OMASeg': 91.82},
    'Eye R': {'TotalSeg': 90.90, 'OMASeg': 92.16},
    'Larynx': {'TotalSeg': 36.44, 'OMASeg': 80.24},
    'Optic nerve L': {'TotalSeg': 61.21, 'OMASeg': 66.60},
    'Optic nerve R': {'TotalSeg': 62.51, 'OMASeg': 67.46},
    'Parotid gland L': {'TotalSeg': 62.99, 'OMASeg': 82.39},
    'Parotid gland R': {'TotalSeg': 63.56, 'OMASeg': 84.58},
    'Submandibular gland L': {'TotalSeg': 81.22, 'OMASeg': 85.53},
    'Submandibular gland R': {'TotalSeg': 84.72, 'OMASeg': 84.11},
    
    'Aorta': {'TotalSeg': 89.55, 'OMASeg': 89.74},
    'Urinary bladder': {'TotalSeg': 90.54, 'OMASeg': 89.94},
    'Brain': {'TotalSeg': 96.39, 'OMASeg': 96.52},
    'Esophagus': {'TotalSeg': 85.21, 'OMASeg': 83.20},
    'Humerus L': {'TotalSeg': 95.44, 'OMASeg': 95.46},
    'Humerus R': {'TotalSeg': 95.16, 'OMASeg': 95.17},
    'Kidney L': {'TotalSeg': 90.06, 'OMASeg': 90.09},
    'Kidney R': {'TotalSeg': 90.91, 'OMASeg': 90.66},
    'Liver': {'TotalSeg': 96.45, 'OMASeg': 96.51},
    'Lung L': {'TotalSeg': 97.09, 'OMASeg': 96.70},
    
    'Lung R': {'TotalSeg': 97.23, 'OMASeg': 96.93},
    'Prostate': {'TotalSeg': 72.48, 'OMASeg': 75.69},
    'Spinal cord': {'TotalSeg': 88.66, 'OMASeg': 90.08},
    'Spleen': {'TotalSeg': 90.36, 'OMASeg': 90.17},
    'Stomach': {'TotalSeg': 88.76, 'OMASeg': 87.87},
    'Thyroid': {'TotalSeg': 82.37, 'OMASeg': 79.40},
    'Trachea': {'TotalSeg': 73.89, 'OMASeg': 70.43},
    'Inferior vena cava': {'TotalSeg': 71.14, 'OMASeg': 65.87},
    'Heart': {'TotalSeg': 77.93, 'OMASeg': 81.22},
    'Optic chiasm': {'TotalSeg': None, 'OMASeg': 30.49},
    
    'Glottis': {'TotalSeg': None, 'OMASeg': 37.11},
    'Lacrimal gland L': {'TotalSeg': None, 'OMASeg': 60.03},
    'Lacrimal gland R': {'TotalSeg': None, 'OMASeg': 52.78},
    'Mandible': {'TotalSeg': None, 'OMASeg': 88.81},
    'Oral cavity': {'TotalSeg': None, 'OMASeg': 83.60},
    'Pituitary gland': {'TotalSeg': None, 'OMASeg': 58.02},
    'Rectum': {'TotalSeg': None, 'OMASeg': 71.71},
    'Seminal vesicle': {'TotalSeg': None, 'OMASeg': 82.12}
}

import matplotlib.pyplot as plt
import numpy as np
import random
from adjustText import adjust_text

random.seed(42)

plt.figure(figsize=(20, 16))

shared_organs = []
unique_organs = []
texts = []

for organ, values in dice_data.items():
    if values['TotalSeg'] is not None:
        shared_organs.append(organ)
    else:
        unique_organs.append(organ)

# random colors for shared and unique organs
organ_colors = {}
for organ in shared_organs + unique_organs:
    h = random.random()
    s = 0.6 + random.random() * 0.4
    v = 0.6 + random.random() * 0.4
    organ_colors[organ] = plt.cm.hsv(h)

# diagonal line and fill area
min_val = 25
max_val = 100
plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5)

green_patch = plt.fill_between([-5, max_val], [-5, max_val], [max_val, max_val], 
                             alpha=0.1, color='green', label='CADS > TotalSeg')
red_patch = plt.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val], 
                           alpha=0.1, color='red', label='TotalSeg > CADS')

legend_shared = plt.scatter([], [], s=500, c='gray', alpha=0.6, label='Shared structures')
legend_unique = plt.scatter([], [], marker='D', s=200, c='gray', alpha=0.6, label='Unique structures')

# shared structures
for organ in shared_organs:
    x = dice_data[organ]['TotalSeg']
    y = dice_data[organ]['OMASeg']
    diff = y - x
    size = abs(diff) * 100 + 500  # increase the size of the markers
    
    plt.scatter(x, y, s=size, c=[organ_colors[organ]], alpha=0.7)
    texts.append(plt.text(x, y, organ, fontsize=18))  # increase font size

# unique structures
for organ in unique_organs:
    y = dice_data[organ]['OMASeg']
    plt.scatter(0, y, marker='D', c=[organ_colors[organ]], s=300, alpha=0.7)
    texts.append(plt.text(0, y, organ, fontsize=20))

# adjust text positions
adjust_text(texts,
           force_points=(2.0, 2.0),  # increase point-to-point repulsion
           force_text=(2.0, 2.0),    # incrfease text-to-text repulsion
           expand_points=(2.5, 2.5),  # inrease point-to-text repulsion
           expand_text=(2.5, 2.5),    # increase text-to-text repulsion
           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=0.5),
           text_from_point=True)

plt.xlim(-5, 105)
plt.ylim(25, 105)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('TotalSeg Dice Score (%) (0 = Not Supported)', fontsize=24)
plt.ylabel('CADS Dice Score (%)', fontsize=24)

legend_shared = plt.scatter([], [], s=600, c='gray', alpha=1, label='Mutual Structures')
legend_unique = plt.scatter([], [], marker='D', s=400, c='gray', alpha=1, label='Unique Structures')

plt.legend(handles=[green_patch, red_patch, legend_shared, legend_unique],
          bbox_to_anchor=(0.98, 0.02),
          loc='lower right',
          fontsize=22,
          framealpha=0.8,
          markerscale=0.5,
          labelspacing=0.8,
          handletextpad=1.2,
          borderpad=0.8) 

plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig('/mnt/hdda/murong/22k/plots/result_c_usz_scatter.png', dpi=300, bbox_inches='tight')
# plt.savefig('/mnt/hdda/murong/22k/plots/result_c_usz_scatter.eps', bbox_inches='tight')
plt.show()