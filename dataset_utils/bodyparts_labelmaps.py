# Labelmap of each sub-model
# 551
labelmap_part_organs = {
    0: "background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right"
}

# 552
labelmap_part_vertebrae = {
    0: "background",
    1: "vertebrae_L5",
    2: "vertebrae_L4",
    3: "vertebrae_L3",
    4: "vertebrae_L2",
    5: "vertebrae_L1",
    6: "vertebrae_T12",
    7: "vertebrae_T11",
    8: "vertebrae_T10",
    9: "vertebrae_T9",
    10: "vertebrae_T8",
    11: "vertebrae_T7",
    12: "vertebrae_T6",
    13: "vertebrae_T5",
    14: "vertebrae_T4",
    15: "vertebrae_T3",
    16: "vertebrae_T2",
    17: "vertebrae_T1",
    18: "vertebrae_C7",
    19: "vertebrae_C6",
    20: "vertebrae_C5",
    21: "vertebrae_C4",
    22: "vertebrae_C3",
    23: "vertebrae_C2",
    24: "vertebrae_C1"
}

# 553
labelmap_part_cardiac = {
    0: "background",
    1: "esophagus",
    2: "trachea",
    3: "heart_myocardium",
    4: "heart_atrium_left",
    5: "heart_ventricle_left",
    6: "heart_atrium_right",
    7: "heart_ventricle_right",
    8: "pulmonary_artery",
    9: "brain",
    10: "iliac_artery_left",
    11: "iliac_artery_right",
    12: "iliac_vena_left",
    13: "iliac_vena_right",
    14: "small_bowel",
    15: "duodenum",
    16: "colon",
    17: "urinary_bladder",
    18: "face"
}

# 554
labelmap_part_muscles = {
    0: "background",
    1: "humerus_left",
    2: "humerus_right",
    3: "scapula_left",
    4: "scapula_right",
    5: "clavicula_left",
    6: "clavicula_right",
    7: "femur_left",
    8: "femur_right",
    9: "hip_left",
    10: "hip_right",
    11: "sacrum",
    12: "gluteus_maximus_left",
    13: "gluteus_maximus_right",
    14: "gluteus_medius_left",
    15: "gluteus_medius_right",
    16: "gluteus_minimus_left",
    17: "gluteus_minimus_right",
    18: "autochthon_left",
    19: "autochthon_right",
    20: "iliopsoas_left",
    21: "iliopsoas_right"
}

# 555
labelmap_part_ribs = {
    0: "background",
    1: "rib_left_1",
    2: "rib_left_2",
    3: "rib_left_3",
    4: "rib_left_4",
    5: "rib_left_5",
    6: "rib_left_6",
    7: "rib_left_7",
    8: "rib_left_8",
    9: "rib_left_9",
    10: "rib_left_10",
    11: "rib_left_11",
    12: "rib_left_12",
    13: "rib_right_1",
    14: "rib_right_2",
    15: "rib_right_3",
    16: "rib_right_4",
    17: "rib_right_5",
    18: "rib_right_6",
    19: "rib_right_7",
    20: "rib_right_8",
    21: "rib_right_9",
    22: "rib_right_10",
    23: "rib_right_11",
    24: "rib_right_12"
}

# 556: 21 classes organs at risk in house and visceral
labelmap_part_oarrad = {
    0: "background",
    1: "spinal_canal",
    2: "larynx",
    3: "heart",
    4: "bowel_bag",
    5: "sigmoid",
    6: "rectum",
    7: "prostate",
    8: "seminal_vesicle",
    9: "left_mammary_gland",
    10: "right_mammary_gland",
    11: "sternum",
    12: "right psoas major",
    13: "left psoas major",
    14: "right rectus abdominis",
    15: "left rectus abdominis",
}

# 557: 10 classes head koens registration
labelmap_part_head = {
    0: "background",
    1: "white matter",
    2: "gray matter",
    3: "csf",
    4: "scalp",
    5: "eye balls",
    6: "compact bone",
    7: "spongy bone",
    8: "blood",
    9: "head muscles",
}

# 558: 30 lasses Head and neck structures from HanSeg
labelmap_part_headneck = {
    0: "background",
    1: "OAR_A_Carotid_L",
    2: "OAR_A_Carotid_R",
    3: "OAR_Arytenoid",
    4: "OAR_Bone_Mandible",
    5: "OAR_Brainstem",
    6: "OAR_BuccalMucosa",
    7: "OAR_Cavity_Oral",
    8: "OAR_Cochlea_L",
    9: "OAR_Cochlea_R",
    10: "OAR_Cricopharyngeus",
    11: "OAR_Esophagus_S",
    12: "OAR_Eye_AL",
    13: "OAR_Eye_AR",
    14: "OAR_Eye_PL",
    15: "OAR_Eye_PR",
    16: "OAR_Glnd_Lacrimal_L",
    17: "OAR_Glnd_Lacrimal_R",
    18: "OAR_Glnd_Submand_L",
    19: "OAR_Glnd_Submand_R",
    20: "OAR_Glnd_Thyroid",
    21: "OAR_Glottis",
    22: "OAR_Larynx_SG",
    23: "OAR_Lips",
    24: "OAR_OpticChiasm",
    25: "OAR_OpticNrv_L",
    26: "OAR_OpticNrv_R",
    27: "OAR_Parotid_L",
    28: "OAR_Parotid_R",
    29: "OAR_Pituitary",
}

# 559
labelmap_part_bodyregions = {
    0: "background",
    1: "subcutaneous_tissue",
    2: "muscle",
    3: "abdominal_cavity",
    4: "thoracic_cavity",
    5: "bones",
    6: "glands",
    7: "pericardium",
    8: "breast_implant",
    9: "mediastinum",
    10: "spinal_cord",
}

map_taskid_to_labelmaps = {
    551: labelmap_part_organs,
    552: labelmap_part_vertebrae,
    553: labelmap_part_cardiac,
    554: labelmap_part_muscles,
    555: labelmap_part_ribs,
    556: labelmap_part_oarrad,
    557: labelmap_part_head,
    558: labelmap_part_headneck,
    559: labelmap_part_bodyregions,
}

labelmap_all_structure = {
    1: 'spleen',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'gallbladder',
    5: 'liver',
    6: 'stomach',
    7: 'aorta',
    8: 'inferior_vena_cava',
    9: 'portal_vein_and_splenic_vein',
    10: 'pancreas',
    11: 'adrenal_gland_right',
    12: 'adrenal_gland_left',
    13: 'lung_upper_lobe_left',
    14: 'lung_lower_lobe_left',
    15: 'lung_upper_lobe_right',
    16: 'lung_middle_lobe_right',
    17: 'lung_lower_lobe_right',
    18: 'vertebrae_L5',
    19: 'vertebrae_L4',
    20: 'vertebrae_L3',
    21: 'vertebrae_L2',
    22: 'vertebrae_L1',
    23: 'vertebrae_T12',
    24: 'vertebrae_T11',
    25: 'vertebrae_T10',
    26: 'vertebrae_T9',
    27: 'vertebrae_T8',
    28: 'vertebrae_T7',
    29: 'vertebrae_T6',
    30: 'vertebrae_T5',
    31: 'vertebrae_T4',
    32: 'vertebrae_T3',
    33: 'vertebrae_T2',
    34: 'vertebrae_T1',
    35: 'vertebrae_C7',
    36: 'vertebrae_C6',
    37: 'vertebrae_C5',
    38: 'vertebrae_C4',
    39: 'vertebrae_C3',
    40: 'vertebrae_C2',
    41: 'vertebrae_C1',
    42: 'esophagus',
    43: 'trachea',
    44: 'heart_myocardium',
    45: 'heart_atrium_left',
    46: 'heart_ventricle_left',
    47: 'heart_atrium_right',
    48: 'heart_ventricle_right',
    49: 'pulmonary_artery',
    50: 'brain',
    51: 'iliac_artery_left',
    52: 'iliac_artery_right',
    53: 'iliac_vena_left',
    54: 'iliac_vena_right',
    55: 'small_bowel',
    56: 'duodenum',
    57: 'colon',
    58: 'urinary_bladder',
    59: 'face',
    60: 'humerus_left',
    61: 'humerus_right',
    62: 'scapula_left',
    63: 'scapula_right',
    64: 'clavicula_left',
    65: 'clavicula_right',
    66: 'femur_left',
    67: 'femur_right',
    68: 'hip_left',
    69: 'hip_right',
    70: 'sacrum',
    71: 'gluteus_maximus_left',
    72: 'gluteus_maximus_right',
    73: 'gluteus_medius_left',
    74: 'gluteus_medius_right',
    75: 'gluteus_minimus_left',
    76: 'gluteus_minimus_right',
    77: 'autochthon_left',
    78: 'autochthon_right',
    79: 'iliopsoas_left',
    80: 'iliopsoas_right',
    81: 'rib_left_1',
    82: 'rib_left_2',
    83: 'rib_left_3',
    84: 'rib_left_4',
    85: 'rib_left_5',
    86: 'rib_left_6',
    87: 'rib_left_7',
    88: 'rib_left_8',
    89: 'rib_left_9',
    90: 'rib_left_10',
    91: 'rib_left_11',
    92: 'rib_left_12',
    93: 'rib_right_1',
    94: 'rib_right_2',
    95: 'rib_right_3',
    96: 'rib_right_4',
    97: 'rib_right_5',
    98: 'rib_right_6',
    99: 'rib_right_7',
    100: 'rib_right_8',
    101: 'rib_right_9',
    102: 'rib_right_10',
    103: 'rib_right_11',
    104: 'rib_right_12',
    105: 'spinal_canal',
    106: 'larynx',
    107: 'heart',
    108: 'bowel_bag',
    109: 'sigmoid',
    110: 'rectum',
    111: 'prostate',
    112: 'seminal_vesicle',
    113: 'left_mammary_gland',
    114: 'right_mammary_gland',
    115: 'sternum',
    116: 'right psoas major',
    117: 'left psoas major',
    118: 'right rectus abdominis',
    119: 'left rectus abdominis',
    120: 'white matter',
    121: 'gray matter',
    122: 'csf',
    123: 'scalp',
    124: 'eye balls',
    125: 'compact bone',
    126: 'spongy bone',
    127: 'blood',
    128: 'head muscles',
    129: 'OAR_A_Carotid_L',
    130: 'OAR_A_Carotid_R',
    131: 'OAR_Arytenoid',
    132: 'OAR_Bone_Mandible',
    133: 'OAR_Brainstem',
    134: 'OAR_BuccalMucosa',
    135: 'OAR_Cavity_Oral',
    136: 'OAR_Cochlea_L',
    137: 'OAR_Cochlea_R',
    138: 'OAR_Cricopharyngeus',
    139: 'OAR_Esophagus_S',
    140: 'OAR_Eye_AL',
    141: 'OAR_Eye_AR',
    142: 'OAR_Eye_PL',
    143: 'OAR_Eye_PR',
    144: 'OAR_Glnd_Lacrimal_L',
    145: 'OAR_Glnd_Lacrimal_R',
    146: 'OAR_Glnd_Submand_L',
    147: 'OAR_Glnd_Submand_R',
    148: 'OAR_Glnd_Thyroid',
    149: 'OAR_Glottis',
    150: 'OAR_Larynx_SG',
    151: 'OAR_Lips',
    152: 'OAR_OpticChiasm',
    153: 'OAR_OpticNrv_L',
    154: 'OAR_OpticNrv_R',
    155: 'OAR_Parotid_L',
    156: 'OAR_Parotid_R',
    157: 'OAR_Pituitary',
    158: 'subcutaneous_tissue',
    159: 'muscle',
    160: 'abdominal_cavity',
    161: 'thoracic_cavity',
    162: 'bones',
    163: 'glands',
    164: 'pericardium',
    165: 'breast_implant',
    166: 'mediastinum',
    167: 'spinal_cord',
}

labelmap_all_structure_renamed = {
    1: 'Spleen',
    2: 'Kidney R',
    3: 'Kidney L',
    4: 'Gallbladder',
    5: 'Liver',
    6: 'Stomach',
    7: 'Aorta',
    8: 'Inferior vena cava',
    9: 'Portal and splenic vein',
    10: 'Pancreas',
    11: 'Adrenal gland R',
    12: 'Adrenal gland L',
    13: 'Lung upper lobe L',
    14: 'Lung lower lobe L',
    15: 'Lung upper lobd R',
    16: 'Lung middle lobe R',
    17: 'Lung lower lobe R',
    18: 'Vertebrae L5',
    19: 'Vertebrae L4',
    20: 'Vertebrae L3',
    21: 'Vertebrae L2',
    22: 'Vertebrae L1',
    23: 'Vertebrae T12',
    24: 'Vertebrae T11',
    25: 'Vertebrae T10',
    26: 'Vertebrae T9',
    27: 'Vertebrae T8',
    28: 'Vertebrae T7',
    29: 'Vertebrae T6',
    30: 'Vertebrae T5',
    31: 'Vertebrae T4',
    32: 'Vertebrae T3',
    33: 'Vertebrae T2',
    34: 'Vertebrae T1',
    35: 'Vertebrae C7',
    36: 'Vertebrae C6',
    37: 'Vertebrae C5',
    38: 'Vertebrae C4',
    39: 'Vertebrae C3',
    40: 'Vertebrae C2',
    41: 'Vertebrae C1',
    42: 'Esophagus',
    43: 'Trachea',
    44: 'Heart myocardium',
    45: 'Heart atrium L',
    46: 'Heart ventricle L',
    47: 'Heart atrium R',
    48: 'Heart ventricle R',
    49: 'Pulmonary artery',
    50: 'Brain',
    51: 'Iliac artery L',
    52: 'Iliac artery R',
    53: 'Iliac vena L',
    54: 'Iliac vena R',
    55: 'Small bowel',
    56: 'Duodenum',
    57: 'Colon',
    58: 'Urinary bladder',
    59: 'Face',
    60: 'Humerus L',
    61: 'Humerus R',
    62: 'Scapula L',
    63: 'Scapula R',
    64: 'Clavicula L',
    65: 'Clavicula R',
    66: 'Femur L',
    67: 'Femur R',
    68: 'Hip L',
    69: 'Hip R',
    70: 'Sacrum',
    71: 'Gluteus maximus L',
    72: 'Gluteus maximus R',
    73: 'Gluteus medius L',
    74: 'Gluteus medius R',
    75: 'Gluteus minimus L',
    76: 'Gluteus minimus R',
    77: 'Autochthon L',
    78: 'Autochthon R',
    79: 'Iliopsoas L',
    80: 'Iliopsoas R',
    81: 'Rib-1 L',
    82: 'Rib-2 L',
    83: 'Rib-3 L',
    84: 'Rib-4 L',
    85: 'Rib-5 L',
    86: 'Rib-6 L',
    87: 'Rib-7 L',
    88: 'Rib-8 L',
    89: 'Rib-9 L',
    90: 'Rib-10 L',
    91: 'Rib-11 L',
    92: 'Rib-12 L',
    93: 'Rib-1 R',
    94: 'Rib-2 R',
    95: 'Rib-3 R',
    96: 'Rib-4 R',
    97: 'Rib-5 R',
    98: 'Rib-6 R',
    99: 'Rib-7 R',
    100: 'Rib-8 R',
    101: 'Rib-9 R',
    102: 'Rib-10 R',
    103: 'Rib-11 R',
    104: 'Rib-12 R',
    105: 'Spinal canal',
    106: 'Larynx',
    107: 'Heart',
    108: 'Bowel bag',
    109: 'Sigmoid',
    110: 'Rectum',
    111: 'Prostate',
    112: 'Seminal vesicle',
    113: 'Mammary gland L',
    114: 'Mammary gland R',
    115: 'Sternum',
    116: 'Psoas major R',
    117: 'Psoas major L',
    118: 'Rectus abdominis R',
    119: 'Rectus abdominis L',
    120: 'White matter',
    121: 'Gray matter',
    122: 'CSF',
    123: 'Scalp',
    124: 'Eye balls',
    125: 'Compact bone',
    126: 'Spongy bone',
    127: 'Blood',
    128: 'Head muscles',
    129: 'Carotid artery L',
    130: 'Carotid artery R',
    131: 'Arytenoids',
    132: 'Mandible',
    133: 'Brainstem',
    134: 'Buccal mucosa',
    135: 'Oral cavity',
    136: 'Cochlea L',
    137: 'Cochlea R',
    138: 'Cricopharyngeal inlet',
    139: 'Cervical esophagus',
    140: 'Anterior eyeball L',
    141: 'Anterior eyeball R',
    142: 'Posterior eyeball L',
    143: 'Posterior eyeball R',
    144: 'Lacrimal gland L',
    145: 'Lacrimal gland R',
    146: 'Submandibular gland L',
    147: 'Submandibular gland R',
    148: 'Thyroid',
    149: 'Larynx-glottis',
    150: 'Larynx-supraglottic',
    151: 'Lips',
    152: 'Optic chiasm',
    153: 'Optic nerve L',
    154: 'Optic nerve R',
    155: 'Parotid gland L',
    156: 'Parotid gland R',
    157: 'Pituitary gland',
    158: 'Subcutaneous tissue',
    159: 'Muscle',
    160: 'Abdominal cavity',
    161: 'Thoracic cavity',
    162: 'Bones',
    163: 'Glands',
    164: 'Pericardium',
    165: 'Breast implant',
    166: 'Mediastinum',
    167: 'Spinal cord',
}

except_labels_combine = [
    'background',
    'brain',  # place for gray-white matter
    'face',  # for visualization
    'glands',  # from saros
    'scalp',  # cover all head structures
    'eye balls',  # repeated in hanseg
    'compact bone',  # overlaps with hanseg structures
    'spongy bone',  # overlaps with hanseg structures
    'blood',  # overlaps with hanseg structures
    'head muscles',  # overlaps with hanseg structures
    "muscle",
    "abdominal_cavity",
    "bones",  # overlaps for vertebrae
    "pericardium",
    "breast_implant",
    "mediastinum",
    "spinal_cord",
    "heart",
    "bowel_bag",
]

# Structures not included in TotalSegmentator v2
totalseg_exclude_to_compare = [
    'Psoas major R',
    'Psoas major L',
    'Rectus abdominis R',
    'Rectus abdominis L',
    'Spinal canal', 
    'Bowel bag',
    'Sigmoid',
    'Rectum',
    'Seminal vesicle',
    'Mammary gland L',
    'Mammary gland R',
    'Arytenoids',
    'Mandible',
    'Buccal mucosa',
    'Oral cavity',
    'Cochlea L',
    'Cochlea R',
    'Cricopharyngeal inlet',
    'Cervical esophagus',
    'Anterior eyeball L',
    'Anterior eyeball R',
    'Posterior eyeball L',
    'Posterior eyeball R',
    'Lacrimal gland L',
    'Lacrimal gland R',
    'Larynx-glottis',
    'Larynx-supraglottic',
    'Lips',
    'Optic chiasm',
    'Pituitary gland',
    'White matter',
    'Gray matter',
    'CSF',
    'head bone',  # not in our list but need to exclude due to dataset structure
    'Scalp',
    'Eye balls',
    'Compact bone',
    'Spongy bone',
    'Blood',
    'Head muscles',
    'Subcutaneous tissue',
    'Muscle',
    'Abdominal cavity',
    'Thoracic cavity',
    'Bones',
    'Glands',
    'Pericardium',
    'Breast implant',
    'Mediastinum',
]