splits = {0: 'test', 1: 'train', 2: 'val', 3: 'no_orig_labels_retrain'}

# List of datasets
datasets = [
    '0001_visceral_gc',
    '0002_visceral_sc',
    '0003_kits21',
    '0004_lits',
    '0005_bcv_abdomen',
    '0006_bcv_cervix',
    '0007_chaos',
    '0008_ctorg',
    '0009_abdomenct1k',
    '0010_verse',
    '0011_exact',
    '0012_cad_pe',
    '0013_ribfrac',
    '0014_learn2reg',
    '0015_lndb',
    '0016_lidc',
    '0017_lola11',
    '0018_sliver07',
    '0019_tcia_ct_lymph_nodes',
    '0020_tcia_cptac_ccrcc',
    '0021_tcia_cptac_luad',
    '0022_tcia_ct_images_covid19',
    '0023_tcia_nsclc_radiomics',
    '0024_pancreas_ct',
    '0025_pancreatic_ct_cbct_seg',
    '0026_rider_lung_ct',
    '0027_tcia_tcga_kich',
    '0028_tcia_tcga_kirc',
    '0029_tcia_tcga_kirp',
    '0030_tcia_tcga_lihc',
    '0032_stoic2021',
    '0033_tcia_nlst',
    '0034_empire',
    '0035_ct_tri',
    '0037_totalsegmentator',
    '0038_amos',
    '0039_han_seg',
    '0040_saros',
    '0041_3k',
]

# Datasets with GT labels
datasets_with_gt = [
    '0001_visceral_gc',
    '0001_visceral_gc_new',
    '0002_visceral_sc',
    '0003_kits21',
    '0004_lits',
    '0005_bcv_abdomen',
    '0006_bcv_cervix',
    '0007_chaos',
    '0008_ctorg',
    '0009_abdomenct1k',
    '0010_verse',
    '0014_learn2reg',
    '0018_sliver07',
    '0034_empire',
    '0037_totalsegmentator',
    '0038_amos',
    '0039_han_seg',
    '0040_saros',
]

# 0001_visceral_gc
visceral_labelmap = {
    0: 'background',
    1: 'liver',
    2: 'spleen',
    3: 'pancreas',
    4: 'gall bladder',  # map to 'gallbladder'
    5: 'urinary bladder',  # map to 'urinary_bladder'
    6: 'aorta',
    7: 'trachea',
    8: 'right lung',  # not exists in 9 models (but 5x lung lobes)
    9: 'left lung',  # not exists in 9 models (but 5x lung lobes)
    10: 'sternum',
    11: 'thyroid gland',  # map to 'OAR_Glnd_Thyroid'
    12: 'first lumbar vertebrae',  # not exists in 9 models
    13: 'right kidney',  # map to 'kidney_right'
    14: 'left kidney',  # map to 'kidney_left'
    15: 'right adrenal gland',  # map to 'adrenal_gland_right'
    16: 'left adrenal gland',  # map to 'adrenal_gland_left'
    17: 'right psoas major',
    18: 'left psoas major',
    19: 'right rectus abdominis',
    20: 'left rectus abdominis',
}

# 0001_visceral_gc_new
visceralnew_labelmap = {
    0: 'background',
    1: 'brainstem',
    2: 'spinal_canal',
    3: 'left_parotid_gland',
    4: 'right_parotid_gland',
    5: 'left_submandibular gland',
    6: 'right_submandibular_gland',
    7: 'larynx',
    8: 'heart',
    9: 'esophagus',
    10: 'stomach',
    11: 'bowel_bag',
    12: 'sigmoid',
    13: 'rectum',
    14: 'prostate',
    15: 'seminal_vesicle',
    16: 'left_mammary_gland',
    17: 'right_mammary_gland',
    18: 'sternum',
    19: 'right psoas major',
    20: 'left psoas major',
    21: 'right rectus abdominis',
    22: 'left rectus abdominis',
}

# KITS2021
kits_labelmap = {
    0: 'background',
    1: 'kidney',  # not exists in 9x models (but kiendy_left, kidney_right)
    2: 'lesion',
    3: 'cyst',
}
# LITS
lits_labelmap = {
    0: 'background',
    1: 'liver',
    2: 'lesion'
}

# BVC Abdomen
bvc_abdomen_labelmap = {
    0: 'background',
    1: 'spleen',
    2: 'right kidney',  # map to 'kidney_right'
    3: 'left kidney',  # map to 'kidney_left'
    4: 'gallbladder',
    5: 'esophagus',
    6: 'liver',
    7: 'stomach',
    8: 'aorta',
    9: 'inferior vena cava',  # map to 'inferior_vena_cava'
    10: 'portal and splenic_vein',  # map to 'portal_vein_and_splenic_vein'
    11: 'pancreas',
    12: 'right adrenal gland',  # map to 'adrenal_gland_right'
    13: 'left adrenal gland',  # map to 'adrenal_gland_left'
}

# BVC Cervix
bvc_cervix_labelmap = {
    0: 'background',
    1: 'urinary bladder',  # map to 'urinary_bladder'
    2: 'uterus',  # not exists in 9x models
    3: 'rectum',
    4: 'small bowel',  # map to 'small_bowel'
}

# CHAOS
chaos_labelmap = {
    0: 'background',
    1: 'liver',
}

# EMPIRE
empire_labelmap = {
    0: 'background',
    1: 'lungs',  # not exists in 9 models (but 5x lung lobes)
}


# Learn2Reg
learn2reg_labelmap = {
    0: 'background',
    1: 'liver',
    2: 'spleen',
    3: 'right kidney',  # map to 'kidney_right'
    4: 'left kidney',  # map to 'kidney_left'
}

# CTORG
ctorg_labelmap = {
    0: 'background',
    1: 'liver',
    2: 'urinary bladder',  # map to 'urinary_bladder'
    3: 'lungs',  # not exists in 9 models (but 5x lung lobes)
    4: 'kidneys',  # not exists in 9x models (but kiendy_left, kidney_right)
    # not exists in totalsegmentator {ribs + vertebrae is the closest}
    5: 'bone',
    # 6: 'brain',  # annotation doesn't exist in our available data
}

# Abdominal 1k
abdominal1k_labelmap = {0: 'background',
                        1: 'liver',
                        # not exists in 9x models (but kiendy_left, kidney_right)
                        2: 'kidney',
                        3: 'spleen',
                        4: 'pancreas',
                        }


# Abdominal 1k
sliver_labelmap = {0: 'background',
                   1: 'liver'
                   }

# verse
verse_labelmap = {
    0: 'background',
    1: 'vertebrae_C1',
    2: 'vertebrae_C2',
    3: 'vertebrae_C3',
    4: 'vertebrae_C4',
    5: 'vertebrae_C5',
    6: 'vertebrae_C6',
    7: 'vertebrae_C7',
    8: 'vertebrae_T1',
    9: 'vertebrae_T2',
    10: 'vertebrae_T3',
    11: 'vertebrae_T4',
    12: 'vertebrae_T5',
    13: 'vertebrae_T6',
    14: 'vertebrae_T7',
    15: 'vertebrae_T8',
    16: 'vertebrae_T9',
    17: 'vertebrae_T10',
    18: 'vertebrae_T11',
    19: 'vertebrae_T12',
    20: 'vertebrae_L1',
    21: 'vertebrae_L2',
    22: 'vertebrae_L3',
    23: 'vertebrae_L4',
    24: 'vertebrae_L5',
    25: 'vertebrae_L6',  # not exists in 9x models, skip transitional
    26: 'Sacrum',  # not exists in either GT annotation & 9x models, ignore
    27: 'Cocc',  # not exists in either GT annotation & 9x models, ignore
    28: 'vertebrae_T13'  # not exists in 9x models, skip transitional
}

# AMOS
amos_labelmap = {0: 'background',
                 1: 'spleen',
                 2: 'right kidney',  # map to 'kidney_right'
                 3: 'left kidney',  # map to 'kidney_left'
                 4: 'gall bladder',  # map to 'gallbladder'
                 5: 'esophagous',  # map to 'esophagus'
                 6: 'liver',
                 7: 'stomach',
                 8: 'aorta',
                 9: 'postcava',  # map to 'inferior_vena_cava'
                 10: 'pancreas',
                 11: 'right adrenal gland',  # map to 'adrenal_gland_right'
                 12: 'left adrenal gland',  # map to 'adrenal_gland_left'
                 13: 'duodenum',
                 14: 'bladder',  # map to 'urinary_bladder'
                 15: 'prostate/uterus'  # map to 'prostate'
                 }


totalsegmentator_labelmap = {
    0: 'background',
    1: 'spleen',
    2: 'right kidney',  # map to 'kidney_right'
    3: 'left kidney',  # map to 'kidney_left'
    4: 'gall bladder',  # map to 'gallbladder'
    5: 'liver',
    6: 'stomach',
    7: 'aorta',
    8: 'inferior vena cava',  # map to 'inferior_vena_cava'
    9: 'portal vein and splenic vein',  # map to 'portal_vein_and_splenic_vein'
    10: 'pancreas',
    11: 'right adrenal gland',  # map to 'adrenal_gland_right'
    12: 'left adrenal gland',  # map to 'adrenal_gland_left'
    13: 'lung upper lobe_left',  # map to 'lung_upper_lobe_left'
    14: 'lung lower lobe left',  # map to 'lung_lower_lobe_left'
    15: 'lung upper lobe right',  # map to 'lung_upper_lobe_right'
    16: 'lung middle lobe right',  # map to 'lung_middle_lobe_right'
    17: 'lung lower lobe right',  # map to 'lung_lower_lobe_right'
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
    42: 'esophagous',  # map to 'esophagus'
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
    58: 'rib_left_1',
    59: 'rib_left_2',
    60: 'rib_left_3',
    61: 'rib_left_4',
    62: 'rib_left_5',
    63: 'rib_left_6',
    64: 'rib_left_7',
    65: 'rib_left_8',
    66: 'rib_left_9',
    67: 'rib_left_10',
    68: 'rib_left_11',
    69: 'rib_left_12',
    70: 'rib_right_1',
    71: 'rib_right_2',
    72: 'rib_right_3',
    73: 'rib_right_4',
    74: 'rib_right_5',
    75: 'rib_right_6',
    76: 'rib_right_7',
    77: 'rib_right_8',
    78: 'rib_right_9',
    79: 'rib_right_10',
    80: 'rib_right_11',
    81: 'rib_right_12',
    82: 'humerus_left',
    83: 'humerus_right',
    84: 'scapula_left',
    85: 'scapula_right',
    86: 'clavicula_left',
    87: 'clavicula_right',
    88: 'femur_left',
    89: 'femur_right',
    90: 'hip_left',
    91: 'hip_right',
    92: 'sacrum',
    93: 'face',
    94: 'gluteus_maximus_left',
    95: 'gluteus_maximus_right',
    96: 'gluteus_medius_left',
    97: 'gluteus_medius_right',
    98: 'gluteus_minimus_left',
    99: 'gluteus_minimus_right',
    100: 'autochthon_left',
    101: 'autochthon_right',
    102: 'iliopsoas_left',
    103: 'iliopsoas_right',
    104: 'urinary bladder'  # map to 'urinary_bladder'
}

head_labelmap = {
    0: 'background',
    1: 'white matter',
    2: 'gray matter',
    3: 'csf',
    4: 'head bone',
    5: 'scalp',
    6: 'eye balls',
    7: 'compact bone',
    8: 'spongy bone',
    9: 'blood',
    10: 'head muscles',
}

han_seg_labelmap = {
    0: 'background',
    1: 'OAR_A_Carotid_L',  # Carotid artery (left)
    2: 'OAR_A_Carotid_R',  # Carotid artery (right)
    3: 'OAR_Arytenoid',  # Arytenoids
    4: 'OAR_Bone_Mandible',  # Mandible
    5: 'OAR_Brainstem',  # Brainstem
    6: 'OAR_BuccalMucosa',  # Buccal mucosa
    7: 'OAR_Cavity_Oral',  # Oral cavity 
    8: 'OAR_Cochlea_L',  # Cochlea (left) 
    9: 'OAR_Cochlea_R',  # Cochlea (right) 
    10: 'OAR_Cricopharyngeus',  # Cricopharyngeal inlet
    11: 'OAR_Esophagus_S',  # Cervical esophagus
    12: 'OAR_Eye_AL',  # Anterior eyeball segment (left)
    13: 'OAR_Eye_AR',  # Anterior eyeball segment (right)
    14: 'OAR_Eye_PL',  # Posterior eyeball segment (left)
    15: 'OAR_Eye_PR',  # Posterior eyeball segment (right)
    16: 'OAR_Glnd_Lacrimal_L',  # Lacrimal gland (left)
    17: 'OAR_Glnd_Lacrimal_R',  # Lacrimal gland (right)
    18: 'OAR_Glnd_Submand_L',  # Submandibular gland (left)
    19: 'OAR_Glnd_Submand_R',  # Submandibular gland (right)
    20: 'OAR_Glnd_Thyroid',  # Thyroid
    21: 'OAR_Glottis',  # Larynx—glottis
    22: 'OAR_Larynx_SG',  # Larynx—supraglottic
    23: 'OAR_Lips',  # Lips
    24: 'OAR_OpticChiasm',  # Optic chiasm
    25: 'OAR_OpticNrv_L',  # Optic nerve (left)
    26: 'OAR_OpticNrv_R',  # Optic nerve (right)
    27: 'OAR_Parotid_L',  # Parotid gland (left)
    28: 'OAR_Parotid_R',  # Parotid gland (right)
    29: 'OAR_Pituitary',  # Pituitary gland
    30: 'OAR_SpinalCord',  # Spinal cord
}

saros_seg_labelmap = {
    0: 'background',
    1: 'subcutaneous_tissue',
    2: 'muscle',
    3: 'abdominal_cavity',
    4: 'thoracic_cavity',
    5: 'bones',
    6: 'glands',
    7: 'pericardium',
    8: 'breast_implant',
    9: 'mediastinum',
    10: 'brain',
    11: 'spinal_cord',
    255: 'nolabel'  # not exists in 9x models
}

void_labelmap = {0: None}

dataset2labelmap = {'0001_visceral_gc': visceral_labelmap,
                    '0001_visceral_gc_new': visceralnew_labelmap,
                    '0002_visceral_sc': visceral_labelmap,
                    '0003_kits21': kits_labelmap,
                    '0004_lits': lits_labelmap,
                    '0005_bcv_abdomen': bvc_abdomen_labelmap,
                    '0006_bcv_cervix': bvc_cervix_labelmap,
                    '0007_chaos': chaos_labelmap,
                    '0008_ctorg': ctorg_labelmap,
                    '0009_abdomenct1k': abdominal1k_labelmap,
                    '0010_verse': verse_labelmap,
                    '0011_exact': void_labelmap,
                    '0012_cad_pe': void_labelmap,
                    '0013_ribfrac': void_labelmap,
                    '0014_learn2reg': learn2reg_labelmap,
                    '0015_lndb': void_labelmap,
                    '0016_lidc': void_labelmap,
                    '0017_lola11': void_labelmap,
                    '0018_sliver07': sliver_labelmap,
                    '0019_tcia_ct_lymph_nodes': void_labelmap,
                    '0020_tcia_cptac_ccrcc': void_labelmap,
                    '0021_tcia_cptac_luad': void_labelmap,
                    '0022_tcia_ct_images_covid19': void_labelmap,
                    '0023_tcia_nsclc_radiomics': void_labelmap,
                    '0024_pancreas_ct': void_labelmap,
                    '0025_pancreatic_ct_cbct_seg': void_labelmap,
                    '0026_rider_lung_ct': void_labelmap,
                    '0027_tcia_tcga_kich': void_labelmap,
                    '0028_tcia_tcga_kirc': void_labelmap,
                    '0029_tcia_tcga_kirp': void_labelmap,
                    '0030_tcia_tcga_lihc': void_labelmap,
                    '0031_tcia_tcga_sarc': void_labelmap,
                    '0032_stoic2021': void_labelmap,
                    '0033_tcia_nlst': void_labelmap,
                    '0034_empire': empire_labelmap,
                    '0035_ct_tri': void_labelmap,
                    '0036_visceral_gc_new_annotations': visceralnew_labelmap,
                    '0037_totalsegmentator': totalsegmentator_labelmap,
                    '0038_amos': amos_labelmap,
                    '0039_han_seg': han_seg_labelmap,
                    '0039_han_seg_reg': head_labelmap,
                    '0040_saros': saros_seg_labelmap, }


dataset_times = {
    '0001_visceral_gc': '0-02:00:00',
    '0002_visceral_sc': '0-05:00:00',
    '0003_kits21': '0-08:00:00',
    '0004_lits': '0-05:00:00',
    '0005_bcv_abdomen': '0-02:00:00',
    '0006_bcv_cervix': '0-02:00:00',
    '0007_chaos': '0-02:00:00',
    '0008_ctorg': '0-05:00:00',
    '0009_abdomenct1k': '0-10:00:00',
    '0010_verse': '0-05:00:00',
    '0011_exact': '0-01:00:00',
    '0012_cad_pe': '0-01:00:00',
    '0013_ribfrac': '0-10:00:00',
    '0014_learn2reg': '0-01:00:00',
    '0015_lndb': '0-05:00:00',
    '0016_lidc': '0-10:00:00',
    '0017_lola11': '0-01:00:00',
    '0018_sliver07': '0-01:00:00',
    '0019_tcia_ct_lymph_nodes': '0-04:00:00',
    '0020_tcia_cptac_ccrcc': '0-05:00:00',
    '0021_tcia_cptac_luad': '0-03:00:00',
    '0022_tcia_ct_images_covid19': '0-03:00:00',
    '0023_tcia_nsclc_radiomics': '0-03:00:00',
    '0024_pancreas_ct': '0-02:00:00',
    '0025_pancreatic_ct_cbct_seg': '0-02:00:00',
    '0026_rider_lung_ct': '0-01:00:00',
    '0027_tcia_tcga_kich': '0-04:00:00',
    '0028_tcia_tcga_kirc': '0-05:00:00',
    '0029_tcia_tcga_kirp': '0-01:00:00',
    '0030_tcia_tcga_lihc': '0-05:00:00',
    '0032_stoic2021': '1-10:00:00',
    '0033_tcia_nlst': '1-10:00:00',
    '0034_empire': '0-01:00:00',
    '0035_ct_tri': '1-00:00:00',
    '0037_totalsegmentator': '1-00:00:00',
    '0038_amos': '0-03:00:00',
    '0039_han_seg': '0-03:00:00',
    '0040_saros': '1-00:00:00',
    '0041_3k': '1-00:00:00',
}

# Extra test sets
segthor_labelmap = {
    0: 'background',
    1: 'esophagus',
    2: 'heart',
    3: 'trachea',
    4: 'aorta',
}

ribseg_labelmap = {
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

word_labelmap = {
    0: 'background',
    1: 'liver',
    2: 'spleen',
    3: 'kidney_left',
    4: 'kidney_right',
    5: 'stomach',
    6: 'gallbladder',
    7: 'esophagus',
    8: 'pancreas',
    9: 'duodenum',
    10: 'colon',
    11: 'intestine',
    12: 'adrenal',
    13: 'rectum',
    14: 'urinary_bladder',
    15: 'head_of_femur_left',
    16: 'head_of_femur_right',
}

VNet_labelmap = {
    0: 'background',
    1: 'spleen',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'gallbladder',
    5: 'esophagus',
    6: 'liver',
    7: 'stomach',
    8: 'aorta',
    9: 'inferior_vena_cava',
    10: 'portal_vein_and_splenic_vein',
    11: 'pancreas',
    12: 'adrenal_gland_right',
    13: 'adrenal_gland_left',
    14: 'duodenum',
}

CTPelvic1K_labelmap = {
    0: 'background',
    1: 'sacrum', 
    2: 'hip_left',
    3: 'hip_right',
    4: 'lumbar_spine',
}

CTSpine1K_labelmap = {
    0: 'background',
    1: 'vertebrae_C1',
    2: 'vertebrae_C2',
    3: 'vertebrae_C3',
    4: 'vertebrae_C4',
    5: 'vertebrae_C5',
    6: 'vertebrae_C6',
    7: 'vertebrae_C7',
    8: 'vertebrae_T1',
    9: 'vertebrae_T2',
    10: 'vertebrae_T3',
    11: 'vertebrae_T4',
    12: 'vertebrae_T5',
    13: 'vertebrae_T6',
    14: 'vertebrae_T7',
    15: 'vertebrae_T8',
    16: 'vertebrae_T9',
    17: 'vertebrae_T10',
    18: 'vertebrae_T11',
    19: 'vertebrae_T12',
    20: 'vertebrae_L1',
    21: 'vertebrae_L2',
    22: 'vertebrae_L3',
    23: 'vertebrae_L4',
    24: 'vertebrae_L5',
    25: 'vertebrae_L6',
}

dataset2labelmap_extra_test = {
    '0080_SegTHOR': segthor_labelmap,
    '0081_ribseg': ribseg_labelmap,
    '0082_word': word_labelmap,
    '0083_word_lits': word_labelmap,
    '0084_BTCV_VNet': VNet_labelmap,
    '0085_TCIA_VNet': VNet_labelmap,
    '0086_Private_CTPelvic1K': CTPelvic1K_labelmap,
    '0087_COVID19_CTSpine1K': CTSpine1K_labelmap,
    '0088_MSD_CTSpine1K': CTSpine1K_labelmap,
}

dataset_renamed = {
    '0001_visceral_gc': 'VISCERAL Gold Corpus',
    '0001_visceral_gc_new': 'VISCERAL Gold Corpus-Extra',
    '0002_visceral_sc': 'VISCERAL Silver Corpus',
    '0003_kits21': 'KiTS',
    '0004_lits': 'LiTS ',
    '0005_bcv_abdomen': 'BTCV-Abdomen',
    '0006_bcv_cervix': 'BTCV-Cervix',
    '0007_chaos': 'CHAOS',
    '0008_ctorg': 'CT-ORG',
    '0009_abdomenct1k': 'AbdomenCT-1K',
    '0010_verse': 'VerSe',
    '0011_exact': 'EXACT09',
    '0012_cad_pe': 'CAD-PE',
    '0013_ribfrac': 'RibFrac ',
    '0014_learn2reg': 'Learn2reg',
    '0015_lndb': 'LNDb',
    '0016_lidc': 'LIDC-IDRI ',
    '0017_lola11': 'LOLA11',
    '0018_sliver07': 'SLIVER07',
    '0019_tcia_ct_lymph_nodes': 'CT Lymph Nodes',
    '0020_tcia_cptac_ccrcc': 'CPTAC-CCRCC',
    '0021_tcia_cptac_luad': 'CPTAC-LUAD',
    '0022_tcia_ct_images_covid19': 'CT Images in COVID-19',
    '0023_tcia_nsclc_radiomics': 'NSCLC Radiogenomics',
    '0024_pancreas_ct': 'Pancreas-CT',
    '0025_pancreatic_ct_cbct_seg': 'Pancreatic-CT-CBCT-SEG',
    '0026_rider_lung_ct': 'RIDER Lung CT',
    '0027_tcia_tcga_kich': 'TCGA-KICH',
    '0028_tcia_tcga_kirc': 'TCGA-KIRC',
    '0029_tcia_tcga_kirp': 'TCGA-KIRP',
    '0030_tcia_tcga_lihc': 'TCGA-LIHC',
    '0032_stoic2021': 'STOIC2021',
    '0033_tcia_nlst': 'National Lung Screening Trial (NLST)',
    '0034_empire': 'EMPIRE10',
    '0035_ct_tri': 'CT-TRI',
    '0037_totalsegmentator': 'Total-Segmentator',
    '0038_amos': 'AMOS',
    '0039_han_seg': 'HaN-Seg',
    '0039_han_seg_reg': 'HaN-Seg Extra Brain Labels',
    '0040_saros': 'SAROS',
    '0041_3k': 'Private Hospital Data',
}