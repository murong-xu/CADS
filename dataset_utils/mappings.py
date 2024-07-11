# For aligning the naming conventions across: each dataset -> 9x models
replacements = {
    'gall bladder': 'gallbladder',
    'urinary bladder': 'urinary_bladder',
    'right kidney': 'kidney_right',
    'left kidney': 'kidney_left',
    # 'brainstem': 'OAR_Brainstem',  # not do it here, we have 2 classes existing
    'left parotid gland': 'left_parotid_gland',
    'right parotid gland': 'right_parotid_gland',
    'right adrenal gland': 'adrenal_gland_right',
    'left adrenal gland': 'adrenal_gland_left',
    'portal and splenic_vein': 'portal_vein_and_splenic_vein',
    'portal vein and splenic vein': 'portal_vein_and_splenic_vein',
    'postcava': 'inferior_vena_cava',
    'inferior vena cava': 'inferior_vena_cava',
    'small bowel': 'small_bowel',
    'esophagous': 'esophagus',
    'prostate/uterus': 'prostate',
    'bladder': 'urinary_bladder', 
    # 0037_totalseg labels -> 9 models' predictions
    'lung upper lobe_left': 'lung_upper_lobe_left',
    'lung lower lobe left': 'lung_lower_lobe_left',
    'lung upper lobe right': 'lung_upper_lobe_right',
    'lung middle lobe right': 'lung_middle_lobe_right',
    'lung lower lobe right': 'lung_lower_lobe_right',
    'urinary bladder': 'urinary_bladder',
    # other new updates
    'thyroid gland': 'OAR_Glnd_Thyroid', # 0001_visceral_gc
    }


def replace_labelmap(labelmap, replacements):
    mapping = replacements
    # Iterate through the keys and values in labelmap1
    for key, value in labelmap.items():
        if value in mapping.keys():
            labelmap[key] = mapping[value]
    return labelmap


def map_labels(labelmap1, labelmap2, check_identical=False):
    label_mapping = {}
    # Iterate through the keys and values in labelmap1
    for key1, value1 in labelmap1.items():
        # Search for a similar label in labelmap2
        for key2, value2 in labelmap2.items():
            if value1.lower() in value2.lower() and value1 != 'background' and value2 != 'background':
                # If a similar label is found, add it to the mapping dictionary
                if value1 != value2 and check_identical:
                    continue
                else:
                    label_mapping[int(key1)] = key2
                    break

    return label_mapping

FULLY_ANNOTATED_DATASETS = ['0001_visceral_gc', '0001_visceral_gc_new', '0002_visceral_sc', '0003_kits21', '0004_lits', '0005_bcv_abdomen',
                            '0006_bcv_cervix', '0007_chaos', '0008_ctorg', '0009_abdomenct1k', '0014_learn2reg', '0034_empire', '0038_amos', '0039_han_seg', '0039_han_seg_reg']
TOL_MISSING_VOLLUME_PERCENTAGE = 0.1
