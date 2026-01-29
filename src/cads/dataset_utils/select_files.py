import os
import json
import glob


def select_imgs(folder, split):
    """
    This function scans a folder for JSON files, reads the metadata to determine the split ('train', 'val', or 'test'), and selects corresponding image files if they exist.
    """
    json_files = glob.glob(folder + '/*.json')
    json_files.sort()

    images = []
    for jsonfile in json_files:
        file = open(jsonfile)
        metadata = json.load(file)
        file.close()
        if metadata['train'] == split:
            base = os.path.basename(jsonfile).split(
                '/')[-1].split('.json')[-2][:-5]
            imagefile = os.path.join(folder, base + '_0000.nii.gz')
            if os.path.exists(imagefile):
                images.append(imagefile)

    print(f'Number of images: {len(images)}')
    return images


def select_labels(folder, datasetname, split):
    ct_data_dir = os.path.join(folder, datasetname, 'images')

    json_files = glob.glob(ct_data_dir + '/*.json')
    json_files.sort()

    label_ids = []
    label = []
    label_split = []
    for jsonfile in json_files:
        file = open(jsonfile)
        metadata = json.load(file)
        file.close()
        if metadata["train"] == split:  # 1 train 0 test 2 validation
            base = os.path.basename(jsonfile).split('.')[0][:-5]
            labelfile = os.path.join(
                folder, datasetname, 'labels', base + '.nii.gz')
            if os.path.exists(labelfile):
                label_ids.append(base)
                label_split.append(metadata["train"])
                label.append(labelfile)
    return label, label_ids, label_split


def select_files_total_seg(datasetname, split):
    ctfolder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm'
    labelsfolder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/gt_data/Totalsegmentor_dataset_combined'

    ct_data_dir = os.path.join(ctfolder, datasetname, 'images')

    json_files = glob.glob(ct_data_dir + '/*.json')
    json_files.sort()

    folder_ids = []
    folders = []
    folder_split = []
    for jsonfile in json_files:
        file = open(jsonfile)
        metadata = json.load(file)
        file.close()
        if metadata["train"] == split:  # 1 train 0 test 2 validation
            base = os.path.basename(jsonfile).split('.')[0][:-5]
            imagefile = os.path.join(
                labelsfolder, base, base + '_combined.nii.gz')
            if os.path.exists(imagefile):
                folder_ids.append(base)
                folders.append(os.path.join(labelsfolder, base))
                folder_split.append(metadata["train"])

    print(f'Number of selected images in split ({split}): {len(folder_ids)}')
    return folders, folder_ids, folder_split

def select_files_total_seg_corrected(datasetname, split, labelsfolder):
    """
    After rib (255) and vertebrae (252) correction. 
    """
    ctfolder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm'
    ct_data_dir = os.path.join(ctfolder, datasetname, 'images')

    json_files = glob.glob(ct_data_dir + '/*.json')
    json_files.sort()

    folder_ids = []
    folders = []
    folder_split = []
    for jsonfile in json_files:
        file = open(jsonfile)
        metadata = json.load(file)
        file.close()
        if metadata["train"] == split:  # 1 train 0 test 2 validation
            base = os.path.basename(jsonfile).split('.')[0][:-5]
            labelfolder = os.path.join(labelsfolder, base)
            if os.path.exists(labelfolder):
                folder_ids.append(base)
                folders.append(os.path.join(labelsfolder, base))
                folder_split.append(metadata["train"])

    print(f'Number of selected images in split ({split}): {len(folder_ids)}')
    print(f'Using label folder: {labelsfolder}')
    return folders, folder_ids, folder_split


def select_labels_from_gt_data(folder, datasetname, gt_datasetname, split):
    labelsfolder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/gt_data'

    ct_data_dir = os.path.join(folder, datasetname, 'images')

    json_files = glob.glob(ct_data_dir + '/*.json')
    json_files.sort()

    label_ids = []
    label = []
    label_split = []
    for jsonfile in json_files:
        file = open(jsonfile)
        metadata = json.load(file)
        file.close()
        if metadata["train"] == split:  # 1 train 0 test 2 validation
            base = os.path.basename(jsonfile).split('.')[0][:-5]
            labelfile = os.path.join(
                labelsfolder, gt_datasetname, 'labels', base + '.nii.gz')
            if os.path.exists(labelfile):
                label_ids.append(base)
                label_split.append(metadata["train"])
                label.append(labelfile)
    return label, label_ids, label_split
