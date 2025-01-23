import glob
import os
from typing import Optional, List, Union
import nibabel as nib
import numpy as np
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
import seg_metrics.seg_metrics as sg
import logging
import pickle

from dataset_utils.datasets_labelmap import dataset2labelmap, dataset2labelmap_extra_test
from dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps
from dataset_utils.mappings import replace_labelmap, map_labels, replacements, FULLY_ANNOTATED_DATASETS, TOL_MISSING_VOLLUME_PERCENTAGE
from dataset_utils.postprocessing_for_eval import PostprocessingMetric, RecalculateAvgOrganVolume
from dataset_utils.select_files import select_labels, select_files_total_seg, select_labels_from_gt_data

from utils.metrics import compute_max_HD_distance, save_metric, save_missing_structure_check
from utils.libs import time_it

partname_mapping = {
        551: 251,
        552: 252,
        553: 253,
        554: 254,
        555: 255,
        556: 256,
        557: 257,
        558: 258,
        559: 259
    }


@time_it
def compute_metrics_other(input_dir, output_folder, path_avg_organ_volume, split, score_penalize_FP):
    name_split = input_dir.split('/')
    dataset = name_split[-2]

    labelmap = dataset2labelmap[dataset]
    labelmap = replace_labelmap(labelmap, replacements)

    # Select files
    if dataset == "0001_visceral_gc_new":
        gtfiles, ids, splits = select_labels(
            folder='/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm', datasetname=dataset, split=split)
        pseudo_files_range = "[1, 3, 6, 8]"
    elif dataset in ["0003_kits21", "0004_lits", "0034_empire"]:
        gtfiles, ids, splits = select_labels(
            folder='/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm', datasetname=dataset, split=split)
        pseudo_files_range = "[1]"
    elif dataset == "0039_han_seg":
        gtfiles, ids, splits = select_labels(
            folder='/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm', datasetname=dataset, split=split)
        pseudo_files_range = "[8, 9]"
    elif dataset == "0039_han_seg_reg":
        gtfiles, ids, splits = select_labels_from_gt_data(
            folder='/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm', datasetname="0039_han_seg", gt_datasetname="257_han_seg", split=split)
        pseudo_files_range = "[7]"
    elif dataset == "0040_saros":
        gtfiles, ids, splits = select_labels(
            folder='/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm', datasetname=dataset, split=split)
        pseudo_files_range = "[3, 9]"
    elif dataset == "0037_totalsegmentator":
        raise ValueError(
            "This function is not used for 0037_totalsegmentator.")
    else:
        gtfiles, ids, splits = select_labels(
            folder='/net/cephfs/shares/menze.dqbm.uzh/murong/20k/ct_ready_1mm', datasetname=dataset, split=split)
        pseudo_files_range = "[1-9]"

    # Use average organ size for setting thresholds
    with open(path_avg_organ_volume, "rb") as f:
        avg_organ_volume = pickle.load(f)
        f.close()
    recalculate_avg_organ_volume = RecalculateAvgOrganVolume(
        avg_volume=avg_organ_volume, tolerance=TOL_MISSING_VOLLUME_PERCENTAGE)
    avg_organ_volume = recalculate_avg_organ_volume.recalculate()

    # Init missing-structure checks
    postprocessing = PostprocessingMetric(datasetname=dataset)
    count_FP = {}
    count_FN = {}
    count_FN_ignore = {}
    count_FP_ignore = {}

    # Init metrics
    dicematrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1
    nsdmatrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1
    hdmatrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1
    hd95matrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1

    for i, (gtfile, base, split) in enumerate(zip(gtfiles, ids, splits)):
        print(f'Processing {i + 1}/{len(gtfiles)}: {base}')
        labelmap = dataset2labelmap[dataset]
        labelmap = replace_labelmap(labelmap, replacements)

        gt = nib.load(gtfile)
        spacing = gt.header.get_zooms()
        gt = gt.get_fdata()
        gt = gt.astype(np.uint8)
        gt_shape = gt.shape
        labelmap, gt = postprocessing.process_gt(labelmap, gt)
        unique_gt = np.unique(gt)

        pseudofolder = os.path.join(input_dir, base)
        pseudofiles = glob.glob(
            pseudofolder + "/*_part_55" + pseudo_files_range + ".nii.gz")
        pseudofiles.sort()
        for pseudofile in pseudofiles:
            pseudo = nib.load(pseudofile)
            pseudo = pseudo.get_fdata()
            pseudo = pseudo.astype(np.uint8)

            if pseudo.shape != gt_shape:
                print(
                    f'gt shape {gt_shape} and pseudo shape {pseudo.shape} are different')
                logging.info(
                    f"Dataset {dataset}: case {base} gt & pred shape mis-match!")
                continue

            part = int(os.path.basename(
                pseudofile).split('_')[-1].split('.')[0])
            pseudo_labelmap = map_taskid_to_labelmaps[part]
            pseudo_labelmap, pseudo = postprocessing.process_pseudo(
                part, pseudo_labelmap, pseudo)
            unique_pseudo = np.unique(pseudo)

            label_mapping = map_labels(
                labelmap, pseudo_labelmap, check_identical=True)

            new_unique_pseudo = []
            for key, value in label_mapping.items():
                if value in unique_pseudo:
                    new_unique_pseudo.append(key)
            new_unique_pseudo = np.array(new_unique_pseudo, dtype=np.uint8)

            # 1) FN
            ind_FN = np.intersect1d(np.setdiff1d(
                unique_gt, new_unique_pseudo), np.array(list(label_mapping.keys())))
            if len(ind_FN) != 0:
                for ind in ind_FN:
                    gt_structure = labelmap[ind]
                    array_ind_FN = np.argwhere(gt == ind)
                    FN_pred = pseudo[array_ind_FN[:, 0],
                                     array_ind_FN[:, 1], array_ind_FN[:, 2]]
                    volume = len(FN_pred)
                    pred_structure_label, pred_structure_count = np.unique(
                        FN_pred, return_counts=True)
                    pred_structure = {pseudo_labelmap[pred_structure_label[i]]: pred_structure_count[i] for i in range(
                        len(pred_structure_label))}
                    # 1.1) FN_ignored
                    if volume <= avg_organ_volume[partname_mapping[part]][gt_structure]:
                        if gt_structure not in count_FN_ignore.keys():
                            count_FN_ignore[gt_structure] = [
                                {base: [{"GT_vol": volume}, pred_structure]}]
                        else:
                            count_FN_ignore[gt_structure].append(
                                {base: [{"GT_vol": volume}, pred_structure]})
                        hd95matrix[i, ind] = -2
                        hdmatrix[i, ind] = -2
                        dicematrix[i, ind] = -2
                        nsdmatrix[i, ind] = -2
                    # 1.2) True FN
                    else:
                        if gt_structure not in count_FN.keys():
                            count_FN[gt_structure] = [
                                {base: [{"GT_vol": volume}, pred_structure]}]
                        else:
                            count_FN[gt_structure].append(
                                {base: [{"GT_vol": volume}, pred_structure]})
                        max_HD = compute_max_HD_distance(gt_shape, spacing)
                        hd95matrix[i, ind] = max_HD
                        hdmatrix[i, ind] = max_HD
                        dicematrix[i, ind] = 0
                        nsdmatrix[i, ind] = 0

            # 2) FP
            ind_FP = np.intersect1d(np.setdiff1d(
                new_unique_pseudo, unique_gt), np.array(list(label_mapping.keys())))
            if len(ind_FP) != 0:
                for ind in ind_FP:
                    ind_pred = label_mapping[ind]
                    array_ind_FP = np.argwhere(pseudo == ind_pred)
                    FP_pred = gt[array_ind_FP[:, 0],
                                 array_ind_FP[:, 1], array_ind_FP[:, 2]]
                    volume = len(FP_pred)
                    pred_structure = labelmap[ind]
                    gt_structure_label, gt_structure_count = np.unique(
                        FP_pred, return_counts=True)
                    gt_structure = {labelmap[gt_structure_label[i]]: gt_structure_count[i] for i in range(
                        len(gt_structure_label))}
                    # 2.1) True FP
                    if score_penalize_FP and dataset in FULLY_ANNOTATED_DATASETS:
                        if pred_structure not in count_FP.keys():
                            count_FP[pred_structure] = [
                                {base: [{"FP_pred_vol": volume}, gt_structure]}]
                        else:
                            count_FP[pred_structure].append(
                                {base: [{"FP_pred_vol": volume}, gt_structure]})
                        max_HD = compute_max_HD_distance(gt_shape, spacing)
                        hd95matrix[i, ind] = max_HD
                        hdmatrix[i, ind] = max_HD
                        dicematrix[i, ind] = 0
                        nsdmatrix[i, ind] = 0
                    # 2.2) FP_ignored
                    else:
                        if pred_structure not in count_FP_ignore.keys():
                            count_FP_ignore[pred_structure] = [
                                {base: [{"FP_pred_vol": volume}, gt_structure]}]
                        else:
                            count_FP_ignore[pred_structure].append(
                                {base: [{"FP_pred_vol": volume}, gt_structure]})
                        hd95matrix[i, ind] = -3
                        hdmatrix[i, ind] = -3
                        dicematrix[i, ind] = -3
                        nsdmatrix[i, ind] = -3

            ind_no_gt_and_pred = np.setdiff1d(
                np.array(list(labelmap.keys())), unique_gt)
            exceptions = np.concatenate([ind_FN, ind_FP, ind_no_gt_and_pred])
            for key, value in label_mapping.items():
                if key not in exceptions:
                    pred_binary = np.copy(pseudo)
                    gt_binary = np.copy(gt)

                    gt_binary[gt_binary != key] = 0
                    gt_binary[gt_binary == key] = 1

                    pred_binary[pred_binary != value] = 0
                    pred_binary[pred_binary == value] = 1

                    labels = [0, 1]
                    metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                               gdth_img=gt_binary,
                                               pred_img=pred_binary,
                                               # csv_file='cvsfile.csv',
                                               spacing=spacing,
                                               metrics=['dice', 'hd', 'hd95'])

                    hd95matrix[i, key] = metrics[0]['hd95'][0]
                    hdmatrix[i, key] = metrics[0]['hd'][0]
                    dicematrix[i, key] = metrics[0]['dice'][0]

                    gt_binary = gt_binary.astype(bool)
                    pred_binary = pred_binary.astype(bool)

                    # hausdorff_distance= directed_hausdorff(pred_binary,gt_binary)
                    normalized_surface_distance = compute_surface_dice_at_tolerance(
                        compute_surface_distances(gt_binary, pred_binary, spacing_mm=spacing), tolerance_mm=3)
                    nsdmatrix[i, key] = normalized_surface_distance

    # save metrics
    save_metric(replace_labelmap(dataset2labelmap[dataset], replacements), ids,
                hd95matrix, hdmatrix, nsdmatrix, dicematrix, output_folder, dataset, splits)
    # save checking results of missing structure
    count = {"FP": count_FP, "FN": count_FN,
             "FN_ignore": count_FN_ignore, "FP_ignore": count_FP_ignore}
    save_missing_structure_check(count, output_folder, dataset)

@time_it
def compute_metrics_totalseg(input_dir, output_folder, path_avg_organ_volume, split, score_penalize_FP, parts: Optional[Union[str, List[int]]]):
    name_split = input_dir.split('/')
    dataset = name_split[-2]
    print(f'Processing {dataset}')
    print(f'Input dir {input_dir}')

    gtfolders, ids, splits = select_files_total_seg(
        datasetname=dataset, split=split)
    if parts == 'all':
        parts = np.arange(551, 556)

    with open(path_avg_organ_volume, "rb") as f:
        avg_organ_volume = pickle.load(f)
        f.close()
    recalculate_avg_organ_volume = RecalculateAvgOrganVolume(
        avg_volume=avg_organ_volume, tolerance=TOL_MISSING_VOLLUME_PERCENTAGE)
    avg_organ_volume = recalculate_avg_organ_volume.recalculate()

    for part in parts:
        print(part)
        labelmap = map_taskid_to_labelmaps[part]
        labelmap = replace_labelmap(labelmap, replacements)

        # init
        count_FP = {}
        count_FN = {}
        count_FN_ignore = {}
        count_FP_ignore = {}

        dicematrix = np.ones(
            shape=(len(gtfolders), len(labelmap.values()))) * -1
        nsdmatrix = np.ones(
            shape=(len(gtfolders), len(labelmap.values()))) * -1
        hdmatrix = np.ones(shape=(len(gtfolders), len(labelmap.values()))) * -1
        hd95matrix = np.ones(
            shape=(len(gtfolders), len(labelmap.values()))) * -1

        for i, (gtfolder, base, split) in enumerate(zip(gtfolders, ids, splits)):
            print(f'Processing {i + 1}/{len(gtfolders)}: {base}')
            labelmap = map_taskid_to_labelmaps[part]
            labelmap = replace_labelmap(labelmap, replacements)
            pseudo_labelmap = map_taskid_to_labelmaps[part]
            pseudo_labelmap = replace_labelmap(pseudo_labelmap, replacements)

            gtfile = os.path.join(gtfolder, f'{base}_part_{str(partname_mapping[part])}.nii.gz')
            gt = nib.load(gtfile)
            spacing = gt.header.get_zooms()
            gt = gt.get_fdata()
            gt = gt.astype(np.uint8)
            gt_shape = gt.shape
            unique_gt = np.unique(gt)

            pseudofile = os.path.join(
                input_dir, base, f'{base}_part_{str(part)}.nii.gz')
            pseudo = nib.load(pseudofile)
            pseudo = pseudo.get_fdata()
            pseudo = pseudo.astype(np.uint8)
            unique_pseudo = np.unique(pseudo)

            if pseudo.shape != gt_shape:
                print(
                    f'gt shape {gt_shape} and pseudo shape {pseudo.shape} are different')
                logging.info(
                    f"Dataset {dataset}: case {base} gt & pred shape mis-match!")
                continue

            label_mapping = map_labels(
                labelmap, pseudo_labelmap, check_identical=True)

            new_unique_pseudo = []
            for key, value in label_mapping.items():
                if value in unique_pseudo:
                    new_unique_pseudo.append(key)
            new_unique_pseudo = np.array(new_unique_pseudo, dtype=np.uint8)

            # 1) FN
            ind_FN = np.intersect1d(np.setdiff1d(
                unique_gt, new_unique_pseudo), np.array(list(label_mapping.keys())))
            if len(ind_FN) != 0:
                for ind in ind_FN:
                    gt_structure = labelmap[ind]
                    array_ind_FN = np.argwhere(gt == ind)
                    FN_pred = pseudo[array_ind_FN[:, 0],
                                     array_ind_FN[:, 1], array_ind_FN[:, 2]]
                    volume = len(FN_pred)
                    pred_structure_label, pred_structure_count = np.unique(
                        FN_pred, return_counts=True)
                    pred_structure = {pseudo_labelmap[pred_structure_label[i]]: pred_structure_count[i] for i in range(
                        len(pred_structure_label))}
                    # 1.1) FN_ignored
                    if volume <= avg_organ_volume[partname_mapping[part]][gt_structure]:
                        if gt_structure not in count_FN_ignore.keys():
                            count_FN_ignore[gt_structure] = [
                                {base: [{"GT_vol": volume}, pred_structure]}]
                        else:
                            count_FN_ignore[gt_structure].append(
                                {base: [{"GT_vol": volume}, pred_structure]})
                        hd95matrix[i, ind] = -2
                        hdmatrix[i, ind] = -2
                        dicematrix[i, ind] = -2
                        nsdmatrix[i, ind] = -2
                    # 1.2) True FN
                    else:
                        if gt_structure not in count_FN.keys():
                            count_FN[gt_structure] = [
                                {base: [{"GT_vol": volume}, pred_structure]}]
                        else:
                            count_FN[gt_structure].append(
                                {base: [{"GT_vol": volume}, pred_structure]})
                        max_HD = compute_max_HD_distance(gt_shape, spacing)
                        hd95matrix[i, ind] = max_HD
                        hdmatrix[i, ind] = max_HD
                        dicematrix[i, ind] = 0
                        nsdmatrix[i, ind] = 0

            # 2) FP
            ind_FP = np.intersect1d(np.setdiff1d(
                new_unique_pseudo, unique_gt), np.array(list(label_mapping.keys())))
            if len(ind_FP) != 0:
                for ind in ind_FP:
                    ind_pred = label_mapping[ind]
                    array_ind_FP = np.argwhere(pseudo == ind_pred)
                    FP_pred = gt[array_ind_FP[:, 0],
                                 array_ind_FP[:, 1], array_ind_FP[:, 2]]
                    volume = len(FP_pred)
                    pred_structure = labelmap[ind]
                    gt_structure_label, gt_structure_count = np.unique(
                        FP_pred, return_counts=True)
                    gt_structure = {labelmap[gt_structure_label[i]]: gt_structure_count[i] for i in range(
                        len(gt_structure_label))}
                    # 2.1) True FP
                    if score_penalize_FP and dataset in FULLY_ANNOTATED_DATASETS:
                        if pred_structure not in count_FP.keys():
                            count_FP[pred_structure] = [
                                {base: [{"FP_pred_vol": volume}, gt_structure]}]
                        else:
                            count_FP[pred_structure].append(
                                {base: [{"FP_pred_vol": volume}, gt_structure]})
                        max_HD = compute_max_HD_distance(gt_shape, spacing)
                        hd95matrix[i, ind] = max_HD
                        hdmatrix[i, ind] = max_HD
                        dicematrix[i, ind] = 0
                        nsdmatrix[i, ind] = 0
                    # 2.2) FP_ignored
                    else:
                        if pred_structure not in count_FP_ignore.keys():
                            count_FP_ignore[pred_structure] = [
                                {base: [{"FP_pred_vol": volume}, gt_structure]}]
                        else:
                            count_FP_ignore[pred_structure].append(
                                {base: [{"FP_pred_vol": volume}, gt_structure]})
                        hd95matrix[i, ind] = -3
                        hdmatrix[i, ind] = -3
                        dicematrix[i, ind] = -3
                        nsdmatrix[i, ind] = -3

            ind_no_gt_and_pred = np.setdiff1d(
                np.array(list(labelmap.keys())), unique_gt)
            exceptions = np.concatenate([ind_FN, ind_FP, ind_no_gt_and_pred])
            for key, value in label_mapping.items():
                if key not in exceptions:
                    pred_binary = np.copy(pseudo)
                    gt_binary = np.copy(gt)

                    gt_binary[gt_binary != key] = 0
                    gt_binary[gt_binary == key] = 1

                    pred_binary[pred_binary != value] = 0
                    pred_binary[pred_binary == value] = 1

                    labels = [0, 1]
                    metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                               gdth_img=gt_binary,
                                               pred_img=pred_binary,
                                               # csv_file='cvsfile.csv',
                                               spacing=spacing,
                                               metrics=['dice', 'hd', 'hd95'])

                    hd95matrix[i, key] = metrics[0]['hd95'][0]
                    hdmatrix[i, key] = metrics[0]['hd'][0]
                    dicematrix[i, key] = metrics[0]['dice'][0]

                    gt_binary = gt_binary.astype(bool)
                    pred_binary = pred_binary.astype(bool)

                    # hausdorff_distance= directed_hausdorff(pred_binary,gt_binary)
                    normalized_surface_distance = compute_surface_dice_at_tolerance(
                        compute_surface_distances(gt_binary, pred_binary, spacing_mm=spacing), tolerance_mm=3)
                    nsdmatrix[i, key] = normalized_surface_distance

        # save metrics
        save_metric(labelmap, ids, hd95matrix, hdmatrix, nsdmatrix,
                    dicematrix, output_folder, dataset, splits, part=str(part))
        # save checking results of missing structure
        count = {"FP": count_FP, "FN": count_FN,
                 "FN_ignore": count_FN_ignore, "FP_ignore": count_FP_ignore}
        save_missing_structure_check(count, output_folder, dataset)


def compute_metrics(input_dir, output_folder, path_avg_organ_volume, split, score_penalize_FP, totalseg_group: Optional[Union[str, List[int]]] = None):
    name_split = input_dir.split('/')
    dataset = name_split[-2]
    print(f'Processing {dataset}')
    print(f'Input dir {input_dir}')

    if dataset == '0037_totalsegmentator':
        compute_metrics_totalseg(
            input_dir, output_folder, path_avg_organ_volume, split, score_penalize_FP, totalseg_group)
    else:
        compute_metrics_other(input_dir, output_folder,
                              path_avg_organ_volume, split, score_penalize_FP)


@time_it
def compute_metrics_from_extra_test_set(input_dir, output_folder, path_avg_organ_volume, score_penalize_FP, pseudo_files_range: str = "[1-9]"):
    name_split = input_dir.split('/')
    dataset = name_split[-2]

    labelmap = dataset2labelmap_extra_test[dataset]
    labelmap = replace_labelmap(labelmap, replacements)

    # Select files
    ids = glob.glob(input_dir + '/*')
    ids = [i.split('/')[-1] for i in ids]
    ids.sort()
    gt_all_folder = '/net/cephfs/shares/menze.dqbm.uzh/murong/20k/extra_test_data'  # TODO: 
    gt_folder = os.path.join(gt_all_folder, dataset, 'labels')
    gtfiles = [os.path.join(gt_folder, i+'.nii.gz') for i in ids]
    splits = [0] * len(gtfiles)

    # Use average organ size for setting thresholds
    with open(path_avg_organ_volume, "rb") as f:
        avg_organ_volume = pickle.load(f)
        f.close()
    recalculate_avg_organ_volume = RecalculateAvgOrganVolume(
        avg_volume=avg_organ_volume, tolerance=TOL_MISSING_VOLLUME_PERCENTAGE)
    avg_organ_volume = recalculate_avg_organ_volume.recalculate()

    # Init missing-structure checks
    postprocessing = PostprocessingMetric(datasetname=dataset)
    count_FP = {}
    count_FN = {}
    count_FN_ignore = {}
    count_FP_ignore = {}

    # Init metrics
    dicematrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1
    nsdmatrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1
    hdmatrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1
    hd95matrix = np.ones(shape=(len(gtfiles), len(labelmap.values()))) * -1

    for i, (gtfile, base, split) in enumerate(zip(gtfiles, ids, splits)):
        print(f'Processing {i + 1}/{len(gtfiles)}: {base}')
        labelmap = dataset2labelmap_extra_test[dataset]
        labelmap = replace_labelmap(labelmap, replacements)

        gt = nib.load(gtfile)
        spacing = gt.header.get_zooms()
        gt = gt.get_fdata()
        gt = gt.astype(np.uint8)
        gt_shape = gt.shape
        labelmap, gt = postprocessing.process_gt(labelmap, gt)
        unique_gt = np.unique(gt)

        pseudofolder = os.path.join(input_dir, base)
        pseudofiles = glob.glob(
            pseudofolder + "/*_part_55" + pseudo_files_range + ".nii.gz")
        pseudofiles.sort()
        for pseudofile in pseudofiles:
            pseudo = nib.load(pseudofile)
            pseudo = pseudo.get_fdata()
            pseudo = pseudo.astype(np.uint8)

            if pseudo.shape != gt_shape:
                print(
                    f'gt shape {gt_shape} and pseudo shape {pseudo.shape} are different')
                logging.info(
                    f"Dataset {dataset}: case {base} gt & pred shape mis-match!")
                continue

            part = int(os.path.basename(
                pseudofile).split('_')[-1].split('.')[0])
            pseudo_labelmap = map_taskid_to_labelmaps[part]
            pseudo_labelmap, pseudo = postprocessing.process_pseudo(
                part, pseudo_labelmap, pseudo)
            unique_pseudo = np.unique(pseudo)

            label_mapping = map_labels(
                labelmap, pseudo_labelmap, check_identical=True)

            new_unique_pseudo = []
            for key, value in label_mapping.items():
                if value in unique_pseudo:
                    new_unique_pseudo.append(key)
            new_unique_pseudo = np.array(new_unique_pseudo, dtype=np.uint8)

            # 1) FN
            ind_FN = np.intersect1d(np.setdiff1d(
                unique_gt, new_unique_pseudo), np.array(list(label_mapping.keys())))
            if len(ind_FN) != 0:
                for ind in ind_FN:
                    gt_structure = labelmap[ind]
                    array_ind_FN = np.argwhere(gt == ind)
                    FN_pred = pseudo[array_ind_FN[:, 0],
                                     array_ind_FN[:, 1], array_ind_FN[:, 2]]
                    volume = len(FN_pred)
                    pred_structure_label, pred_structure_count = np.unique(
                        FN_pred, return_counts=True)
                    pred_structure = {pseudo_labelmap[pred_structure_label[i]]: pred_structure_count[i] for i in range(
                        len(pred_structure_label))}
                    # 1.1) FN_ignored
                    if volume <= avg_organ_volume[partname_mapping[part]][gt_structure]:
                        if gt_structure not in count_FN_ignore.keys():
                            count_FN_ignore[gt_structure] = [
                                {base: [{"GT_vol": volume}, pred_structure]}]
                        else:
                            count_FN_ignore[gt_structure].append(
                                {base: [{"GT_vol": volume}, pred_structure]})
                        hd95matrix[i, ind] = -2
                        hdmatrix[i, ind] = -2
                        dicematrix[i, ind] = -2
                        nsdmatrix[i, ind] = -2
                    # 1.2) True FN
                    else:
                        if gt_structure not in count_FN.keys():
                            count_FN[gt_structure] = [
                                {base: [{"GT_vol": volume}, pred_structure]}]
                        else:
                            count_FN[gt_structure].append(
                                {base: [{"GT_vol": volume}, pred_structure]})
                        max_HD = compute_max_HD_distance(gt_shape, spacing)
                        hd95matrix[i, ind] = max_HD
                        hdmatrix[i, ind] = max_HD
                        dicematrix[i, ind] = 0
                        nsdmatrix[i, ind] = 0

            # 2) FP
            ind_FP = np.intersect1d(np.setdiff1d(
                new_unique_pseudo, unique_gt), np.array(list(label_mapping.keys())))
            if len(ind_FP) != 0:
                for ind in ind_FP:
                    ind_pred = label_mapping[ind]
                    array_ind_FP = np.argwhere(pseudo == ind_pred)
                    FP_pred = gt[array_ind_FP[:, 0],
                                 array_ind_FP[:, 1], array_ind_FP[:, 2]]
                    volume = len(FP_pred)
                    pred_structure = labelmap[ind]
                    gt_structure_label, gt_structure_count = np.unique(
                        FP_pred, return_counts=True)
                    gt_structure = {labelmap[gt_structure_label[i]]: gt_structure_count[i] for i in range(
                        len(gt_structure_label))}
                    # 2.1) True FP
                    if score_penalize_FP and dataset in FULLY_ANNOTATED_DATASETS:
                        if pred_structure not in count_FP.keys():
                            count_FP[pred_structure] = [
                                {base: [{"FP_pred_vol": volume}, gt_structure]}]
                        else:
                            count_FP[pred_structure].append(
                                {base: [{"FP_pred_vol": volume}, gt_structure]})
                        max_HD = compute_max_HD_distance(gt_shape, spacing)
                        hd95matrix[i, ind] = max_HD
                        hdmatrix[i, ind] = max_HD
                        dicematrix[i, ind] = 0
                        nsdmatrix[i, ind] = 0
                    # 2.2) FP_ignored
                    else:
                        if pred_structure not in count_FP_ignore.keys():
                            count_FP_ignore[pred_structure] = [
                                {base: [{"FP_pred_vol": volume}, gt_structure]}]
                        else:
                            count_FP_ignore[pred_structure].append(
                                {base: [{"FP_pred_vol": volume}, gt_structure]})
                        hd95matrix[i, ind] = -3
                        hdmatrix[i, ind] = -3
                        dicematrix[i, ind] = -3
                        nsdmatrix[i, ind] = -3

            ind_no_gt_and_pred = np.setdiff1d(
                np.array(list(labelmap.keys())), unique_gt)
            exceptions = np.concatenate([ind_FN, ind_FP, ind_no_gt_and_pred])
            for key, value in label_mapping.items():
                if key not in exceptions:
                    pred_binary = np.copy(pseudo)
                    gt_binary = np.copy(gt)

                    gt_binary[gt_binary != key] = 0
                    gt_binary[gt_binary == key] = 1

                    pred_binary[pred_binary != value] = 0
                    pred_binary[pred_binary == value] = 1

                    labels = [0, 1]
                    metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                               gdth_img=gt_binary,
                                               pred_img=pred_binary,
                                               # csv_file='cvsfile.csv',
                                               spacing=spacing,
                                               metrics=['dice', 'hd', 'hd95'])

                    hd95matrix[i, key] = metrics[0]['hd95'][0]
                    hdmatrix[i, key] = metrics[0]['hd'][0]
                    dicematrix[i, key] = metrics[0]['dice'][0]

                    gt_binary = gt_binary.astype(bool)
                    pred_binary = pred_binary.astype(bool)

                    # hausdorff_distance= directed_hausdorff(pred_binary,gt_binary)
                    normalized_surface_distance = compute_surface_dice_at_tolerance(
                        compute_surface_distances(gt_binary, pred_binary, spacing_mm=spacing), tolerance_mm=3)
                    nsdmatrix[i, key] = normalized_surface_distance

    # save metrics
    save_metric(replace_labelmap(dataset2labelmap_extra_test[dataset], replacements), ids,
                hd95matrix, hdmatrix, nsdmatrix, dicematrix, output_folder, dataset, splits)
    # save checking results of missing structure
    count = {"FP": count_FP, "FN": count_FN,
             "FN_ignore": count_FN_ignore, "FP_ignore": count_FP_ignore}
    save_missing_structure_check(count, output_folder, dataset)