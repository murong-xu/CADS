import argparse
import os
from utils.compute_metrics import compute_metrics

# import debugpy
# debugpy.listen(('0.0.0.0', 4444))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('You can debug your script now')


def all_or_names(value):
    if value == 'all':
        return value
    try:
        return [i for i in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Value must be 'all' or a comma-separated list of dataset names")


def all_or_indices(value):
    if value == 'all':
        return value
    try:
        return [int(i) for i in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Value must be 'all' or a comma-separated list of numbers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--preddir", required=True)
    parser.add_argument("-out", "--outputdir", required=True)
    parser.add_argument("-split", "--split",
                        help="val=2, test=0, train=1", required=True)
    parser.add_argument("-dataset", "--dataset",
                        help="'all' or names (comma-separated)", type=all_or_names, required=True)
    parser.add_argument("-totalseg_group", "--totalseg_group",
                        help="'all' or indices (comma-separated)", type=all_or_indices, required=False)
    parser.add_argument("-penalize_FP", "--penalize_FP",
                        help="True for giving penalization on FP predictions in metric scores (Dice, HD, etc.).", required=True)
    args = parser.parse_args()

    pred_dir = args.preddir
    output_dir = args.outputdir
    eval_datasets = args.dataset
    split = args.split
    score_penalize_FP = args.penalize_FP
    if args.totalseg_group is not None:
        totalseg_group = args.totalseg_group
    else:
        totalseg_group = 'all'

    if split == "0":
        split_name = "test_0"
    elif split == "2":
        split_name = "val_2"
    elif split == '1':
        split_name = 'train_1'
    else:
        raise ValueError(
            "Invalid split number. Choose from 0 (test), 1 (train), 2 (validation).")

    if score_penalize_FP == "True":
        score_penalize_FP = True
    elif score_penalize_FP == "False":
        score_penalize_FP = False
    else:
        raise ValueError("score_penalize_FP should be either True/False.")

    output_folder = os.path.join(output_dir, split_name)
    pred_folder = pred_dir
    path_avg_organ_volume = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/monai_models/avg_organ_volume/organ_volume_clipped_median.pkl"  # TODO: check

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    datasetfolders = []
    folders = os.listdir(pred_folder)
    for folder in folders:
        if eval_datasets == 'all':
            splits_folders = os.listdir(os.path.join(pred_folder, folder))
            for split_ in splits_folders:
                if split_ == split:
                    current_folder = os.path.join(pred_folder, folder, split_)
                    if len(os.listdir(current_folder)) > 0:
                        datasetfolders.append(current_folder)
        else:
            for eval_dataset in eval_datasets:
                if eval_dataset in folder:
                    splits_folders = os.listdir(
                        os.path.join(pred_folder, folder))
                    for split_ in splits_folders:
                        if split_ == split:
                            current_folder = os.path.join(
                                pred_folder, folder, split_)
                            if len(os.listdir(current_folder)) > 0:
                                datasetfolders.append(current_folder)

    datasetfolders.sort()
    for datasetfolder in datasetfolders:
        compute_metrics(datasetfolder, output_folder, path_avg_organ_volume, int(
            split), score_penalize_FP, totalseg_group)


if __name__ == "__main__":
    main()
