import argparse
import os
from cads.utils.compute_metrics import compute_metrics, compute_metrics_from_extra_test_set

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
    
def get_split_name(split):
    split_mapping = {"0": "test_0", "1": "train_1", "2": "val_2"}
    if split in split_mapping:
        return split_mapping[split]
    raise ValueError("Invalid split number. Choose from 0 (test), 1 (train), 2 (validation).")

def create_output_folder(output_dir, split_name=None):
    output_folder = os.path.join(output_dir, split_name) if split_name else output_dir
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def get_dataset_folders(pred_folder, eval_datasets, split):
    datasetfolders = []
    for folder in os.listdir(pred_folder):
        if eval_datasets == 'all' or any(eval_dataset in folder for eval_dataset in eval_datasets):
            for split_folder in os.listdir(os.path.join(pred_folder, folder)):
                if split is None or split_folder == split:
                    current_folder = os.path.join(pred_folder, folder, split_folder)
                    if os.listdir(current_folder):
                        datasetfolders.append(current_folder)
    return sorted(datasetfolders)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--preddir", required=True)
    parser.add_argument("-out", "--outputdir", required=True)
    parser.add_argument("-split", "--split",
                        help="val=2, test=0, train=1", required=False)
    parser.add_argument("-dataset", "--dataset",
                        help="'all' or names (comma-separated)", type=all_or_names, required=True)
    parser.add_argument("-totalseg_group", "--totalseg_group",
                        help="'all' or indices (comma-separated)", type=all_or_indices, required=False)
    parser.add_argument("-penalize_FP", "--penalize_FP",
                        help="True for giving penalization on FP predictions in metric scores (Dice, HD, etc.).", required=True)
    parser.add_argument("-extra_test", "--extra_test", action="store_true", default=False)
    parser.add_argument('--extra_test_pseudo_files_range', type=str, default="[1-9]", help='Range of pseudo files to consider')

    args = parser.parse_args()

    pred_dir = args.preddir
    output_dir = args.outputdir
    eval_datasets = args.dataset
    split = args.split
    score_penalize_FP = args.penalize_FP.lower() == "true"
    totalseg_group = args.totalseg_group if args.totalseg_group is not None else 'all'

    split_name = get_split_name(split) if split else None
    output_folder = create_output_folder(output_dir, split_name)

    path_avg_organ_volume = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/monai_models/avg_organ_volume/organ_volume_clipped_median.pkl"  # TODO: check
    
    datasetfolders = get_dataset_folders(pred_dir, eval_datasets, split)

    for datasetfolder in datasetfolders:
        if not args.extra_test:
            compute_metrics(datasetfolder, output_folder, path_avg_organ_volume, int(split), score_penalize_FP, totalseg_group)
        else:
            compute_metrics_from_extra_test_set(datasetfolder, output_folder, path_avg_organ_volume, score_penalize_FP, pseudo_files_range=args.extra_test_pseudo_files_range)



if __name__ == "__main__":
    main()
