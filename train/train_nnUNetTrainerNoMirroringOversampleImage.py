from nnunetv2.run.run_training import run_training_entry
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-prob_minority', type=float, default=None, required=False)
    args = parser.parse_args()
    return args

def main():
    sys.argv = ['/home/murong/22k/OMASeg_sync/OMASeg/train/train.py',
                '553', 
                '3d_fullres', 
                'all',
                '-tr', 'nnUNetTrainerNoMirroringOversampleImage',
                '-p', 'nnUNetResEncUNetLPlans',
                '-prob_minority', '0.35'
                ]
    run_training_entry()
    print('a')

if __name__ == "__main__":
    main()