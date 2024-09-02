# import os
# os.environ['nnUNet_raw'] = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/final_nnUNets/nnUNet_raw"
# os.environ['nnUNet_preprocessed'] = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/final_nnUNets/nnUNet_preprocessed"
# os.environ['nnUNet_results'] = "/net/cephfs/shares/menze.dqbm.uzh/murong/20k/final_nnUNets/nnUNet_results"

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
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    args = parser.parse_args()
    return args

def main():
    # sys.argv = ['/data/muxu/20k/OMASeg/train/test.py',
    #             '551', 
    #             '3d_fullres', 
    #             'all',
    #             '-tr', 'nnUNetTrainer_4000epochs_NoMirroring',
    #             '-p', 'nnUNetResEncUNetLPlans',
    #             ]
    run_training_entry()

if __name__ == "__main__":
    main()