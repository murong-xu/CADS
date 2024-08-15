import os
import sys
import torch
import time
from pathlib import Path
import numpy as np
import nibabel as nib
from functools import partial
from p_tqdm import p_map
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from dataset_utils.bodyparts_labelmaps import labelmap_all_structure, map_taskid_to_labelmaps, except_labels_combine
from utils.snapshot import generate_snapshot
from utils.libs import time_it


def get_task_model_folder(task_id: int, model_folder: str, trainer: str):
    """
    Find the sub-directory containing 'nnUNetResEncUNetLPlans' for a given task in the model folder.
    """
    model_folder_path = Path(model_folder)
    dirs = [d for d in model_folder_path.glob("*") if d.is_dir()]
    for d in dirs:
        if f'Dataset{task_id:03d}' in d.name:
            subfolder = d.resolve()
            subdirs = [sd for sd in subfolder.glob(
                "*") if sd.is_dir() and 'nnUNetResEncUNetLPlans' in sd.name]
            if subdirs:
                for subdir in subdirs:
                    if trainer in subdir.name:
                        print(f'Inference using model {subdir.name}')
                        return str(subdir.resolve())
                else:
                    raise ValueError(
                        f"No subdirectory containing {trainer} found in {subfolder}")
            else:
                raise ValueError(
                    f"No subdirectory containing 'nnUNetResEncUNetLPlans' found in {subfolder}")

    raise ValueError(
        f'task id {task_id} not found in model folder {model_folder}')


class nnUNetv2Predictor():
    """
    A class to handle predictions using nnUNetv2 models.
    This class initializes an nnUNetv2 predictor for a specified task and provides methods for both batch and single image predictions.
    """

    def __init__(self, model_folder, task_id, device, batch_predict=False, folds='all', checkpoint='checkpoint_final.pth', trainer='nnUNetTrainerNoMirroring', num_threads_preprocessing=6, num_threads_nifti_save=2, verbose=False):
        self.num_threads_preprocessing = num_threads_preprocessing
        self.num_threads_nifti_save = num_threads_nifti_save
        self.device = device

        # Use default settings recommended by nnUNet (also same as TotalSeg_v2)
        self.predictor = nnUNetPredictor(tile_step_size=0.5,
                                         use_gaussian=True,
                                         use_mirroring=False,
                                         perform_everything_on_device=True,
                                         device=self.device,
                                         verbose=verbose,
                                         allow_tqdm=True,
                                         verbose_preprocessing=verbose)
        task_id = int(task_id)
        try:
            task_model_folder = get_task_model_folder(task_id, model_folder, trainer)
        except ValueError as e:
            print(e)
            sys.exit(1)

        self.predictor.initialize_from_trained_model_folder(
            task_model_folder, folds, checkpoint)

        if batch_predict:
            self.predict = self._nnUNetv2_batch_predict
        else:
            self.predict = self._nnUNetv2_predict

    @time_it
    def _nnUNetv2_batch_predict(self, folder_in, folder_out):
        """
        Identical to command nnUNetv2_predict, works for batch predictions (predicting many images at once), supposed to be faster.
        """
        # TODO: check
        self.predictor.predict_from_files(folder_in, folder_out, save_probabilities=False,
                                          overwrite=False,
                                          num_processes_preprocessing=self.num_threads_preprocessing,
                                          num_processes_segmentation_export=self.num_threads_nifti_save,
                                          folder_with_segs_from_prev_stage=None,
                                          num_parts=1, part_id=0)  # TODO: num_parts and part_id?

    @time_it
    def _nnUNetv2_predict(self, file_in, file_out):
        """
        Identical to command nnUNetv2_predict, works for predicting one image at a time.
        """
        img, props = SimpleITKIO().read_images([file_in])
        self.predictor.predict_single_npy_array(
            img, props, None, file_out, False)


def save_target_to_nifti(target, seg, output_folder, labelmap_inv, basename, original_affine):
    include_value = labelmap_inv[target]
    mask = np.isin(seg, include_value)
    binary_img = np.where(mask, 1, 0)

    output_path = os.path.join(output_folder, f'{basename}_{target}.nii.gz')
    nib.save(nib.Nifti1Image(binary_img.astype(
        np.uint8), original_affine), output_path)


def save_targets_to_nifti(save_separate_targets, output_targets_dir, affine, labelmap, seg, patient_id, nr_threads_saving):
    """
    Save a specific target segmentation to a binary nifti file.
    """
    targets = [structure for structure in save_separate_targets if structure in list(
        labelmap.values())]
    if targets:
        labelmap_inv = {v: k for k, v in labelmap.items()}
        _ = p_map(partial(save_target_to_nifti, seg=seg, output_folder=output_targets_dir, labelmap_inv=labelmap_inv,
                  basename=patient_id, original_affine=affine), targets, num_cpus=nr_threads_saving)


def predict(files_in, folder_out, model_folder, task_ids, trainer, folds='all', preprocess_omaseg=False, save_all_combined_seg=True, snapshot=True, save_separate_targets=False, num_threads_preprocessing=1, nr_threads_saving=6, verbose=False):
    """
    Loop images and use nnUNetv2 models to predict. 
    """
    device = torch.device('cuda')
    # multithreading in torch doesn't help nnU-Net if run on GPU
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    labelmap_all_structure_inv = {v: k for k,
                                  v in labelmap_all_structure.items()}

    # Init nnUNetv2 predictor
    models = {}
    for task_id in task_ids:
        models[task_id] = nnUNetv2Predictor(model_folder, task_id, device, batch_predict=False, folds=folds, checkpoint='checkpoint_final.pth', trainer=trainer,
                                            num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=nr_threads_saving, verbose=verbose)

    # Loop images
    for i, file_in in enumerate(files_in):
        if os.path.basename(file_in)[-11:] == "0000.nii.gz":
            patient_id = os.path.basename(file_in)[:-12]
        else:
            patient_id = os.path.basename(file_in)[:-7]
        dataset_name = file_in.split('/')[-3]
        print("Predicting file {}/{}   ".format(i+1, len(files_in)), patient_id)
        start = time.time()

        # TODO:
        if preprocess_omaseg:
            # Reorient to RAS, resampling to 1.5, remove rotation and translation
            print('Preprocessing')

        output_dir = os.path.join(folder_out, dataset_name, patient_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if save_separate_targets:
            output_targets_dir = os.path.join(output_dir, 'targets')
            if not os.path.exists(output_targets_dir):
                os.makedirs(output_targets_dir, exist_ok=True)

        # Inference and save segmentations: loop 9x models
        for task_id in task_ids:
            models[task_id].predict(file_in, os.path.join(
                output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz'))

        # Combine all classes into a single segmentation nii file
        if save_all_combined_seg:
            init = False
            for task_id in task_ids:
                labelmap = map_taskid_to_labelmaps[task_id]
                seg = nib.load(os.path.join(
                    output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')).get_fdata()
                if not init:
                    seg_combined = np.zeros(seg.shape, dtype=np.uint8)
                    affine = nib.load(os.path.join(
                        output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')).affine
                    init = True
                for class_index, class_name in labelmap.items():
                    if class_name in except_labels_combine:
                        continue
                    seg_combined[seg ==
                                 class_index] = labelmap_all_structure_inv[class_name]

                # Save structures into separate binary segmentations
                if save_separate_targets:
                    save_targets_to_nifti(save_separate_targets, output_targets_dir,
                                          affine, labelmap, seg, patient_id, nr_threads_saving)

            file_seg_combined = os.path.join(
                output_dir, patient_id+'_combined.nii.gz')
            nib.save(nib.Nifti1Image(seg_combined, affine), file_seg_combined)

        # Generate a snapshot of combined segmentations
        if snapshot:
            try:
                file_snapshot = os.path.join(
                    output_dir, patient_id + '_snapshot.png')
                if not os.path.isfile(file_seg_combined):
                    raise FileNotFoundError(
                        f"The file '{file_seg_combined}' does not exist.")
                generate_snapshot(file_in, file_seg_combined, file_snapshot)
            except FileNotFoundError as e:
                print(f"Error: {e}")

        print(f"Finished in {time.time() - start:.2f}s")
