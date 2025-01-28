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

from omaseg.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, map_taskid_to_labelmaps, except_labels_combine
from omaseg.dataset_utils.preprocessing import preprocess_nifti, restore_seg_in_orig_format
from omaseg.dataset_utils.postprocessing import _do_outlier_postprocessing_groups, postprocess_seg, postprocess_head, postprocess_head_and_neck
from omaseg.utils.snapshot import generate_snapshot
from omaseg.utils.libs import time_it, cleanup_temp_files

TRAINERS = {
    551: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    552: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    553: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    554: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    555: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    556: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    557: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    558: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    559: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
}

def get_task_model_folder(task_id: int, model_folder: str):
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
            trainer = TRAINERS[task_id]
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

    def __init__(self, model_folder, task_id, device, batch_predict=False, folds='all', checkpoint='checkpoint_final.pth', num_threads_preprocessing=6, num_threads_nifti_save=2, verbose=False):
        self.num_threads_preprocessing = num_threads_preprocessing
        self.num_threads_nifti_save = num_threads_nifti_save
        self.device = device

        # Use default settings recommended by nnUNet (also same as TotalSeg_v2)
        self.predictor = nnUNetPredictor(tile_step_size=0.5,  # TODO: 0.5, 0.8
                                         use_gaussian=True,
                                         use_mirroring=False,
                                         perform_everything_on_device=True,
                                         device=self.device,
                                         verbose=verbose,
                                         allow_tqdm=True,
                                         verbose_preprocessing=verbose)
        task_id = int(task_id)
        try:
            task_model_folder = get_task_model_folder(task_id, model_folder)
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
        # This is suitable for processing a bunch of files, and will produce some side-files (predict_from_raw_data_args.json ...)
        self.predictor.predict_from_files(folder_in, folder_out, save_probabilities=False,
                                          overwrite=True,
                                          num_processes_preprocessing=self.num_threads_preprocessing,
                                          num_processes_segmentation_export=self.num_threads_nifti_save,
                                          folder_with_segs_from_prev_stage=None,
                                          num_parts=1, part_id=0)

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


def predict(files_in, folder_out, model_folder, task_ids, folds='all', run_in_slicer=False, use_cpu=False, preprocess_omaseg=False, postprocess_omaseg=False, save_all_combined_seg=True, snapshot=True, save_separate_targets=False, num_threads_preprocessing=4, nr_threads_saving=6, verbose=False):
    """
    Loop images and use nnUNetv2 models to predict. 
    """
    if use_cpu:
        device = torch.device('cpu') 
    else:
        device = torch.device('cuda')

    # TODO: really need to set up torch threads?
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    labelmap_all_structure_inv = {v: k for k,
                                  v in labelmap_all_structure.items()}

    # Init nnUNetv2 predictor
    models = {}
    for task_id in task_ids:
        models[task_id] = nnUNetv2Predictor(model_folder, task_id, device, batch_predict=False, folds=folds, checkpoint='checkpoint_final.pth',
                                            num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=nr_threads_saving, verbose=verbose)
    if any(task in task_ids for task in [557, 558]) and 553 not in task_ids:
        models[553] = nnUNetv2Predictor(model_folder, 553, device, batch_predict=False, folds=folds, checkpoint='checkpoint_final.pth',
                                            num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=nr_threads_saving, verbose=verbose)
    if 558 in task_ids and 552 not in task_ids:
        models[552] = nnUNetv2Predictor(model_folder, 552, device, batch_predict=False, folds=folds, checkpoint='checkpoint_final.pth',
                                            num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=nr_threads_saving, verbose=verbose)

    # Loop images
    for i, file_in in enumerate(files_in):
        if os.path.basename(file_in)[-11:] == "0000.nii.gz":
            patient_id = os.path.basename(file_in)[:-12]
        else:
            patient_id = os.path.basename(file_in)[:-7]
        print("Predicting file {}/{}   ".format(i+1, len(files_in)), patient_id)
        start = time.time()

        temp_dir = None
        if preprocess_omaseg:
            # Reorient to RAS, resampling to 1.5, remove rotation and translation
            temp_dir, file_in, metadata_orig, preprocessing_done = preprocess_nifti(file_in, spacing=1.5, num_threads_preprocessing=num_threads_preprocessing)

        if run_in_slicer:
            patient_id = 'segmentation'  # make the output filename general
            output_dir = folder_out
        else: 
            output_dir = os.path.join(folder_out, patient_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if save_separate_targets:
            output_targets_dir = os.path.join(output_dir, 'targets')
            if not os.path.exists(output_targets_dir):
                os.makedirs(output_targets_dir, exist_ok=True)

        # Inference and save segmentations: loop 9x models
        task_ids.sort()
        for task_id in task_ids:
            file_out = os.path.join(output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')
            # models[task_id].predict([[file_in]], [file_out])  # for batch_predict
            models[task_id].predict(file_in, file_out)

            if postprocess_omaseg:
                if task_id in _do_outlier_postprocessing_groups:
                    if run_in_slicer:
                        postprocess_seg(file_out, task_id, file_out)
                    else:
                        from omaseg.dataset_utils.TPTBox import postprocess_seg_TPTBox  #TODO: simplfy import
                        postprocess_seg_TPTBox(file_out, task_id, file_out)
                if task_id in [557, 558]:
                    file_seg_brain_group = os.path.join(output_dir, patient_id+'_part_'+str(553)+'.nii.gz')
                    if not os.path.exists(file_seg_brain_group):
                        print(f"Task {task_id} needs pre-segmentation from task 553, generating segmentations...")
                        # models[553].predict([[file_in]], [file_seg_brain_group])  # for batch_predict
                        models[553].predict(file_in, file_seg_brain_group)

                    # For group 558, also need cervical vertebrae as reference
                    if task_id == 558:
                        file_seg_vertebrae_group = os.path.join(output_dir, patient_id+'_part_'+str(552)+'.nii.gz')
                        # First check if brain exists and is sufficient
                        if not postprocess_head_and_neck(task_id, file_seg_brain_group, None, file_out):
                            # Only predict vertebrae if brain check failed
                            if not os.path.exists(file_seg_vertebrae_group):
                                print(f"Task {task_id} needs pre-segmentation from task 552 (spine), generating segmentations...")
                                # models[552].predict([[file_in]], [file_seg_vertebrae_group])  # for batch_predict
                                models[552].predict(file_in, file_seg_vertebrae_group)
                            postprocess_head_and_neck(task_id, file_seg_brain_group, file_seg_vertebrae_group, file_out)
                    else:
                        postprocess_head(task_id, file_seg_brain_group, file_out)

        # reverse pre-processing
        for task_id in task_ids:
            file_out = os.path.join(output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')
            if preprocess_omaseg and preprocessing_done:
                restore_seg_in_orig_format(file_out, metadata_orig, num_threads_preprocessing=num_threads_preprocessing)

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

        if temp_dir:
            cleanup_temp_files(temp_dir)
        print(f"Finished in {time.time() - start:.2f}s")
