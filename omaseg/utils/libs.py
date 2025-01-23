import os
import contextlib
import sys
import shutil
import zipfile
import datetime
from pathlib import Path
from urllib.request import urlopen

"""
Helpers to suppress stdout prints from nnunet
https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
"""

def time_it(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        duration_in_s = duration.total_seconds()
        print(f'{func.__name__} took {duration_in_s:.6f} seconds to run')
        return result
    return wrapper


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout(verbose=False):
    if not verbose:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout
    else:
        yield


def download_pretrained_weights(task_id):

    config_dir = Path(os.path.join(os.getcwd()[:-3], 'models')) / "nnunet/results/nnUNet/3d_fullres"
    config_dir.mkdir(exist_ok=True, parents=True)

    old_weights = [
        "Task223_my_test"
    ]

    if task_id == 251:
        weights_path = config_dir / "Task251_TotalSegmentator_part1_organs_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
    elif task_id == 252:
        weights_path = config_dir / "Task252_TotalSegmentator_part2_vertebrae_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip?download=1"
    elif task_id == 253:
        weights_path = config_dir / "Task253_TotalSegmentator_part3_cardiac_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip?download=1"
    elif task_id == 254:
        weights_path = config_dir / "Task254_TotalSegmentator_part4_muscles_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip?download=1"
    elif task_id == 255:
        weights_path = config_dir / "Task255_TotalSegmentator_part5_ribs_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip?download=1"
    elif task_id == 256:
        weights_path = config_dir / "Task256_TotalSegmentator_3mm_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip?download=1"

    for old_weight in old_weights:
        if (config_dir / old_weight).exists():
            shutil.rmtree(config_dir / old_weight)

    if WEIGHTS_URL is not None and not weights_path.exists():
        print(f"Downloading pretrained weights for Task {task_id} (~230MB) ...")

        data = urlopen(WEIGHTS_URL).read()
        with open(config_dir / "tmp_download_file.zip", "wb") as weight_file:
            weight_file.write(data)

        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(config_dir)
            print(config_dir)

        # delete tmp file
        (config_dir / "tmp_download_file.zip").unlink()


def setup_nnunet_env():
    config_dir = Path(os.path.join(Path(os.getcwd()).parent, 'models'))
    (config_dir / "nnUNet/nnUNet_results").mkdir(exist_ok=True, parents=True)
    weights_dir = config_dir / "nnUNet/nnUNet_results"

    # This variables will only be active during the python script execution. Therefore
    # do not have to unset them in the end.
    os.environ["nnUNet_raw"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_results"] = str(weights_dir)
