import contextlib
import datetime
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

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


def get_model_weights_dir():
    if "CADS_WEIGHTS_PATH" in os.environ:
        model_dir = Path(os.environ["CADS_WEIGHTS_PATH"])
    else:
        codebase_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = Path(codebase_dir) / 'cads' / 'model_weights'
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)

def setup_nnunet_env():
    weights_dir = get_model_weights_dir()

    os.environ["nnUNet_raw"] = str(weights_dir)  # not needed, placeholder
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, placeholder
    os.environ["nnUNet_results"] = str(weights_dir)
    
    # Disable torch.compile to avoid compatibility issues (fp32 TypeError)
    # Users can override by setting nnUNet_compile=t before running if needed
    if "nnUNet_compile" not in os.environ:
        os.environ["nnUNet_compile"] = "f"

def check_or_download_model_weights(task_id):
    weights_dir = Path(get_model_weights_dir())
    weights_dir.mkdir(parents=True, exist_ok=True)
    # CADS-model v1.0.0
    weights_info = {
        551: ("Dataset551_Totalseg251", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset551_Totalseg251.zip"),
        552: ("Dataset552_Totalseg252", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset552_Totalseg252.zip"),
        553: ("Dataset553_Totalseg253", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset553_Totalseg253.zip"),
        554: ("Dataset554_Totalseg254", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset554_Totalseg254.zip"),
        555: ("Dataset555_Totalseg255", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset555_Totalseg255.zip"),
        556: ("Dataset556_GC256", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset556_GC256.zip"),
        557: ("Dataset557_Brain257", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset557_Brain257.zip"),
        558: ("Dataset558_OAR258", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset558_OAR258.zip"),
        559: ("Dataset559_Saros259", "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset559_Saros259.zip"),
    }
    
    folder_name, url = weights_info[task_id]
    weights_path = weights_dir / folder_name
    
    if not os.path.exists(weights_path):
        print(f"Downloading model for Task {task_id} ...")

        tempfile = weights_dir / f"tmp_download_file_{task_id}.zip"
        try:
            with open(tempfile, 'wb') as f:
                response = requests.get(url, stream=True, allow_redirects=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in response.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
                progress_bar.close()

            print("Download finished. Extracting...")
            with zipfile.ZipFile(tempfile, 'r') as zip_f:
                for member in zip_f.namelist():
                    if not member.startswith('__MACOSX') and not os.path.basename(member).startswith('._'):
                        zip_f.extract(member, weights_dir)
            print(f"Model weights extracted to {weights_path}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading weights: {e}")
            if os.path.exists(weights_path):
                os.rmdir(weights_path)
            raise
        except zipfile.BadZipFile:
            print("Error: Downloaded file is not a valid zip file")
            if os.path.exists(weights_path):
                os.rmdir(weights_path)
            raise
        finally:
            if os.path.exists(tempfile):
                os.remove(tempfile)

def cleanup_temp_files(dir_to_remove):
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)
