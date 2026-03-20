import os
import pickle
from pathlib import Path
import json
from utils import LOGGER, colorstr



def read_dataset(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_jsonl_dataset(file_path: str, split: str=""):
    # read jsonl to list of tuples
    dialogs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            dialog = json.loads(line)
            dialogs.append(tuple(dialog))
    return dialogs


def write_dataset(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def txt_read(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def txt_write(path, data):
    with open(path, 'w') as f:
        f.writelines(data)
        

def make_project_dir(config, is_rank_zero=False):
    """
    Make project folder.

    Args:
        config: yaml config.
        is_rank_zero (bool): make folder only at the zero-rank device.

    Returns:
        (path): project folder path.
    """
    prefix = colorstr('make project folder')
    project = config.project
    name = config.name

    save_dir = os.path.join(project, f"{name}-{config.epochs}-{config.batch_size}-{config.max_len}")

    if os.path.exists(save_dir):
        if is_rank_zero:
            LOGGER.info(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        #save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    if is_rank_zero:
        os.makedirs(project, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    
    return Path(save_dir)


def yaml_save(file='data.yaml', data=None, header=''):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    save_path = Path(file)
    print(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        LOGGER.info(f"Config is saved at {save_path}")