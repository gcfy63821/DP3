import numpy as np
import os
from termcolor import cprint

def read_npy_file(file_path):
    if not os.path.exists(file_path):
        cprint(f"File {file_path} does not exist.", 'red')
        return

    try:
        data_dict = np.load(file_path, allow_pickle=True).item()
        if not isinstance(data_dict, dict):
            cprint(f"File {file_path} does not contain a dictionary.", 'red')
            return

        cprint(f"Keys and values in {file_path}:", 'green')
        for key, value in data_dict.items():
            cprint(f"Key: {key}", 'yellow')
            cprint(f"Value: {value}", 'cyan')
    except Exception as e:
        cprint(f"Error reading {file_path}: {e}", 'red')

if __name__ == "__main__":
    # 指定要读取的 .npy 文件路径
    file_path = '/media/robotics/ST_16T/crq/data/record_data/20250120/20:28/20:28_000_data.npy'
    read_npy_file(file_path)