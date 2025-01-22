import os
import zarr
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import tqdm
import random

def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

# 输入路径（包含.npy文件）
expert_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/record_data/20250120'
save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_data.zarr'
# N = 5  # 每隔 N 个文件选择 1 个文件，设置为你需要的间隔值

# 获取目录下所有子文件夹
subfolders = [os.path.join(expert_data_path, f) for f in os.listdir(expert_data_path) if os.path.isdir(os.path.join(expert_data_path, f))]
subfolders = sorted(subfolders)
N = 5
random.seed(42)

# storage
total_count = 0
img_arrays = []
state_arrays = []
force_arrays = []
action_arrays = []
timestamp_arrays = []
episode_ends_arrays = []

if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

for subfolder in subfolders:
    print(subfolder)
    # 遍历子文件夹里的 .npy 文件
    npy_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.npy')]
    print(len(npy_files))
    npy_files = sorted(npy_files)  # 确保文件按时间顺序排列
    
    current_count = 0
    start_index = len(state_arrays)
    for k, npy_file in enumerate(npy_files):
        # k += random.randint(0, 9)
        
        total_count += 1
        current_count += 1
        cprint(f'Processing {npy_file}', 'green')
        
        
        # 加载 .npy 文件
        data_dict = np.load(npy_file, allow_pickle=True).item()

        obs_image = data_dict['image']
        obs_image = preproces_image(obs_image)
        force = np.concatenate([data_dict['ft_compensated'],  data_dict['netft']], axis=-1)
        robot_state = np.concatenate([data_dict['agent_pos'], data_dict['position'], data_dict['orientation']], axis=-1)
        timestamp = data_dict['timestamp'] # 记录时间戳
        
        img_arrays.append(obs_image)
        force_arrays.append(force)
        state_arrays.append(robot_state)
        timestamp_arrays.append(timestamp)
        
    episode_ends_arrays.append(total_count)

    # 如果需要填充action数组
    for j in range(current_count):
        if j < N:
            continue
        action_arrays.append(state_arrays[j + start_index])  # 根据需要的逻辑填充
    for j in range(N):
        action_arrays.append(state_arrays[-1])  # 填充最后一个状态

# 创建 zarr 文件
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

# 将数据堆叠到数组中
img_arrays = np.stack(img_arrays, axis=0)
# if img_arrays.shape[1] == 3:  # make channel last
#     img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
# 假设 img 是形状为 [256, 84, 84, 3] 的张量
img_arrays = np.transpose(img_arrays, (0, 3, 1, 2))

action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
force_arrays = np.stack(force_arrays, axis=0)
timestamp_arrays = np.array(timestamp_arrays)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])

if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    action_chunk_size = (100,)

# 将数据写入 zarr 文件
zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('force', data=force_arrays, chunks=(100, force_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('timestamp', data=timestamp_arrays, chunks=(100,), dtype='float64', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# 打印输出
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'force shape: {force_arrays.shape}, range: [{np.min(force_arrays)}, {np.max(force_arrays)}]', 'green')
cprint(f'timestamp shape: {timestamp_arrays.shape}, range: [{np.min(timestamp_arrays)}, {np.max(timestamp_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')