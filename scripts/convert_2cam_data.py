import os
import zarr
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import tqdm
from scipy.spatial.transform import Rotation as R
import copy
import cv2
import pytorch3d.transforms as pt

def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数转换为旋转矩阵
    """
    r = R.from_quat(quaternion)
    return r.as_matrix()

def preproces_image1(image):
    img_size = 84
    
    # 将图像转换为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.unsqueeze(0)  # 添加通道维度，变为 1xHxW

    min_x, max_x = 100, 480
    min_y, max_y = 50, 330

    image = image[:, min_y:max_y, min_x:max_x]

    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.cpu().numpy()
    return image

def preproces_image2(image):
    img_size = 64
       
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW


    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def quaternion_multiply(q1, q2):
    """
    计算两个四元数的乘积
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quaternion_inverse(q):
    """
    计算四元数的逆
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (w**2 + x**2 + y**2 + z**2)

def quaternion_to_rotation_6d(quaternion):
    """
    将四元数转换为6D旋转表示
    """
    rotation_matrix = pt.quaternion_to_matrix(torch.tensor(quaternion).view(1, 4))
    rotation_6d = pt.matrix_to_rotation_6d(rotation_matrix).squeeze().numpy()
    return rotation_6d

def extract_timestamp(file_path):
    # 提取文件名中的时间戳部分
    file_name = os.path.basename(file_path)
    time_str = file_name.split('_')[1]
    return int(time_str)

# 输入路径（包含.npy文件）
# expert_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/record_data/20250120'
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data.zarr'
expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/new_neck'
save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_2cam_3.zarr'
N = 10  # 采样间隔
T = 3
# 获取目录下所有子文件夹
subfolders = [os.path.join(expert_data_path, f) for f in os.listdir(expert_data_path) if os.path.isdir(os.path.join(expert_data_path, f))]
subfolders = sorted(subfolders)

# storage
total_count = 0
img_arrays = []
img2_arrays = []

state_arrays = []
state2_arrays = []
rotation_arrays = []

force_arrays = []
action_arrays = []
action2_arrays = []
action3_arrays = []

timestamp_arrays = []
episode_ends_arrays = []





if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    # user_input = input()
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
    npy_files = sorted(npy_files, key=extract_timestamp)  # 确保文件按时间顺序排列



    for sample in range(N):
        current_count = 0
        start_index = len(state_arrays)
        prev_position = None
        prev_state = None
        prev_rpy = None
        prev_timestamp = None
        for k, npy_file in enumerate(npy_files):
            # if (k + sample) % N != 0:
            #     continue  # 每隔 N 个文件选择 1 个

            if current_count==0:
                if k < sample:
                    continue
            
            # 加载 .npy 文件
            data_dict = np.load(npy_file, allow_pickle=True).item()

            if current_count == 0:
                initial_position = copy.deepcopy(data_dict['position'])
                initial_rpy = copy.deepcopy(data_dict['euler'])

            current_position = data_dict['position']
            current_rpy = data_dict['euler']
            timestamp = data_dict['timestamp']
            current_orientation = data_dict['orientation']

            if prev_timestamp is not None:
                if timestamp - prev_timestamp < 0.2:
                    continue
            
            cprint(f'Processing {npy_file}', 'green')
            total_count += 1
            current_count += 1

            position_to_initial = current_position - initial_position
            rpy_to_initial = current_rpy - initial_rpy

            delta_position = current_position - prev_position if prev_position is not None else np.zeros(3)
            delta_rpy = current_rpy - prev_rpy if prev_rpy is not None else np.zeros(3)


            # 计算速度
            if prev_position is not None and prev_timestamp is not None:
                delta_time = timestamp - prev_timestamp
                velocity = delta_position / delta_time if delta_time > 0 else np.zeros(3)
                w = delta_rpy / delta_time if delta_time > 0 else np.zeros(3)
            else:
                velocity = np.zeros(3)
                w = np.zeros(3)

            prev_position = copy.deepcopy(current_position)
            prev_rpy = copy.deepcopy(current_rpy)
            prev_timestamp = copy.deepcopy(timestamp)



            us_image = data_dict['image']
            us_image = preproces_image1(us_image)
            realsense_image = data_dict['combined_image']
            realsense_image = preproces_image2(realsense_image)
            # force = np.concatenate([data_dict['ft_compensated'], data_dict['netft']], axis=-1)
            force = data_dict['ft_compensated']
            # robot_state = np.concatenate([initial_position, initial_orientation, delta_position, delta_orientation], axis=-1)
            # robot_state = np.concatenate([position_to_initial, rpy_to_initial, delta_position, delta_rpy], axis=-1)
            # robot_state = np.concatenate([position_to_initial, rpy_to_initial, current_position, current_rpy, velocity, w], axis=-1)
            
            timestamp = data_dict['timestamp']  # 记录时间戳

            # 将四元数转换为旋转矩阵，并提取前两列
            # rotation_matrix = quaternion_to_rotation_matrix(current_orientation)
            # rotation_6d = rotation_matrix[:, :2].flatten()  # 提取前两列并展平为 1D 数组
            rotation_6d = quaternion_to_rotation_6d(current_orientation)

            

            # action_state = np.concatenate([delta_position, delta_rpy, force], axis=-1)
            # robot_state = np.concatenate([current_position, current_orientation, velocity, position_to_initial], axis=-1) # 3+4+3+3=13
            robot_state = np.concatenate([current_position, rotation_6d, velocity, w, position_to_initial], axis=-1) # 9+6+3=18
            robot_state2 = np.concatenate([current_position, rotation_6d], axis=-1) # 9
            robot_rotation = rotation_6d

            # action_state = np.concatenate([delta_position, current_position, current_orientation, force], axis=-1) # 7 + 6 = 13
            action_state = np.concatenate([current_position, rotation_6d, delta_position, delta_rpy, force], axis=-1) # 9+6+6=21
            action_state2 = np.concatenate([delta_position, current_rpy, force], axis=-1) # 12
            action_state3 = np.concatenate([current_position, rotation_6d, force], axis=-1) # 
            
            img_arrays.append(us_image)
            img2_arrays.append(realsense_image)
            force_arrays.append(force)

            state_arrays.append(robot_state)
            state2_arrays.append(robot_state2)
            rotation_arrays.append(robot_rotation)

            timestamp_arrays.append(timestamp)
            if current_count >= T:
                action_arrays.append(action_state)

        episode_ends_arrays.append(total_count)

        while len(action_arrays) < len(force_arrays):
            action_arrays.append(action_state)
            action2_arrays.append(action_state2)
            action3_arrays.append(action_state3)
        

        

# 创建 zarr 文件
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

# 将数据堆叠到数组中
img_arrays = np.stack(img_arrays, axis=0)
# if img_arrays.shape[1] == 3:  # make channel last
#     img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
# 假设 img 是形状为 [256, 84, 84, 3] 的张量
# img_arrays = np.transpose(img_arrays, (0, 3, 1, 2))

img2_arrays = np.stack(img2_arrays, axis=0)
img2_arrays = np.transpose(img2_arrays, (0, 3, 1, 2))

action_arrays = np.stack(action_arrays, axis=0)
action2_arrays = np.stack(action2_arrays, axis=0)
action3_arrays = np.stack(action3_arrays, axis=0)

state_arrays = np.stack(state_arrays, axis=0)
state2_arrays = np.stack(state2_arrays, axis=0)
rotation_arrays = np.stack(rotation_arrays, axis=0)

force_arrays = np.stack(force_arrays, axis=0)
timestamp_arrays = np.array(timestamp_arrays)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
img2_chunk_size = (100, img2_arrays.shape[1], img2_arrays.shape[2], img2_arrays.shape[3])

if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    action_chunk_size = (100,)

# 将数据写入 zarr 文件
zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('img2', data=img2_arrays, chunks=img2_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action2', data=action2_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action3', data=action3_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state2', data=state_arrays, chunks=(100, state2_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('rotation', data=state_arrays, chunks=(100, rotation_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)

zarr_data.create_dataset('force', data=force_arrays, chunks=(100, force_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('timestamp', data=timestamp_arrays, chunks=(100,), dtype='float64', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# 打印输出
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'img2 shape: {img2_arrays.shape}, range: [{np.min(img2_arrays)}, {np.max(img2_arrays)}]', 'green')

cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'force shape: {force_arrays.shape}, range: [{np.min(force_arrays)}, {np.max(force_arrays)}]', 'green')
cprint(f'timestamp shape: {timestamp_arrays.shape}, range: [{np.min(timestamp_arrays)}, {np.max(timestamp_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')