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

def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    # 如果图像是 bgra8 格式，删除透明通道
    if image.shape[0] == 4:
        image = image[:3, :, :] # 只保留 BGR 通道
    
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image



def farthest_point_sampling(points, K, use_cuda=False):
    points = torch.tensor(points, dtype=torch.float32)
    if use_cuda and torch.cuda.is_available():
        points = points.cuda()
    else:
        points = points.cpu()
    
    sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
    sampled_points = sampled_points.squeeze(0)
    sampled_points = sampled_points.cpu().numpy()
    return sampled_points, indices.cpu().numpy()

def preprocess_point_cloud(points, num_points=1024, use_cuda=False):

    # qw = 0.722037047140051
    # qx = -0.6677685888235922
    # qy = -0.12216735823105529
    # qz = 0.13350187609419234
    # x = 0.6649676791489088
    # y = -1.0718525225295776
    # z = 0.20009733257876774

    # # 计算旋转矩阵
    # R = np.array([
    #     [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
    #     [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
    #     [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    # ])

    # # 组合齐次变换矩阵
    # T = np.array([
    #     [R[0, 0], R[0, 1], R[0, 2], x],
    #     [R[1, 0], R[1, 1], R[1, 2], y],
    #     [R[2, 0], R[2, 1], R[2, 2], z],
    #     [0, 0, 0, 1]
    # ])

    # extrinsics_matrix = T

    WORK_SPACE = [
        [0.02, 0.25],
        [-0.3, 0.05],
        [0.15, 0.4] 
    ]

    # scale
    point_xyz = points[..., :3] * 0.2500000118743628
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    # point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    # print(f"Point cloud range before crop: x[{points[..., 0].min()}, {points[..., 0].max()}], "
    #       f"y[{points[..., 1].min()}, {points[..., 1].max()}], z[{points[..., 2].min()}, {points[..., 2].max()}]")

    
    # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                             (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                             (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    # sample_indices = sample_indices.cpu()
    sample_indices = torch.tensor(sample_indices, dtype=torch.long).cpu()  # 确保 sample_indices 是一个 torch.Tensor
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    
    return points_xyz

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


# 输入路径（包含.npy文件）
# expert_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/record_data/20250120'
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data.zarr'
expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/point_cloud'
save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_pc.zarr'
# N = 10  # 采样间隔
T = 1

# 获取目录下所有子文件夹
subfolders = [os.path.join(expert_data_path, f) for f in os.listdir(expert_data_path) if os.path.isdir(os.path.join(expert_data_path, f))]
subfolders = sorted(subfolders)

# storage
total_count = 0
img_arrays = []
state_arrays = []
force_arrays = []
action_arrays = []
timestamp_arrays = []
episode_ends_arrays = []
action_state_arrays = []
point_cloud_arrays = []

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
    npy_files = sorted(npy_files)  # 确保文件按时间顺序排列
    
    current_count = 0
    start_index = len(state_arrays)
    prev_position = None
    prev_state = None
    prev_rpy = None
    for k, npy_file in enumerate(npy_files):

        cprint(f'Processing {npy_file}', 'green')
        
        # 加载 .npy 文件
        data_dict = np.load(npy_file, allow_pickle=True).item()

        if current_count == 0:
            initial_position = copy.deepcopy(data_dict['position'])
            initial_rpy = copy.deepcopy(data_dict['euler'])
        total_count += 1
        current_count += 1

        current_position = data_dict['position']
        current_rpy = data_dict['euler']
        timestamp = data_dict['timestamp']

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
        prev_timestamp = timestamp
        
        obs_image = data_dict['image']
        obs_image = preproces_image(obs_image)
        obs_point_cloud = data_dict['point_cloud']
        obs_point_cloud = preprocess_point_cloud(obs_point_cloud)
        force = data_dict['ft_compensated']
        # robot_state = np.concatenate([position_to_initial, rpy_to_initial, current_position, current_rpy, delta_position, delta_rpy], axis=-1)
        robot_state = np.concatenate([position_to_initial, rpy_to_initial, current_position, current_rpy, velocity, w], axis=-1)
        timestamp = data_dict['timestamp']  # 记录时间戳
        action_state = np.concatenate([current_position, current_rpy, data_dict['ft_compensated']], axis=-1)
        # action_state = np.concatenate([delta_position, delta_rpy, force], axis=-1)
    
        img_arrays.append(obs_image)
        force_arrays.append(force)
        state_arrays.append(robot_state)
        point_cloud_arrays.append(obs_point_cloud)
        timestamp_arrays.append(timestamp)
        if current_count >= T:
            action_arrays.append(action_state)

    episode_ends_arrays.append(total_count)

    while len(action_arrays) < len(img_arrays):
        action_arrays.append(action_state)
        

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
point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])

if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    action_chunk_size = (100,)
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

# 将数据写入 zarr 文件
zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('force', data=force_arrays, chunks=(100, force_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('timestamp', data=timestamp_arrays, chunks=(100,), dtype='float64', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# 打印输出
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'force shape: {force_arrays.shape}, range: [{np.min(force_arrays)}, {np.max(force_arrays)}]', 'green')
cprint(f'timestamp shape: {timestamp_arrays.shape}, range: [{np.min(timestamp_arrays)}, {np.max(timestamp_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')