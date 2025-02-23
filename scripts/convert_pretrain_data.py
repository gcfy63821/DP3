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

def apply_mask(depth_image, combined_image, min_depth, max_depth):
    """
    对深度图像应用掩码，只保留在指定阈值范围内的像素值，并应用到组合图像上
    :param depth_image: 深度图像
    :param combined_image: 组合图像
    :param min_depth: 最小深度阈值
    :param max_depth: 最大深度阈值
    :return: 应用掩码后的组合图像
    """
    mask = (depth_image >= min_depth) & (depth_image <= max_depth)
    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)  # 扩展 mask 以匹配 combined_image 的形状
    masked_image = np.where(mask, combined_image, 0)
    return masked_image

def extract_timestamp(file_path):
    # 提取文件名中的时间戳部分
    file_name = os.path.basename(file_path)
    time_str = file_name.split('_')[1]
    return int(time_str)

# 输入路径（包含.npy文件）
# expert_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/record_data/20250120'
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data.zarr'
expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/20250223/data_for_pretrain'
save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/data_for_pretrain.zarr'
N = 5  # 采样间隔
T = 5
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

label_arrays = []


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

for i, subfolder in enumerate(subfolders):
    if i < 1:
        continue
    print(subfolder)
    # 遍历子文件夹里的 .npy 文件
    npy_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.npy')]
    print(len(npy_files))
    npy_files = sorted(npy_files, key=extract_timestamp)  # 确保文件按时间顺序排列




        
    for k, npy_file in enumerate(npy_files):
        # if (k + sample) % N != 0:
        #     continue  # 每隔 N 个文件选择 1 个


        
        # 加载 .npy 文件
        data_dict = np.load(npy_file, allow_pickle=True).item()

        cprint(f'Processing {npy_file}', 'green')

        us_image = data_dict['image']
        us_image = preproces_image1(us_image)
        depth_image = data_dict['depth']
        realsense_image = data_dict['combined_image']
        max_depth_value = np.max(depth_image)
        min_depth = 100
        max_depth = max_depth_value * 0.1
        realsense_image = apply_mask(depth_image, realsense_image, min_depth, max_depth)
        realsense_image = preproces_image2(realsense_image)

        label = data_dict['task_state']

            
        img_arrays.append(us_image)
        img2_arrays.append(realsense_image)
        label_arrays.append(label)

        

        

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
label_arrays = np.stack(label_arrays, axis=0)


compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
img2_chunk_size = (100, img2_arrays.shape[1], img2_arrays.shape[2], img2_arrays.shape[3])



# 将数据写入 zarr 文件
zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('img2', data=img2_arrays, chunks=img2_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('label', data=label_arrays, chunks=(100, ), dtype='float32', overwrite=True, compressor=compressor)

# 打印输出
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'img2 shape: {img2_arrays.shape}, range: [{np.min(img2_arrays)}, {np.max(img2_arrays)}]', 'green')
cprint(f'label shape: {label_arrays.shape}, range: [{np.min(label_arrays)}, {np.max(label_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')