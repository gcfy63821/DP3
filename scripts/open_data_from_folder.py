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

def apply_depth_mask(depth_image, min_depth, max_depth):
    """
    对深度图像应用掩码，只保留在指定阈值范围内的像素值
    :param depth_image: 深度图像
    :param min_depth: 最小深度阈值
    :param max_depth: 最大深度阈值
    :return: 应用掩码后的深度图像
    """
    mask = (depth_image >= min_depth) & (depth_image <= max_depth)
    masked_depth_image = np.where(mask, depth_image, 0)
    return masked_depth_image

def remove_red_color(depth_colormap):
    """
    去除深度图像中的红色部分
    :param depth_colormap: 深度图像的彩色映射
    :return: 去除红色部分后的深度图像
    """
    # 定义红色的范围
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([50, 50, 255])

    # 创建掩码，去除红色部分
    mask = cv2.inRange(depth_colormap, lower_red, upper_red)
    depth_colormap[mask != 0] = [0, 0, 0]

    return depth_colormap

def extract_timestamp(file_path):
    # 提取文件名中的时间戳部分
    file_name = os.path.basename(file_path)
    time_str = file_name.split('_')[1]
    return int(time_str)

# 输入路径（包含.npy文件）
# expert_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/record_data/20250120'
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data.zarr'
expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/neck23'
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_2cam.zarr'
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
force_arrays = []
action_arrays = []
timestamp_arrays = []
episode_ends_arrays = []
action_state_arrays = []


for subfolder in subfolders:
    print(subfolder)
    npy_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.npy')]
    print(len(npy_files))
    npy_files = sorted(npy_files, key=extract_timestamp)

    for k, npy_file in enumerate(npy_files):
        data_dict = np.load(npy_file, allow_pickle=True).item()

        cprint(f'Processing {npy_file}', 'green')

        timestamp = data_dict['timestamp']
        us_image = data_dict['image']
        realsense_image = data_dict['image2']
        depth_image = data_dict['depth']

        # 计算并输出深度图像中的最大和最小深度
        # masked_depth_image = apply_depth_mask(depth_image, min_depth, max_depth)
        max_depth_value = np.max(depth_image)
        min_depth_value = np.min(depth_image)  # 忽略掩码中的零值
        print(f"Max depth: {max_depth_value}, Min depth: {min_depth_value}")

        min_depth = 100
        max_depth = max_depth_value * 0.1
        depth_image = apply_depth_mask(depth_image, min_depth, max_depth)
        depth_image = remove_red_color(depth_image)
        
        
        # 将深度图像归一化到0-255范围

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # depth_colormap = apply_depth_mask(depth_colormap, min_depth, max_depth)
        depth_colormap = remove_red_color(depth_colormap)



        # 显示图像
        cv2.imshow('US Image', us_image)
        cv2.imshow('Depth Image', depth_colormap)
        cv2.imshow('RealSense Image', realsense_image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

        
