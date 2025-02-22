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

def extract_timestamp(file_path):
    # 提取文件名中的时间戳部分
    file_name = os.path.basename(file_path)
    time_str = file_name.split('_')[1]
    return int(time_str)

# 输入路径（包含.npy文件）
# expert_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/record_data/20250120'
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data.zarr'
expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/new_neck'
save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_2cam.zarr'
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

        # 将深度图像归一化到0-255范围

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
        # 显示图像
        cv2.imshow('US Image', us_image)
        cv2.imshow('Depth Image', depth_colormap)
        cv2.imshow('RealSense Image', realsense_image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

        
