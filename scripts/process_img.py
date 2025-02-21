import os
import numpy as np
import torch
import torchvision
from termcolor import cprint
from PIL import Image

def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    # 如果图像是 bgra8 格式，删除透明通道
    if image.shape[0] == 4:
        image = image[:3, :, :] # 只保留 BGR 通道

    min_x, max_x = 100, 480
    min_y, max_y = 50, 330

    image = image[:, min_y:max_y, min_x:max_x]

    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def save_channel_images(image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 将 PIL 图像转换为 NumPy 数组
    image = np.array(image)
    
    # 提取每个通道
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    # 保存每个通道的图像
    Image.fromarray(red_channel.astype(np.uint8)).save(os.path.join(output_dir, 'red_channel.png'))
    Image.fromarray(green_channel.astype(np.uint8)).save(os.path.join(output_dir, 'green_channel.png'))
    Image.fromarray(blue_channel.astype(np.uint8)).save(os.path.join(output_dir, 'blue_channel.png'))
    cprint(f"Channel images saved to {output_dir}", 'green')

def process_and_save_image(input_path, output_path, output_dir):
    # 读取 PNG 文件
    image = Image.open(input_path)
    image = np.array(image)

    # 处理图像
    processed_image = preproces_image(image)

    # 保存处理后的图像
    processed_image = Image.fromarray(processed_image.astype(np.uint8))
    processed_image.save(output_path)
    cprint(f"Processed image saved to {output_path}", 'green')

    # 保存每个通道的图像
    save_channel_images(processed_image, output_dir)

if __name__ == "__main__":
    input_path = 'img.png'
    output_path = 'processed.png'
    output_dir = 'channels'
    process_and_save_image(input_path, output_path, output_dir)