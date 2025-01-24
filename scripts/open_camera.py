# 0120 可以读取超声图像。原本大小： 640 * 480， 60fps
import cv2
import os
import datetime

# 设置视频捕获设备，0 是本地摄像头
camera_id = 0 # 这个是超声的
cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print(f'original width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
print(f'original height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

# Set desired resolution
desired_width = 1024
desired_height =768

# Set the resolution for the input video stream (camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        
# Verify the resolution settings
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution set to: {actual_width}x{actual_height}")


fps = cap.get(cv2.CAP_PROP_FPS)  
print('fps:',fps)


# 设置视频保存的文件夹路径和文件名
save_dir = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data'
if not os.path.exists(save_dir):  
    os.makedirs(save_dir)

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_file = os.path.join(save_dir, f'{current_time}.avi')  # 使用当前时间命名文件


# 设置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
out = cv2.VideoWriter(output_file, fourcc, fps, (actual_width , actual_height))


while True:
    # 从摄像头读取帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法获取帧，退出")
        break
    
    # 显示视频流
    cv2.imshow("Video Stream", frame)
    
    # 将帧写入到视频文件
    out.write(frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()