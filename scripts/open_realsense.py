import pyrealsense2 as rs
import numpy as np
import cv2

def initialize_realsense():
    # 配置 RealSense 流
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用彩色流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 启用深度流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 开始流
    pipeline.start(config)
    
    return pipeline

def get_frames(pipeline):
    # 等待一帧数据
    frames = pipeline.wait_for_frames()
    
    # 获取彩色帧
    color_frame = frames.get_color_frame()
    # 获取深度帧
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        return None, None
    
    # 将图像转换为 NumPy 数组
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    return color_image, depth_image

def main():
    # 初始化 RealSense 相机
    pipeline = initialize_realsense()
    
    try:
        while True:
            # 获取彩色图像和深度图像
            color_image, depth_image = get_frames(pipeline)
            
            if color_image is None or depth_image is None:
                continue
            
            # 显示彩色图像
            cv2.imshow('Color Image', color_image)
            # 显示深度图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Depth Image', depth_colormap)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()