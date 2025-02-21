import rospy
import cv2
import numpy as np
import os
import time
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge
from os.path import join
import tf
import tf2_ros
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

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


def quaternion_to_euler(quaternion):
    """
    将四元数转换为欧拉角roll, pitch, yaw
    """
    r = R.from_quat(quaternion)
    return r.as_euler('xyz', degrees=False)


def get_tf_mat(i, dh):
    """Calculate the transformation matrix for the given joint based on DH parameters."""
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    
    # Transformation matrix based on DH parameters
    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]])

def get_franka_fk_solution(joint_angles):
    """Calculate the forward kinematics solution and return the end-effector position (xyz) and orientation (rotation matrix or quaternion)."""
    
    # Define the DH parameters for the 7 DOF robotic arm (example for a robot like Panda)
    dh_params = [
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi/2, joint_angles[1]],
        [0, 0.316, np.pi/2, joint_angles[2]],
        [0.0825, 0, np.pi/2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
        [0, 0, np.pi/2, joint_angles[5]],
        [0.088, 0, np.pi/2, joint_angles[6]],
        [0, 0.107, 0, 0],  # End-effector (gripper) offset (typically the last transformation)
        [0, 0, 0, -np.pi/4],  # Some additional offsets if needed
        [0.0, 0.1034, 0, 0]
    ]

    # Initialize the transformation matrix as identity matrix
    T = np.eye(4)
    
    # Calculate the transformation matrix for each joint using the DH parameters
    for i in range(len(dh_params)):
        T = T @ get_tf_mat(i, dh_params)
    
    # Extract the position (xyz) and orientation (rotation matrix)
    position = T[:3, 3]  # The position is the last column of the transformation matrix
    rotation_matrix = T[:3, :3]  # The orientation is the top-left 3x3 matrix (rotation part)

    # Convert the rotation matrix to a quaternion (optional, if you need orientation as quaternion)
    orientation = rotation_matrix_to_quaternion(rotation_matrix)
    
    return position, orientation

def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])



class DataCollector:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('data_collector', anonymous=True)
        print('inited ')
        # 设置文件保存路径
        # self.root = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data'
        self.root = '/media/robotics/ST_16T/crq/data'
        self.date = time.strftime('%Y%m%d', time.localtime())
        self.date2 = time.strftime('%H:%M:%S', time.localtime())
        self.imwrite_dir = join(self.root, 'record_data', self.date, self.date2)
        os.makedirs(self.imwrite_dir, exist_ok=True)
        
        ########################################
        # 设置视频捕获设备，0 是本地摄像头
        camera_id = 2 # 这个是超声的
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)


        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()

        print(f'cam original width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
        print(f'cam original height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

        # Set desired resolution
        desired_width = 640 # 1024
        desired_height = 480 # 768

        # Set the resolution for the input video stream (camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

                
        # Verify the resolution settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution set to: {actual_width}x{actual_height}")


        fps = self.cap.get(cv2.CAP_PROP_FPS)  
        print('fps:',fps)

        save_dir = self.imwrite_dir
        output_file = os.path.join(save_dir, f'ultrasound_video.avi')  # 使用当前时间命名文件


        # 设置视频写入器
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
        self.out = cv2.VideoWriter(output_file, self.fourcc, fps, (actual_width , actual_height))

        ###########################################
        
        self.bridge = CvBridge()
        self.ft_compensated_data = []
        self.netft_data = []
        self.agent_pos_data = []
        # self.ee_pos_data = []
        
        
        rospy.Subscriber("/ft_sensor/ft_compensated", WrenchStamped, self.ft_comp_callback)
        rospy.Subscriber("/ft_sensor/netft_data", WrenchStamped, self.netft_callback)
        rospy.Subscriber("/joint_states", JointState, self.agent_pos_callback)
        
        # 发布图像到 ROS 话题
        # self.pub_image = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)

        self.tf_listener = tf.TransformListener()

        # 计数器和同步机制
        self.count = 0
        self.rate = rospy.Rate(100)  # 采集频率
        cv2.namedWindow("Video Stream")
        self.running = True
        cv2.setMouseCallback("Video Stream", self.on_mouse_click)

        self.pipeline = initialize_realsense()


    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Left mouse button clicked - exiting data collection")
            self.running = False
        
    def get_panda_EE_transform(self):
        try:
            self.tf_listener.waitForTransform("/panda_link0", "/panda_EE", rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform("/panda_link0", "/panda_EE", rospy.Time(0))
            return np.array(trans), np.array(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("TF Exception")
            return None, None

    def ft_comp_callback(self, msg):
        ft_compensated = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])
        self.ft_compensated_data.append((msg.header.stamp.to_sec(), ft_compensated))  # 使用 ROS 时间戳

    def netft_callback(self, msg):
        netft = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])
        self.netft_data.append((msg.header.stamp.to_sec(), netft))  # 使用 ROS 时间戳


    def agent_pos_callback(self, msg):
        # 处理机械臂末端位置和关节数据
        joint_positions = np.array(msg.position)
        self.agent_pos_data.append((rospy.get_time(), joint_positions))  # 保存时间戳和关节数据

    def collect_data(self):
        print('=> Data collection started')
        
        while not rospy.is_shutdown():

            # ultrasound
            ret, frame = self.cap.read()
            # realsense
            color_image, depth_image = get_frames(self.pipeline)

            if ret and color_image is not None:
                # 显示视频流
                cv2.imshow("Video Stream", frame)
                cv2.imshow('Color Image', color_image)
                # 显示深度图像
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colormap)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if self.running == False:
                    break
                
                # 将帧写入到视频文件
                self.out.write(frame)

                current_time = rospy.get_time()

                # 获取当前时间戳，查找与此时间戳最接近的数据（同步）
                synced_ft_compensated = self.get_closest_data(self.ft_compensated_data, current_time)
                # synced_netft = self.get_closest_data(self.netft_data, current_time)
                synced_agent_pos = self.get_closest_data(self.agent_pos_data, current_time)
                

                # 创建字典存储图像和对应的数据
                data_dict = {}

                # 发布图像到 ROS 话题
                # ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                # ros_image.header.stamp = rospy.Time.now()  # 设置时间戳
                # self.pub_image.publish(ros_image)  # 发布图像消息

                # 将同步的图像和数据存储到字典
                if synced_ft_compensated and synced_agent_pos:
                    ft, ft_timestamp = synced_ft_compensated
                    # netft, netft_timestamp = synced_netft
                    agent_pos, agent_pos_timestamp = synced_agent_pos
                    position, orientation = self.get_panda_EE_transform()
                    if position is None or orientation is None:
                        continue
                    euler = quaternion_to_euler(orientation)
                    # 将彩色图像和深度图像合并为一个4通道图像
                    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image_normalized = depth_image_normalized.astype(np.uint8)
                    combined_image = np.dstack((color_image, depth_image_normalized))
                    
                    # 将图像数据转为字典存储
                    data_dict['image'] = frame  # 图像数据是 OpenCV 格式的 NumPy 数组
                    data_dict['ft_compensated'] = ft 
                    # data_dict['netft'] = netft
                    data_dict['agent_pos'] = agent_pos  # 机械臂关节数据
                    data_dict['position'] = position # cartesian 
                    data_dict['orientation'] = orientation
                    data_dict['timestamp'] = current_time  # 当前时间戳
                    data_dict['euler'] = euler # RPY
                    data_dict['image2'] = color_image
                    data_dict['depth'] = depth_image
                    data_dict['combined_image'] = combined_image

                    # 保存字典数据到文件
                    data_filename = join(self.imwrite_dir, f'{self.date2}_{self.count:03d}_data.npy')
                    np.save(data_filename, data_dict)  # 保存字典为 .npy 文件

                    print(f"Saved data {data_filename}")
                

                self.count += 1  # 增加计数器
            # self.rate.sleep() 
            self.rate.sleep()  # 控制采集频率
        self.cap.release()
        self.out.release()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def get_closest_data(self, data_list, timestamp):
        # 获取与当前时间戳最接近的数据
        closest_data = None
        min_time_diff = float('inf')
        
        for data_timestamp, data in data_list:
            time_diff = abs(data_timestamp - timestamp)
            if time_diff < min_time_diff:
                closest_data = (data, data_timestamp)
                min_time_diff = time_diff
        
        return closest_data

if __name__ == "__main__":
    data_collector = DataCollector()
    try:
        data_collector.collect_data()  # 开始数据收集
    except rospy.ROSInterruptException:
        pass
    finally:
        print(data_collector.ft_compensated_data)
        print("Data collection completed.")
        cv2.destroyAllWindows()
