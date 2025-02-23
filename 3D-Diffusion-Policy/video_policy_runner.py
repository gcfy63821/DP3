if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import rospy
import torch
import cv2
import numpy as np
import torchvision
import pathlib
import copy
import hydra
import dill
import sys
from termcolor import cprint
from omegaconf import OmegaConf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from diffusion_policy_3d.policy.ultrasound_policy import UltrasoundDP
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.ultrasound_dataset import UltrasoundDataset
from diffusion_policy_3d.policy.dp3 import DP3
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped, WrenchStamped
import tf
import tf2_ros
from collections import deque
from torch.utils.data import DataLoader
import time
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import pytorch3d.transforms as pt
import collections


OmegaConf.register_new_resolver("eval", eval, replace=True)

def extract_timestamp(file_path):
    # 提取文件名中的时间戳部分
    file_name = os.path.basename(file_path)
    time_str = file_name.split('_')[1]
    return int(time_str)

def quaternion_to_rotation_6d(quaternion):
    """
    将四元数转换为6D旋转表示
    """
    rotation_matrix = pt.quaternion_to_matrix(torch.tensor(quaternion).view(1, 4))
    rotation_6d = pt.matrix_to_rotation_6d(rotation_matrix).squeeze().numpy()
    return rotation_6d


def rotation6d_to_quaternion(rotation6d):
    """
    将6D旋转表示转换为四元数
    :param rotation6d: 6D旋转表示 形状为 (6,)
    :return: 四元数，形状为 (4,)
    """
    # 将6D旋转表示转换为旋转矩阵
    rotation_matrix = pt.rotation_6d_to_matrix(torch.tensor(rotation6d).view(1, 6))
    
    # 将旋转矩阵转换为四元数
    quaternion = pt.matrix_to_quaternion(rotation_matrix).squeeze().numpy()
    
    return quaternion


class PolicyROSRunner:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(self.cfg.training.device)
        self.output_dir = 'data/outputs/ultrasound_2cam_scan-ultrasound_dp_2cam-0223-1_seed0'

        # 初始化 ROS 节点
        rospy.init_node('ultrasound_policy_runner', anonymous=True)

        self.n_obs_steps = self.cfg.policy.n_obs_steps
        self.n_action_steps = self.cfg.policy.n_action_steps
        self.history = deque(maxlen=self.n_obs_steps)

        


        self.model: UltrasoundDP = hydra.utils.instantiate(self.cfg.policy)
        self.ema_model: DP3 = None
        if self.cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(self.cfg.policy)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)


        ###################        
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        self.policy = self.model
        if self.cfg.training.use_ema:
            self.policy = self.ema_model
        self.policy.eval()
        self.policy.cuda()


        #######################################
        
        # 发布的 ROS 话题
        self.pose_pub = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10)
        self.wrench_pub = rospy.Publisher("/desired_wrench", WrenchStamped, queue_size=10)

        # 存储数据
        self.bridge = CvBridge()
        self.agent_pos_data = []
        self.ft_compensated_data = []
        self.netft_data = []
        self.initial_position = None
        self.initial_orientation = None

        self.curr_force = np.zeros(6)
        self.curr_img = None
        self.curr_agent_pos = np.zeros(7)

        self.prev_position = None
        self.prev_rpy = None
        self.prev_timestamp = None
        ########################################
        # 设置视频捕获设备，0 是本地摄像头
        # camera_id = 0 # 这个是超声的
        # self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        # if not self.cap.isOpened():
        #     print("无法打开摄像头")
        #     sys.exit(1)

        # print(f'cam original width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
        # print(f'cam original height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')


        # fps = self.cap.get(cv2.CAP_PROP_FPS)  
        # print('fps:',fps)

        # 订阅 ROS 话题
        rospy.Subscriber("/ft_sensor/ft_compensated", WrenchStamped, self.ft_comp_callback)
        rospy.Subscriber("/joint_states", JointState, self.agent_pos_callback)
        print('Subscribed Topics')

        
        self.rate = rospy.Rate(10) # 10Hz
        self.tf_listener = tf.TransformListener()

        # self.pipeline = initialize_realsense()



        
    def get_panda_EE_transform(self):
        try:
            self.tf_listener.waitForTransform("/panda_link0", "/panda_EE", rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform("/panda_link0", "/panda_EE", rospy.Time(0))
            return np.array(trans), np.array(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("TF Exception")
            return None, None

    def collect_observations(self, obs_dict):
        self.history.append(copy.deepcopy(obs_dict))

    def get_history(self):
        return list(self.history)

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
    
    def preproces_image1(self, image):
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
    
    def preproces_image2(self, image):
        img_size = 64
        
        image = image.astype(np.float32)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW

        image = torchvision.transforms.functional.resize(image, (img_size, img_size))
        # image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
        image = image.cpu().numpy()
        return image

    def obs2dp_obs(self, data_dict):
        timestamp = time.time()
        if self.curr_force is None:
            return
        # point_cloud = self.curr_pointcloud.copy()
        # point_cloud = point_cloud_sampling(point_cloud, num_points=128, method="fps")
        # img = self.curr_img.copy()
        # img = self.preproces_image(img)
        # obs_history = {"point_cloud": point_cloud, "img": img}

        # obs_history = {"img": None,"img2":None}

        # position, orientation = self.get_panda_EE_transform()
        position = data_dict['position']
        orientation = data_dict['orientation']
        curr_force = data_dict['ft_compensated']

        if position is None or orientation is None:
            return
        euler = R.from_quat(orientation).as_euler('xyz', degrees=False)
        rotation_6d = quaternion_to_rotation_6d(orientation)

                
        if self.initial_position is None:
            self.initial_position = copy.deepcopy(position)
            self.initial_rpy = copy.deepcopy(euler)

        if self.prev_position is None:
            delta_position = np.zeros_like(position)
            delta_rpy = np.zeros_like(euler)
            position_to_initial = np.zeros_like(position)
            rpy_to_initial = np.zeros_like(euler)
            
        else:
            delta_position = position - self.prev_position
            delta_rpy = euler - self.prev_rpy
            position_to_initial = position - self.initial_position
            rpy_to_initial = euler - self.initial_rpy

        # 计算速度
        if self.prev_position is not None and self.prev_timestamp is not None:
            delta_time = timestamp - self.prev_timestamp
            velocity = delta_position / delta_time if delta_time > 0 else np.zeros(3)
            w = delta_rpy / delta_time if delta_time > 0 else np.zeros(3)
        else:
            velocity = np.zeros(3)
            w = np.zeros(3)
        
        
        # state = np.concatenate([position_to_initial, rpy_to_initial, position, euler, velocity, w], axis=-1)
        # state = np.concatenate([position, euler, velocity, w], axis=-1)
        state = np.concatenate([position, rotation_6d, velocity, w, position_to_initial], axis=-1)
        # force = self.curr_force.copy()
        force = curr_force

        # obs_history["state"] = state
        # obs_history["force"] = force
        self.prev_rpy = euler
        self.prev_position = position
        self.prev_timestamp = timestamp

        # self.state_queue.append(state)
        # self.force_queue.append(force)

        # return obs_history
        return state, force

    def run(self):
        # 处理输入并运行 policy
        # rate = rospy.Rate(30)  # 10Hz
        print('start running')
        prev_position = None
        prev_orientation = None
        current_time = None
        prev_timestamp = None
        last_obs = None
        desired_position = None
        desired_rpy = None

        obs_queue = collections.deque(maxlen=self.n_obs_steps)

        expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/neck23'
        subfolders = [os.path.join(expert_data_path, f) for f in os.listdir(expert_data_path) if os.path.isdir(os.path.join(expert_data_path, f))]
        subfolders = sorted(subfolders)

        

        for subfolder in subfolders:
            print(subfolder)
            npy_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.npy')]
            print(len(npy_files))
            npy_files = sorted(npy_files, key=extract_timestamp)

            for k, npy_file in enumerate(npy_files):
                data_dict = np.load(npy_file, allow_pickle=True).item()

                # cprint(f'opened {npy_file}', 'green')

                timestamp = data_dict['timestamp']
                frame = data_dict['image']
                color_image = data_dict['image2']
                depth_image = data_dict['depth']

                # 将深度图像归一化到0-255范围

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    
                # 显示图像
                cv2.imshow('US Image', frame)
                cv2.imshow('Depth Image', depth_colormap)
                cv2.imshow('RealSense Image', color_image)
                self.rate.sleep()

                state, force = self.obs2dp_obs(data_dict)



                curr_img = self.preproces_image1(frame)

                depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_normalized = depth_image_normalized.astype(np.uint8)
                combined_image = np.dstack((color_image, depth_image_normalized))
                curr_img2 = self.preproces_image2(combined_image)
                
                if curr_img is None:
                    continue


                device = self.device
                img = torch.from_numpy(curr_img).to(device)  # 将图像移动到 GPU
                state = torch.from_numpy(state).to(device)  # 将状态移动到 GPU
                force = torch.from_numpy(force).to(device)  # 将力的张量移动到 GPU
                img2 = torch.from_numpy(curr_img2).to(device)  # 将动作张量移动到 GPU（如果需要）
                
                obs_queue.append({
                    'img': img,
                    'state': state,
                    'force': force,
                    'img2' : img2,
                    # 'point_cloud': point_cloud
                })

                
                # 确保队列长度与 n_obs_steps 匹配
                if len(obs_queue) < self.n_obs_steps:
                    continue  # 如果队列长度不足，等待更多数据

                obs_dict = {
                    'img': torch.stack([obs['img'] for obs in obs_queue]).unsqueeze(0),
                    'state': torch.stack([obs['state'] for obs in obs_queue]).unsqueeze(0),
                    'force': torch.stack([obs['force'] for obs in obs_queue]).unsqueeze(0),
                    'img2': torch.stack([obs['img2'] for obs in obs_queue]).unsqueeze(0),
                    
                    # 'point_cloud': torch.stack([obs['point_cloud'] for obs in obs_queue]).unsqueeze(0)
                }
                

                with torch.no_grad():
                    print('start_inference')
                    start_time = rospy.get_time()
                    action_dict = self.policy.predict_action(obs_dict)
                    print('time_cost:', rospy.get_time() - start_time)
                    action_output = action_dict["action"].cpu().numpy().flatten()
                
                try:
                    if desired_position is None:
                        curr_state = state.cpu().numpy()
                        desired_position = curr_state[:3]
                        desired_rotation6d = curr_state[3:9]

                    # for i in range(self.n_action_steps):
                    for i in range(1):
                        this_action = action_output[i*21 : i*21 + 21]
                        # desired:
                        # desired_position += nactions[9:12]
                        # desired_position = nactions[:3]
                        
                        desired_position = this_action[:3]
                        delta_position = this_action[9:12]
                        delta_rpy = this_action[12:15]

                        # desired_position += delta_position
                        desired_rotation6d = this_action[3:9]
                        desired_orientation = rotation6d_to_quaternion(desired_rotation6d)

                        output_position = desired_position
                        output_orientation = desired_orientation
                        output_wrench = this_action[15:21]
                        
                        self.publish_tf(output_position, output_orientation, frame_id="panda_link0", child_id=f"action_frame_{i}")

                        # self.publish_pose_and_wrench(output_wrench, output_position, output_orientation)
                        curr_state = state.cpu().numpy()
                        print(1)

                        delta_position_length = np.linalg.norm(output_position - curr_state[:3])
                        force_magnitude = np.linalg.norm(output_wrench)
                        print('delta_position_lenth:',delta_position_length, 'force:', force_magnitude)
                        
                        if delta_position_length < 0.04:
                            self.publish_pose_and_wrench(output_wrench, output_position, output_orientation)
                            self.publish_tf(output_position, output_orientation, frame_id="panda_link0", child_id=f"action_frame")

                        else: # back to stable
                            desired_position = curr_state[:3]
                            desired_rotation6d = curr_state[3:9]
                            print('going back')


                        self.rate.sleep()

                    
                except Exception as e:
                    print(e)
                    continue

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        

                



    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype


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


    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
        
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if key in self.__dict__:
                    try:
                        self.__dict__[key].load_state_dict(value, **kwargs)
                    except ValueError as e:
                        cprint(f"Error loading state_dict for {key}: {e}", 'red')
                else:
                    cprint(f"Warning: Key '{key}' not found in self.__dict__. Skipping.", 'yellow')
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
        
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    def publish_tf(self, position, orientation, frame_id, child_id):
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = frame_id
        t.child_frame_id = child_id

        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.x = orientation[0]
        t.transform.rotation.y = orientation[1]
        t.transform.rotation.z = orientation[2]
        t.transform.rotation.w = orientation[3]

        br.sendTransform(t)

    def publish_pose_and_wrench(self, wrench, position, orientation):
        pose = np.concatenate([position, orientation])
        # 创建 WrenchStamped 消息
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.header.frame_id = "panda_link0"
        wrench_msg.wrench.force.x = wrench[0]
        wrench_msg.wrench.force.y = wrench[1]
        wrench_msg.wrench.force.z = wrench[2]
        wrench_msg.wrench.torque.x = wrench[3]
        wrench_msg.wrench.torque.y = wrench[4]
        wrench_msg.wrench.torque.z = wrench[5]

        # 发布 WrenchStamped 消息
        self.wrench_pub.publish(wrench_msg)
        # rospy.loginfo(f"Published WrenchStamped: {wrench_msg}")
        rospy.loginfo_throttle(1, f"Published WrenchStamped: {wrench_msg}")

        # 创建 PoseStamped 消息
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]
        pose_msg.pose.orientation.x = pose[3]
        pose_msg.pose.orientation.y = pose[4]
        pose_msg.pose.orientation.z = pose[5]
        pose_msg.pose.orientation.w = pose[6]

        # 发布 PoseStamped 消息
        self.pose_pub.publish(pose_msg)
        # rospy.loginfo(f"Published PoseStamped: {pose_msg}")
        rospy.loginfo_throttle(1, f"Published PoseStamped: {pose_msg}")


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

def quarternion_to_euler(q):
    """
    Convert quaternion to euler angles
    """
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=False)




@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    runner = PolicyROSRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
