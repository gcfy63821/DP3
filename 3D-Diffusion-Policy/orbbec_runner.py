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
from cv_bridge import CvBridge, CvBridgeError
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


from sensor_msgs.msg import Image


OmegaConf.register_new_resolver("eval", eval, replace=True)
 

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



class PolicyROSRunner:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(self.cfg.training.device)
        self.output_dir = 'data/outputs/ultrasound_force-ultrasound_dp-0213-4_seed0'

        # 初始化 ROS 节点
        rospy.init_node('ultrasound_policy_runner', anonymous=True)

        self.n_obs_steps = self.cfg.policy.n_obs_steps
        self.history = deque(maxlen=self.n_obs_steps)

        


        self.model: DP3 = hydra.utils.instantiate(self.cfg.policy)
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

        self.ft_compensated_data = []
        self.agent_pos_data = []
        self.img_data = []
        self.initial_position = None
        self.initial_rpy = None
        

        # 订阅 ROS 话题
        rospy.Subscriber("/ft_sensor/ft_compensated", WrenchStamped, self.ft_comp_callback)
        rospy.Subscriber("/joint_states", JointState, self.agent_pos_callback)
        rospy.Subscriber("/k4a/rgb/image_raw", Image, self.img_callback)
        
        self.bridge = CvBridge()
        print('Subscribed Topics')

        
        self.rate = rospy.Rate(10) # 30Hz
        self.tf_listener = tf.TransformListener()

        self.state_queue = deque(maxlen=self.n_obs_steps)
        self.force_queue = deque(maxlen=self.n_obs_steps)
        self.img_queue = deque(maxlen=self.n_obs_steps)
        
        

        
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
    
    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            self.img_data.append((rospy.get_time(), img))  # 保存时间戳和图像数据
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")
    
    def preproces_image(self, image):
        img_size = 84
        image = image.astype(np.float32)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
        # 如果图像是 bgra8 格式，删除透明通道
        if image.shape[0] == 4:
            image = image[:3, :, :] # 只保留 BGR 通道
        image = torchvision.transforms.functional.resize(image, (img_size, img_size))
        # image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
        image = image.cpu().numpy()
        return image

    def run(self):
        # 处理输入并运行 policy
        # rate = rospy.Rate(30)  # 10Hz
        print('start running')
        prev_position = None
        prev_orientation = None
        current_time = None

        while not rospy.is_shutdown():
            
            current_time = rospy.get_time()

            # 获取当前时间戳，查找与此时间戳最接近的数据（同步）
            synced_ft_compensated = self.get_closest_data(self.ft_compensated_data, current_time)
            # synced_netft = self.get_closest_data(self.netft_data, current_time)
            synced_agent_pos = self.get_closest_data(self.agent_pos_data, current_time)
            synced_img = self.get_closest_data(self.img_data, current_time)

            if synced_ft_compensated  and synced_agent_pos and synced_img:
            # if synced_agent_pos:
                ft, ft_timestamp = synced_ft_compensated
                # ft = np.zeros(6)
                # netft, netft_timestamp = synced_netft
                agent_pos, agent_pos_timestamp = synced_agent_pos
                frame , img_timestamp = synced_img
                # position, orientation = get_franka_fk_solution(agent_pos)
                position, orientation = self.get_panda_EE_transform()
                if position is None or orientation is None:
                    continue
                euler = quarternion_to_euler(orientation)


                # 初始化
                if self.initial_position is None:
                    # self.initial_position = position
                    self.initial_position = copy.deepcopy(position)
                    # self.initial_orientation = orientation
                    # self.initial_rpy = euler
                    self.initial_rpy = copy.deepcopy(euler)

                if prev_position is None:
                    delta_position = np.zeros_like(position)
                    # delta_orientation = np.zeros_like(orientation)
                    delta_rpy = np.zeros_like(euler)
                    position_to_initial = np.zeros_like(position)
                    rpy_to_initial = np.zeros_like(euler)
                else:
                    delta_position = position - prev_position
                    # delta_orientation = quaternion_multiply(orientation, quaternion_inverse(prev_orientation))
                    delta_rpy = euler - prev_rpy
                    position_to_initial = position - self.initial_position
                    rpy_to_initial = euler - self.initial_rpy

                # state = np.concatenate([self.initial_position, self.initial_orientation, position, orientation], axis=-1)
                # state = np.concatenate([self.initial_position, self.initial_orientation, delta_position, delta_orientation], axis=-1)
                # state = np.concatenate([self.initial_position, self.initial_rpy, delta_position, delta_rpy], axis=-1)
                # state = np.concatenate([position_to_initial, rpy_to_initial, delta_position, delta_rpy], axis=-1)
                # state = np.concatenate([position_to_initial, rpy_to_initial, position, euler], axis=-1)
                state = np.concatenate([position_to_initial, rpy_to_initial, position, euler, delta_position, delta_rpy], axis=-1)
                force = ft
                obs_image = self.preproces_image(frame)

                self.state_queue.append(state)
                self.force_queue.append(force)
                self.img_queue.append(obs_image)

                # 确保队列长度与 n_obs_steps 匹配
                if len(self.state_queue) < self.n_obs_steps:
                    continue  # 如果队列长度不足，等待更多数据

                # 将队列中的数据转换为张量
                state_tensor = torch.tensor(list(self.state_queue), dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)
                force_tensor = torch.tensor(list(self.force_queue), dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)
                img_tensor = torch.tensor(list(self.img_queue), dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)


                # state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0).to(self.device)
                # force_tensor = torch.tensor(force, dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0).to(self.device)
                # img_tensor = torch.tensor(obs_image, dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0).to(self.device)


                obs_dict = {
                    "state": state_tensor,
                    "force": force_tensor,
                    "img": img_tensor
                }

                # self.collect_observations(obs_dict)

                # if len(self.history) == self.n_obs_steps:
                #     obs_dict = {
                #         "state": torch.cat([obs["state"] for obs in self.get_history()], dim=1), 
                #         "force": torch.cat([obs["force"] for obs in self.get_history()], dim=1),
                #         "img": torch.cat([obs["img"] for obs in self.get_history()], dim=1)
                #     }
                # else:
                #     continue

                with torch.no_grad():
                    action_dict = self.policy.predict_action(obs_dict)
                    # print('action_dict:', action_dict)
                    action_output = action_dict["action"].cpu().numpy().flatten()
                    # print('action_output:', action_output.shape)
                    delta_position = action_output[:3]
                    # delta_orientation = action_output[3:7]
                    delta_rpy = action_output[3:6]

                    # output_wrench = action_output[7:19]
                    output_wrench = action_output[6:12]
                    # output_position = delta_position + position
                    output_position = action_output[:3]
                    # output_rpy = delta_rpy + euler
                    output_rpy = action_output[3:6]
                    output_orientation = R.from_euler('xyz', output_rpy).as_quat()

                prev_position = copy.deepcopy(position)
                # prev_orientation = orientation
                prev_rpy = copy.deepcopy(euler)

                # 发布 action 的 ROS 话题
                
                # print('publish action output:', action_output)
                self.publish_pose_and_wrench(output_wrench, output_position, output_orientation)
                
                # action_msg = Float32MultiArray()
                # action_msg.data = action_output.tolist()
                # self.pose_pub.publish(action_msg)

                
                # rospy.loginfo(f"Published action: {action_output}")

                # 发布 TF 变换
                self.publish_tf(output_position, output_orientation, frame_id="panda_link0", child_id="action_frame")
            else:
                if synced_ft_compensated is None:
                    rospy.logwarn("No synced ft compensated data")
                if synced_agent_pos is None:
                    rospy.logwarn("No synced agent pos data")


            self.rate.sleep()


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
