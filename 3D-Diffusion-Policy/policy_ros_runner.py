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
        self.output_dir = 'data/outputs/ultrasound_scan-ultrasound_dp-0213-5_seed0'

        # 初始化 ROS 节点
        rospy.init_node('ultrasound_policy_runner', anonymous=True)

        self.n_obs_steps = self.cfg.policy.n_obs_steps
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
        camera_id = 0 # 这个是超声的
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            print("无法打开摄像头")
            sys.exit(1)

        print(f'cam original width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
        print(f'cam original height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')


        fps = self.cap.get(cv2.CAP_PROP_FPS)  
        print('fps:',fps)

        # 订阅 ROS 话题
        rospy.Subscriber("/ft_sensor/ft_compensated", WrenchStamped, self.ft_comp_callback)
        rospy.Subscriber("/joint_states", JointState, self.agent_pos_callback)
        print('Subscribed Topics')

        
        self.rate = rospy.Rate(10) # 10Hz
        self.tf_listener = tf.TransformListener()
        
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
    
    def preproces_image(self, image):
        img_size = 84
        image = image.astype(np.float32)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
        image = torchvision.transforms.functional.resize(image, (img_size, img_size))
        # image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
        image = image.cpu().numpy()
        return image

    def obs2dp_obs(self):
        timestamp = time.time()
        if self.curr_force is None:
            return
        # point_cloud = self.curr_pointcloud.copy()
        # point_cloud = point_cloud_sampling(point_cloud, num_points=128, method="fps")
        # img = self.curr_img.copy()
        # img = self.preproces_image(img)
        # obs_history = {"point_cloud": point_cloud, "img": img}

        obs_history = {"img": None}

        position, orientation = self.get_panda_EE_transform()
        if position is None or orientation is None:
            return
        euler = R.from_quat(orientation).as_euler('xyz', degrees=False)
                
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
        
        state = np.concatenate([position_to_initial, rpy_to_initial, position, euler, velocity, w], axis=-1)
        # state = np.concatenate([position, euler, velocity, w], axis=-1)
        force = self.curr_force.copy()

        obs_history["state"] = state
        obs_history["force"] = force
        self.prev_rpy = euler
        self.prev_position = position
        self.prev_timestamp = timestamp

        return obs_history

    def run(self):
        # 处理输入并运行 policy
        # rate = rospy.Rate(30)  # 10Hz
        print('start running')
        prev_position = None
        prev_orientation = None
        current_time = None
        prev_timestamp = None
        last_obs = None
        
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                # 显示视频流
                cv2.imshow("Video Stream", frame)
                if current_time is None:
                    # 避免开始的时候有绿色的帧
                    time.sleep(1)
                current_time = rospy.get_time()
                curr_obs = self.obs2dp_obs()

                curr_img = self.preproces_image(frame)
                
                if curr_obs is None or curr_img is None:
                    continue
                curr_obs["img"] = curr_img
                
                if last_obs is None:
                    obs_history = {
                        k: [np.array(v, dtype=self._get_type(v))] * self.cfg.n_obs_steps for k, v in curr_obs.items()
                    }
                else:
                    obs_history = {}
                    obs_history["state"] = [
                        np.array(last_obs["state"], dtype=np.float32),
                        np.array(curr_obs["state"], dtype=np.float32),
                    ]
                    obs_history["force"] = [
                        np.array(last_obs["force"], dtype=np.float32),
                        np.array(curr_obs["force"], dtype=np.float32),
                    ]
                    obs_history["img"] = [
                        np.array(last_obs["img"], dtype=np.float32),
                        np.array(curr_obs["img"], dtype=np.float32),
                    ]


                prepped_data = {k: torch.tensor(np.array([v]), device="cuda") for k, v in obs_history.items()}

                with torch.no_grad():
                    print('start_inference')
                    start_time = rospy.get_time()
                    action_dict = self.policy.predict_action(prepped_data)
                    print('time_cost:', rospy.get_time() - start_time)
                    action_output = action_dict["action"].cpu().numpy().flatten()
                    # delta_position = action_output[:3]
                    # delta_rpy = action_output[3:6]
                    # output_wrench = action_output[6:12]
                    # # direct control:
                    # # output_position = action_output[:3]
                    # # output_rpy = action_output[3:6]
                    # # delta control:
                    # output_position = delta_position + curr_obs['state'][6:9]
                    # output_rpy = delta_rpy + curr_obs['state'][9:12]

                    # output_orientation = R.from_euler('xyz', output_rpy).as_quat()
                    print(action_output)
                    for i in range(3):
                        try:
                            delta_position = action_output[i*12:i*12 +3]
                            delta_rpy = action_output[i*12+3:i*12 +6]
                            output_wrench = -1 * action_output[i*12+6:i*12+12]
                            # direct control:
                            # output_position = action_output[:3]
                            # output_rpy = action_output[3:6]
                            # delta control:
                            output_position = delta_position + curr_obs['state'][i*12+6:i*12+9]
                            output_rpy = delta_rpy + curr_obs['state'][i*12+9:i*12+12]
                            output_orientation = R.from_euler('xyz', output_rpy).as_quat()

                            self.publish_pose_and_wrench(output_wrench, output_position, output_orientation)
                            
                            # 发布 TF 变换
                            self.publish_tf(output_position, output_orientation, frame_id="panda_link0", child_id="action_frame")
                            
                            self.rate.sleep()

                        except Exception as e:
                            continue

                last_obs = curr_obs.copy()

                if self.curr_force is None:
                    rospy.logwarn("No synced ft compensated data")

                self.rate.sleep()
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

        self.cap.release()
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
