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
from diffusion_policy_3d.policy.dp3 import DP3
from geometry_msgs.msg import TransformStamped
import tf2_ros

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

class PolicyROSRunner:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(self.cfg.training.device)
        self.output_dir = '3D-Diffusion-Policy/data/outputs/ultrasound_scan-ultrasound_dp-0120_seed0'

        # 初始化 ROS 节点
        rospy.init_node('ultrasound_policy_runner', anonymous=True)

        
        self.model: DP3 = hydra.utils.instantiate(self.cfg.policy)
        self.ema_model: DP3 = None
        if self.cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(self.cfg.policy)
        
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
        self.action_pub = rospy.Publisher("/action_cmd", Float32MultiArray, queue_size=10)

        # 存储数据
        self.bridge = CvBridge()
        self.agent_pos_data = []
        self.ft_compensated_data = []
        self.netft_data = []
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
        rospy.Subscriber("/ft_sensor/netft_data", WrenchStamped, self.netft_callback)
        rospy.Subscriber("/joint_states", JointState, self.agent_pos_callback)
        print('Subscribed Topics')

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

    def run(self):
        # 处理输入并运行 policy
        # rate = rospy.Rate(30)  # 10Hz
        print('start running')
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                # 显示视频流
                cv2.imshow("Video Stream", frame)
                current_time = rospy.get_time()

                # 获取当前时间戳，查找与此时间戳最接近的数据（同步）
                synced_ft_compensated = self.get_closest_data(self.ft_compensated_data, current_time)
                synced_netft = self.get_closest_data(self.netft_data, current_time)
                synced_agent_pos = self.get_closest_data(self.agent_pos_data, current_time)

                if synced_ft_compensated and synced_netft and synced_agent_pos:
                    ft, ft_timestamp = synced_ft_compensated
                    netft, netft_timestamp = synced_netft
                    agent_pos, agent_pos_timestamp = synced_agent_pos
                    position, orientation = get_franka_fk_solution(agent_pos)

                    state = np.concatenate([agent_pos, position, orientation], axis=-1)
                    force = np.concatenate([ft, netft], axis=-1)
                    obs_image = self.preproces_image(frame)

                    state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)
                    force_tensor = torch.tensor(force, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)
                    img_tensor = torch.tensor(obs_image, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)


                    obs_dict = {
                        "state": state_tensor,
                        "force": force_tensor,
                        "img": img_tensor
                    }

                    with torch.no_grad():
                        action_dict = self.policy.predict_action(obs_dict)
                        action_output = action_dict["action"].cpu().numpy().flatten()

                    # 发布 action 的 ROS 话题
                    action_msg = Float32MultiArray()
                    action_msg.data = action_output.tolist()
                    self.action_pub.publish(action_msg)
                    
                    rospy.loginfo(f"Published action: {action_output}")

                    # 发布 TF 变换
                    self.publish_tf(action_output, frame_id="base_link", child_id="action_frame")


            # rate.sleep()
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


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
                self.__dict__[key].load_state_dict(value, **kwargs)
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

    def publish_tf(self, action, frame_id, child_id):
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = frame_id
        t.child_frame_id = child_id

        # 假设 action 数据是 [x, y, z, qx, qy, qz, qw]
        t.transform.translation.x = action[7]
        t.transform.translation.y = action[8]
        t.transform.translation.z = action[9]
        t.transform.rotation.x = action[10]
        t.transform.rotation.y = action[11]
        t.transform.rotation.z = action[12]
        t.transform.rotation.w = action[13]

        br.sendTransform(t)


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
