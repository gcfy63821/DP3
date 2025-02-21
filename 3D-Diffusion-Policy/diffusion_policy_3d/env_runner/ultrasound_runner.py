import torch
import tqdm
import numpy as np
from termcolor import cprint
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
import diffusion_policy_3d.common.logger_util as logger_util
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import zarr  # 用于读取zarr文件
import collections
from scipy.spatial.transform import Rotation as R
import tf
import tf2_ros
import rospy
from geometry_msgs.msg import TransformStamped

# eval_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_data.zarr'
eval_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_neck.zarr'
# eval_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_force_position.zarr'
cam2_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_2cam.zarr'


class UltrasoundRunner(BaseRunner):
    def __init__(self,
                 eval_episodes,
                 output_dir,
                 n_obs_steps=2,
                 n_action_steps=3,
                 data_path=eval_data_path,  # 数据集路径
                 batch_size=100,  # 每批次的数据量
                 device="cuda:0"):
        super().__init__(output_dir)
        
        self.data_path = data_path  # 从数据集读取
        self.batch_size = batch_size
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        rospy.init_node('ultrasound_policy_runner', anonymous=True)

        
        # 日志记录器
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        # Load the Zarr data
        self.zarr_data = zarr.open(self.data_path, mode='r')['data']
        self.zarr_meta = zarr.open(self.data_path, mode='r')['meta']
        
        self.img_data = self.zarr_data['img']
        self.action_data = self.zarr_data['action']
        self.state_data = self.zarr_data['state']
        self.force_data = self.zarr_data['force']
        # self.point_cloud_data = self.zarr_data['point_cloud']
        self.timestamp_data = self.zarr_data['timestamp']
        self.episode_ends = self.zarr_meta['episode_ends']
        self.eval_episodes = len(self.episode_ends)
        
    def run(self, policy: BasePolicy, save_video=False):
        """
        从数据集读取数据并进行预测，进行评估。
        :param policy: 评估的策略模型
        :param save_video: 是否保存视频
        :return: 日志数据
        """
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc="Evaluating"):
            episode_start = self.episode_ends[episode_idx]
            episode_end = self.episode_ends[episode_idx + 1] if episode_idx + 1 < len(self.episode_ends) else len(self.force_data)
            
            images = self.img_data[episode_start:episode_end]
            actions = self.action_data[episode_start:episode_end]
            states = self.state_data[episode_start:episode_end]
            forces = self.force_data[episode_start:episode_end]
            # point_clouds = self.point_cloud_data[episode_start:episode_end]
            timestamps = self.timestamp_data[episode_start:episode_end]
            
            traj_reward = 0
            is_success = False
            done = False
            desired_position = None

            obs_queue = collections.deque(maxlen=self.n_obs_steps)
            action_queue = collections.deque(maxlen=self.n_action_steps)

            print('episode id: ',episode_idx)
            for t in range(len(actions)):
                img = torch.from_numpy(images[t]).to(device)  # 将图像移动到 GPU
                state = torch.from_numpy(states[t]).to(device)  # 将状态移动到 GPU
                force = torch.from_numpy(forces[t]).to(device)  # 将力的张量移动到 GPU
                action = torch.from_numpy(actions[t]).to(device)  # 将动作张量移动到 GPU（如果需要）
                # point_cloud = torch.from_numpy(point_clouds[t]).to(device)
                
                
                # Prepare the full state information (image + force + state)

                obs_queue.append({
                    'img': img,
                    'state': state,
                    'force': force,
                    # 'point_cloud': point_cloud
                })
                action_queue.append(action)

                # 如果观测数据队列长度不足 self.n_obs_steps，则继续收集数据
                if len(obs_queue) < self.n_obs_steps or len(action_queue) < self.n_action_steps:
                    continue

                # 将观测数据队列转换为模型输入
                obs_dict_input = {
                    'img': torch.stack([obs['img'] for obs in obs_queue]).unsqueeze(0),
                    'state': torch.stack([obs['state'] for obs in obs_queue]).unsqueeze(0),
                    'force': torch.stack([obs['force'] for obs in obs_queue]).unsqueeze(0),
                    # 'point_cloud': torch.stack([obs['point_cloud'] for obs in obs_queue]).unsqueeze(0)
                }
                
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)


                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                nactions = np_action_dict['action'].squeeze(0).flatten()
                # predicted_action = np_action_dict['action_pred'].squeeze(0)
                
                curr_state = state.cpu().numpy()
                
                if desired_position is None:
                    # desired_position = curr_state[6:9]
                    # desired_rpy = curr_state[9:12]
                    desired_position = curr_state[:3]
                    desired_orientation = curr_state[3:7]

                real_action = actions[t:t+self.n_action_steps].flatten()
                min_length = min(len(real_action), len(nactions))
                real_action = real_action[:min_length]
                nactions = nactions[:min_length]
                # 计算奖励和结束条件
                reward = self.calculate_reward(real_action, nactions)
                traj_reward += reward
                # not delta:
                # output_position = nactions[:3]
                # output_rpy = nactions[3:6]
                # for delta:
                
                # desired:
                desired_position += nactions[:3]
                desired_orientation = nactions[6:10]
                desired_orientation = desired_orientation / np.linalg.norm(desired_orientation)

                output_position = desired_position
                # output_rpy = desired_rpy
                # output_orientation = R.from_euler('xyz', output_rpy).as_quat()
                output_orientation = desired_orientation

                data_action_position = real_action[:3] + curr_state[:3]
                data_action_orientation = curr_state[3:7]


                try:
                    # output_position2 = nactions[12:15]+ curr_state[6:9]
                    # output_rpy2 = nactions[15:18]+ curr_state[9:12]
                    # output_orientation2 = R.from_euler('xyz', output_rpy2).as_quat()

                    # output_position3 = nactions[24:27]+ curr_state[6:9]
                    # output_rpy3 = nactions[27:30]+ curr_state[9:12]
                    # output_orientation3 = R.from_euler('xyz', output_rpy3).as_quat()

                    position = state[:3].cpu()
                    orientation = state[3:7].cpu()
                    orientation = orientation / np.linalg.norm(orientation)

                    self.publish_tf(output_position, output_orientation, frame_id="panda_link0", child_id="action_frame")
                    self.publish_tf(position, orientation, frame_id="panda_link0", child_id="current_state_frame")
                    # self.publish_tf(output_position2, output_orientation2, frame_id="panda_link0", child_id="action_frame2")
                    # self.publish_tf(output_position3, output_orientation3, frame_id="panda_link0", child_id="action_frame3")
                    self.publish_tf(data_action_position, data_action_orientation, frame_id="panda_link0", child_id="real_action")
                    
                except Exception as e:
                    continue
                
                if episode_end: # not implemented
                    done = True
                    is_success = True  # 如果有 `episode_end` 标志为成功
                
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

        # 计算评估结果的平均值
        log_data = {}
        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        # 记录成功率数据
        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        return log_data
    
    def calculate_reward(self, predicted_action, true_action):
        """
        计算当前步骤的奖励。
        :param predicted_action: 模型预测的动作
        :param true_action: 数据集中的真实动作
        :return: 奖励值
        """
        # 这里的奖励计算方式可以自定义，例如可以基于动作的相似度或其他方式
        reward = -np.linalg.norm(predicted_action - true_action)  # 这里使用了 L2 范数作为奖励
        return reward

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


class Ultrasound2CamRunner(BaseRunner):
    def __init__(self,
                 eval_episodes,
                 output_dir,
                 n_obs_steps=2,
                 n_action_steps=3,
                 data_path=cam2_data_path,  # 数据集路径
                 batch_size=100,  # 每批次的数据量
                 device="cuda:0"):
        super().__init__(output_dir)
        
        self.data_path = data_path  # 从数据集读取
        self.batch_size = batch_size
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        rospy.init_node('ultrasound_policy_runner', anonymous=True)

        
        # 日志记录器
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        # Load the Zarr data
        self.zarr_data = zarr.open(self.data_path, mode='r')['data']
        self.zarr_meta = zarr.open(self.data_path, mode='r')['meta']
        
        self.img_data = self.zarr_data['img']
        self.img2_data = self.zarr_data['img2']
        self.action_data = self.zarr_data['action']
        self.state_data = self.zarr_data['state']
        self.force_data = self.zarr_data['force']
        # self.point_cloud_data = self.zarr_data['point_cloud']
        self.timestamp_data = self.zarr_data['timestamp']
        self.episode_ends = self.zarr_meta['episode_ends']
        self.eval_episodes = len(self.episode_ends)
        
    def run(self, policy: BasePolicy, save_video=False):
        """
        从数据集读取数据并进行预测，进行评估。
        :param policy: 评估的策略模型
        :param save_video: 是否保存视频
        :return: 日志数据
        """
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc="Evaluating"):
            episode_start = self.episode_ends[episode_idx]
            episode_end = self.episode_ends[episode_idx + 1] if episode_idx + 1 < len(self.episode_ends) else len(self.force_data)
            
            images = self.img_data[episode_start:episode_end]
            images2 = self.img2_data[episode_start:episode_end]
            actions = self.action_data[episode_start:episode_end]
            states = self.state_data[episode_start:episode_end]
            forces = self.force_data[episode_start:episode_end]
            # point_clouds = self.point_cloud_data[episode_start:episode_end]
            timestamps = self.timestamp_data[episode_start:episode_end]
            
            traj_reward = 0
            is_success = False
            done = False

            obs_queue = collections.deque(maxlen=self.n_obs_steps)
            action_queue = collections.deque(maxlen=self.n_action_steps)

            desired_position = None
            desired_rpy = None

            print('episode id: ',episode_idx)
            for t in range(len(actions)):
                img = torch.from_numpy(images[t]).to(device)  # 将图像移动到 GPU
                img2 = torch.from_numpy(images2[t]).to(device)  # 将图像移动到 GPU
                state = torch.from_numpy(states[t]).to(device)  # 将状态移动到 GPU
                force = torch.from_numpy(forces[t]).to(device)  # 将力的张量移动到 GPU
                action = torch.from_numpy(actions[t]).to(device)  # 将动作张量移动到 GPU（如果需要）
                # point_cloud = torch.from_numpy(point_clouds[t]).to(device)
                
                
                # Prepare the full state information (image + force + state)

                obs_queue.append({
                    'img': img,
                    'img2': img2,
                    'state': state,
                    'force': force,
                    # 'point_cloud': point_cloud
                })
                action_queue.append(action)

                # 如果观测数据队列长度不足 self.n_obs_steps，则继续收集数据
                if len(obs_queue) < self.n_obs_steps or len(action_queue) < self.n_action_steps:
                    continue

                # 将观测数据队列转换为模型输入
                obs_dict_input = {
                    'img': torch.stack([obs['img'] for obs in obs_queue]).unsqueeze(0),
                    'img2': torch.stack([obs['img2'] for obs in obs_queue]).unsqueeze(0),
                    'state': torch.stack([obs['state'] for obs in obs_queue]).unsqueeze(0),
                    'force': torch.stack([obs['force'] for obs in obs_queue]).unsqueeze(0),
                    # 'point_cloud': torch.stack([obs['point_cloud'] for obs in obs_queue]).unsqueeze(0)
                }
                
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)


                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                nactions = np_action_dict['action'].squeeze(0).flatten()
                curr_state = state.cpu().numpy()

                if desired_position is None:
                    # desired_position = curr_state[6:9]
                    # desired_rpy = curr_state[9:12]
                    desired_position = curr_state[:3]
                    desired_orientation = curr_state[3:7]

                real_action = actions[t:t+self.n_action_steps].flatten()
                min_length = min(len(real_action), len(nactions))
                real_action = real_action[:min_length]
                nactions = nactions[:min_length]
                # 计算奖励和结束条件
                reward = self.calculate_reward(real_action, nactions)
                traj_reward += reward
                # not delta:
                # output_position = nactions[:3]
                # output_rpy = nactions[3:6]
                # for delta:
                
                # desired:
                desired_position += nactions[:3]
                desired_orientation = nactions[6:10]
                desired_orientation = desired_orientation / np.linalg.norm(desired_orientation)

                output_position = desired_position
                output_position1 = nactions[3:6]
                # output_rpy = desired_rpy
                # output_orientation = R.from_euler('xyz', output_rpy).as_quat()
                output_orientation = desired_orientation

                data_action_position = real_action[:3] + curr_state[:3]
                data_action_orientation = curr_state[3:7]


                try:
                    # output_position2 = nactions[12:15]+ curr_state[6:9]
                    # output_rpy2 = nactions[15:18]+ curr_state[9:12]
                    # output_orientation2 = R.from_euler('xyz', output_rpy2).as_quat()

                    # output_position3 = nactions[24:27]+ curr_state[6:9]
                    # output_rpy3 = nactions[27:30]+ curr_state[9:12]
                    # output_orientation3 = R.from_euler('xyz', output_rpy3).as_quat()

                    position = state[:3].cpu()
                    orientation = state[3:7].cpu()
                    orientation = orientation / np.linalg.norm(orientation)

                    self.publish_tf(output_position, output_orientation, frame_id="panda_link0", child_id="action_frame")
                    self.publish_tf(position, orientation, frame_id="panda_link0", child_id="current_state_frame")
                    self.publish_tf(output_position1, output_orientation, frame_id="panda_link0", child_id="action_frame_direct")
                    # self.publish_tf(output_position3, output_orientation3, frame_id="panda_link0", child_id="action_frame3")
                    self.publish_tf(data_action_position, data_action_orientation, frame_id="panda_link0", child_id="real_action")
                    
                except Exception as e:
                    continue
                
                if episode_end: # not implemented
                    done = True
                    is_success = True  # 如果有 `episode_end` 标志为成功
                
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

        # 计算评估结果的平均值
        log_data = {}
        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        # 记录成功率数据
        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        return log_data
    
    def calculate_reward(self, predicted_action, true_action):
        """
        计算当前步骤的奖励。
        :param predicted_action: 模型预测的动作
        :param true_action: 数据集中的真实动作
        :return: 奖励值
        """
        # 这里的奖励计算方式可以自定义，例如可以基于动作的相似度或其他方式
        reward = -np.linalg.norm(predicted_action - true_action)  # 这里使用了 L2 范数作为奖励
        return reward

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