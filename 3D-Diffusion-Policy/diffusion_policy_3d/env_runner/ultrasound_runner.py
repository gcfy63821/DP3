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


eval_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_data.zarr'

class UltrasoundRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 data_path=eval_data_path,  # 数据集路径
                 batch_size=100,  # 每批次的数据量
                 device="cuda:0"):
        super().__init__(output_dir)
        
        self.data_path = data_path  # 从数据集读取
        self.batch_size = batch_size
        self.device = device
        
        # 日志记录器
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        # Load the Zarr data
        self.zarr_data = zarr.open(self.data_path, mode='r')['data']
        self.zarr_meta = zarr.open(self.data_path, mode='r')['meta']
        
        # Prepare the data (images, actions, states, etc.)
        self.img_data = self.zarr_data['img']
        self.action_data = self.zarr_data['action']
        self.state_data = self.zarr_data['state']
        self.force_data = self.zarr_data['force']
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
            # Load the data for one episode
            episode_start = self.episode_ends[episode_idx]
            episode_end = self.episode_ends[episode_idx + 1] if episode_idx + 1 < len(self.episode_ends) else len(self.img_data)
            
            images = self.img_data[episode_start:episode_end]
            actions = self.action_data[episode_start:episode_end]
            states = self.state_data[episode_start:episode_end]
            forces = self.force_data[episode_start:episode_end]
            timestamps = self.timestamp_data[episode_start:episode_end]
            
            traj_reward = 0
            is_success = False
            done = False

            for t in range(len(images)):
                img = torch.from_numpy(images[t]).to(device)  # 将图像移动到 GPU
                state = torch.from_numpy(states[t]).to(device)  # 将状态移动到 GPU
                force = torch.from_numpy(forces[t]).to(device)  # 将力的张量移动到 GPU
                action = torch.from_numpy(actions[t]).to(device)  # 将动作张量移动到 GPU（如果需要）

                
                
                # Preprocess the image
                # img_input = self.preprocess_input(img)
                
                # Prepare the full state information (image + force + state)
                with torch.no_grad():
                    obs_dict_input = {'img': img.unsqueeze(0), 'state': state.unsqueeze(0), 'force':force.unsqueeze(0)}
                    action_dict = policy.predict_action(obs_dict_input) # to predict

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                # action = np_action_dict['action'].squeeze(0)
                predicted_action = np_action_dict['action_pred'].squeeze(0)

                # 计算奖励和结束条件
                reward = self.calculate_reward(predicted_action, action)  # 你可以定义一个奖励计算函数
                traj_reward += reward

                if episode_end:
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
