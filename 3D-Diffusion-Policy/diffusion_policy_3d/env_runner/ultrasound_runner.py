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
                 eval_episodes=20,
                 data_path=eval_data_path,  # 数据集路径
                 batch_size=32,  # 每批次的数据量
                 tqdm_interval_sec=5.0,
                 device="cuda:0"):
        super().__init__(output_dir)
        
        self.eval_episodes = eval_episodes
        self.data_path = data_path  # 从数据集读取
        self.batch_size = batch_size
        self.device = device
        self.tqdm_interval_sec = tqdm_interval_sec
        
        # 日志记录器
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        # 加载数据集
        self.zarr_data = zarr.open(self.data_path, mode='r')  # 打开zarr文件
        
        self.img_data = self.zarr_data['data/img']
        self.state_data = self.zarr_data['data/state']
        self.action_data = self.zarr_data['data/action']
        self.timestamp_data = self.zarr_data['data/timestamp']
        self.episode_ends_data = self.zarr_data['data/episode_ends']
        
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

        # 总评估回合
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), 
                                      desc="Eval from Dataset", 
                                      leave=False, mininterval=self.tqdm_interval_sec):
            # 获取当前回合的索引
            start_idx = episode_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.img_data))

            # 初始化回合数据
            traj_reward = 0
            is_success = False
            done = False

            # 在数据集上执行评估
            for idx in range(start_idx, end_idx):
                # 从数据集中加载数据
                img = torch.from_numpy(self.img_data[idx]).to(device=device, dtype=dtype)
                state = torch.from_numpy(self.state_data[idx]).to(device=device, dtype=dtype)
                action = torch.from_numpy(self.action_data[idx]).to(device=device, dtype=dtype)
                timestamp = torch.from_numpy(self.timestamp_data[idx]).to(device=device, dtype=dtype)
                episode_end = self.episode_ends_data[idx]

                # 输入数据准备
                obs_dict_input = {}
                obs_dict_input['img'] = img.unsqueeze(0)  # 加上批次维度
                obs_dict_input['state'] = state.unsqueeze(0)

                # 使用模型预测动作
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                predicted_action = np_action_dict['action'].squeeze(0)

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
