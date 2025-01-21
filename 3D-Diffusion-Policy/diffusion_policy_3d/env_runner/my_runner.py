import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

class CustomRunner:
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 tqdm_interval_sec=5.0,
                 device="cuda:0"
                 ):
        self.output_dir = output_dir
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, dataset, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc="Evaluating Dataset", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            dataset_sample = dataset[episode_idx]
            obs = dataset_sample['obs']
            policy.reset()
            traj_reward = 0
            is_success = False

            for step in range(self.max_steps):
                obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                action = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())['action']

                reward = dataset_sample['reward'][step]
                done = dataset_sample['done'][step]
                is_success = dataset_sample['success'][step]

                traj_reward += reward
                if done:
                    break

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            
        log_data = {
            'mean_traj_rewards': np.mean(all_traj_rewards),
            'mean_success_rates': np.mean(all_success_rates),
            'test_mean_score': np.mean(all_success_rates)
        }
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        if save_video:
            videos_wandb = wandb.Video(dataset_sample['video'], fps=10, format="mp4")
            log_data[f'dataset_video_eval'] = videos_wandb

        return log_data
