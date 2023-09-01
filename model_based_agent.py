# References
#        Replay Buffer: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import copy
from collections import namedtuple
import cv2
import gymnasium as gym
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal, Independent
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from gymnasium.wrappers import TimeLimit, PixelObservationWrapper, TransformObservation
from utils import *
from planet import Planet


Transition = namedtuple('Transition',
                        ('observation', 'action', 'reward'))


class ModelBasedLearner:
    def __init__(self, params):
        self.params = params
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        exp_tag, self.env, self.action_dim = self.get_env()
        self.replay_buffer = ReplayBuffer(params=self.params)
        self.world_model = Planet(params=self.params, action_dim=self.action_dim).to(self.d_type).to(self.device)
        print(f'Initialized {self.world_model} ({count_parameters(self.world_model)}) as the world-model')
        self.optimizer = Adam(params=self.world_model.parameters(), lr=self.params['lr'], eps=self.params['adam_epsilon'])
        self.logger = SummaryWriter(comment=exp_tag)

    def get_env(self):
        if self.params['api_name'] == 'gym':
            return self.get_gym_env()
        elif self.params['api_name'] == 'dmc':
            return self.get_dm_control_env()
        else:
            raise NotImplementedError(f'{self.params["api_name"]} is not implemented')
        
    def get_gym_env(self):
        exp_tag = '_' + self.params['env_name'].lower() + '_'
        env = gym.make(self.params['env_name'], render_mode='rgb_array')
        env = ActionRepeat(env, n_repeat=self.params['action_repeat'])
        env = TimeLimit(env, max_episode_steps=self.params['max_episode_step'])
        env = PixelObservationWrapper(env)
        env = TransformObservation(env, lambda obs: self.process_gym_observation(obs['pixels']))
        env.reset(seed=self.params['rng_seed'])
        action_dim = env.action_space.shape[0]
        print(f'Initialized {self.params["env_name"]} as environment')
        return exp_tag, env, action_dim

    def get_dm_control_env(self):
        exp_tag = '_' + self.params['domain_name'] + '_' + self.params['task_name'] + '_'
        env = suite.load(
            domain_name=self.params['domain_name'],
            task_name=self.params['task_name'],
            task_kwargs={'random': self.params['rng_seed']}
        )
        env = pixels.Wrapper(env=env, render_kwargs={
            'height': self.params['observation_resolution'],
            'width': self.params['observation_resolution'],
            'camera_id': 0
        })
        env = GymWrapper(env)
        env = ActionRepeatDM(env, n_repeat=self.params['action_repeat'])
        env = TransformObservationDM(env, obs_transformation=self.process_dm_observation)
        env.reset()
        action_dim = env.action_space.shape[0]
        print(f'Initialized {self.params["domain_name"]}-{self.params["task_name"]} as environment')
        return exp_tag, env, action_dim

    def process_gym_observation(self, raw_obs):
        bits = self.params['pixel_bit']
        visual_resolution = self.params['observation_resolution']
        resized_obs = cv2.resize(raw_obs, dsize=(visual_resolution, visual_resolution), interpolation=cv2.INTER_AREA)
        bins = 2 ** bits
        norm_ob = np.float16(resized_obs)
        if bits < 8:
            norm_ob = np.floor(norm_ob / 2 ** (8 - bits))
        norm_ob = (norm_ob / bins) - 0.5
        processed_obs = torch.tensor(norm_ob, dtype=self.d_type)
        processed_obs = processed_obs.transpose(0, 2)
        return processed_obs

    def process_dm_observation(self, raw_obs):
        bits = self.params['pixel_bit']
        bins = 2 ** bits
        norm_ob = np.float16(raw_obs)
        if bits < 8:
            norm_ob = np.floor(norm_ob / 2 ** (8 - bits))
        norm_ob = (norm_ob / bins) - 0.5
        processed_obs = torch.tensor(norm_ob, dtype=self.d_type)
        processed_obs = processed_obs.transpose(0, 2)
        return processed_obs

    def collect_seed_episodes(self):
        print('\n')
        while len(self.replay_buffer.memory) < self.params['n_seed_episodes']:
            print(f'\rCollecting seed episodes ({1+len(self.replay_buffer.memory)}/{self.params["n_seed_episodes"]}) ... ', end='')
            prev_obs, _ = self.env.reset()
            episode_transitions = list()
            while True:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_transitions.append(
                    Transition(
                        observation=prev_obs,
                        action=torch.tensor(action, dtype=self.d_type),
                        reward=torch.tensor([reward], dtype=self.d_type)
                    )
                )
                prev_obs = copy.deepcopy(obs)
                if terminated or truncated:
                    break
            self.replay_buffer.push(episode_transitions)
        print(f'\rCollected {self.params["n_seed_episodes"]} episodes as initial seed data!')

    def collect_episode(self):
        print('\rCollecting a new episode with CEM-based planning ...', end='')
        self.world_model.eval()
        prev_obs, _ = self.env.reset()
        # h_state = torch.zeros((1, self.params['h_dim']), dtype=self.d_type, device=self.device)
        h_state = self.world_model.get_init_h_state(batch_size=1)
        episode_transitions = list()
        while True:
            # Inject observation noise
            noisy_prev_obs = (1/pow(2, self.params['pixel_bit']))*torch.randn_like(prev_obs) + prev_obs
            # Get posterior states using observation
            encoded_obs = self.world_model.obs_encoder(noisy_prev_obs.unsqueeze(dim=0).to(self.device))
            posterior_z = self.world_model.repr_model(h_state, encoded_obs)
            z_state = posterior_z.sample()
            # Get best action by planning in latent space through open-loop prediction
            with torch.no_grad():
                action = self.plan_action_with_cem(h_state, z_state)
            exploration_noise = Normal(loc=0.0, scale=self.params['action_epsilon']).sample(sample_shape=torch.Size(action.shape)).to(self.d_type).to(self.device)
            noisy_action = action + exploration_noise
            obs, reward, terminated, truncated, info = self.env.step(noisy_action.to('cpu').numpy())
            # Get next latent state
            h_state = self.world_model.rnn_model(h_state, z_state, noisy_action.unsqueeze(dim=0))
            # Save environment transition
            episode_transitions.append(
                Transition(
                    observation=prev_obs,
                    action=noisy_action.to('cpu'),
                    reward=torch.tensor([reward], dtype=self.d_type)
                )
            )
            prev_obs = copy.deepcopy(obs)
            if terminated or truncated:
                break
        print('\rCollected a new episode!' + 50*' ')
        return episode_transitions

    def learn_with_planet(self):
        global_step = 0
        for learning_step in range(self.params['n_steps']):
            print(f'\n\nLearning step: {1+learning_step}/{self.params["n_steps"]}\n')
            self.world_model.train()
            for update_step in range(self.params['collect_interval']):
                print(f'\rFitting world model : ({update_step+1}/{self.params["collect_interval"]})', end='')
                sampled_episodes = self.replay_buffer.sample(self.params['batch_size'])
                dist_predicted = self.world_model(sampled_episodes=sampled_episodes)
                loss, (recon_loss, kl_loss, reward_loss) = self.world_model.compute_loss(target=sampled_episodes, dist_predicted=dist_predicted)
                loss.backward()
                nn.utils.clip_grad_value_(self.world_model.parameters(), clip_value=self.params['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.add_scalar('Reward/max_from_train_batch', sampled_episodes['reward'].max(), global_step)
                self.logger.add_scalar('TrainLoss/obs_recon', recon_loss, global_step)
                self.logger.add_scalar('TrainLoss/kl_div', kl_loss, global_step)
                self.logger.add_scalar('TrainLoss/reward_prediction', reward_loss, global_step)
                global_step += 1
            print('\rUpdated world model!' + 50*' ')
            with torch.no_grad():
                self.replay_buffer.push(self.collect_episode())
                self.evaluate_learning(step=learning_step+1)
                self.evaluate_video_prediction(step=learning_step+1)

    def evaluate_learning(self, step):
        print('\rEvaluating learning progress ...', end='')
        self.world_model.eval()
        prev_obs, _ = self.env.reset()
        # h_state = torch.zeros((1, self.params['h_dim']), dtype=self.d_type, device=self.device)
        h_state = self.world_model.get_init_h_state(batch_size=1)
        observed_frames, reconstructed_frames = list(), list()
        ep_reward = 0
        while True:
            observed_frames.append(prev_obs)
            # Get posterior states using observation
            encoded_obs = self.world_model.obs_encoder(prev_obs.unsqueeze(dim=0).to(self.device))
            posterior_z = self.world_model.repr_model(h_state, encoded_obs)
            z_state = posterior_z.sample()
            # Get best action by planning in latent space through open-loop prediction
            action = self.plan_action_with_cem(h_state, z_state)
            obs, reward, terminated, truncated, info = self.env.step(action.to('cpu').numpy())
            ep_reward += reward
            # Reconstruct observation
            recon_obs = self.world_model.decoder_model(h_state, z_state).mean
            reconstructed_frames.append(recon_obs.squeeze())
            # Get next latent state
            h_state = self.world_model.rnn_model(h_state, z_state, action.unsqueeze(dim=0))
            prev_obs = copy.deepcopy(obs)
            if terminated or truncated:
                break
        observed_frames = torch.stack(observed_frames).unsqueeze(dim=0) + 0.5
        reconstructed_frames = torch.clip(torch.stack(reconstructed_frames).unsqueeze(dim=0) + 0.5, min=0.0, max=1.0)
        self.logger.add_scalar('Reward/test_episodes', ep_reward, step)
        if step % self.params['eval_gif_freq'] == 0:
            # self.logger.add_video(f'TestEpisodes/after_training_step', reconstructed_frames.transpose(3, 4), global_step=step)
            self.logger.add_video(f'ObservedTestEpisode/{step}', observed_frames.transpose(3, 4))
            self.logger.add_video(f'ReconstructedTestEpisode/{step}', reconstructed_frames.transpose(3, 4))
            print('\rLearning progress evaluation complete! Saved the episode!')
        else:
            print('\rLearning progress evaluation is complete!')

    def evaluate_video_prediction(self, step):
        if step % self.params['vp_eval_freq'] == 0:
            print('\rEvaluating video prediction ability ...', end='')
            n_context_frames = 5
            n_predicted_frames = 50
            self.world_model.eval()
            prev_obs, _ = self.env.reset()
            # h_state = torch.zeros((1, self.params['h_dim']), dtype=self.d_type, device=self.device)
            h_state = self.world_model.get_init_h_state(batch_size=1)

            observed_frames, predicted_frames = list(), list()
            observed_frames.append(prev_obs)
            # Feed context
            for _ in range(n_context_frames):
                encoded_obs = self.world_model.obs_encoder(prev_obs.unsqueeze(dim=0).to(self.device))
                posterior_z = self.world_model.repr_model(h_state, encoded_obs)
                z_state = posterior_z.sample()
                # Get best action by planning in latent space through open-loop prediction
                action = self.plan_action_with_cem(h_state, z_state)
                obs, reward, terminated, truncated, info = self.env.step(action.to('cpu').numpy())
                observed_frames.append(obs)
                # Reconstruct observation
                recon_obs = self.world_model.decoder_model(h_state, z_state).mean
                predicted_frames.append(recon_obs.squeeze())
                h_state = self.world_model.rnn_model(h_state, z_state, action.unsqueeze(dim=0))
                prev_obs = copy.deepcopy(obs)

                # Generate prediction
                for _ in range(n_predicted_frames):
                    prior_z = self.world_model.transition_model(h_state)
                    z_state = prior_z.sample()
                    action = self.plan_action_with_cem(h_state, z_state)
                    obs, reward, terminated, truncated, info = self.env.step(action.to('cpu').numpy())
                    observed_frames.append(obs)
                    # Reconstruct observation
                    recon_obs = self.world_model.decoder_model(h_state, z_state).mean
                    predicted_frames.append(recon_obs.squeeze())
                    h_state = self.world_model.rnn_model(h_state, z_state, action.unsqueeze(dim=0))

            observed_frames = 0.5 + torch.stack(observed_frames[:-1]).to(self.device)
            predicted_frames = torch.clip(0.5 + torch.stack(predicted_frames), min=0.0, max=1.0)
            overlay_frames = torch.clip(0.5*(1 - observed_frames) + 0.5*predicted_frames, min=0.0, max=1.0)

            combined_frame = torch.cat([observed_frames, predicted_frames, overlay_frames], dim=3).unsqueeze(dim=0)
            # self.logger.add_video(f'VideoPrediction/after_training_step', combined_frame.transpose(3, 4), global_step=step)
            self.logger.add_video(f'VideoPrediction/after_training_step_{step}', combined_frame.transpose(3, 4))
            print('\rVideo prediction evaluation is complete! Saved the episode!')

    def plan_action_with_cem(self, init_h_state, init_z_state):
        action_dist = Independent(Normal(loc=torch.zeros(self.params['planning_horizon'], self.action_dim), scale=1.0), reinterpreted_batch_ndims=2)
        for _ in range(self.params['plan_optimization_iter']):
            reward_buffer = list()
            h_state = torch.clone(init_h_state).repeat(self.params['n_plans'], 1)
            z_state = torch.clone(init_z_state).repeat(self.params['n_plans'], 1)
            candidate_plan = torch.clip_(
                action_dist.sample(sample_shape=torch.Size([self.params['n_plans']])).to(self.d_type).to(self.device),
                min=self.params['min_action'], max=self.params['max_action'])
            for time_step in range(self.params['planning_horizon']):
                batched_ts_action = candidate_plan[:, time_step, :]
                # Use learnt dynamics to get next hidden state
                h_state = self.world_model.rnn_model(h_state, z_state, batched_ts_action)
                # Get latent variables from transition model (prior)
                prior_z = self.world_model.transition_model(h_state)
                z_state = prior_z.sample()
                predicted_reward = self.world_model.reward_model(h_state, z_state)
                sampled_reward = torch.clip(predicted_reward.mean,
                                            min=self.params['min_reward'], max=(1+self.params['action_repeat'])*self.params['max_reward'])
                reward_buffer.append(sampled_reward)
            plan_reward = torch.stack(reward_buffer).squeeze().sum(dim=0)
            chosen_actions = candidate_plan[torch.topk(plan_reward, k=self.params['top_k']).indices]
            action_mu, action_std = chosen_actions.mean(dim=0), chosen_actions.std(dim=0)
            action_dist = Independent(Normal(loc=action_mu, scale=action_std+1e-6), reinterpreted_batch_ndims=2)
        optimized_next_action = action_dist.mean[0]
        return optimized_next_action


class ReplayBuffer:
    def __init__(self, params):
        self.params = params
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        # self.chunk_len = params['chunk_length']
        self.memory = list()

    def __len__(self):
        return len(self.memory)

    def push(self, episode):
        if len(episode) >= self.params['chunk_length']:
            self.memory.append(episode)

    def sample(self, n):
        sampled_indices = np.random.choice(len(self.memory), n, replace=True)
        chunked_episodes = list()
        for ep_idx in sampled_indices:
            start_idx = np.random.randint(low=0, high=len(self.memory[ep_idx])-self.params['chunk_length'])
            chunked_episodes.append(self.memory[ep_idx][start_idx:start_idx+self.params['chunk_length']])
        serialized_episodes = self.serialize_episode(chunked_episodes)
        return serialized_episodes

    def serialize_episode(self, list_episodes):
        batched_ep_obs, batched_ep_action, batched_ep_reward = list(), list(), list()
        for episode in list_episodes:
            ep_obs = torch.stack([transition.observation for transition in episode])
            ep_action = torch.stack([transition.action for transition in episode])
            ep_reward = torch.stack([transition.reward for transition in episode])
            batched_ep_obs.append(ep_obs)
            batched_ep_action.append(ep_action)
            batched_ep_reward.append(ep_reward)
        batched_ep_obs = torch.stack(batched_ep_obs).to(self.d_type).to(self.device)
        batched_ep_action = torch.stack(batched_ep_action).to(self.d_type).to(self.device)
        batched_ep_reward = torch.stack(batched_ep_reward).to(self.d_type).to(self.device)
        return {'obs': batched_ep_obs.transpose(0, 1), 'action': batched_ep_action.transpose(0, 1), 'reward': batched_ep_reward.transpose(0, 1)}
