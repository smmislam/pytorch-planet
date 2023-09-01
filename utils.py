import random
import torch
import numpy as np
import gymnasium
from moviepy.editor import ImageSequenceClip


def get_device(tag):
    if tag == 'cpu':
        return 'cpu'
    elif tag[:4] == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        NotImplementedError(f'{tag} was not found as a valid device!')


def get_dtype(precision):
    if precision == 16:
        return torch.float16
    elif precision == 32:
        return torch.float32
    else:
        NotImplementedError(f'{precision} as precision is not implemented!')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)


def count_parameters(model, human_format=True):
    n_params = np.sum([torch.numel(p) for p in model.parameters() if p.requires_grad])
    if human_format:
        base = 1000.0
        units = ['', 'K', 'M', 'B', 'T', 'QD']
        order = int(np.emath.logn(n=base, x=n_params))
        return f'{n_params/(base**order):0.2f}{units[order]}'
    else:
        return f'{n_params}'


def create_video(target, prediction, fps=2):
    random_index = np.random.choice(len(target), 1)[0]
    target_ep, reconstructed_ep = post_process(target[random_index].detach().to('cpu')), post_process(prediction[random_index].detach().to('cpu'))
    stitched_ep = torch.cat([target_ep, torch.zeros_like(target_ep), reconstructed_ep], dim=2).transpose(1, 3).numpy()
    list_frames = [np.uint8(frame) for frame in stitched_ep]
    video_clip = ImageSequenceClip(list_frames, fps=fps)
    return video_clip


def post_process(raw_output, bits=5):
    bins = 2 ** bits
    processed_output = bins * (raw_output + 0.5) * (256.0/bins)
    processed_output = torch.clip(processed_output, min=0.0, max=255.0)
    return processed_output


class ActionRepeat(gymnasium.Wrapper):
    def __init__(self, env, n_repeat: int):
        super(ActionRepeat, self).__init__(env)
        assert n_repeat >= 0, 'n_repeat must be positive'
        self.n_repeat = n_repeat

    def step(self, action):
        # Regular step
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        net_reward = reward
        # Repeated steps
        if not(terminated or truncated):
            for _ in range(self.n_repeat):
                obs, reward, terminated, truncated, info = self.env.step(action=action)
                net_reward += reward
                if terminated or truncated:
                    break
        return obs, net_reward, terminated, truncated, info


class GymWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def observation_space(self):
        obs_spec = self.env.observation_spec()
        return gymnasium.spaces.Box(0, 255, obs_spec['pixels'].shape, dtype=np.uint8)

    @property
    def action_space(self):
        action_spec = self.env.action_spec()
        return gymnasium.spaces.Box(action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def reset(self):
        time_step = self.env.reset()
        obs = time_step.observation['pixels']
        info = {'discount': time_step.discount}
        return obs, info

    def step(self, action):
        time_step = self.env.step(action)
        obs = time_step.observation['pixels']
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False
        info = {'discount': time_step.discount}
        return obs, reward, terminated, truncated, info


class ActionRepeatDM(GymWrapper):
    def __init__(self, env, n_repeat: int):
        super(ActionRepeatDM, self).__init__(env)
        assert n_repeat >= 0, 'n_repeat must be positive'
        self.n_repeat = n_repeat

    def reset(self):
        return self.env.reset()

    def step(self, action):
        net_reward = 0
        # Regular step
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        net_reward += reward
        # Repeated steps
        if not(terminated or truncated):
            for _ in range(self.n_repeat):
                obs, reward, terminated, truncated, info = self.env.step(action=action)
                net_reward += reward
                if terminated or truncated:
                    break
        return obs, net_reward, terminated, truncated, info


class TransformObservationDM(GymWrapper):
    def __init__(self, env, obs_transformation):
        super(TransformObservationDM, self).__init__(env)
        self.transformation = (lambda x: x) if obs_transformation is None else obs_transformation

    def reset(self):
        obs, info = self.env.reset()
        return self.transformation(obs), info

    def step(self, action):
        # Regular step
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        return self.transformation(obs), reward, terminated, truncated, info

