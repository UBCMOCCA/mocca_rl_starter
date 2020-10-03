import torch
import torch.nn.functional as F
from gym import Env
from torch.optim import Adam


class RNDEnvWrapper(Env):
    """
    Wraps around an environment with Random Network Distillation for intrinsic reward calculation
    """

    def __init__(self, env, encoder, target_encoder, intrinsic_weight, n_epochs):
        self.target_encoder = target_encoder
        self.encoder = encoder
        self.env = env
        self.intrinsic_weight = intrinsic_weight
        self.n_epochs = n_epochs

        # Set up optimization stuff
        trainable_parameters = []
        trainable_parameters.extend(list(self.encoder.parameters()))
        self.optimizer = Adam(lr=3e-4, params=trainable_parameters)

    def render(self, mode='human'):
        return self.render(mode=mode)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, extrinsic_reward, done, info = self.env.step(action)

        # Compute intrinsic reward based on distillation error
        obs_tensor = torch.tensor(obs.reshape(1, -1)).float()
        encoded_obs = self.encoder.forward(obs_tensor)
        target_encoded_obs = self.target_encoder.forward(obs_tensor)
        loss = F.mse_loss(encoded_obs, target_encoded_obs)
        intrinsic_reward = loss.item()
        if extrinsic_reward < 1e-7 and done:
            # don't add reward if failed
            reward = 0
        else:
            reward = extrinsic_reward + self.intrinsic_weight * intrinsic_reward

        return obs, reward, done, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
