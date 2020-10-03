from argparse import ArgumentParser

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

np.set_printoptions(formatter={'float': "{:0.3f}".format})


class EnvFactory:

    def __init__(self, env_name):
        self.env_name = env_name

    def make_env(self):
        return gym.make(self.env_name, render=True)


def main(args):
    expert = None
    expert_state_dim = 0
    if args.policy_path is not None:
        policy_path = args.policy_path
        expert = PPO.load(policy_path)
        expert_state_dim = expert.observation_space.shape[0]

    factory = EnvFactory(args.env)
    env = DummyVecEnv([factory.make_env])
    if args.stats_path is not None:
        env = VecNormalize.load(args.stats_path, env)
        env.training = False
    else:
        env = VecNormalize(env, training=False)

    obs = env.reset()
    env.render()
    total_reward = 0
    while True:
        if expert is None:
            action = env.action_space.sample()
            action = np.zeros_like(action)
        else:
            good_obs = obs[:, :expert_state_dim]
            action, _ = expert.predict(good_obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        reward = env.get_original_reward()
        total_reward += reward[0]
        if done:
            print("Total reward: {:.3f}".format(total_reward))
            obs = env.reset()
            total_reward = 0
        # Uncomment below to slow down rendering
        # import time; time.sleep(0.05)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", help="Name of the environment as defined in __init__.py somewhere", type=str, required=True)
    parser.add_argument("--policy_path", help="Path to policy zip file, if any. Otherwise compute null actions.", type=str)
    parser.add_argument("--stats_path", help="Path to policy normalization stats.", type=str)
    args = parser.parse_args()
    main(args)
