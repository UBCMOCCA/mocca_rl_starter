import os
from argparse import ArgumentParser
from shutil import copyfile

import gym
import numpy as np
import torch
import wandb
from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnv, DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from torch import nn

np.set_printoptions(formatter={'float': "{:0.3f}".format})
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


class EnvFactory:
    """
    Factory pattern helps use parallel processing support more elegantly
    """

    def __init__(self, env_name):
        self.env_name = env_name

    def make_env(self):
        return gym.make(self.env_name, render=False)


def get_cumulative_rewards_from_vecenv_results(reward_rows, done_rows):
    """
    rewards and dones are 2d numpy arrays.
    each row corresponds to a process running the environment.
    each column corresponds to a timestep.
    for each row, we accumulate up to the first case where done=True
    """
    cumulative_rewards = []
    for reward_row, done_row in zip(reward_rows, done_rows):
        cumulative_reward = 0
        for reward, done in zip(reward_row, done_row):
            cumulative_reward += reward
            if done:
                break
        cumulative_rewards.append(cumulative_reward)
    return np.array(cumulative_rewards)


class WAndBEvalCallback(BaseCallback):

    def __init__(self, render_env: Env, eval_every: int, envs: VecNormalize, verbose=0):
        self.render_env = render_env  # if render with rgb_array is implemented, use this to collect images
        self.eval_every = eval_every
        self.best_cumulative_rewards_mean = -np.inf
        self.envs = envs
        super().__init__(verbose)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        run_dir = os.path.join("checkpoints", "{:s}_{:s}".format(args.project_name, args.run_name))
        os.makedirs(run_dir, exist_ok=True)

        # save policy weights
        self.model.save(os.path.join(wandb.run.dir, "latest.zip"))
        self.model.save(os.path.join(run_dir, "latest.zip".format(args.project_name, args.run_name)))
        copyfile(os.path.join(run_dir, "latest.zip".format(args.project_name, args.run_name)),
                 os.path.join(run_dir, "second_latest.zip".format(args.project_name, args.run_name)))

        # save stats for normalization
        self.envs.save(os.path.join(wandb.run.dir, "latest_stats.pth"))
        stats_path = os.path.join(run_dir, "latest_stats.pth")
        self.envs.save(stats_path)
        copyfile(os.path.join(run_dir, "latest_stats.pth".format(args.project_name, args.run_name)),
                 os.path.join(run_dir, "second_latest_stats.pth".format(args.project_name, args.run_name)))

        metrics = {"n_calls": self.n_calls}
        if self.n_calls % self.eval_every == 0:
            obs_column = self.envs.reset()
            reward_columns = []
            done_columns = []
            actions = []
            # We can optionally gather images and render as video
            # images = []
            self.envs.training = False
            for i in range(1000):
                action_column, states = self.model.predict(obs_column, deterministic=True)
                next_obs_column, old_reward_column, done_column, info = self.envs.step(action_column)
                for a in action_column:
                    actions.append(a)
                reward_column = self.envs.get_original_reward()
                reward_columns.append(reward_column)
                done_columns.append(done_column)
                obs_column = next_obs_column

            self.envs.training = True
            reward_rows = np.stack(reward_columns).transpose()
            done_rows = np.stack(done_columns).transpose()
            cumulative_rewards = get_cumulative_rewards_from_vecenv_results(reward_rows, done_rows)
            cumulative_rewards_mean = np.mean(cumulative_rewards)
            # Also can compute standard deviation of rewards across different inits
            # cumulative_rewards_std = np.std(cumulative_rewards)
            # Uploads images as video
            # images = np.stack(images)
            # metrics.update({"video": wandb.Video(images, fps=24, format="mp4")})

            # Can also do other things like upload plots, etc.

            if cumulative_rewards_mean > self.best_cumulative_rewards_mean:
                self.best_cumulative_rewards_mean = cumulative_rewards_mean
                self.model.save(os.path.join(wandb.run.dir, "best.zip"))
                self.model.save(os.path.join(run_dir, "best.zip"))

                self.envs.save(os.path.join(wandb.run.dir, "best_stats.pth"))
                self.envs.save(os.path.join(run_dir, "best_stats.pth"))

            metrics.update({"cumulative_rewards_mean": cumulative_rewards_mean})

        wandb.log(metrics)


def main(args):
    wandb.init(project=args.project_name, name=args.run_name)
    n_envs = len(os.sched_getaffinity(0))
    factory = EnvFactory(args.env)

    # Wrap the
    render_env = factory.make_env()  # for rendering

    callback = CallbackList([])

    # Wrap the environment around parallel processing friendly wrapper, unless debug is on
    if args.debug:
        envs = DummyVecEnv([factory.make_env for _ in range(n_envs)])
    else:
        envs = SubprocVecEnv([factory.make_env for _ in range(n_envs)])
    #
    if args.stats_path is None:
        envs = VecNormalize(envs)
    else:
        envs = VecNormalize.load(args.stats_path, envs)
    eval_callback = WAndBEvalCallback(render_env, args.eval_every, envs)
    callback.callbacks.append(eval_callback)

    # We use PPO by default, but it should be easy to swap out for other algorithms.
    if args.pretrained_path is not None:
        pretrained_path = args.pretrained_path
        learner = PPO.load(pretrained_path, envs)
        learner.learn(total_timesteps=10000000, callback=callback)
    else:
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(
                vf=[256, 256],
                pi=[256, 256]
            )
            ],
            log_std_init=args.log_std_init,
            squash_output=False

        )
        learner = PPO(MlpPolicy, envs, n_steps=args.n_steps, verbose=1, policy_kwargs=policy_kwargs)
        learner.learn(total_timesteps=args.total_timesteps, callback=callback)

    render_env.close()
    envs.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project_name", help="Weights & Biases project name", required=True, type=str)
    parser.add_argument("--run_name", help="Weights & Biases run name", required=True, type=str)
    parser.add_argument("--env", help="Name of the environment as registered in __init__.py somewhere", required=True,
                        type=str)
    parser.add_argument("--n_steps", help="Number of timesteps in each rollouts when training with PPO", required=True,
                        type=int)
    parser.add_argument("--total_timesteps", help="Total timesteps to train with PPO", required=True,
                        type=int)
    parser.add_argument("--eval_every", help="Evaluate current policy every eval_every episodes", required=True,
                        type=int)
    parser.add_argument("--pretrained_path", help="Path to the pretrained policy zip file, if any", type=str)
    parser.add_argument("--stats_path", help="Path to the pretrained policy normalizer stats file, if any", type=str)
    parser.add_argument("--log_std_init", help="Initial Gaussian policy exploration level", type=float, default=-2.0)
    parser.add_argument("--debug", help="Set true to disable parallel processing and run debugging programs",
                        action="store_true")
    args = parser.parse_args()
    main(args)
