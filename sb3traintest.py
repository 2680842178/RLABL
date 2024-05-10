import gymnasium as gym
from gymnasium import spaces
import minigrid
import argparse
import time
import datetime
import numpy as np
import tensorboard
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn
from torchvision import transforms

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


'''
说明：minigrid环境被我修改过，使用的时候需要在gym.make()中指定reward_type参数：
reward_type = 0: 原minigrid环境
reward_type = 1: 拿钥匙，钥匙奖励为1.0，拿到钥匙后终止episode
reward_type = 2: 已经有钥匙，开门奖励为1.0，开门后终止episode
reward_type = 3: 门已开，走到终点奖励为1.0，终止episode
'''

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--train", type=int, default=0)
parser.add_argument("--env-name", type=str, default='MiniGrid-ConfigWorld-v0')
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--reward-type", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=512)
# reward_type:
# 0: goal=1, lava=-1. 
# 1: key=1&terminated, lava = -1
# 2: have key, door = 1 & terminated, lava=-1
# 3：door is open, goal = 1 & terminated, lava=-1
# 4: lava=1
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--totaltimesteps", type=int, default=100000000,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.00025,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
# parser.add_argument("--optim-alpha", type=float, default=0.99,
#                     help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=12, stride=6, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    

class CustomCallback(BaseCallback):
    """
    A custom callback that evaluates the agent and saves
    the best model every `eval_freq` steps and also saves
    the model every `save_freq` steps.
    """
    def __init__(self, eval_env, eval_freq, save_freq, best_model_save_path, regular_save_path, n_eval_episodes=5, deterministic=True, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_model_save_path = best_model_save_path
        self.regular_save_path = regular_save_path
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent and save if it's the best
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_save_path)

        if self.n_calls % self.save_freq == 0:
            # Save the current model
            self.model.save(self.regular_save_path)

        return True
    
def preprocess_obs(obs):
    obs = transforms.ToTensor()(obs)
    resize = transforms.Resize((256, 256))
    obs = resize(obs)
    return obs.cpu().numpy()

class MyWrapper(gym.Wrapper):
    def __init__(self, env_name='MiniGrid-ConfigWorld-v0', reward_type=1, max_steps=10000, seed=1, render_mode=None):
        env = gym.make(env_name, reward_type=reward_type, max_steps=max_steps, render_mode=render_mode)
        super().__init__(env)
        self.env = env
        self.action_space = spaces.Discrete(7)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, 256, 256), dtype=np.uint8)
        self.counter = 0 ###

    def reset(self, seed=1):
        self.counter = 0 ###
        state, _ = self.env.reset()
        state = state['image']
        state = preprocess_obs(state)
        info = {}
        return state, info

    def step(self, action):
        self.counter += 1 ###
        state, reward, terminated, truncated, info = self.env.step(action)
        state = state['image']
        state = preprocess_obs(state)
        if reward > 0:
            print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        return state, reward, terminated, truncated, info
    
    def render(self):
        # 输出测试图像，测试环境的predict和step有bug，不能输出正确的动作，使用evaluate_policy进行测试和输出，调用render输出图像：
        image = self.env.render()
        if self.render_mode == 'rgb_array':
            plt.figure()
            plt.imshow(image)
            plt.savefig(f"./pictures/opendoor/frame_{self.counter}.png")
        return image
    
def make_my_env(env_name='MiniGrid-ConfigWorld-v0', reward_type=1, max_steps=10000):
    env = MyWrapper(env_name, reward_type, max_steps)
    return env  

if __name__ == "__main__":
    args = parser.parse_args()
    TRAIN = args.train
    print(TRAIN)
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env_name}_seed{args.seed}_{date}"
    model_dir = f"./ppo_models/{default_model_name}"
    model_tensorboard_dir = f"./ppo_models/tensorboard/"
    env = make_vec_env(lambda **kwargs: make_my_env(args.env_name, reward_type=args.reward_type), n_envs=args.num_envs, env_kwargs={'max_steps': 10000})
    eval_env = MyWrapper(args.env_name, reward_type=args.reward_type, max_steps=10000, render_mode='rgb_array')
    policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
    # eval_callback = EvalCallback(eval_env, best_model_save_path=f"./ppo_models/models/{default_model_name}",
    #                             log_path=f"./ppo_models/logs/{default_model_name}", eval_freq=10000,
    #                             deterministic=True, render=False)
    my_callback = CustomCallback(eval_env, eval_freq=10000, save_freq=10000, 
                                 best_model_save_path=f"./ppo_models/models/best_{default_model_name}", 
                                 regular_save_path=f"./ppo_models/models/last_{default_model_name}", 
                                 n_eval_episodes=5, deterministic=True, verbose=1)
    model = PPO("CnnPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                seed=args.seed, 
                learning_rate=args.lr,
                tensorboard_log=model_tensorboard_dir,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                gamma=args.discount,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_eps,
                ent_coef=args.entropy_coef,
                )
    new_env = model.get_env()
    if TRAIN == 1:
        model.learn(total_timesteps=args.totaltimesteps, 
                    callback=my_callback,
                    tb_log_name=default_model_name,
                    # progress_bar=True,
                    )
        model.save(f"./ppo_models/models/last_{default_model_name}")
    else:
        eval_model = PPO.load("./ppo_models/models/MiniGrid-ConfigWorld-v0_seed1_24-04-12-15-38-07/best_model.zip",
                              env = new_env)
        # 若不想输出图像render改为False
        mean_reward, std_reward = evaluate_policy(eval_model, eval_env, n_eval_episodes=1, render=True)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        # frames = []
        # obs, _ = eval_env.reset()
        # frame = eval_env.render()
        # for i in range(20):
        #     frames.append(frame)
        #     action, _ = eval_model.predict(obs)
        #     obs, _, terminated, truc, _ = eval_env.step(action)
        #     frame = eval_env.render()
        #     # plt.figure()
        #     # plt.imshow(obs)
        #     # plt.savefig(f"frame_{i}.png")
        #     if terminated:
        #         break
        # animate_frames(frames)
