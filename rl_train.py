# -*- coding: utf-8 -*-
import pygame
import gymnasium as gym
from gymnasium import Env
import math
import random
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import numpy as np
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from collections import deque
from rl_agent import Agent
import time
import queue
import argparse



def Img_to_State(img):

    img = cv2.resize(img, (300, 300))
    R, G, B = cv2.split(img)

    return np.stack([R,G,B])

def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    #x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    x_t = cv2.resize(observation, (84, 84))
    r, g, b = np.dsplit(x_t, 3)
    return [r, g, b]



def init_state(processed_obs):
    return np.stack((processed_obs[0],processed_obs[1],processed_obs[2],
                     processed_obs[0], processed_obs[1], processed_obs[2],
                     processed_obs[0], processed_obs[1], processed_obs[2],
                     processed_obs[0], processed_obs[1], processed_obs[2],
                     ), axis=0)


def dqn(n_episodes=100000, max_t=200, eps_start=1.0, eps_end=0.05, eps_decay=0.9995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode, maximum frames
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env.reset(seed=args.seed)
        img = env.render()
        state = Img_to_State(img)
        state=np.squeeze(state)

        score = 0
        for t in range(max_t):

            action = agent.act(state, eps)
            _, reward, done,truncated, _ = env.step(action)
            next_img = env.render()
            next_state=Img_to_State(next_img)
            next_state=np.squeeze(next_state)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\tEpsilon now : {:.2f}'.format(eps))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rEpisode {}\tThe length of replay buffer now: {}'.format(i_episode, len(agent.memory)))

        if np.mean(scores_window) >= 0.9:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint/dqn_checkpoint_solved.pth')
            break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint/dqn_checkpoint_6.pth')
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        # default="MiniGrid-MultiRoom-N6-v0",
        default="MiniGrid-ConfigWorld-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        default=False,
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )
    args = parser.parse_args()
    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="rgb_array",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )
    env.reset(seed=args.seed)
    img = env.render()
    state_size = env.observation_space.shape

    action_size = env.action_space.n
    print('Original state shape: ', state_size)
    print('Number of actions: ', env.action_space.n)
    agent = Agent((16, 3, 300, 300), action_size, seed=6)  # state size (batch_size, 4 frames, img_height, img_width)
    TRAIN = True  # train or test flag


    if TRAIN:
        start_time = time.time()
        scores = dqn()
        print('COST: {} min'.format((time.time() - start_time)/60))
        print("Max score:", np.max(scores))

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint/dqn_checkpoint_6.pth'))
        rewards = []
        memory_buffer=queue.Queue()
        for i in range(10):  # episodes, play ten times
            total_reward = 0
            obs_dict, _ = env.reset()
            obs = obs_dict['image']
            plt.imsave('1.jpg', obs)
            obs = pre_process(obs)
            state = init_state(obs)
            state = np.squeeze(state)
            for j in range(200):  # frames, in case stuck in one frame
                action = agent.act(state)
                env.render()
                next_state_dict, reward, done, truncated, _ = env.step(action)
                next_state = next_state_dict['image']
                state = np.stack((
                    state[3], state[4], state[5],
                    state[6], state[7], state[8],
                    state[9], state[10], state[11],
                    np.squeeze(pre_process(next_state)[0]),
                    np.squeeze(pre_process(next_state)[1]),
                    np.squeeze(pre_process(next_state)[2]),), axis=0)
                total_reward += reward
                # time.sleep(0.01)
                if done or truncated :
                    print(action)
                    rewards.append(total_reward)
                    break
        print("Test rewards are:", rewards)
        print("Average reward:", np.mean(rewards))
        env.close()
        img = np.concatenate((obs[0], obs[1], obs[2]), axis=2)
        plt.imsave('1.jpg',img)