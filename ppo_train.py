import os
from datetime import datetime
import time

import torch
import numpy as np
import cv2

import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper

from ppo_agent import PPOAgent as Agent

import matplotlib.pyplot as plt

env_name = "MiniGrid-ConfigWorld-v0"
env = gym.make(env_name, 
                tile_size=32, 
                screen_size="640",
                max_steps=15000)

env = RGBImgObsWrapper(env)

obs_dict, _ = env.reset()
obs = obs_dict['image']
plt.imshow(obs)
plt.show()

max_ep_len = 15000
max_training_timesteps = int(1e6)

print_freq = max_ep_len * 1
log_freq = max_ep_len * 2
save_model_freq = int(5e4)

update_timestep = max_ep_len * 2
K_epochs = 80
eps_clip = 0.2
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001
random_seed = 0

state_size = env.observation_space['image'].shape
action_size = env.action_space.n
print('Original state shape: ', state_size)
print('Number of actions: ', env.action_space.n) # 7

agent = Agent((12, 84, 84), action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, random_seed)

TRAIN = True

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

def train():
    print("============================================================================================")
    
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    lod_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
    
    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    
        ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    
    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################
    
    print("============================================================================================")
    
    # training 
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')
    
    print_running_reward = 0
    print_running_episodes = 0
    
    log_running_reward = 0  
    log_running_episodes = 0
    
    time_step = 0
    i_episode = 0
    
    # training loop
    while time_step <= max_training_timesteps:
        state_dict, _ = env.reset()
        state = state_dict['image']
        state = pre_process(state)
        state = init_state(state)
        state = np.squeeze(state)
        current_ep_reward = 0
        
        for t in range(max_ep_len):
            action = agent.select_action(state)
            next_state_dict, reward, done, truncated, _ = env.step(action)
            next_state=next_state_dict['image']
            next_state = np.stack((
                state[3], state[4], state[5],
                state[6], state[7], state[8],
                state[9], state[10], state[11],
                np.squeeze(pre_process(next_state)[0]),
                np.squeeze(pre_process(next_state)[1]),
                np.squeeze(pre_process(next_state)[2]),
            ), axis=0)
            next_state=np.squeeze(next_state)
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            if time_step % update_timestep == 0:
                agent.update()
                
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
                
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0
                
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
    
            state = next_state
    
            if done or truncated:
                break
            
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        
    log_f.close()
    env.close()
    
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    
def test():    
    print("============================================================================================")
    # preTrained weights directory

    run_num_pretrained = 0      #### set this to load a particular checkpoint num
    total_test_episodes = 10
    render = True
    frame_delay = 0

    directory = "PPO_preTrained" + '\\' + env_name + '\\'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    for ep in range(total_test_episodes):
        ep_reward = 0
        state_dict, _ = env.reset()
        state = state_dict['image']
        state = pre_process(state)
        state = init_state(state)
        state = np.squeeze(state)
        
        for t in range(max_ep_len):
            action = agent.select_action(state)
            next_state_dict, reward, done, truncated, _ = env.step(action)
            next_state=next_state_dict['image']
            next_state = np.stack((
                state[3], state[4], state[5],
                state[6], state[7], state[8],
                state[9], state[10], state[11],
                np.squeeze(pre_process(next_state)[0]),
                np.squeeze(pre_process(next_state)[1]),
                np.squeeze(pre_process(next_state)[2]),
            ), axis=0)
            next_state=np.squeeze(next_state)
            ep_reward += reward
            
            if render:
                env.render()
                time.sleep(frame_delay)
            
            if done:
                break
            
        agent.buffer.clear()
        
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        
    env.close()
    
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == "__main__":
    if TRAIN:
        train()
    else:
        test()
    
    
    
