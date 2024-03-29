import pygame
import gymnasium as gym
from gymnasium import Env
import math
import random
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# from itertools import count
# import torch
import numpy as np
# from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper,RGBImgPartialObsWrapper
# from collections import deque
# from rl_agent import Agent
# from rl_train import pre_process,init_state
# import time
# import queue
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def Img_to_State(img):

    img = cv2.resize(img, (300, 300))
    R, G, B = cv2.split(img)

    return np.stack([R,G,B])



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))



if __name__ == "__main__":
    import argparse

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


    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)
    print(gym.envs.registry.keys())

    state_size = env.observation_space.shape
    action_size = env.action_space.n
    env.reset(seed=args.seed)
    img0=env.render()
    plt.imsave('trace/0.jpg', img0)
    action = random.choice(np.arange(action_size))
    action = 2
    env.step(action)
    img1=env.render()
    plt.imsave('trace/1.jpg', img1)

    img2= img1-img0
    plt.imsave('trace/2.jpg', img2)














    # total_reward = 0
    # event_list=[]
    # memory=[]
    # memory.append(img)
    # state=Img_to_State(img)
    # print(state.shape)
    # state_seq=[]
    # act_seq=[]



    # for j in range(100):  # frames, in case stuck in one frame
    #
    #     action=random.choice(np.arange(action_size))
    #     next_state, reward, done, truncated, _ = env.step(action)
    #     next_img=env.render()
    #     memory.append(next_img)
    #     total_reward += reward
    #     # time.sleep(0.01)
    #     if done or truncated:
    #         print(total_reward)
    #         break
    # print(len(memory))
    # for i in range(len(memory)):
    #     plt.imsave('trace/{i}.jpg'.format(i=i), memory[i])




# ##SAM model
#     from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
#
#     # 加载模型
#     sam_checkpoint = 'E:\SAM_model\meta_sam\sam_vit_h_4b8939.pth'
#     model_type = "vit_h"
#
#     device = "cuda"
#
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     image=memory[-1]
#     # 调用全局分割模型
#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=32,
#         points_per_batch=256,
#         pred_iou_thresh=0.88,  # IOU阈值
#         stability_score_thresh=0.92,  # 稳定性得分阈值
#         crop_n_layers=1,
#         crop_n_points_downscale_factor=2,
#         min_mask_region_area=100,  # Requires open-cv to run post-processing
#     )
#
#     masks = mask_generator.generate(image)
#     print(len(masks))  # 产生的掩码数量
#     print(masks[0].keys())  # 第1个掩码内的相关属性
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_anns(masks)
#     plt.axis('off')
#     plt.show()


#
# state_size = env.observation_space.shape
# action_size = env.action_space.n
# agent = Agent((32, 12, 84, 84), action_size, seed=1)  # state size (batch_size, 4 frames, img_height, img_width)
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint/dqn_checkpoint_8.pth'))
#
# total_reward = 0
# memory=[]
#
# obs_dict, _ = env.reset()
# obs = obs_dict['image']
#
#
# # memory.append(obs)
# obs = pre_process(obs)
# state = init_state(obs)
# state = np.squeeze(state)
# plt.imsave('trace/1.jpg', obs)


# #DQN
# for j in range(200):  # frames, in case stuck in one frame
#     action = agent.act(state)
#     env.render()
#     next_state_dict, reward, done, truncated, _ = env.step(action)
#     next_state = next_state_dict['image']
#     memory.append(next_state)
#     state = np.stack((
#         state[3], state[4], state[5],
#         state[6], state[7], state[8],
#         state[9], state[10], state[11],
#         np.squeeze(pre_process(next_state)[0]),
#         np.squeeze(pre_process(next_state)[1]),
#         np.squeeze(pre_process(next_state)[2]),), axis=0)
#     total_reward += reward
#     # time.sleep(0.01)
#     if done or truncated:
#         print(total_reward)
#         break

# #random
# for j in range(200):  # frames, in case stuck in one frame
#
#     action=random.choice(np.arange(action_size))
#     env.render()
#     next_state_dict, reward, done, truncated, _ = env.step(action)
#     next_state = next_state_dict['image']
#     memory.append(next_state)
#     total_reward += reward
#     # time.sleep(0.01)
#     if done or truncated:
#         print(total_reward)
#         break
#
# for i in range(len(memory)):
#     plt.imsave('trace/{i}.jpg'.format(i=i), memory[i])
