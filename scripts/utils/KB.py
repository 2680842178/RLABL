import networkx as nx
import torch
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch.distributions.categorical import Categorical
import sys
sys.path.append("..")
import numpy
import random
from .abl_trace import abl_trace

G = nx.DiGraph()
# 添加对应的边和点
for i in range(5):
    G.add_node(i, desc='v' + str(i))  # 结点名称不能为str,desc为标签即结点名称
G.add_edge(0 , 1)  # 添加边， 参数name为边权值
G.add_edge(0, 4)
G.add_edge(1, 2)
G.add_edge(1, 4)
G.add_edge(2, 3)
G.add_edge(2, 4)



def get_state(env):
    return env.Current_state()

def get_fsm():
    return G


# def abl_trace(input_trace,output_trace):

#     return G

def Choose_agent(FSM_id):
    if FSM_id == 0:
        return 0
    if FSM_id == 1:
        return 1
    if FSM_id == 2:
        return 2
    if FSM_id == 3:
        return 2
    if FSM_id == 4:
        return 2


def Mutiagent_collect_experiences(env, acmodels,StateNN,device,num_frames_per_proc, discount, gae_lambda, preprocess_obss):

    def obs_To_state(model,pre_obs,obs):
        pre_image_data=preprocess_obss([pre_obs], device=device)
        image_data=preprocess_obss([obs], device=device)
        input_tensor = image_data.image[0]-pre_image_data.image[0]
        input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        output = model(input_batch)
        # print(output)
        with torch.no_grad():
            _, predicted = torch.max(output, 1)
        return predicted.item()
    
    def pre_obs_softmax(model, obs):
        image_data=preprocess_obss([obs], device=device)
        input_tensor = image_data.image[0]
        input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        output = model(input_batch)
        prob = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
        # print("prob", prob)
        return prob

    StateNN.eval()
    agent_num=len(acmodels)
    obs=env.gen_obs()
    pre_obs=obs

    mask_trace=[]
    obs_trace=[]
    state_trace=[]
    action_trace=[]
    value_trace=[]
    reward_trace=[]
    log_prob_trace=[]

    mask=torch.ones(1, device=device)

    done_counter=0
    log_return=[0]
    log_num_frames=[0]
    episode_reward_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device)

    # current_state=stateNN(env)
    current_state=obs_To_state(StateNN,pre_obs,obs)

    for i in range(num_frames_per_proc):
        mask_trace.append(mask)
        preprocessed_obs = preprocess_obss([obs], device=device)
        obs_trace.append(obs)

        state_trace.append(current_state)
        agent=acmodels[Choose_agent(current_state)]
        with torch.no_grad():
            dist, value = agent.acmodel(preprocessed_obs)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

        # next_state=stateNN(preprocess_obss([next_obs], device=device)).item()
        t=obs_To_state(StateNN, obs, next_obs)
        if t==0:
            next_state=current_state
        else:
            next_state = t

        if terminated or truncated:
            env.reset()

        done = terminated|truncated
        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        obs=next_obs
        current_state=next_state

        episode_reward_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.tensor(1, device=device)

        if done:
            done_counter += 1
            log_return.append(episode_reward_return.item())
            log_num_frames.append(log_episode_num_frames.item())

        episode_reward_return *= mask
        log_episode_num_frames *= mask

        action_trace.append(action)
        value_trace.append(value)
        reward_trace.append(torch.tensor(reward, device=device,dtype=torch.float))
        log_prob_trace.append(dist.log_prob(action))

    keep1 = max(done_counter,1)
    log1 = {
        "return_per_episode": log_return[-keep1:],
        "num_frames_per_episode": log_num_frames[-keep1:],
        "num_frames": num_frames_per_proc
    }
    # random_list = random.sample(range(len(state_trace)), 20)
    # for i in random_list:
    #     state_trace[i] = 2
    print("before abl", state_trace)

    # ABL修正state_trace
    state_list = [2, 3] # 课程路径
    trace_list=[]
    obs_trace_list=[]
    end_list=[]
    start_index=0
    for t in range(len(state_trace)):
        if(mask_trace[t]==0):
            trace_list.append(state_trace[start_index:t+1])
            obs_trace_list.append(obs_trace[start_index:t+1])
            start_index=t+1
            if reward_trace[t - 1] > 0:
                end_list.append(3)
            else:
                end_list.append(4)
    last_trace = state_trace[start_index:]
    last_obs_trace_list = obs_trace[start_index:]
    for i in range(len(last_trace)):
        last_trace[i] = 2
    # def testStateNN(obs):
    #     return [1, 1, 1, 1, 1, 1]
    def get_prob(obs):
        return pre_obs_softmax(StateNN, obs)
    for m in range(len(trace_list)):
        _, trace_list[m] = abl_trace(trace_list[m], state_list, end_list[m], obs_trace_list[m], get_prob)
    # last_trace = abl_trace(last_trace, [0, 1], 1, last_obs_trace_list, get_prob)
    ### abl over
    ### abl
    after_abl_list = trace_list + [last_trace]
    print("after abl:", after_abl_list)

    state_trace = []
    for m in range(len(trace_list)):
        state_trace.extend(trace_list[m])
    state_trace.extend(last_trace)
    ###

    for i in range(len(state_trace) - 1):
        if state_trace[i] != state_trace[i + 1]:
            # mental reward
            if reward_trace[i] == 0 and state_trace[i] != 3 and state_trace[i] != 4:
                reward_trace[i] = torch.tensor(3, device=device, dtype=torch.float)

    next_value=value_trace[-1]
    advantage_trace=[0]*len(action_trace)
    for i in reversed(range(num_frames_per_proc)):
        next_mask = mask_trace[i + 1] if i < num_frames_per_proc - 1 else mask
        next_value = value_trace[i + 1] if i < num_frames_per_proc - 1 else next_value
        next_advantage = advantage_trace[i + 1] if i < num_frames_per_proc - 1 else 0

        delta = reward_trace[i] + discount * next_value * next_mask - value_trace[i]
        advantage_trace[i] = delta + discount * gae_lambda * next_advantage * next_mask

    # print("After abl", state_trace)
    exps_list=[]

    for i in range(agent_num):
        exps=DictList()
        exps.obs=[]
        exps.action=[]
        exps.reward=[]
        exps.value=[]
        exps.advantage=[]
        exps.log_prob=[]
        exps_list.append(exps)

    # print(state_trace)
    # print([int(i.item()) for i in mask_trace])
    start_index=0
    for i in range(len(state_trace)-1):
        if state_trace[i]!=state_trace[i+1]:
            id=state_trace[start_index]
            # mental reward
            if id!=3 and id!=4:
                exps_list[id].obs.extend(obs_trace[start_index:i+1])
                exps_list[id].action.extend(action_trace[start_index:i+1])
                exps_list[id].reward.extend(reward_trace[start_index:i+1])
                exps_list[id].value.extend(value_trace[start_index:i+1])
                exps_list[id].advantage.extend(advantage_trace[start_index:i+1])
                exps_list[id].log_prob.extend(log_prob_trace[start_index:i+1])
            start_index=i+1
    if start_index<len(state_trace)-2:
        id = state_trace[start_index]
        exps_list[id].obs.extend(obs_trace[start_index:])
        exps_list[id].action.extend(action_trace[start_index:])
        exps_list[id].reward.extend(reward_trace[start_index:])
        exps_list[id].value.extend(value_trace[start_index:])
        exps_list[id].advantage.extend(advantage_trace[start_index:])
        exps_list[id].log_prob.extend(log_prob_trace[start_index:])

    for i in range(agent_num):
        exp_len=len(exps_list[i].obs)
        if exp_len:
            exps_list[i].obs = preprocess_obss(exps_list[i].obs, device=device)
            exps_list[i].action = torch.tensor(exps_list[i].action, device=device, dtype=torch.int)
            exps_list[i].reward = torch.tensor(exps_list[i].reward, device=device)
            exps_list[i].value = torch.tensor(exps_list[i].value, device=device)
            exps_list[i].advantage = torch.tensor(exps_list[i].advantage, device=device)
            exps_list[i].log_prob = torch.tensor(exps_list[i].log_prob, device=device)
            exps_list[i].returnn = exps_list[i].value + exps_list[i].advantage
    # print([int(i.item()) for i in reward_trace])
    log_reshaped_return=[0]
    log_done_counter=0
    log_episode_reshaped_return=torch.zeros(1, device=device)
    for i in range(len(reward_trace)):
        log_episode_reshaped_return+=reward_trace[i]
        if mask_trace[i].item() == 0 or i==len(reward_trace)-1:
            log_done_counter += 1
            log_reshaped_return.append(log_episode_reshaped_return.item())
        log_episode_reshaped_return *= mask_trace[i]

    keep2 = max(log_done_counter, 1)
    log2={
                "reshaped_return_per_episode": log_reshaped_return[-keep2:],
            }
    # print(log_reshaped_return[-keep2:])
    # print(action_trace)
    # print([i.item() for i in mask_trace])

    image_trace = preprocess_obss(obs_trace, device=device).image

    for i in range(len(image_trace)-1, 0, -1):
        image_trace[i]=image_trace[i]-image_trace[i-1]
    image_trace[0] = image_trace[0] - image_trace[0]

    for i in range(len(state_trace)-1, 0, -1):
        if state_trace[i]==state_trace[i-1]:
            state_trace[i]=0

    # print("state_trace:", state_trace)
    statenn_exps={
                "img": image_trace,
                "label": state_trace,
            }
    return exps_list, {**log1, **log2}, statenn_exps











