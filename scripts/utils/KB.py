import networkx as nx
import torch
from torchvision import transforms
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch.distributions.categorical import Categorical
import sys
import heapq
import copy
sys.path.append("..")
import numpy
import random
from .abl_trace import abl_trace
from .env import copy_env
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def get_state(env):
    return env.Current_state()

def sample_from_selected_dimensions(logits, selected_dims):
    # 创建一个与logits形状相同的mask，其值全部初始化为一个非常小的负数
    mask = torch.full_like(logits, fill_value=-float('Inf'))
    # 将选定维度的mask值设置为0，这样softmax后它们的概率不会变成0
    mask[:, selected_dims] = 0
    # 应用mask并执行softmax
    masked_logits = logits + mask
    probabilities = torch.softmax(masked_logits, dim=1)
    # 根据概率分布进行采样
    samples = torch.multinomial(probabilities, num_samples=1)
    return samples

def obs_To_state(current_state,
                preprocess_obss,
                anomalyNN, 
                contrast, 
                G: nx.DiGraph,
                pre_obs,
                obs,
                device):
    pre_image_data=preprocess_obss([pre_obs], device=device)
    image_data=preprocess_obss([obs], device=device)
    input_tensor = image_data.image[0]-pre_image_data.image[0]
    mutation = numpy.squeeze(input_tensor).cpu().numpy().astype(numpy.uint8)
    anomaly_mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)  
    # input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    # print(anomalyNN(input_batch))
    # return current_state
    ####### 下面是原先的代码
    if anomalyNN(anomaly_mutation)[0, 0] >= anomalyNN(anomaly_mutation)[0, 1]:
        return current_state
    similiarity = []
    for next_state in list(G.successors(current_state)):
        similiarity.append((next_state, contrast(mutation, G.nodes[next_state]['state'].mutation)))  
    output = max(similiarity, key=lambda x: x[1]) 
    if output[1] < 0.98:
        return current_state
    output = output[0]
    return output

def Mutiagent_collect_experiences(env, 
                                algos,
                                contrast,
                                G: nx.DiGraph,
                                start_node,
                                anomalyNN,  
                                device,
                                num_frames_per_proc, 
                                discount, 
                                gae_lambda, 
                                preprocess_obss):

    # def pre_obs_softmax(model, obs):
    #     image_data=preprocess_obss([obs], device=device)
    #     input_tensor = image_data.image[0]
    #     input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    #     output = model(input_batch)
    #     prob = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
    #     # print("prob", prob)
    #     return prob

    agent_num=len(algos)
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
    done=0

    done_counter=0
    log_return=[0]
    log_num_frames=[0]
    episode_reward_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device)


    env_ini_state = start_node 
    current_state=env_ini_state
    state_ini_flag=False

    for _ in range(num_frames_per_proc):

        preprocessed_obs = preprocess_obss([obs], device=device)

        # t,prob_dist=obs_To_state(StateNN, pre_obs, obs)

        # if t==0:
        #     current_state=current_state
        # else:
        #     candidate_list=Candidate(current_state)
        #     t=sample_from_selected_dimensions(prob_dist,candidate_list)
        #     current_state = t
        if not state_ini_flag:
            current_state = obs_To_state(current_state,
                                     preprocess_obss, 
                                     anomalyNN, 
                                     contrast, 
                                     G, 
                                     pre_obs, 
                                     obs, 
                                     device)


        if state_ini_flag:
            current_state=env_ini_state
            state_ini_flag=False
        if current_state != 0 and current_state != 1:
            agent = G.nodes[current_state]['state'].agent

            with torch.no_grad():
                dist, value = agent.acmodel(preprocessed_obs)
            action = dist.sample()

        if done:
            if reward > 0:
                current_state = 1
            if reward < 0:
                current_state = 0
            env.reset()
            next_obs=env.gen_obs()
            reward=0
            terminated=0
            truncated=0
            state_ini_flag=True
        else:
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

        state_trace.append(current_state)
        mask_trace.append(mask)
        obs_trace.append(obs)
        action_trace.append(action)
        value_trace.append(value)
        reward_trace.append(torch.tensor(reward, device=device,dtype=torch.float))
        log_prob_trace.append(dist.log_prob(action))

#####################################################
        done = terminated|truncated
        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        pre_obs=obs
        obs=next_obs

        episode_reward_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.tensor(1, device=device)

        if done:
            done_counter += 1
            log_return.append(episode_reward_return.item())
            log_num_frames.append(log_episode_num_frames.item())

        episode_reward_return *= mask
        log_episode_num_frames *= mask



    keep1 = max(done_counter,1)
    log1 = {
        "return_per_episode": log_return[-keep1:],
        "num_frames_per_episode": log_num_frames[-keep1:],
        "num_frames": num_frames_per_proc
    }


    ############################################################
    # print("before abl", state_trace)
    #
    # # ABL修正state_trace
    # state_list = [1, 2, 3] # 课程路径
    # trace_list=[]
    # obs_trace_list=[]
    # end_list=[]
    # start_index=0
    # for t in range(len(state_trace)):
    #     if(mask_trace[t]==0):
    #         trace_list.append(state_trace[start_index:t+1])
    #         obs_trace_list.append(obs_trace[start_index:t+1])
    #         start_index=t+1
    #         if reward_trace[t - 1] > 0:
    #             end_list.append(3)
    #         else:
    #             end_list.append(4)
    # last_trace = state_trace[start_index:]
    # last_obs_trace_list = obs_trace[start_index:]
    # # for i in range(len(last_trace)):
    # #     last_trace[i] = 2
    # # def testStateNN(obs):
    # #     return [1, 1, 1, 1, 1, 1]
    # def get_prob(obs):
    #     return pre_obs_softmax(StateNN, obs)
    # for m in range(len(trace_list)):
    #     _, trace_list[m] = abl_trace(trace_list[m], state_list, end_list[m], obs_trace_list[m], get_prob)
    # if len(last_trace) > 0:
    #     last_state_list = []
    #     for i in state_list:
    #         if i <= last_trace[-1]:
    #             last_state_list.append(i)
    #
    #     _, last_trace = abl_trace(last_trace, last_state_list, last_trace[-1], last_obs_trace_list, get_prob)
    #
    # # last_trace = abl_trace(last_trace, [0, 1], 1, last_obs_trace_list, get_prob)
    # ### abl over
    # ### abl
    # after_abl_list = trace_list + [last_trace]
    # print("after abl:", after_abl_list)
    #
    # state_trace = []
    # for m in range(len(trace_list)):
    #     state_trace.extend(trace_list[m])
    # state_trace.extend(last_trace)
    #################################################################

    for i in range(len(state_trace) - 1):
        if state_trace[i] != state_trace[i + 1]:
            # mental reward
            if reward_trace[i] == 0 and state_trace[i] != 0 and state_trace[i] != 1:
                reward_trace[i] = torch.tensor(1, device=device, dtype=torch.float)

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

    for i in range(agent_num + 2):
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
            if id!=0 and id!=1:
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
        if mask_trace[i-1].item() == 0:
            image_trace[i] = image_trace[i] - image_trace[i]
        else:
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
    for i in range(len(state_trace)):
        if state_trace[i] == 3:
            img=image_trace[i].cpu().numpy().astype(numpy.uint8)
            plt.imsave('trace/{i}.jpg'.format(i=3), img)
        if state_trace[i] == 4:
            img = image_trace[i].cpu().numpy().astype(numpy.uint8)
            plt.imsave('trace/{i}.jpg'.format(i=4), img)
    return exps_list, {**log1, **log2}, statenn_exps



def Mutiagent_collect_experiences_q(env, 
                                algos,
                                contrast,
                                G: nx.DiGraph,
                                start_node,
                                anomalyNN,  
                                device,
                                num_frames_per_proc, 
                                preprocess_obss,
                                epsilon):

    # def pre_obs_softmax(model, obs):
    #     image_data=preprocess_obss([obs], device=device)
    #     input_tensor = image_data.image[0]
    #     input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    #     output = model(input_batch)
    #     prob = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
    #     # print("prob", prob)
    #     return prob

    agent_num=len(algos)
    obs=env.gen_obs()
    pre_obs=obs

    mask_trace=[]
    obs_trace=[]
    state_trace=[]
    action_trace=[]
    q_value_trace=[]
    reward_trace=[]
    next_obs_trace = []

    mask=torch.ones(1, device=device)
    done=0

    done_counter=0
    log_return=[0]
    log_num_frames=[0]
    episode_reward_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device)


    env_ini_state = start_node 
    current_state=env_ini_state
    state_ini_flag=False

    for _ in range(num_frames_per_proc):

        preprocessed_obs = preprocess_obss([obs], device=device)

        # t,prob_dist=obs_To_state(StateNN, pre_obs, obs)

        # if t==0:
        #     current_state=current_state
        # else:
        #     candidate_list=Candidate(current_state)
        #     t=sample_from_selected_dimensions(prob_dist,candidate_list)
        #     current_state = t
        current_state = obs_To_state(current_state,
                                     preprocess_obss,
                                     anomalyNN, 
                                     contrast, 
                                     G, 
                                     pre_obs, 
                                     obs, 
                                     device)


        if state_ini_flag:
            current_state=env_ini_state
            state_ini_flag=False
        if current_state != 0 and current_state != 1:
            agent = G.nodes[current_state]['state'].agent

            with torch.no_grad():
                q_values = agent.acmodel(preprocessed_obs)
            if random.random() < epsilon:
                action = torch.tensor([random.choice(range(env.action_space.n))], device=device)
            else:
                action = q_values.argmax(dim=1)

        if done:
            if reward > 0:
                current_state = 1
            if reward < 0:
                current_state = 0
            env.reset()
            next_obs=env.gen_obs()
            reward=0
            terminated=0
            truncated=0
            state_ini_flag=True
        else:
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

        state_trace.append(current_state)
        mask_trace.append(mask)
        obs_trace.append(obs)
        action_trace.append(action)
        q_value_trace.append(q_values)
        reward_trace.append(torch.tensor(reward, device=device,dtype=torch.float))
        next_obs_trace.append(next_obs)

#####################################################
        done = terminated|truncated
        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        pre_obs=obs
        obs=next_obs

        episode_reward_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.tensor(1, device=device)

        if done:
            done_counter += 1
            log_return.append(episode_reward_return.item())
            log_num_frames.append(log_episode_num_frames.item())

        episode_reward_return *= mask
        log_episode_num_frames *= mask



    keep1 = max(done_counter,1)
    log1 = {
        "return_per_episode": log_return[-keep1:],
        "num_frames_per_episode": log_num_frames[-keep1:],
        "num_frames": num_frames_per_proc
    }

    for i in range(len(state_trace) - 1):
        if state_trace[i] != state_trace[i + 1]:
            # mental reward
            if reward_trace[i] == 0 and state_trace[i] != 0 and state_trace[i] != 1:
                reward_trace[i] = torch.tensor(0.5, device=device, dtype=torch.float)

    # print("After abl", state_trace)
    exps_list=[]

    for i in range(agent_num):
        exps=DictList()
        exps.obs=[]
        exps.action=[]
        exps.reward=[]
        exps.mask = []
        exps.obs_ = []
        exps_list.append(exps)

    # print(state_trace)
    # print([int(i.item()) for i in mask_trace])
    start_index=0
    for i in range(len(state_trace)-1):
        if state_trace[i]!=state_trace[i+1]:
            id=state_trace[start_index]
            # mental reward
            if id!=0 and id!=1:
                exps_list[id].obs.extend(obs_trace[start_index:i+1])
                exps_list[id].action.extend(action_trace[start_index:i+1])
                exps_list[id].reward.extend(reward_trace[start_index:i+1])
                exps_list[id].mask.extend(mask_trace[start_index:i + 1])
                exps_list[id].obs_.extend(next_obs_trace[start_index:i + 1])
            start_index=i+1
    if start_index<len(state_trace)-2:
        id = state_trace[start_index]
        exps_list[id].obs.extend(obs_trace[start_index:])
        exps_list[id].action.extend(action_trace[start_index:])
        exps_list[id].reward.extend(reward_trace[start_index:])
        exps_list[id].mask.extend(mask_trace[start_index:])
        exps_list[id].obs_.extend(next_obs_trace[start_index:])

    for i in range(agent_num):
        exp_len=len(exps_list[i].obs)
        if exp_len:
            exps_list[i].obs = preprocess_obss(exps_list[i].obs, device=device)
            exps_list[i].action = torch.tensor(exps_list[i].action, device=device, dtype=torch.int)
            exps_list[i].reward = torch.tensor(exps_list[i].reward, device=device)
            exps_list[i].mask = torch.tensor(exps_list[i].mask, device=device)
            exps_list[i].obs_ = preprocess_obss(exps_list[i].obs_, device=device)
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
        if mask_trace[i-1].item() == 0:
            image_trace[i] = image_trace[i] - image_trace[i]
        else:
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
    for i in range(len(state_trace)):
        if state_trace[i] == 3:
            img=image_trace[i].cpu().numpy().astype(numpy.uint8)
            plt.imsave('trace/{i}.jpg'.format(i=3), img)
        if state_trace[i] == 4:
            img = image_trace[i].cpu().numpy().astype(numpy.uint8)
            plt.imsave('trace/{i}.jpg'.format(i=4), img)
    return exps_list, {**log1, **log2}, statenn_exps


def collect_experiences_mutation(algo, 
                                    start_env, 
                                    get_mutation_score, 
                                    mutation_buffer, 
                                    mutation_value, 
                                    contrast,
                                    known_mutation_buffer,
                                    arrived_state_buffer,
                                    preprocess_obss,
                                    env_name):
    """Collects rollouts and computes advantages.

    Runs several environments concurrently. The next actions are computed
    in a batch mode for all environments at the same time. The rollouts
    and advantages from all environments are concatenated together.

    Returns
    -------
    exps : DictList
        Contains actions, rewards, advantages etc as attributes.
        Each attribute, e.g. `exps.reward` has a shape
        (algo.num_frames_per_proc * num_envs, ...). k-th block
        of consecutive `algo.num_frames_per_proc` frames contains
        data obtained from the k-th environment. Be careful not to mix
        data from different environments!
    logs : dict
        Useful stats about the training process, including the average
        reward, policy loss, value loss, etc.
    """
    env = copy_env(start_env, env_name)
    parallel_env = ParallelEnv([env])
    done = (True,)
    last_done = (True,)
    for i in range(algo.num_frames_per_proc):
        # Do one agent-environment interaction
        last_done = done

        preprocessed_obs = preprocess_obss(algo.obs, device=algo.device)

        with torch.no_grad():
            if algo.acmodel.recurrent:
                dist, value, memory = algo.acmodel(preprocessed_obs, algo.memory * algo.mask.unsqueeze(1))
            else:
                dist, value = algo.acmodel(preprocessed_obs)

        x=dist.probs+dist.probs
        combined_dist = Categorical(probs=x)
        action = combined_dist.sample()
        obs, reward, terminated, truncated, _ = parallel_env.step(action.cpu().numpy())
        if reward[0] > 0:
            arrived_state_buffer.append(1)
        done = tuple(a | b for a, b in zip(terminated, truncated))
        if done[0]:
            env = copy_env(start_env, env_name)
            parallel_env = ParallelEnv([env])
            obs = parallel_env.gen_obs()

        # print(obs)
        # print(algo.obss)
        the_preprocessed_obs = preprocess_obss(obs, device=algo.device)
        # plt.imshow(numpy.squeeze(the_preprocessed_obs.image).cpu().numpy().astype(numpy.uint8))
        # plt.show()
        # print("reward", reward)
        # print("action", action)
        
        if last_done[0]:
            mutation = the_preprocessed_obs.image - the_preprocessed_obs.image
        else:
            mutation = the_preprocessed_obs.image - preprocessed_obs.image
        mutation = numpy.squeeze(mutation)
        mutation = mutation.cpu().numpy().astype(numpy.uint8)
        # print(mutation.shape)
        if get_mutation_score(mutation) > mutation_value and reward[0] == 0:
            for _, (idx, mutation_) in enumerate(known_mutation_buffer):
                if contrast(mutation, mutation_) > 0.99:
                    arrived_state_buffer.append(idx)
                    reward = 1
                    done = (True,)
                    break
            is_in_buffer = False
            for idx, (score_, mutation_, times_, env_) in enumerate(mutation_buffer):
                if contrast(mutation, mutation_) > 0.99:
                    mutation_buffer[idx] = (score_, mutation_, times_ + 1, copy.deepcopy(algo.env))
                    is_in_buffer = True
                    break
            if not is_in_buffer:
                # plt.imshow(mutation)
                # plt.show()
                #print(get_mutation_score(mutation).dtype)
                # heapq.heappush(mutation_buffer, (get_mutation_score(mutation), mutation, 1, copy.deepcopy(algo.env)))
                mutation_buffer.append((get_mutation_score(mutation), mutation, 1, copy.deepcopy(algo.env)))
        # Update experiences values

        algo.obss[i] = algo.obs
        algo.obs = obs
        if algo.acmodel.recurrent:
            algo.memories[i] = algo.memory
            algo.memory = memory
        algo.masks[i] = algo.mask
        algo.mask = 1 - torch.tensor(done, device=algo.device, dtype=torch.float)
        algo.actions[i] = action
        algo.values[i] = value
        if algo.reshape_reward is not None:
            algo.rewards[i] = torch.tensor([
                algo.reshape_reward(obs_, action_, reward_, done_)
                for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
            ], device=algo.device)
        else:
            algo.rewards[i] = torch.tensor(reward, device=algo.device)
        algo.log_probs[i] = dist.log_prob(action)

        # Update log values

        algo.log_episode_return += torch.tensor(reward, device=algo.device, dtype=torch.float)
        algo.log_episode_reshaped_return += algo.rewards[i]
        algo.log_episode_num_frames += torch.ones(algo.num_procs, device=algo.device)

        for i, done_ in enumerate(done):
            if done_:
                algo.log_done_counter += 1
                algo.log_return.append(algo.log_episode_return[i].item())
                algo.log_reshaped_return.append(algo.log_episode_reshaped_return[i].item())
                algo.log_num_frames.append(algo.log_episode_num_frames[i].item())

        algo.log_episode_return *= algo.mask
        algo.log_episode_reshaped_return *= algo.mask
        algo.log_episode_num_frames *= algo.mask

    # Add advantage and return to experiences

    preprocessed_obs = preprocess_obss(algo.obs, device=algo.device)
    with torch.no_grad():
        if algo.acmodel.recurrent:
            _, next_value, _ = algo.acmodel(preprocessed_obs, algo.memory * algo.mask.unsqueeze(1))
        else:
            _, next_value = algo.acmodel(preprocessed_obs)

    for i in reversed(range(algo.num_frames_per_proc)):
        next_mask = algo.masks[i+1] if i < algo.num_frames_per_proc - 1 else algo.mask
        next_value = algo.values[i+1] if i < algo.num_frames_per_proc - 1 else next_value
        next_advantage = algo.advantages[i+1] if i < algo.num_frames_per_proc - 1 else 0

        delta = algo.rewards[i] + algo.discount * next_value * next_mask - algo.values[i]
        algo.advantages[i] = delta + algo.discount * algo.gae_lambda * next_advantage * next_mask

    # Define experiences:
    #   the whole experience is the concatenation of the experience
    #   of each process.
    # In comments below:
    #   - T is algo.num_frames_per_proc,
    #   - P is algo.num_procs,
    #   - D is the dimensionality.
    exps = DictList()
    exps.obs = [algo.obss[i][j]
                for j in range(algo.num_procs)
                for i in range(algo.num_frames_per_proc)]
    if algo.acmodel.recurrent:
        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = algo.memories.transpose(0, 1).reshape(-1, *algo.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = algo.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
    # for all tensors below, T x P -> P x T -> P * T
    exps.action = algo.actions.transpose(0, 1).reshape(-1)
    exps.value = algo.values.transpose(0, 1).reshape(-1)
    exps.reward = algo.rewards.transpose(0, 1).reshape(-1)
    exps.advantage = algo.advantages.transpose(0, 1).reshape(-1)
    exps.returnn = exps.value + exps.advantage
    exps.log_prob = algo.log_probs.transpose(0, 1).reshape(-1)

    # Preprocess experiences

    exps.obs = algo.preprocess_obss(exps.obs, device=algo.device)

    # Log some values
    keep = max(algo.log_done_counter, algo.num_procs)
    logs = {
        "return_per_episode": algo.log_return[-keep:],
        "reshaped_return_per_episode": algo.log_reshaped_return[-keep:],
        "num_frames_per_episode": algo.log_num_frames[-keep:],
        "num_frames": algo.num_frames
    }

    algo.log_done_counter = 0
    algo.log_return = algo.log_return[-algo.num_procs:]
    algo.log_reshaped_return = algo.log_reshaped_return[-algo.num_procs:]
    algo.log_num_frames = algo.log_num_frames[-algo.num_procs:]

    return exps, logs








