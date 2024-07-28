import argparse
import yaml
import time
import copy
import datetime
import torch_ac
import tensorboardX
from torchvision import transforms
import sys
import networkx as nx
from typing import Tuple
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

import utils
from utils import *
from utils import device
from model import ACModel, CNN, QNet

parser = argparse.ArgumentParser()
parser.add_argument("--task-config", required=True,
                    help="the task config to use, including the graph(knowledge)")
parser.add_argument("--discover", required=True, type=int,
                    help="if this task need to discover new state")
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--AnomalyNN", default=None,
                    help="name of the anomalyNN")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=32,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=256,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--buffer-size", type=int, default=10000,
                    help="buffer size for dqn")
parser.add_argument("--target-update", type=int, default=10,
                    help="frequency to update target net")


G = nx.DiGraph()
args = parser.parse_args()
test_logs = {"num_frames_per_episode": [], "return_per_episode": []}

class stateNode():
    def __init__(self, 
                 id, 
                 mutation = None, 
                 agent: ACModel = None):
        self.id = id
        self.mutation = mutation
        self.agent = agent

# need to get parameters to get the graph
    
def contrast(mutation1, mutation2) -> float: # that means the similarity between two mutations
    if mutation1 is None or mutation2 is None:
        return 0
    # print(mutation1.shape)
    return ssim(mutation1, mutation2, multichannel=True, channel_axis=2)
    # print(mutation1, mutation1.shape)
    # gray1 = cv2.cvtColor(mutation1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(mutation2, cv2.COLOR_BGR2GRAY)
    # orb1 = cv2.ORB_create()
    # orb2 = cv2.ORB_create()
    # kp1, des1 = orb1.detectAndCompute(gray1, None)
    # kp2, des2 = orb2.detectAndCompute(gray2, None)
    # similarity_matrix = cosine_similarity(des1, des2)
    # print(similarity_matrix.mean())
    # return similarity_matrix.mean()
    
    
    # return similiarity_list (length=count of states, means mutation to known states and a common mutation),
    # and most similiar mutation/state(0 means new mutation, -1 means common mutation)
    
def obs_To_mutation(pre_obs, obs, preprocess_obss):
    pre_image_data=preprocess_obss([pre_obs], device=device).image
    image_data=preprocess_obss([obs], device=device).image
    input_tensor = image_data - pre_image_data
    input_tensor = numpy.squeeze(input_tensor)
    # input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return input_tensor

def test(env, 
            start_node: int, 
            episodes: int = 1,
            max_steps_per_episode: int = 256,
            preprocess_obss = None):
    mutation_buffer = []
    for node, stateNode in G.nodes(data=True):
        print(stateNode['state'].agent)
        if stateNode['state'].mutation is not None:
            mutation_buffer.append((node, stateNode['state'].mutation))
    
    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device) 
    
    current_state = start_node
    print(start_node, "start_node")
    obss, _ = env.reset()
    obss = env.gen_obs()
    # print(obss)
    pre_obss = obss
    stop_env = env
    stop_obss = obss

    for _ in range(episodes):
        step_count = 0
        while step_count < max_steps_per_episode:
            mutation = obs_To_mutation(pre_obss, obss, preprocess_obss) 
            mutation = mutation.cpu().numpy().astype(numpy.uint8)
            for node_num, node_mutation in mutation_buffer:
                if contrast(node_mutation, mutation):
                    current_state = node_num
                    stop_env = env
                    stop_obss = obss
                    break
                
            preprocessed_obss = preprocess_obss([obss], device=device)
            with torch.no_grad():
                dist, _ = G.nodes[current_state]['state'].agent.acmodel(preprocessed_obss)
            #actions = dist.sample() #dist.probs.max(1, keepdim=True)[1]
            actions = dist.probs.max(1, keepdim=True)[1]
            obss, rewards, terminateds, truncateds, _ = env.step(actions)
            
            step_count += 1
            dones = terminateds | truncateds # tuple(a | b for a, b in zip(terminateds, truncateds))
            # G.nodes[current_state]['state'].agent.analyze_feedbacks(rewards, dones)
            
            log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
            log_episode_num_frames += torch.ones(args.procs, device=device)
            
            # print(dones)
            if dones: 
                log_done_counter += 1
                test_logs["return_per_episode"].append(log_episode_return[0].item())
                test_logs["num_frames_per_episode"].append(log_episode_num_frames[0].item())
                # print(test_logs["return_per_episode"])
            if rewards > 0:
                # print("successful test.")
                break
            # print(actions, rewards)
    if len(test_logs["return_per_episode"]) == 0:
        print("unsuccessful test, last state: ", current_state) 
        return current_state, stop_env, stop_obss
    return_per_episode = utils.synthesize(test_logs["return_per_episode"])
    if return_per_episode["mean"] > 0:
        print("successful test!")
        return 1, 1, None
    else:
        print("unsuccessful test, last state: ", current_state)
        return current_state, stop_env, stop_obss

def random_discover(env,
                    start_obss, 
                    start_node: int, 
                    initial_state: bool = True,
                    steps: int = 1e8,
                    anomalyNN = None,
                    preprocess_obss = None):
    known_mutation_buffer = []
    self_mutation_buffer = []
    new_node = len(G.nodes)
    G.add_node(new_node, state=stateNode(new_node, None, None))
    for node in G.nodes:
        if G.nodes[node]['state'].mutation is not None:
            known_mutation_buffer.append((node, G.nodes[node]['state'].mutation))
    print("known_mutation_buffer: ", known_mutation_buffer)
    pre_obss = start_obss   
    obss = start_obss
    for _ in range(int(steps)):
        action = env.action_space.sample()  
        pre_obss = obss
        obss, rewards, terminateds, truncateds, _ = env.step(action)
        # print("obs", obss)
        mutation = obs_To_mutation(pre_obss, obss, preprocess_obss)
        mutation = mutation.cpu().numpy().astype(numpy.uint8)
        anomaly_mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)
        # print(mutation.shape)
        # print(anomalyNN(anomaly_mutation))
        #plt.imshow(mutation)
        #plt.show()
        ##### find the new state and its next state.
        #print(mutation.shape)
        #print(anomalyNN(mutation))
        # 此处anomalyNN接受的shape：(1, 3, 300, 300)
        # imshow接受的shape: (300, 300, 3)
        # 所以需要将mutation(shape=300, 300, 3)转换为(1, 3, 300, 300)

        if terminateds:
            if (new_node, 0) not in G.edges:    
                G.add_edge(new_node, 0)
            env.reset()
            obss = env.gen_obs()
            pre_obss = obss
            continue
        if rewards > 0:
            print("arrive the state 1 without any mutation! The test may have some problems.")
            return None
            # G.add_node(len(G.nodes), state=stateNode(len(G.nodes), mutation)) 
            # if initial_state:
            #     G.add_edge(new_node, 1)
            # else:
            #     for successor in list(G.successors(start_node)):
            #         print(successor)
            #         if successor != 0:
            #             G.remove_edge(start_node, successor)
            #     G.add_edge(start_node, new_node)
            #     G.add_edge(new_node, 1)
            #     if len(self_mutation_buffer) == 0:
            #         print("can't find a mutation.")
            #         nx.draw(G, with_labels=True)    
            #         plt.show()
            #         return None 
            #     G.nodes[new_node]['state'].mutation = self_mutation_buffer[0][0]
            # nx.draw(G, with_labels=True)
            # plt.show()
            # return new_node
        if anomalyNN(anomaly_mutation)[0, 1] > anomalyNN(anomaly_mutation)[0, 0]:
            print("new mutation: ", anomalyNN(anomaly_mutation))
            plt.imshow(mutation)
            plt.show()
            #pre_image_data = preprocess_obss([pre_obss], device=device).image
            #pre_image_data = numpy.squeeze(pre_image_data).cpu().numpy().astype(numpy.uint8)
            #image_data = preprocess_obss([obss], device=device).image
            #image_data = numpy.squeeze(image_data).cpu().numpy().astype(numpy.uint8)
            #plt.imshow(image_data)
            #plt.imshow(pre_image_data)
            #plt.show()
            self_mutation_buffer.append((mutation, anomalyNN(anomaly_mutation)))
            mutation_env = copy.deepcopy(env)

            # 此处逻辑（对于初始状态）：发现新的突变之后，如果这个突变与过去已知的突变相似
            # 那么将新的节点指向这个已知的节点，直接返回。
            # 如果是新的突变（不与任何已知突变相似），那么这个突变属于起始节点（过去没有突变）
            if initial_state:
                for node_num, node_mutation in known_mutation_buffer:
                    if contrast(node_mutation, mutation) > 0.95:
                        # G.add_node(len(G.nodes), state=stateNode(len(G.nodes), mutation))
                        G.add_edge(start_node, node_num)
                        nx.draw(G, with_labels=True)
                        plt.show()
                        return new_node
                G.add_edge(new_node, start_node)
                G.nodes[start_node]['state'].mutation = self_mutation_buffer[0][0]
                return new_node
            # plt.imshow(mutation_env.gen_obs()['image'])
            # plt.show()
            for _ in range(int(steps)):
                action = env.action_space.sample()  
                pre_obss = obss
                obss, rewards, terminateds, truncateds, _ = env.step(action)
                # print("obs", obss)
                # plt.imshow(obss['image'])
                # plt.show()
                mutation = obs_To_mutation(pre_obss, obss, preprocess_obss)
                mutation = mutation.cpu().numpy().astype(numpy.uint8)
                anomaly_mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)
                #print(rewards)
                if rewards > 0:
                    print("arrive the state 1!")
                    G.add_edge(new_node, 1)
                    G.add_edge(start_node, new_node)
                    # if initial_state:
                    #     G.add_edge(start_node, node_num)
                    # else:
                    #     for successor in list(G.successors(start_node)):
                    #         print(successor)
                    #         if successor != 0:
                    #             G.remove_edge(start_node, successor)
                    #     G.add_edge(start_node, new_node)
                    #     G.add_edge(new_node, 1)
                    #     G.nodes[new_node]['state'].mutation = self_mutation_buffer[0][0]
                    nx.draw(G, with_labels=True)
                    plt.show()
                    return new_node 
                if terminateds:
                    #print("terminatied!")
                    env = copy.deepcopy(mutation_env)
                    obss = env.gen_obs()
                    pre_obss = obss
                    # plt.imshow(obss['image'])
                    # plt.show()
                    continue
                if anomalyNN(anomaly_mutation)[0, 1] > anomalyNN(anomaly_mutation)[0, 0]:
                    print("new mutation1: ", anomalyNN(anomaly_mutation))
                    plt.imshow(mutation)
                    plt.show()
                    
                    for node_num, node_mutation in known_mutation_buffer:
                        if contrast(node_mutation, mutation) > 0.95:
                            # G.add_node(len(G.nodes), state=stateNode(len(G.nodes), mutation))
                            # if initial_state:
                            #     G.add_edge(start_node, node_num)
                            # else:
                            for successor in list(G.successors(start_node)):
                                if successor != 0:
                                    G.remove_edge(start_node, successor)
                            G.add_edge(start_node, new_node)
                            G.add_edge(new_node, node_num)
                            G.nodes[new_node]['state'].mutation = self_mutation_buffer[0][0]
                            nx.draw(G, with_labels=True)
                            plt.show()
                            return new_node
                    print("There are two new mutations, can't find the new state!")
                    self_mutation_buffer.append((mutation, anomalyNN(anomaly_mutation)))
                
            
        # for node_num, node_mutation in known_mutation_buffer:
        #     if contrast(node_mutation, mutation) > 0.95:
        #         G.add_node(len(G.nodes), state=stateNode(len(G.nodes), mutation))
        #         if initial_state:
        #             G.add_edge(start_node, node_num)
        #         else:
        #             for successor in list(G.successors(start_node)):
        #                 if successor != 0:
        #                     G.remove_edge(start_node, successor)
        #             G.add_edge(start_node, len(G.nodes) - 1)
        #             G.add_edge(len(G.nodes) - 1, node_num)
        #             G.nodes[len(G.nodes) - 1]['state'].mutation = self_mutation_buffer[0][0]
        #         nx.draw(G, with_labels=True)
        #         plt.show()
        #         return len(G.nodes) - 1 
        
    return None 
    # return the new state number.
    # return stop state, stop env
    
def main():
    # task_path: the path of the last task, the last folder name is "task"+task_number.
    # task_config_path: the path of the last task config(a yaml file).
    # new_task_path: the path of the new task, the last folder name is "task"+new_task_number.
    task_path = os.path.join("config", args.model, args.task_config)
    task_config_path = os.path.join("config", args.model, args.task_config, "config.yaml")
    task_number = int(args.task_config[-1])
    new_task_number = task_number + 1
    new_task_name = args.task_config[:-1] + str(new_task_number)
    new_task_path = os.path.join("config", args.model, new_task_name)
    with open(task_config_path, "r") as file:
        task_config = yaml.safe_load(file)
    # get the graph structure
    for node_id in task_config['graph']['nodes']:
        G.add_node(node_id, state=stateNode(node_id, agent=None, mutation=None))
    for edge in task_config['graph']['edges']:
        #nx.draw(G, with_labels=True)
        G.add_edge(edge["from"], edge["to"])
    start_node = task_config['graph']['start_node'] 
    # nx.draw(G, with_labels=True)
    # plt.show()
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    AnomalyNN_model_name = args.AnomalyNN or model_name
    AnomalyNN_model_dir = utils.get_StateNN_model_dir(AnomalyNN_model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")
    
    # Load environments

    envs = []
    initial_img = None
    for i in range(args.procs):
        env=utils.make_env(args.env, args.seed + 10000 * i)
        initial_img, _ = env.reset()
        envs.append(env)
    txt_logger.info("Environments loaded\n")
    last_initial_img_dir = task_path + "/initial_image.jpg"
    initial_img = initial_img['image']
    # plt.imshow(initial_img)
    #plt.show()
    if not os.path.exists(new_task_path):
        os.makedirs(new_task_path)
    if args.discover == 0:
        plt.imsave(task_path + "/initial_image.jpg", initial_img)
    else:
        plt.imsave(new_task_path + "/initial_image.jpg", initial_img)
    print("The same contrast:", contrast(initial_img, initial_img))
    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")
    
    agent_num = task_config['agent_num']
    acmodels=[]
    for i in range(agent_num):
        if args.algo == "a2c" or args.algo == "ppo":
            acmodel = ACModel(obs_space, envs[0].action_space, args.text)
        elif args.algo == "dqn":
            acmodel = QNet(obs_space, envs[0].action_space, args.text)
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"][i])
        acmodel.to(device)
        acmodels.append(acmodel)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodels[0]))

    algos=[]
    algos.append(None)
    algos.append(None)
    for i in range(agent_num):
        # Load algo
        if args.algo == "a2c":
            algo = torch_ac.A2CAlgo(envs, acmodels[i], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_alpha, args.optim_eps, preprocess_obss)
        elif args.algo == "ppo":
            algo = torch_ac.PPOAlgo(envs, acmodels[i], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        elif args.algo == "dqn":
            algo = torch_ac.DQNAlgo(envs, acmodels[i], device, args.frames_per_proc, args.discount, args.lr,
                                    args.max_grad_norm,
                                    args.optim_eps, args.epochs, args.buffer_size, args.batch_size, args.target_update)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")
        algos.append(algo)
        print("G.nodes[i + 2]", G.nodes[i + 2])
        G.nodes[i + 2]['state'].agent = algo

    AnomalyNN = CNN(num_classes=2)
    try: 
        # AnomalyNN.load_state_dict(torch.load(AnomalyNN_model_dir))
        AnomalyNN = torch.load(AnomalyNN_model_dir)
        AnomalyNN.to(device)
        # print(AnomalyNN(preprocess_obss(initial_img)))
    except OSError:
        AnomalyNN = lambda x: [[1.0, 0]]
        
    # load the mutations 
    for node in G.nodes:
        if list(G.predecessors(node)):
            if node != 0 and node != 1:
                G.nodes[node]['state'].mutation = plt.imread(task_path + "/mutation" + str(node) + ".bmp")
    
    print(G.nodes)
    print(G.edges)
    if args.discover != 0:
        stop_state, stop_env, stop_obss = test(envs[0], start_node, 1, 128, preprocess_obss)
        if stop_state == 1:
            print("successful test!")
            return
        last_initial_img = plt.imread(last_initial_img_dir)
        print(contrast(last_initial_img, initial_img))

        if contrast(last_initial_img, initial_img) > 0.95:
            print("contrast! New node will after the initial node!")
            new_node = random_discover(env = stop_env, start_obss = stop_obss, \
                start_node=stop_state, initial_state=False, steps=1e7, anomalyNN = AnomalyNN, preprocess_obss=preprocess_obss)
            if new_node != None:
                agent_num += 1
            print("New node: ", new_node)
        else:
            print("Not contrast! New node will be added before the initial node!")
            nx.draw(G, with_labels=True)
            plt.show()
            print("nodes", G.nodes)
            print("edges", G.edges)
            new_node = random_discover(env = stop_env, start_obss = stop_obss, \
                start_node=stop_state, initial_state=True, steps=1e7, anomalyNN = AnomalyNN, preprocess_obss=preprocess_obss)
            start_node = new_node
            if new_node != None:
                agent_num += 1
            print("New node: ", new_node)   
        # save the graph to yaml
        
                
        new_acmodel = ACModel(obs_space, envs[0].action_space, args.text)
        new_acmodel.to(device)
        acmodels.append(new_acmodel)
        if args.algo == "a2c":
            algo = torch_ac.A2CAlgo(envs, acmodels[new_node - 2], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_alpha, args.optim_eps, preprocess_obss)
        elif args.algo == "ppo":
            algo = torch_ac.PPOAlgo(envs, acmodels[new_node - 2], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        elif args.algo == "dqn":
            algo = torch_ac.DQNAlgo(envs, acmodels[new_node - 2], device, args.frames_per_proc, args.discount, args.lr,
                                    args.max_grad_norm,
                                    args.optim_eps, args.epochs, args.buffer_size, args.batch_size, args.target_update)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))
        G.nodes[new_node]['state'].agent = algo
        algos.append(algo)
        # save the new graph
        new_yaml = task_config
        new_yaml['graph']['nodes'].append(new_node)
        new_yaml['graph']['start_node'] = start_node
        for new_node_successor in G.successors(new_node):
            new_yaml['graph']['edges'].append({"from": new_node, "to": new_node_successor})
        for new_node_successor in G.predecessors(new_node):
            new_yaml['graph']['edges'].append({"from": new_node_successor, "to": new_node})
        # initial_img_name = 'after_' + task_config['initial_image']
        # initial_img_dir = './initial_images/' + initial_img_name
        # initial_img_jpg = initial_img.cpu().numpy().astype(numpy.uint8)
        with open(new_task_path + '/config.yaml', 'w') as file:
            yaml.dump(data = new_yaml, stream = file, allow_unicode = True)
        #yaml.dump(new_yaml, open(new_yaml_name, 'w'))
        ### save the mutation
        for node in G.nodes:
            if G.nodes[node]['state'].mutation is not None:
                plt.imsave(new_task_path + "/mutation" + str(node) + ".bmp", G.nodes[node]['state'].mutation)
    nx.draw(G, with_labels=True)
    plt.show()
    
    # train the model.
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        envs[0].reset()
        # ini_agent
        epsilon = 0.3 * (1 - num_frames / args.frames)
        if args.algo == "a2c" or args.algo == "ppo":
            exps_list, logs1, statenn_exps = Mutiagent_collect_experiences(env=envs[0], 
                                                                           algos=algos, 
                                                                           contrast=contrast, 
                                                                           G=G,
                                                                           device=device,
                                                                           start_node=start_node,
                                                                           anomalyNN=AnomalyNN,
                                                                       num_frames_per_proc=args.frames_per_proc, 
                                                                       discount=args.discount,
                                                                       gae_lambda=args.gae_lambda, 
                                                                       preprocess_obss=preprocess_obss)
        elif args.algo == "dqn":
            exps_list, logs1, statenn_exps = Mutiagent_collect_experiences_q(env=envs[0], 
                                                                           algos=algos,
                                                                           contrast=contrast, 
                                                                           G=G,
                                                                           device=device,
                                                                           start_node=start_node,
                                                                           anomalyNN=AnomalyNN,
                                                                       num_frames_per_proc=args.frames_per_proc,
                                                                       preprocess_obss=preprocess_obss,
                                                                       epsilon=epsilon)
        # #每个algo更新
        logs2_list = [None] * (agent_num + 2)
        for i in range(2, agent_num + 2):
            if len(exps_list[i].obs) and i != 0 and i != 1:
                logs2 = algos[i].update_parameters(exps_list[i])
                logs2_list[i] = logs2
        logs2 = {}
        if args.algo == "a2c" or args.algo == "ppo":
            entropy_list = [None] * (agent_num + 2)
            value_list = [None] * (agent_num + 2)
            policy_loss_list = [None] * (agent_num + 2)
            value_loss_list = [None] *(agent_num + 2)
            grad_norm_list = [None] * (agent_num + 2)
            for i in range(2, agent_num + 2):
                if len(exps_list[i].obs):
                    entropy_list[i] = logs2_list[i]["entropy"]
                    value_list[i] = logs2_list[i]["value"]
                    policy_loss_list[i] = logs2_list[i]["policy_loss"]
                    value_loss_list[i] = logs2_list[i]["value_loss"]
                    grad_norm_list[i] = logs2_list[i]["grad_norm"]
            logs2 = {
                "entropy": entropy_list,
                "value": value_list,
                "policy_loss": policy_loss_list,
                "value_loss": value_loss_list,
                "grad_norm": grad_norm_list
            }
        elif args.algo == "dqn":
            loss_list = [None] * (agent_num + 2)
            q_value_list = [None] * (agent_num + 2)
            grad_norm_list = [None] * (agent_num + 2)
            for i in range(2, agent_num + 2):
                if len(exps_list[i].obs):
                    loss_list[i] = logs2_list[i]["loss"]
                    q_value_list[i] = logs2_list[i]["q_value"]
                    grad_norm_list[i] = logs2_list[i]["grad_norm"]
            logs2 = {
                "loss": loss_list,
                "grad_norm": grad_norm_list,
                "q_value": q_value_list
            }
        logs = {**logs1, **logs2}

        # logs_list=[0]*agent_num
        # for i in range(agent_num):
        #     if len(exps_list[i].obs):
        #         logs={**logs1, **logs2_list[i]}
        #         logs_list[i]=logs
        #
        # for i in range(agent_num):
        #     if len(exps_list[i].obs):
        #         logs=logs_list[i]
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)

            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            # data += num_frames_per_episode.values()
            if args.algo == "a2c" or args.algo == "ppo":
                header += ["policy_loss", "value_loss"]
                data += [['{:.3f}'.format(item) if item is not None else 'None' for item in logs["policy_loss"]],
                         ['{:.3f}'.format(item) if item is not None else 'None' for item in logs["value_loss"]]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | policy_loss {} "
                    "| value_loss {}".format(*data))
            elif args.algo == "dqn":
                header += ["loss", "q_value"]
                data += [['{:.3f}'.format(item) if item is not None else 'None' for item in logs["loss"]],
                         ['{:.3f}'.format(item) if item is not None else 'None' for item in logs["q_value"]]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | loss {} "
                    "| q_value {}".format(*data))
            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            # for field, value in zip(header, data):
            #     tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update, "agent_num": agent_num,
                      "model_state": [acmodels[i].state_dict() for i in range(agent_num)],
                      "optimizer_state": algo.optimizer.state_dict()}
            # if hasattr(preprocess_obss, "vocab"):
            #     status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

    
    # random discover, save the changes.
    # until get a familiar change(state)
    # change the graph

if __name__ == "__main__":
    main()