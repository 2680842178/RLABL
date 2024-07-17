import argparse
import yaml
import time
import datetime
import torch_ac
import tensorboardX
import sys
import networkx as nx
from typing import Tuple

from utils import *
from utils import device
from model import ACModel,CNN

parser = argparse.ArgumentParser()
parser.add_argument("--task-config", required=True,
                    help="the task config to use, including the graph(knowledge)")
parser.add_argument("--discover", required=True,
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
parser.add_argument("--StateNN", default=None,
                    help="name of the StateNN")

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
    raise NotImplemented
    # return similiarity_list (length=count of states, means mutation to known states and a common mutation),
    # and most similiar mutation/state(0 means new mutation, -1 means common mutation)
    
def anomaly_detection(mutation) -> float:
    raise NotImplemented
    
def obs_To_mutation(pre_obs, obs):
    pre_image_data=preprocess_obss([pre_obs], device=device)
    image_data=preprocess_obss([obs], device=device)
    input_tensor = image_data.image[0]-pre_image_data.image[0]
    input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return input_batch

def train(env, status, num_frames: int = 0):
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    
    while num_frames < args.frames:
        update_start_time = time.time()
        env.reset()
        exps_list, logs1, statenn_exps = Mutiagent_collect_experiences()

def test(env, 
            start_node: int, 
            episodes: int = 1,
            max_steps_per_episode: int = 256) -> Tuple[int, Env]:
    mutation_buffer = []
    for (node, stateNode) in G.nodes(data=True):
        if stateNode.agent is not None:
            mutation_buffer.append({node, stateNode.mutation})
    
    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device) 
    
    current_state = start_node
    obss = env.reset()
    pre_obss = obss
    stop_env = env
    stop_obss = obss

    for _ in range(episodes):
        step_count = 0
        while step_count < max_steps_per_episode:
            mutation = obs_To_mutation([pre_obss], [obss]) 
            for node_num, node_mutation in mutation_buffer:
                if contrast(node_mutation, mutation):
                    current_state = node_num
                    stop_env = env
                    stop_obss = obss
                    break
            actions = G.nodes[current_state]['agent'].get_actions(obss)
            obss, rewards, terminateds, truncateds, _ = env.step(actions)
            if rewards > 0:
                print("successful test.")
            
            step_count += 1
            dones = tuple(a | b for a, b in zip(terminateds, truncateds))
            G.nodes[current_state]['agent'].analyze_feedbacks(rewards, dones)
            
            log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
            log_episode_num_frames += torch.ones(args.procs, device=device)
            
            for i, done in enumerate(dones):
                if done: 
                    log_done_counter += 1
                    test_logs["return_per_episode"].append(log_episode_return[i].item())
                    test_logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
    return_per_episode = utils.synthesize(test_logs["return_per_episode"])
    if return_per_episode["mean"] > 0:
        print("successful test!")
        return 0, None
    else:
        print("unsuccessful test, last state: ", current_state)
        return current_state, stop_env, stop_obss

def random_discover(env,
                    start_obss, 
                    start_node: int, 
                    initial_state: bool = True,
                    steps: int = 2e8):
    known_mutation_buffer = []
    self_mutation_buffer = []
    for (node, stateNode) in G.nodes(data=True):
        if stateNode.agent is not None:
            known_mutation_buffer.append({node, stateNode.mutation})
    pre_obss = start_obss   
    obss = start_obss
    for _ in steps:
        action = env.action_space.sample()  
        pre_obss = obss
        obss, rewards, terminateds, truncateds, _ = env.step(action)
        
        mutation = obs_To_mutation([pre_obss], [obss])
        ##### find the new state and its next state.
        if anomaly_detection(mutation) > 0.5:
            self_mutation_buffer.append({mutation, anomaly_detection(mutation)})
        for node_num, node_mutation in known_mutation_buffer:
            if contrast(node_mutation, mutation):
                G.add_node(len(G.nodes), stateNode(len(G.nodes), mutation))
                if initial_state:
                    if self_mutation_buffer is not None:
                        G.nodes[start_node]['mutation'] = self_mutation_buffer[0]
                    else:
                        G.add_edge(start_node, node_num)
                else:
                    for successor in G.successors(start_node):
                        if successor.node_num != 0:
                            G.remove_edge(start_node, successor)
                    G.add_edge(start_node, len(G.nodes) - 1)
                    G.add_edge(len(G.nodes) - 1, node_num)
                break 
        if rewards > 0:
            G.add_node(len(G.nodes), stateNode(len(G.nodes), mutation)) 
            if initial_state:
                G.add_edge(start_node, node_num)
            else:
                for successor in G.successors(start_node):
                    if successor.node_num != 0:
                        G.remove_edge(start_node, successor)
                G.add_edge(start_node, len(G.nodes) - 1)
                G.add_edge(len(G.nodes) - 1, 1)
                break
        if terminateds:
            G.add_node(len(G.nodes), stateNode(len(G.nodes), mutation))
            G.add_edge(len(G.nodes) - 1, 0)
        
    return None
    # return stop state, stop env
    
def main():
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

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
    for i in range(args.procs):
        env=utils.make_env(args.env, args.seed + 10000 * i)
        env.reset()
        envs.append(env)
    txt_logger.info("Environments loaded\n")
    
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
    
    agent_num = status['agent_num']
    acmodels=[]
    for i in range(agent_num):
        acmodel = ACModel(obs_space, envs[0].action_space, args.text)
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"][i])
        acmodel.to(device)
        acmodels.append(acmodel)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodels[0]))

    algos=[]
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
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")
        algos.append(algo)
    
    # get the graph structure
    task_config_path = os.path.join("config", args.task_config + ".yaml")
    with open(task_config_path, "r") as file:
        task_config = yaml.safe_load(file)
    for node_id in task_config['graph']['nodes']:
        G.add_node(node_id, stateNode(node_id, agent=acmodels[node_id]))
    for edge in task_config['graph']['edges']:
        G.add_edge(edge["from"], edge["to"])
    start_node = task_config['start_node'] 
    
    if args.discover:
        stop_state, stop_env, stop_obss = test(envs[0], G, start_node, 1, 128)
        random_discover(stop_env, stop_obss, stop_state)
        train()    
    else:
        train()
    
    
    # random discover, save the changes.
    # until get a familiar change(state)
    # change the graph