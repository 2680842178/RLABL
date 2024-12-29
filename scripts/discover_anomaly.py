import argparse
import pdb
from typing import Optional
import yaml
import time
import copy
import datetime
import torch_ac
import tensorboardX
from torchvision import transforms
import sys
import networkx as nx
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

import utils
from utils import *
from utils import device
from model import ACModel, CNN, QNet

from graph_test import test, ddm_decision
from utils.anomaly import BoundaryDetector
import math

parser = argparse.ArgumentParser()
parser.add_argument("--task-config", required=True,
                    help="the task config to use, including the graph(knowledge)")
parser.add_argument("--discover", required=True, type=int,
                    help="if this task need to discover new state")
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--curriculum", default=1, type=int,
                    help="Curriculum number(1, 2, 3), used in random env")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--test-interval", type=int, default=10,
                    help="number of updates between two tests (default: 1)")
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
parser.add_argument("--target-update", type=int, default=5,
                    help="frequency to update target net")


G = nx.DiGraph()
args = parser.parse_args()
test_logs = {"num_frames_per_episode": [], "return_per_episode": []}

class stateNode():
    def __init__(self,
                 id,
                 mutation = None,
                 agent: ACModel = None,
                 env_image = None):
        self.id = id
        self.mutation = mutation
        self.agent = agent
        self.env_image = env_image

# need to get parameters to get the graph

def get_discover_probability(mean_reward, test_turns):
    prob = (1 - mean_reward['mean']) / 2.0 + test_turns / 500.0
    return prob

# def get_mutation_score(mutation, anomalyNN):
#     mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)
#     anomaly_score = anomalyNN(mutation)
#     e_x = numpy.exp(anomaly_score - numpy.max(anomaly_score))
#     return (e_x / e_x.sum())[1]

def define_accept_mutation(mutation_score, mutation_times, test_turns, test_mean_reward):
    score = 0.65 * mutation_score + 0.01 * mutation_times + 0.015 * test_turns - (1 + test_mean_reward) / 2
    print("Define accept mutation score: ", score)
    print(mutation_score, mutation_times, test_turns, test_mean_reward)
    if score > 0.3:
        return True
    return False

def contrast(mutation1, mutation2) -> float: # that means the similarity between two mutations
    if mutation1 is None or mutation2 is None:
        return 0
    # return ssim(mutation1, mutation2, multichannel=True, channel_axis=2)
    if mutation1.shape != mutation2.shape:
        target_size = (min(mutation1.shape[0], mutation2.shape[0]),
                       min(mutation1.shape[1], mutation2.shape[1]))
        mutation1 = cv2.resize(mutation1, target_size, interpolation=cv2.INTER_AREA)
        mutation2 = cv2.resize(mutation2, target_size, interpolation=cv2.INTER_AREA)
    hist_1, hist_2 = cv2.calcHist([mutation1], [0], None, [256], [0, 256]), cv2.calcHist([mutation2], [0], None, [256], [0, 256])
    hist_1, hist_2 = cv2.normalize(hist_1, hist_1).flatten(), cv2.normalize(hist_2, hist_2).flatten()
    correlation = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
    similar = abs(correlation)
    return similar

def contrast_ssim(image1, image2) -> float:
    if image1 is None or image2 is None:
        return 0
    return ssim(image1, image2, multichannel=True, channel_axis=2)


def obs_To_mutation(pre_obs, obs, preprocess_obss):
    pre_image_data=preprocess_obss([pre_obs], device=device).image
    image_data=preprocess_obss([obs], device=device).image
    input_tensor = image_data - pre_image_data
    input_tensor = numpy.squeeze(input_tensor)
    # input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return input_tensor

def get_importance_prob(lst):
    total = sum(x for x in lst if x != 0)
    normalized_lst = [x / total if x != 0 else 0 for x in lst]
    return normalized_lst

def calculate_epsilon(num_frames, initial_num_frames, total_frames):
    progress = (num_frames - initial_num_frames) / (total_frames - initial_num_frames)

    # 当进度达到90%时保持最小值
    if progress >= 0.9:
        progress = 0.9

    # 使用指数函数实现快速下降后缓慢下降
    epsilon = 1 - progress ** 0.5

    return epsilon


def discover(start_env,
            start_node,
            algo,
            discover_frames,
            txt_logger,
            mutation_value,
            test_turns,
            test_mean_reward,
            preprocess_obss,
            anomaly_detector,
            discover_csv_logger=None):
    def get_mutation_score(processed_mutation_roi):
        # mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)
        # anomaly_score = anomalyNN(mutation).detach().cpu().numpy()
        # is_anomaly = anomaly_detector.detect_anomaly(mutation)
        if numpy.any(numpy.asarray(processed_mutation_roi.shape) < 5):
            return 0
        is_anomaly = anomaly_detector.is_known_roi(processed_mutation_roi, add_to_buffer=False)
        if is_anomaly:
            return 1
        return 0
        # if is_anomaly:
        #     return 1
        # else:
        #     return 0

    start_env = copy_env(start_env, args.env, args.curriculum)
    # start_env = ParallelEnv([start_env])
    mutation_buffer = []
    heapq.heapify(mutation_buffer)

    txt_logger.info("Optimizer loaded\n")

    num_frames = 0
    update = 0
    start_time = time.time()

    known_mutation_buffer = []
    arrived_state_buffer = []
    for node, data in G.nodes(data=True):
        if data['state'].mutation is not None:
            known_mutation_buffer.append((node, data['state'].mutation))

    txt_logger.info("Start discovering in {} steps.\n".format(discover_frames))
    initial_num_frames = num_frames
    while num_frames < discover_frames:
        epsilon = calculate_epsilon(num_frames, initial_num_frames, args.frames)
        update_start_time = time.time()
        if args.algo == 'ppo' or args.algo == 'a2c':
            exps, logs1 = collect_experiences_mutation(algo,
                                                    start_env,
                                                            get_mutation_score,
                                                            mutation_buffer,
                                                            mutation_value,
                                                            contrast,
                                                            known_mutation_buffer,
                                                            arrived_state_buffer,
                                                            preprocess_obss,
                                                            args.env)
        elif args.algo == 'dqn':
            exps, logs1 = collect_experiences_mutation_q(algo,
                                                    start_env,
                                                            get_mutation_score,
                                                            mutation_buffer,
                                                            mutation_value,
                                                            contrast,
                                                            known_mutation_buffer,
                                                            arrived_state_buffer,
                                                            preprocess_obss,
                                                            args.env,
                                                            epsilon)
        logs2 = algo.update_parameters(exps)

        logs = {**logs1, **logs2}
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
            if args.algo == "a2c" or args.algo == "ppo":
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | entropy {:.3f} | value {:.3f} | policy_loss {:.3f} | value_loss {:.3f} | grad_norm {:.3f}"
                    .format(*data))

            elif args.algo == "dqn":

                header += ["loss", "q_value", "grad_norm"]
                data += [logs["loss"], logs["q_value"], logs["grad_norm"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | loss {:.3f} | q_value {:.3f} | grad_norm {:.3f}"
                    .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if update == 1:
                discover_csv_logger.writerow(header)

            discover_csv_logger.writerow(data)

            if return_per_episode["max"] > 0:
                print("return per episode: {} > 0".format(return_per_episode['max']))
                break

    ####
    # for node, mutation in mutation_buffer:
    #     print("mutation score for node {}: {}".format(node, mutation[0]))
    #     plt.imshow(mutation[1])
    #     plt.show()

    if arrived_state_buffer == []:
        return None, None, None, num_frames
    counter = collections.Counter(arrived_state_buffer)
    most_state, count = counter.most_common(1)[0]
    print("Most state & Count:", most_state, count)
    out_state = most_state

    for idx, (score_, mutation_, times_, env_) in enumerate(mutation_buffer):
        if define_accept_mutation(score_, times_, test_turns, test_mean_reward):
            print(score_, times_, test_turns, test_mean_reward)
            print("Accept mutation with score: ", score_)
            env_img = preprocess_obss(env_.gen_obs(), device=device).image
            env_img = numpy.squeeze(env_img)
            env_img = env_img.cpu().numpy().astype(numpy.uint8)
            return mutation_, env_img, out_state, num_frames
        else:
            print("Reject mutation with score: ", score_)

    print("No mutation detected or accepted.")
    return None, None, None, num_frames
def main():
    print("args", args)
    # task_path: the path of the last task, the last folder name is "task"+task_number.
    # task_config_path: the path of the last task config(a yaml file).
    # new_task_path: the path of the new task, the last folder name is "task"+new_task_number.
    normal_buffer_path = "config/" + args.model + "/buffer/"

    state_img_path = "config/" + args.model + "/"
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

    # AnomalyNN_model_name = args.AnomalyNN or model_name
    # AnomalyNN_model_dir = utils.get_StateNN_model_dir(AnomalyNN_model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    csv_episode_file, csv_episode_logger = utils.get_csv_episode_logger(model_dir)
    # tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)
    print("Seed:", args.seed)
    print("Seed:", args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    initial_img = None
    for i in range(args.procs):
        # kwargs = {"curriculum": args.curriculum}
        env=utils.make_env(args.env, args.seed + 10000 * i, curriculum=args.curriculum)
        initial_img, _ = env.reset()
        envs.append(env)
    txt_logger.info("Environments loaded\n")
    if not os.path.exists(new_task_path):
        os.makedirs(new_task_path)

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

    initial_img = preprocess_obss([initial_img], device=device).image
    initial_img = numpy.squeeze(initial_img)
    initial_img = initial_img.cpu().numpy().astype(numpy.uint8)
    if args.discover == 0:
        plt.imsave(state_img_path + "state{}.bmp".format(start_node), initial_img)

    initial_agent_num = task_config['agent_num']
    if "model_state" not in status:
        initial_agent_num = 0
    agent_num = task_config['agent_num']
    acmodels=[]
    for i in range(agent_num):
        if args.algo == "a2c" or args.algo == "ppo":
            acmodel = ACModel(obs_space, envs[0].action_space, args.text)
        elif args.algo == "dqn":
            acmodel = QNet(obs_space, envs[0].action_space, args.text)
        if "model_state" in status and status["model_state"][i] is not None:
            acmodel.load_state_dict(status["model_state"][i])
        else:
            print(f"This model {i} is None. No load.")
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
                                    args.optim_eps, args.epochs, args.buffer_size, args.batch_size, args.target_update, preprocess_obss)

        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))

        if "optimizer_state" in status and status["optimizer_state"] is not None:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")
        algos.append(algo)
        print("G.nodes[i + 2]", G.nodes[i + 2])
        G.nodes[i + 2]['state'].agent = algo

    if initial_agent_num > 0:
        for i in range(2, initial_agent_num + 2):
            if i < len(algos):
                print(f"Setting trained flag for algo {i}")
                if args.algo == "dqn" and hasattr(algos[i], 'trained'):
                    algos[i].trained = True
    # AnomalyNN = CNN(num_classes=2)
    # try:
    #     # AnomalyNN.load_state_dict(torch.load(AnomalyNN_model_dir))
    #     AnomalyNN = torch.load(AnomalyNN_model_dir)
    #     AnomalyNN.to(device)
    #     # print(AnomalyNN(preprocess_obss(initial_img)))
    # except OSError:
    #     AnomalyNN = lambda x: [[1.0, 0]]
    anomaly_detector = BoundaryDetector(normal_buffer_path)

    # load the mutations
    for node in G.nodes:
        if list(G.predecessors(node)):
            if node != 0 and node != 1:
                # G.nodes[node]['state'].mutation = plt.imread(task_path + "/mutation" + str(node) + ".bmp")
                G.nodes[node]['state'].mutation = cv2.imread(task_path + "/mutation" + str(node) + ".bmp", cv2.IMREAD_GRAYSCALE)
        if node != 0 and node != 1:
            G.nodes[node]['state'].env_image = plt.imread(state_img_path + "/state" + str(node) + ".bmp")

    print(G.nodes)
    print(G.edges)

    if args.discover != 0:
        node_probability_list = [0] * G.number_of_nodes()
        for node, data in G.nodes(data=True):
            if data['state'].env_image is not None:
                node_probability_list[node] = contrast_ssim(data['state'].env_image, initial_img)
        node_probability_list = get_importance_prob(node_probability_list)
        min_stop_state = 0
        ######
        # use ddm to decide whether need to discover
        need_discover, decision_steps, stop_state, stop_env, stop_obss, mean_return = ddm_decision(
            G=G,
            start_env=envs[0],
            env_key=args.env,
            max_decision_steps=100,
            node_probability_list=node_probability_list,
            preprocess_obss=preprocess_obss,
            anomaly_detector=anomaly_detector,
            drift_rate=0.1,
            boundary_separation=1.0,
            starting_point=-0.3,
            args=args
        )
        if not need_discover:
            txt_logger.info("successful test! reward per episode: {}".format(mean_return))
            return "can_solve"
        else:
            txt_logger.info("failed test! need to discover!")


        if args.algo == "a2c" or args.algo == "ppo":
            new_acmodel= ACModel(obs_space, envs[0].action_space, args.text)
        elif args.algo == "dqn":
            new_acmodel = QNet(obs_space, envs[0].action_space, args.text)
        new_acmodel.load_state_dict(status["model_state"][stop_state - 2])
        new_acmodel.to(device)

        if args.algo == "ppo":
            algo = torch_ac.PPOAlgo(envs, new_acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        elif args.algo == "a2c":
            algo = torch_ac.A2CAlgo(envs, new_acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_alpha, args.optim_eps, preprocess_obss)
        elif args.algo == "dqn":
            algo = torch_ac.DQNAlgo(envs, new_acmodel, device, args.frames_per_proc, args.discount, args.lr,
                                    args.max_grad_norm,
                                    args.optim_eps, args.epochs, args.buffer_size, args.batch_size, args.target_update, preprocess_obss)

        if stop_state > min_stop_state:
            min_stop_state = stop_state

        discover_csv_file, discover_csv_logger = utils.get_csv_discover_logger(model_dir=model_dir, agent_idx=agent_num)

        new_mutation, new_state_img, out_state, discover_num_frames = discover(start_env=stop_env,
                                start_node=min_stop_state,
                                algo=algo,
                                discover_frames=500000,
                                txt_logger=txt_logger,
                                mutation_value=0.5,
                                test_turns=decision_steps,
                                test_mean_reward=mean_return,
                                preprocess_obss=preprocess_obss,
                                anomaly_detector=anomaly_detector,
                                discover_csv_logger=discover_csv_logger)
        discover_csv_file.flush()

        if new_mutation is not None:
            new_node_id = len(G.nodes)
            G.add_node(new_node_id, state=stateNode(new_node_id, None, None, stop_env.gen_obs()['image']))
            if min_stop_state == start_node:
                G.add_edge(new_node_id, start_node)
                G.nodes[start_node]['state'].mutation = new_mutation
                plt.imsave(state_img_path + "state{}.bmp".format(new_node_id), initial_img)
                start_node = new_node_id
                print("New start node: ", start_node)
            else:
                plt.imsave(state_img_path + "state{}.bmp".format(new_node_id), new_state_img)
                G.nodes[new_node_id]['state'].mutation = new_mutation
                G.add_edge(stop_state, new_node_id)
                G.add_edge(new_node_id, out_state)
                # for successor in list(G.successors(stop_state)):
                #     if successor != 0:
                #         G.remove_edge(stop_state, successor)
                # G.add_edge(stop_state, new_node_id)
                # G.add_edge(new_node_id, )
                raise NotImplementedError

            acmodels.append(new_acmodel)
            G.nodes[new_node_id]['state'].agent = algo
            algos.append(algo)
            agent_num += 1
            # save the new graph
            new_yaml = task_config
            new_yaml['graph']['nodes'].append(new_node_id)
            new_yaml['graph']['start_node'] = start_node
            new_yaml['agent_num'] = agent_num
            for successor in G.successors(new_node_id):
                new_yaml['graph']['edges'].append({"from": new_node_id, "to": successor})
            for predecessor in G.predecessors(new_node_id):
                new_yaml['graph']['edges'].append({"from": predecessor, "to": new_node_id})
            with open(new_task_path + '/config.yaml', 'w') as file:
                yaml.dump(data = new_yaml, stream = file, allow_unicode = True)
            ### save the mutation
            for node in G.nodes:
                if G.nodes[node]['state'].mutation is not None:
                    # plt.imsave(new_task_path + "/mutation" + str(node) + ".bmp", G.nodes[node]['state'].mutation)
                    cv2.imwrite(new_task_path + "/mutation" + str(node) + ".bmp", G.nodes[node]['state'].mutation)

            nx.draw(G, with_labels=True)
            plt.show()
        else:
            print("Failed to discover, return.")
            return "fail to discover anomaly"
        ######
    # train the model.
    num_frames = status["num_frames"]
    if args.discover != 0:
        num_frames += discover_num_frames
    update = status["update"]
    start_time = time.time()

    # the_max_return = agent_num.copy()
    start_num_frames = copy.deepcopy(num_frames)
    print("G.nodes", G.nodes)
    print("G.edges", G.edges)
    # pdb.set_trace()
    no_csv_head = True
    # 添加变量跟踪最佳测试结果
    best_test_return = float('-inf')
    best_model_states = None
    initial_num_frames = num_frames
    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        envs[0].reset()
        # ini_agent
        epsilon =  calculate_epsilon(num_frames, initial_num_frames, args.frames)
        # print("num_frames", num_frames, "initial_num_frames", initial_num_frames, "args.frames", args.frames)
        # print("epsilon", epsilon)
        if args.algo == "a2c" or args.algo == "ppo":
            exps_list, logs1, statenn_exps = Mutiagent_collect_experiences(env=envs[0],
                                                                           algos=algos,
                                                                           contrast=contrast,
                                                                           G=G,
                                                                           device=device,
                                                                           start_node=start_node,
                                                                           anomaly_detector=anomaly_detector,
                                                                       num_frames_per_proc=args.frames_per_proc * agent_num,
                                                                       discount=args.discount,
                                                                       gae_lambda=args.gae_lambda,
                                                                       preprocess_obss=preprocess_obss,
                                                                       discover=args.discover,)
        elif args.algo == "dqn":
            exps_list, logs1, statenn_exps = Mutiagent_collect_experiences_q(env=envs[0],
                                                                           algos=algos,
                                                                           contrast=contrast,
                                                                           G=G,
                                                                           device=device,
                                                                           start_node=start_node,
                                                                           anomaly_detector=anomaly_detector,
                                                                       num_frames_per_proc=args.frames_per_proc * agent_num,
                                                                       preprocess_obss=preprocess_obss,
                                                                       epsilon=epsilon,
                                                                       discover=args.discover)
        # #每个algo更新
        logs2_list = [None] * (agent_num + 2)
        # print("initial_agent_num", initial_agent_num,"agent_num", agent_num)
        # for i in range(0,len(exps_list)):
        #     print(i, len(exps_list[i].obs))
        if args.algo == "ppo":
            initial_agent_num = 0
        for i in range(initial_agent_num + 2, agent_num + 2):  # 只更新新添加的agent
            if len(exps_list[i].obs):
                logs2 = algos[i].update_parameters(exps_list[i])
                logs2_list[i] = logs2
        logs2 = {}
        if args.algo == "a2c" or args.algo == "ppo":
            entropy_list = [None] * (agent_num + 2)
            value_list = [None] * (agent_num + 2)
            policy_loss_list = [None] * (agent_num + 2)
            value_loss_list = [None] * (agent_num + 2)
            grad_norm_list = [None] * (agent_num + 2)

            # 只记录新agent的日志
            for i in range(initial_agent_num + 2, agent_num + 2):
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

            # 只记录新agent的日志
            for i in range(initial_agent_num + 2, agent_num + 2):
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
            # print(logs['return_per_episode'])
            # print(logs['reshaped_return_per_episode'])
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            # print(rreturn_per_episode, "rreturn_per_episode", return_per_episode, "return_per_episode")
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            header += ["return_" + key for key in return_per_episode.keys()]
            data += rreturn_per_episode.values()
            data += return_per_episode.values()
            # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            # data += num_frames_per_episode.values()
            if args.algo == "a2c" or args.algo == "ppo":
                header += ["policy_loss", "value_loss"]
                data += [['{:.3f}'.format(item) if item is not None else 'None' for item in logs["policy_loss"]],
                         ['{:.3f}'.format(item) if item is not None else 'None' for item in logs["value_loss"]]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | policy_loss {} "
                    "| value_loss {}".format(*data))
                agent1_data = [logs["entropy"][2], logs["value"][2], logs["policy_loss"][2], logs["value_loss"][2], logs["grad_norm"][2]]
            elif args.algo == "dqn":
                header += ["loss", "q_value"]
                data += [['{:.3f}'.format(item) if item is not None else 'None' for item in logs["loss"]],
                         ['{:.3f}'.format(item) if item is not None else 'None' for item in logs["q_value"]]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | loss {} "
                    "| q_value {}".format(*data))
                agent1_data = [logs["loss"][2], logs["q_value"][2], logs["grad_norm"][2]]
            header += ["agent1_entropy", "agent1_value", "agent1_policy_loss", "agent1_value_loss", "agent1_grad_norm"]

            if status["num_frames"] == 0 and no_csv_head:
                csv_logger.writerow(header)
                no_csv_head = False
            csv_logger.writerow(data + agent1_data)
            csv_file.flush()
            # print("reshaped_return_per_episode", len(logs["reshaped_return_per_episode"]))
            # print("num_frames_per_episode", len(logs["num_frames_per_episode"]))
            for episode in range(len(logs["reshaped_return_per_episode"])):
                csv_episode_logger.writerow([logs["reshaped_return_per_episode"][episode], logs["num_frames_per_episode"][episode]])
            csv_episode_file.flush()

            # for field, value in zip(header, data):
            #     tb_writer.add_scalar(field, value, num_frames)
        if args.test_interval > 0 and update % args.test_interval == 0:
            test_return_per_episode, test_num_frames_per_episode, _, _, _ = test(G, envs[0], start_node, 10, 256, args.env, preprocess_obss, anomaly_detector=anomaly_detector, args=args)
            txt_logger.info("U {} | Test reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | Test num frames:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
                            .format(10, *(test_return_per_episode.values()), *(test_num_frames_per_episode.values())))
            # 检查是否获得了更好的测试结果
            current_test_return = test_return_per_episode['mean']
            print("current_test_return", current_test_return)
            print("best_test_return", best_test_return)
            if current_test_return >= best_test_return:
                best_test_return = current_test_return
                # 保存当前最佳模型状态
                best_model_states = [acmodel.state_dict() for acmodel in acmodels]

                # 保存最佳模型
                best_status = {
                    "num_frames": num_frames,
                    "update": update,
                    "agent_num": agent_num,
                    "model_state": best_model_states,
                    "optimizer_state": algo.optimizer.state_dict(),
                    "best_test_return": best_test_return
                }
                utils.save_status(best_status, os.path.join(model_dir, "best_model"))
                txt_logger.info(f"New best model saved with test return: {best_test_return:.2f}")

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update, "agent_num": agent_num,
                      "model_state": [acmodels[i].state_dict() for i in range(agent_num)],
                      "optimizer_state": algo.optimizer.state_dict()}
            # if hasattr(preprocess_obss, "vocab"):
            #     status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
    # save
    # status = {"num_frames": num_frames, "update": update, "agent_num": agent_num,
    #             "model_state": [acmodels[i].state_dict() for i in range(agent_num)],
    #             "optimizer_state": algo.optimizer.state_dict()}
    # # if hasattr(preprocess_obss, "vocab"):
    # #     status["vocab"] = preprocess_obss.vocab.vocab
    # utils.save_status(status, model_dir)
    # txt_logger.info("Status saved")

    for i, acmodel in enumerate(acmodels):
        acmodel.load_state_dict(best_model_states[i])
    txt_logger.info(f"Loaded best model with test return: {best_test_return:.2f}")
    # 训练结束时保存最终状态和最佳状态
    model_state = [acmodel.state_dict() for acmodel in acmodels]
    optimizer_state = algo.optimizer.state_dict()
    if return_per_episode['mean'] <= 0.5:
        print("No save bad model.")
        model_state[-1] = None
        optimizer_state = None
        print(len(model_state))
    final_status = {
        "num_frames": num_frames,
        "update": update,
        "agent_num": agent_num,
        "model_state": model_state, #[acmodel.state_dict() for acmodel in acmodels],
        "optimizer_state": optimizer_state,
        "best_test_return": best_test_return
    }
    utils.save_status(final_status, model_dir)
    txt_logger.info("Final status saved")

    # 如果最终模型不是最佳模型,则加载最佳模型状态
    if return_per_episode['mean'] <= 0.5:
        return "fail train"
    else:
        return "successfull train"
    # random discover, save the changes.
    # until get a familiar change(state)
    # change the graph

if __name__ == "__main__":
    ret_state = main()
    ret_value = 0
    if ret_state == "can_solve":
        ret_value = 0
    elif ret_state == "fail to discover anomaly":
        ret_value = 1
    elif ret_state == "fail train":
        ret_value = 2
    elif ret_state == "successfull train":
        ret_value = 3
    print(f"Return value: {ret_value}")
    sys.exit(ret_value)
# return "can_solve"
# return "fail to discover anomaly"
# return "fail train"
# return "successfull train"
