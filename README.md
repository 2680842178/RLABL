# RLABL 

## 实验脚本：
1. scripts/minigrid_discover.sh 带有状态发现(ABL+RL)
2. scripts/minigrid_RL.sh 纯RL
3. scripts/minigrid_fullDFA.sh 提前构建好状态机的训练。
详细见脚本内部。注意每次运行脚本时要注意脚本内的参数：例如实验环境（随机/固定），实验步数（课程步数和发现步数），实验地图文件（环境的地图会从指定文件中读取）。

# Minigrid

## Installation

安装本地包（在conda内构建符号链接指向该文件夹，不直接安装到conda环境内）
```
cd Minigrid-master
pip install -e .
```

# Torch-ac

## Installation

1. Clone this repository.

2. Install  `torch-ac` RL algorithms:

```
cd torch-ac
pip3 install -e .
```

## 主要参数说明

运行脚本：scripts/discover.py   
参数：--discover 是否进行新状态发现，0为仅训练，1为发现和训练   
--env 环境名称   
--task-config 任务配置文件（yaml格式），位于scripts/config/model名称/任务名称.yaml(任务名称应遵循"task?",'?'为数字的格式，例如，使用task1.yaml配置文件，则参数名称为task1)  
任务配置文件中，主要包含图的结构（知识）和agent数量，若进行发现，还会创建下一个任务配置文件，包含新的图结构以及agent数量。  
config文件夹中还包括存储的突变图像，保存为bmp格式。  
--AnomalyNN 异常检测网络名称，存储在scripts/StateNN文件夹下。  
--model 模型名称，存储在scripts/storage文件夹下。   

## Files

This package contains:
- scripts to:
  - train an agent \
  in `script/train.py` ([more details](#scripts-train))
  - visualize agent's behavior \
  in `script/visualize.py` ([more details](#scripts-visualize))
  - evaluate agent's performances \
  in `script/evaluate.py` ([more details](#scripts-evaluate))
- a default agent's model \
in `model.py` ([more details](#model))
- utilitarian classes and functions used by the scripts \
in `utils`

These files are suited for [`minigrid`](https://github.com/Farama-Foundation/Minigrid) environments and [`torch-ac`](https://github.com/lcswillems/torch-ac) RL algorithms. They are easy to adapt to other environments and RL algorithms by modifying:
- `model.py`
- `utils/format.py`

<h2 id="scripts-train">scripts/train.py</h2>

An example of use:

```bash
python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10 --frames 80000
```

The script loads the model in `storage/DoorKey` or creates it if it doesn't exist, then trains it with the PPO algorithm on the MiniGrid DoorKey environment, and saves it every 10 updates in `storage/DoorKey`. It stops after 80 000 frames.

**Note:** You can define a different storage location in the environment variable `PROJECT_STORAGE`.

More generally, the script has 2 required arguments:
- `--algo ALGO`: name of the RL algorithm used to train
- `--env ENV`: name of the environment to train on

and a bunch of optional arguments among which:
- `--recurrence N`: gradient will be backpropagated over N timesteps. By default, N = 1. If N > 1, a LSTM is added to the model to have memory.
- `--text`: a GRU is added to the model to handle text input.
- ... (see more using `--help`)

During training, logs are printed in your terminal (and saved in text and CSV format):


**Note:** `U` gives the update number, `F` the total number of frames, `FPS` the number of frames per second, `D` the total duration, `rR:μσmM` the mean, std, min and max reshaped return per episode, `F:μσmM` the mean, std, min and max number of frames per episode, `H` the entropy, `V` the value, `pL` the policy loss, `vL` the value loss and `∇` the gradient norm.

During training, logs are also plotted in Tensorboard:


<h2 id="scripts-visualize">scripts/visualize.py</h2>

An example of use:

```
python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

In this use case, the script displays how the model in `storage/DoorKey` behaves on the MiniGrid DoorKey environment.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--argmax`: select the action with highest probability
- ... (see more using `--help`)

<h2 id="scripts-evaluate">scripts/evaluate.py</h2>

An example of use:

```
python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```



In this use case, the script prints in the terminal the performance among 100 episodes of the model in `storage/DoorKey`.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--episodes N`: number of episodes of evaluation. By default, N = 100.
- ... (see more using `--help`)

<h2 id="model">model.py</h2>

The default model is discribed by the following schema:

By default, the memory part (in red) and the langage part (in blue) are disabled. They can be enabled by setting to `True` the `use_memory` and `use_text` parameters of the model constructor.

This model can be easily adapted to your needs.
