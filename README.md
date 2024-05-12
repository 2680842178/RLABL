# Minigrid


## Installation

```
cd Minigrid-master
pip3 install -r requirements.txt
pip install .
```
or 
python setup.py install

# Torch-ac


## Installation

1. Clone this repository.

2. Install  `torch-ac` RL algorithms:

```
cd torch-ac
pip3 install -e .
```

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
