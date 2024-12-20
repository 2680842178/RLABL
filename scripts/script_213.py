import os
from collections import deque
import sys
import shutil
from pathlib import Path
import subprocess

# 初始化任务配置文件内容
START_CONFIG_CONTENT = """graph:  
  nodes:
    - 0
    - 1
    - 2
  # 0 == die, 1 == reward. 
  edges:
    - from: 2
      to: 1
      id: 0
      with_agent: 1
    - from: 2
      to: 0
      id: 1
      with_agent: 0
  start_node: 2
agent_num: 1"""

# 三个任务地图
MAP_1 = """map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, -, -, x, -, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, S, E, x
           x, x, x, x, x, x, x, x"""

MAP_2 = """map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, S, -, x, D, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, -, E, x
           x, x, x, x, x, x, x, x"""

MAP_3 = """map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, K, -, x, D, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, S, E, x
           x, x, x, x, x, x, x, x"""

# 配置参数
LR = 0.0001
DISCOUNT = 0.995
ALGO = "ppo"
EPOCHS = 8
BATCH_SIZE = 128
FRAMES_PER_PROC = 512
CONFIGMAP = "configmap.config"
START_LINE = 15
END_LINE = 22


def edit_configmap(start_line, end_line, content):
    """
    替换配置文件中指定范围的内容
    """
    with open(CONFIGMAP, "r") as f:
        lines = f.readlines()

    with open(CONFIGMAP, "w") as f:
        f.writelines(lines[:start_line - 1])
        f.write(content + "\n")
        f.writelines(lines[end_line:])


def run_discover_anomaly(task_config, discover, env, frames):
    """
    执行 discover_anomaly.py 脚本
    """
    cmd = [
        "python",
        "discover_anomaly.py",
        f"--task-config={task_config}",
        f"--discover={discover}",
        f"--algo={ALGO}",
        f"--env={env}",
        f"--lr={LR}",
        f"--AnomalyNN=test_8",
        f"--model={MODEL_NAME}",
        f"--discount={DISCOUNT}",
        f"--epochs={EPOCHS}",
        f"--frames-per-proc={FRAMES_PER_PROC}",
        f"--frames={frames}",
    ]
    print(f"运行命令: {' '.join(cmd)}")
    result= subprocess.run(cmd)
    return result.returncode


queue = [2, 1, 3]
queue = deque(queue)

def process_queue(queue):
    """
    处理队列逻辑
    """
    temp_queue = deque()  # 临时队列
    frames = 100000
    discover = 0
    task_num = 1
    task_config = f"task{task_num}"

    while queue or temp_queue:
        if not queue:
            queue.extendleft(reversed(temp_queue))
            temp_queue.clear()
        print(f"当前队列: {list(queue)}")
        item = queue.popleft()  # 从队列头部取出

        if item == "3":
            env = "MiniGrid-ConfigWorld-v0"
        else:
            env = "MiniGrid-ConfigWorld-v0-havekey"
        # 处理逻辑
        map = f"MAP_{item}"
        map = globals()[map]
        edit_configmap(START_LINE, END_LINE, map)
        return_code = run_discover_anomaly(task_config, discover, env, frames)
        if return_code == 0 or return_code == 3:
            print("-----Successful play-----")
            if return_code == 3:
                frames += 100000
                if discover == 0:
                    discover = 1
                else:
                    task_num += 1
                    task_config = f"task{task_num}"
            queue.extendleft(reversed(temp_queue))
            temp_queue.clear()
        else:
            print("-----Failed play-----")
            if return_code == 2:
                frames += 100000
            temp_queue.append(item)

    print("所有任务处理完成！")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 2 or sys.argv[1] not in ("0", "1"):
        print("Need a parameter to decide whether to delete old model and config: 0 for no(save), 1 for yes(delete)")
        sys.exit(1)

    delete_old = sys.argv[1] == "1"

    # 设置模型名称和配置文件夹
    global MODEL_NAME
    MODEL_NAME = "20241219-discover-ppo-test"
    MODEL_CONFIG_FOLDER = Path(f"config/{MODEL_NAME}")

    # 删除旧模型和配置
    if delete_old:
        print("Warning: Deleting old model and config...")
        shutil.rmtree(MODEL_CONFIG_FOLDER, ignore_errors=True)
        shutil.rmtree(f"storage/{MODEL_NAME}", ignore_errors=True)
    else:
        print("Use old model and config")

    # 初始化配置文件夹
    if not MODEL_CONFIG_FOLDER.exists():
        print(f"The folder {MODEL_CONFIG_FOLDER} does not exist, creating it...")
        (MODEL_CONFIG_FOLDER / "task1").mkdir(parents=True, exist_ok=True)
        config_file = MODEL_CONFIG_FOLDER / "task1/config.yaml"
        with open(config_file, "w") as f:
            f.write(START_CONFIG_CONTENT)
    else:
        print(f"The folder {MODEL_CONFIG_FOLDER} already exists.")
    process_queue(queue=queue)