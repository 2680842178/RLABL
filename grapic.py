
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



if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    # 读取 CSV 文件
    file_path1 = "multi_1M.csv"  # 替换成你的文件路径
    data1 = pd.read_csv(file_path1, header=0)
    data1 = data1.dropna()
    # 绘制折线图

    # 获取第二列和第十一列数据
    x1_values = data1.iloc[:, 1]  # 第二列数据
    y1_values = data1.iloc[:, 10]  # 第十一列数据
    y1_smoothed = gaussian_filter1d(y1_values, sigma=15)

    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(x1_values, y1_smoothed)

    # 读取 CSV 文件
    file_path2 = "1M.csv"  # 替换成你的文件路径
    data2 = pd.read_csv(file_path2, header=0)
    data2 = data2.dropna()
    x2_values = data2.iloc[:, 1]  # 第二列数据
    y2_values = data2.iloc[:, 10]  # 第十一列数据
    y2_smoothed = gaussian_filter1d(y2_values, sigma=15)
    plt.plot(x2_values, y2_smoothed)

    plt.xlabel(data1.columns[1])  # x 轴标签为第二列数据名字
    plt.ylabel(data1.columns[10])  # y 轴标签为第十一列数据名字
    plt.title("Line Plot")  # 图标题
    plt.grid(True)  # 添加网格线
    plt.legend()
    plt.savefig('result.png')
    plt.show()  # 显示图形
