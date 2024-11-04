import pandas as pd
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()
# 读取 CSV 文件
data = pd.read_csv('./storage/20241105-seed1/log.csv')

window_size = 30
data['return_mean_smooth'] = moving_average(data['return_mean'], window_size)
data['rreturn_mean_smooth'] = moving_average(data['rreturn_mean'], window_size)

# 绘制奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(data['frames'], data['return_mean'], label='return_mean')
plt.plot(data['frames'], data['rreturn_mean'], label='rreturn_mean')
plt.plot(data['frames'], data['return_mean_smooth'], label='return_mean_smooth', alpha=0.4)
plt.plot(data['frames'], data['rreturn_mean_smooth'], label='rreturn_mean_smooth', alpha=0.4)
plt.xlabel('frames')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.legend()
plt.grid(True)
plt.show()