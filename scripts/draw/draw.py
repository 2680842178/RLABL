import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_name = '20241228-discover-ppo-random-test2'

# def moving_average(data, window_size):
#     return data.rolling(window=window_size).mean()
# 读取 CSV 文件
data = pd.read_csv(f'./storage/{model_name}/log.csv')
data_part1 = data[data['frames'] < 200500]
last_step_part1 = data_part1['frames'].iloc[-1]
data_part2 = data[(data['frames'] > 200500) & (data['frames'] < 402000)]
last_step_part2 = data_part2['frames'].iloc[-1]
data_part3 = data[data['frames'] > 402000]

print(last_step_part1.dtype, last_step_part2.dtype)

discover1_data = pd.read_csv(f'./storage/{model_name}/log_discover_1.csv')
discover1_data['frames'] += last_step_part1
discover2_data = pd.read_csv(f'./storage/{model_name}/log_discover_2.csv')
discover2_data['frames'] += last_step_part2
episode_data = pd.read_csv(f'./storage/{model_name}/log_episode.csv')

plt.figure(figsize=(10, 6))
# plt.plot(data_part1['frames'], data_part1['return_mean'], label='real_reward_mean', color='green')
# plt.plot(data_part2['frames'], data_part2['return_mean'], color='green')
# plt.plot(data_part3['frames'], data_part3['return_mean'], color='green')

merged_data = pd.concat([data_part1, discover1_data, data_part2, discover2_data, data_part3])

# plt.plot(data_part1['frames'], data_part1['rreturn_mean'], label='mental_reward_mean', color='blue')
# plt.plot(data_part2['frames'], data_part2['rreturn_mean'], color='blue')
# plt.plot(data_part3['frames'], data_part3['rreturn_mean'], color='blue')

# plt.plot(discover1_data['frames'], discover1_data['rreturn_mean'], label='discover_reward_mean', color='orange')
# plt.plot(discover2_data['frames'], discover2_data['rreturn_mean'], color='orange')

window_size = 10
# data['return_mean_smooth'] = moving_average(data['rreturn_mean'], window_size)
# merged_data['mental_reward_smooth'] = moving_average(merged_data['rreturn_mean'], window_size)

# 绘制奖励曲线
# plt.plot(merged_data['frames'], merged_data['mental_reward_smooth'], label='mental_reward_smooth', color='pink')
plt.plot(merged_data['frames'], merged_data['rreturn_mean'], label='mental_reward', color='pink')
plt.plot(merged_data['frames'], merged_data['return_mean'], label='reward', color='blue')
plt.xlabel('frames')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.legend()
plt.grid(True)
plt.savefig(f"./test_figs/reward_curve_{model_name}.png")
plt.show()

# first_column_data = episode_data.iloc[:, 0]

# # 绘制散点图
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(first_column_data)), first_column_data, label='episode return', color='red', s=10, alpha=0.6)
# plt.xlabel('episode')
# plt.ylabel('return')
# plt.title('episodes return')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(data['frames'], data['agent1_entropy'], label='entropy')
# plt.plot(data['frames'], data['agent1_value'], label='value')
# plt.plot(data['frames'], data['agent1_policy_loss'], label='policy_loss')
# plt.plot(data['frames'], data['agent1_value_loss'], label='value_loss')
# plt.plot(data['frames'], data['agent1_grad_norm'], label='grad_norm')
# plt.xlabel('frames')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()
# plt.grid(True)
# plt.show()
