import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_name = "20241202-seed1-discover"
total_test_times = 30

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

total_RL_data = pd.DataFrame(columns=['frames', 'rreturn_mean', 'return_mean'])
total_discover_data = pd.DataFrame(columns=['frames', 'rreturn_mean', 'return_mean'])

    # data = pd.read_csv(f'./storage/{model_name}-{i}/log.csv')
    # data = data["frames", "rreturn_mean"]
    # data.loc[(data['rreturn_mean'] > 0) & (data['frames'] > 100352) & (data['frames'] < 200000), 'rreturn_mean'] *= 2
    # data.loc[(data['rreturn_mean'] > 0) & (data['frames'] > 200000), 'rreturn_mean'] *= 3

    # episode_data = pd.read_csv(f'./storage/{model_name}/log_episode.csv')
    # data['rreturn_mean_smooth'] = moving_average(data['rreturn_mean'], window_size)

for i in range(1, total_test_times+1):
    # 读取 CSV 文件

    # for idx in range(len(episode_data)):
    #     if idx >= 2000 and episode_data.iloc[idx, 0] == 0:
    #         episode_data.iloc[idx, 0] = -1.0
    #     elif idx >= 2000 and episode_data.iloc[idx, 0] > 1 and episode_data.iloc[idx, 0] < 2:
    #         episode_data.iloc[idx, 0] = episode_data.iloc[idx, 0] - 1
    #     elif idx >= 5000 and episode_data.iloc[idx, 0] == 1:
    #         episode_data.iloc[idx, 0] = -1.0
    #     elif idx >= 5000 and episode_data.iloc[idx, 0] > 2 and episode_data.iloc[idx, 0] < 3:
    #         episode_data.iloc[idx, 0] = episode_data.iloc[idx, 0] - 2

    # window_size = 10
    # episode_data['return_mean_smooth'] = moving_average(episode_data.iloc[:, 0], window_size)
    # data['return_mean_smooth'] = moving_average(data['rreturn_mean'], window_size)

    data = pd.read_csv(f'./storage/{model_name}-{i}/log.csv')
    discover1_data = pd.read_csv(f'./storage/{model_name}-{i}/log_discover_1.csv')
    try:
        discover2_data = pd.read_csv(f'./storage/{model_name}-{i}/log_discover_2.csv')
    except:
        continue
    data_part1 = data[data['frames'] < 101000]
    last_step_part1 = data_part1['frames'].iloc[-1]
    data_part2 = data[(data['frames'] > 101000) & (data['frames'] < 202000)]
    last_step_part2 = data_part2['frames'].iloc[-1]
    data_part3 = data[data['frames'] > 202000]

    discover1_data['frames'] += last_step_part1
    discover2_data['frames'] += last_step_part2
    episode_data = pd.read_csv(f'./storage/{model_name}-{i}/log_episode.csv')
    merged_data = pd.concat([data_part1, discover1_data, data_part2, discover2_data, data_part3])
    # merged_data['mental_reward_smooth'] = moving_average(merged_data['rreturn_mean'], window_size)
    print(merged_data.keys())
    merged_data = merged_data[['frames', 'rreturn_mean', 'return_mean']]
    total_discover_data = pd.concat([total_discover_data, merged_data])
# plt.plot(merged_data['frames'], merged_data['mental_reward_smooth'], label='discover_return', color='pink')
# plt.plot(data['frames'], data['rreturn_mean'], label='rreturn_mean')
# plt.plot(data['frames'], data['return_mean_smooth'], label='return_mean_smooth', alpha=0.4)
# plt.plot(data['frames'], data['rreturn_mean_smooth'], label='rreturn_mean_smooth', alpha=0.4)
sns.lineplot(data=total_discover_data, x='frames', y='rreturn_mean', label='mental_reward')
sns.lineplot(data=total_discover_data, x='frames', y='return_mean', label="real_reward")
plt.xlabel('frames')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.tight_layout()
plt.grid(True)
plt.savefig(f"./{model_name}-reward-curve.png")
plt.show()

first_column_data = episode_data.iloc[:, 0]

# 绘制散点图
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(first_column_data)), first_column_data, label='episode return', color='red', s=10, alpha=0.6)
# plt.plot(range(len(first_column_data)), episode_data['return_mean_smooth'], label='return_mean_smooth')
# plt.xlabel('episode')
# plt.ylabel('return')
# plt.title('episodes return')
# plt.legend()
# plt.grid(True)
# plt.savefig(f"./{model_name}-episode-reward.png")
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
