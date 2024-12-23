import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''

191,98304,132.14838749279062,7,1.9140350818634033,0.036314738591895344,1.765526294708252,1.9621052742004395,"['None', 'None', '0.018', '-0.197']","['None', 'None', '0.021', '0.038']",0.914035090693721,0.03631473501911394,0.765526294708252,0.9621052742004395,1.6448734283447266,0.8350338220596314,0.018370054289698602,0.021372834965586663,0.24515576634155703
192,99328,138.79347115710752,15,1.9191470063965896,0.027456811756652582,1.8602631092071533,1.9621052742004395,"['None', 'None', '0.006', '-0.103']","['None', 'None', '0.019', '0.055']",0.9191470063965896,0.027456814238727575,0.8602631688117981,0.9621052742004395,1.5862174828847249,0.8046717544396719,0.006232564337551594,0.01927686793108781,0.21441181815621704
193,100352,148.02516881287562,22,1.9247453405011086,0.028508201528552796,1.8247368335723877,1.9597368240356445,"['None', 'None', '0.011', '-0.067']","['None', 'None', '0.016', '0.057']",0.9247453366556475,0.02850820223960498,0.8247368335723877,0.9597368240356445,1.4695908029874165,0.8300979137420654,0.010797880279521147,0.01623194493004121,0.1398489972040567

'''
model_name = "20241221-discover-ppo"
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
    data_part1 = data[data['frames'] < 102000]
    last_step_part1 = data_part1['frames'].iloc[-1]
    data_part2 = data[(data['frames'] > 102000) & (data['frames'] < 203000)]
    last_step_part2 = data_part2['frames'].iloc[-1]
    data_part3 = data[data['frames'] > 203000]

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
