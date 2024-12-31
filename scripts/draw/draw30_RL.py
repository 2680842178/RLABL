import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''

191,98304,132.14838749279062,7,1.9140350818634033,0.036314738591895344,1.765526294708252,1.9621052742004395,"['None', 'None', '0.018', '-0.197']","['None', 'None', '0.021', '0.038']",0.914035090693721,0.03631473501911394,0.765526294708252,0.9621052742004395,1.6448734283447266,0.8350338220596314,0.018370054289698602,0.021372834965586663,0.24515576634155703
192,99328,138.79347115710752,15,1.9191470063965896,0.027456811756652582,1.8602631092071533,1.9621052742004395,"['None', 'None', '0.006', '-0.103']","['None', 'None', '0.019', '0.055']",0.9191470063965896,0.027456814238727575,0.8602631688117981,0.9621052742004395,1.5862174828847249,0.8046717544396719,0.006232564337551594,0.01927686793108781,0.21441181815621704
193,100352,148.02516881287562,22,1.9247453405011086,0.028508201528552796,1.8247368335723877,1.9597368240356445,"['None', 'None', '0.011', '-0.067']","['None', 'None', '0.016', '0.057']",0.9247453366556475,0.02850820223960498,0.8247368335723877,0.9597368240356445,1.4695908029874165,0.8300979137420654,0.010797880279521147,0.01623194493004121,0.1398489972040567

'''
model_name = "20241205-seed1-RL"
total_test_time = 30

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()
total_RL_data = pd.DataFrame(columns=['frames', 'rreturn_mean', 'return_mean'])

for i in range(1, total_test_time + 1):
# 读取 CSV 文件
    data = pd.read_csv(f'./storage/{model_name}-{i}/log.csv')
    data = data[['frames', 'rreturn_mean', 'return_mean']]
    data.loc[(data['rreturn_mean'] > 0) & (data['frames'] > 100352) & (data['frames'] < 200000), 'rreturn_mean'] *= 2
    data.loc[(data['rreturn_mean'] > 0) & (data['frames'] > 200000), 'rreturn_mean'] *= 3
    total_RL_data = pd.concat([total_RL_data, data])

    # episode_data = pd.read_csv(f'./storage/{model_name}/log_episode.csv')

sns.lineplot(data=total_RL_data, x='frames', y='return_mean', label='real_reward')

plt.xlabel('frames')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.tight_layout()
plt.grid(True)
plt.savefig(f"./{model_name}-reward-curve.png")
plt.show()


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
