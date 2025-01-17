import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
RL_model_name = "20250112-RL-PPO-taxi"
ABL_model_name = "ABL-PPO-taxi-test"
total_test_time = 1
fig_name = "test_taxi"

total_RL_data = pd.DataFrame(columns=['frames', 'rreturn_mean', 'return_mean'])
total_ABL_data = pd.DataFrame(columns=['frames', 'rreturn_mean', 'return_mean'])

for i in range(1, total_test_time + 1):
# 读取 CSV 文件
    RL_data = pd.read_csv(f'./storage/{RL_model_name}-{i}/log.csv')
    ABL_data = pd.read_csv(f'./storage/{ABL_model_name}-{i}/log.csv')
    discover1_data = pd.read_csv(f'./storage/{ABL_model_name}-{i}/log_discover_1.csv')
    # episode_data = pd.read_csv(f'./storage/{model_name}-{i}/log_episode.csv')
    ABL_merged_data = pd.concat([discover1_data, ABL_data])
    # merged_data['mental_reward_smooth'] = moving_average(merged_data['rreturn_mean'], window_size)
    ABL_merged_data = ABL_merged_data[['frames', 'rreturn_mean', 'return_mean']]
    RL_merged_data = RL_data[['frames', 'rreturn_mean', 'return_mean']]
    
    total_ABL_data = pd.concat([total_ABL_data, ABL_merged_data])
    total_RL_data = pd.concat([total_RL_data, RL_merged_data])

    # episode_data = pd.read_csv(f'./storage/{model_name}/log_episode.csv')

sns.lineplot(data=total_RL_data, x='frames', y='return_mean', label='PPO')
sns.lineplot(data=total_ABL_data, x='frames', y='return_mean', label='ABL+PPO')

plt.xlabel('frames')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.tight_layout()
plt.grid(True)
plt.savefig(f"./{ABL_model_name}-reward-curve.png")
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
