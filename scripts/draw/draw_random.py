import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("muted")

# RL_model_name = "20240112-DQN-easy-small"
# ABL_model_name = "20240112-ABL-DQN-easy-small"
# DFA_model_name = "20250111-DFA-DQN-easy-small"
# fig_name="easy-small-DQN"
# inner_dots = [100500, 200900]

# RL_model_name = "20250113-DQN-easy-large"
ABL_model_name = "ramdom-ABL-PPO-easy-small"
# DFA_model_name = "20240112-DFA-DQN-easy-large"
fig_name="random-easy-small-ppo"
inner_dots = [200500, 400900]

# RL_model_name = "PPO-large"
# ABL_model_name = "ABL-PPO-large"
# DFA_model_name = "20240104-DFA-PPO-easy-large"
# fig_name="easy-large"
# inner_dots = [300500, 600900]

# RL_model_name = "20250104-RL-ppo-easy-small"
# ABL_model_name = "ABL-PPO-small"
# DFA_model_name = "20240104-DFA-PPO-easy-small"
# fig_name="easy-small"
# inner_dots = [100500, 200900]

total_test_times = 2

total_ABL_data = pd.DataFrame(columns=['frames', 'rreturn_mean', 'return_mean'])

for i in range(1, total_test_times+1):
    # 读取 CSV 文件

    ABL_data = pd.read_csv(f'./{ABL_model_name}-{i}/log.csv')
    discover1_data = pd.read_csv(f'./{ABL_model_name}-{i}/log_discover_1.csv')
    discover2_data = pd.read_csv(f'./{ABL_model_name}-{i}/log_discover_2.csv')
    data_part1 = ABL_data[ABL_data['frames'] < inner_dots[0]]
    last_step_part1 = data_part1['frames'].iloc[-1]
    data_part2 = ABL_data[(ABL_data['frames'] > inner_dots[0]) & (ABL_data['frames'] < inner_dots[1])]
    last_step_part2 = data_part2['frames'].iloc[-1]
    data_part3 = ABL_data[ABL_data['frames'] > inner_dots[1]]

    discover1_data['frames'] += last_step_part1
    discover1_data['frames'] += 512
    discover2_data['frames'] += last_step_part2
    discover2_data['frames'] += 512
    # episode_data = pd.read_csv(f'./storage/{model_name}-{i}/log_episode.csv')
    ABL_merged_data = pd.concat([data_part1, discover1_data, data_part2, discover2_data, data_part3])
    # merged_data['mental_reward_smooth'] = moving_average(merged_data['rreturn_mean'], window_size)
    ABL_merged_data = ABL_merged_data[['frames', 'rreturn_mean', 'return_mean']]
    
    total_ABL_data = pd.concat([total_ABL_data, ABL_merged_data])
# plt.plot(merged_data['frames'], merged_data['mental_reward_smooth'], label='discover_return', color='pink')
# plt.plot(data['frames'], data['rreturn_mean'], label='rreturn_mean')
# plt.plot(data['frames'], data['return_mean_smooth'], label='return_mean_smooth', alpha=0.4)
# plt.plot(data['frames'], data['rreturn_mean_smooth'], label='rreturn_mean_smooth', alpha=0.4)
sns.lineplot(data=total_ABL_data, x='frames', y='return_mean', label='Real reward', color='salmon', linewidth=1.5, alpha=0.8)
sns.lineplot(data=total_ABL_data, x='frames', y='rreturn_mean', label='Mental reward', color='steelblue', linewidth=1.5, alpha=0.8)
plt.xlabel('frames')
plt.ylabel('Reward')
plt.title('Reward Curve')

plt.legend(fontsize=8, loc="lower right")
sns.despine()
plt.tight_layout()

plt.grid(True)
plt.savefig(f"./{fig_name}.png", dpi=300)
plt.savefig(f"./{fig_name}.svg", dpi=300)
plt.show()

# first_column_data = episode_data.iloc[:, 0]

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
