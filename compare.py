import numpy as np
from policy_eval import *
import matplotlib.pyplot as plt


def plot_reward_curve(ave_rewards, names):
    if len(ave_rewards) == 0 or len(ave_rewards[0]) == 0:
        raise Exception("Length of reward trace is 0! Please check the data.")
    if len(ave_rewards) != len(names):
        raise Exception("Length of reward is inconsistent! Please check the data.")

    episode_num = len(ave_rewards[0])

    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax1 = fig.add_subplot(111)

    ax1.set(title="Average Cumulative Reward",
            xlim=[-episode_num * 0.05, episode_num * 1.05],
            ylim=[-0.25, 0.05],
            xlabel="Episode",
            ylabel="Average Reward")
    ax1.set_xticks(range(0, (episode_num // 10) * 10 + 1, episode_num // 10))
    ax1.set_xticklabels(range(0, (episode_num // 10) * 10 + 1, episode_num // 10), rotation=45)
    ax1.set_yticks(np.arange(-0.20, 0.05, 0.05))
    for i in np.arange(-0.20, 0.05, 0.05):
        ax1.axhline(i, color="black", alpha=0.4, ls="--")

    for idx, ave_reward_trace in enumerate(ave_rewards):
        ax1.plot(range(episode_num), ave_reward_trace, label=names[idx])

    ax1.legend(ncol=1)
    fig.tight_layout()
    plt.show()
    # plt.close()


if __name__ == '__main__':
    env = Easy21()
    log_path = "./log/"
    fig_path = "./fig/"
    OPTIMAL_POLICY = load_file_dangerous(log_path + "optimal_policy.txt")
    OPTIMAL_VALUE = load_file_dangerous(log_path + "optimal_value.txt")
    ave_rewards, names = [], []

    R = load_reward(log_path + "Q_learning_LR=0.010_e=0.010_100000_ave_reward_100000.txt")
    ave_rewards.append(R)
    names.append("Q-learning, Epsilon, LearningRate=0.01, Epsilon=0.01")
    print(R[:100])

    R = load_reward(log_path + "Q_learning_LR=0.010_DE=0.900_r=0.9999_100000_ave_reward_100000.txt")
    ave_rewards.append(R)
    names.append("Q-learning, DynamicEpsilon, LearningRate=0.01, Epsilon=0.9, DampFactor=0.9999")
    print(R[:100])

    R = load_reward(log_path + "policy_iter_exp_ave_reward_10000000.txt")[:100_000]
    ave_rewards.append(R)
    names.append("Policy Iteration")
    print(R[:100])

    R = load_reward(log_path + "MCMC_ave_reward_2000000.txt")[1_000_000:1_000_000 + 100_000]
    ave_rewards.append(R)
    names.append("Markov Chain Monte-Carlo")
    print(R[:100])

    plot_reward_curve(ave_rewards, names)
