import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from env_easy21 import Easy21


def plot_V(v_table, save_path=None):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')

    # Make data.
    dealer_sum_idx = np.arange(1, 11)
    player_sum_idx = np.arange(1, 22)
    dealer_sum_idx, player_sum_idx = np.meshgrid(dealer_sum_idx, player_sum_idx)

    max_Q = np.ndarray(shape=(21, 10))
    for state in Easy21.non_terminal_state_space:
        player_sum, dealer_sum = state
        max_Q[player_sum - 1][dealer_sum - 1] = v_table.get(state, 0)

    # Plot the surface.
    surf = ax.plot_surface(dealer_sum_idx, player_sum_idx, max_Q, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize plot
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel("Dealer's Initial Sum", fontsize=12)
    plt.ylabel("Player\'s Sum", fontsize=12)
    plt.title("Optimal Value Function", fontsize=16)

    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(1, 22))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_Q(q_table, save_path=None):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')

    # Make data.
    dealer_sum_idx = np.arange(1, 11)
    player_sum_idx = np.arange(1, 22)
    dealer_sum_idx, player_sum_idx = np.meshgrid(dealer_sum_idx, player_sum_idx)

    max_Q = np.ndarray(shape=(21, 10))
    for state in Easy21.non_terminal_state_space:
        player_sum, dealer_sum = state
        max_Q[player_sum - 1][dealer_sum - 1] = max(q_table.get(state, [0, 0]))

    # Plot the surface.
    surf = ax.plot_surface(dealer_sum_idx, player_sum_idx, max_Q, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize plot
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel("Dealer's Initial Sum", fontsize=12)
    plt.ylabel("Player\'s Sum", fontsize=12)
    plt.title("Value Function", fontsize=16)

    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(1, 22))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()


def policy_dif(policy1, policy2):
    """ Return the difference of the given 2 policies. """
    dif_num = 0
    for state in Easy21.non_terminal_state_space:
        if policy1[state] != policy2[state]:
            # print("{}: current policy {}, compared policy {}".format(state, policy1[state], policy2[state]))
            dif_num += 1
    return dif_num


def rms_error(value1, value2):
    """ Return the RMSE (Root Mean Square Error) of the given 2 value function. """
    value1_mat, value2_mat = np.ndarray(shape=(21, 10)), np.ndarray(shape=(21, 10))
    for state in Easy21.non_terminal_state_space:
        player_sum, dealer_sum = state
        value1_mat[player_sum - 1][dealer_sum - 1] = value1.get(state, 0)
        value2_mat[player_sum - 1][dealer_sum - 1] = value2.get(state, 0)
    return np.sqrt((np.power((value1_mat - value2_mat), 2)).mean())


def rms_error_q(value, q_value):
    """ Return the RMSE (Root Mean Square Error) of the given 2 value function. """
    value1_mat, value2_mat = np.ndarray(shape=(21, 10)), np.ndarray(shape=(21, 10))
    for state in Easy21.non_terminal_state_space:
        player_sum, dealer_sum = state
        value1_mat[player_sum - 1][dealer_sum - 1] = value.get(state, 0)
        value2_mat[player_sum - 1][dealer_sum - 1] = max(q_value.get(state, [0, 0]))
    return np.sqrt((np.power((value1_mat - value2_mat), 2)).mean())


def print_policy(policy):
    for i in range(1, 22):
        for j in range(1, 11):
            p = "Stick" if policy[i, j] == 0 else "Hit"
            print("{:>5}".format(p), end=";")
        print()


def load_file_dangerous(file_path):
    #TODO: Dangerous `eval`! Be aware of the content in the file
    with open(file_path, "r") as f:
        content = eval(f.read().strip())
    return content


def load_reward(reward_file_path):
    reward = np.loadtxt(reward_file_path, dtype=np.float16, delimiter=",")
    return reward


def plot_learning_curve(ave_rewards, rmses, names):
    def smooth(l, i, slice_num):
        start = int(i * len(l) / slice_num)
        end = int((i + 1) * len(l) / slice_num)
        length = end - start
        return sum(l[start:end]) / length

    if len(ave_rewards) == 0 or len(ave_rewards[0]) == 0:
        raise Exception("Length of reward trace is 0! Please check the data.")
    if len(rmses) != len(ave_rewards) or len(ave_rewards) != len(names):
        raise Exception("Length of reward is inconsistent! Please check the data.")

    episode_num = len(ave_rewards[0])

    fig = plt.figure(figsize=(14, 6), dpi=160)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

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

    ax2.set(title="RMS Error Curve",
            xlim=[-50 * 0.05, 50 * 1.05],
            ylim=[-0.05, 1.10],
            xlabel="Episode",
            ylabel="RMS Error")
    ax2.set_xticks(range(0, 50 + 1, 5))
    ax2.set_xticklabels(range(0, (episode_num // 10) * 10 + 1, episode_num // 10), rotation=45)
    ax2.set_yticks(np.arange(0.00, 1.05, 0.05))
    for i in np.arange(0.00, 1.05, 0.20):
        ax2.axhline(i, color="black", alpha=0.4, ls="--")

    for idx, rmse in enumerate(rmses):
        ylist = [smooth(rmse, i, 50) for i in range(50)]
        ax2.plot(range(50), ylist)

    ax1.legend(ncol=1)
    fig.tight_layout()
    plt.show()
    # plt.close()

