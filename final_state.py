import numpy as np


def printVec(vec):
    for i in vec:
        print("{:.2f}% |".format(i * 100), end=" ")
    print()


def transfer_matrix(strategy):
    # Stop hitting after k >= strategy
    mat = np.zeros([22, 22], dtype=np.float64)
    for i in range(1, 12):
        mat[i - 1, :] = np.array([1 / 30] * (i - 1) + [0] + [2 / 30] * 10 + [0.] * (11 - i) + [(11 - i) / 30])
    for i in range(12, strategy):
        mat[i - 1, :] = np.array([0] * (i - 11) + [1 / 30] * 10 + [0] + [2 / 30] * (21 - i) + [(i * 2 - 22) / 30])
    for i in range(strategy, 23):
        mat[i - 1, i - 1] = 1.

    return mat


def end_state(init_state, trans_mat, max_iter=10000, delta=1e-9):
    d = init_state
    for _ in range(max_iter):
        d_ = np.matmul(d, trans_mat)
        if np.linalg.norm(d_ - d) < delta:
            break
        d = d_

    for j in range(22):
        if d[j] < 1e-9:
            d[j] = 0

    return d


def winning_rate(distr_a, distr_b):
    """ Calculate Pr{ a > b }. """
    l = len(distr_a)
    prob = 0
    for i in range(l):
        prob += np.sum(distr_b[i] * distr_a[i + 1:])
    return prob


def exp_reward(distr_a, distr_b):
    l = len(distr_a)
    reward = -1 * distr_a[-1]
    for i in range(l - 2, -1, -1):
        reward += np.sum(distr_a[i] * distr_b[:i])  # b < a
        reward += distr_a[i] * distr_b[-1]  # b busts
        reward -= np.sum(distr_a[i] * distr_b[i + 1: -1])  # b > a
    return reward


end_state_list = []
for i in range(10):
    init_s = np.zeros(22, dtype=np.float64)
    init_s[i] = 1.
    tmp_end_state = [init_s]
    for strategy in range(12, 22):
        trans_mat = transfer_matrix(strategy)
        tmp_end_state.append(end_state(init_s, trans_mat))
    end_state_list.append(tmp_end_state)

tot_exp_reward = 0
for player_init_state in range(10):
    for dealer_init_state in range(10):
        best_exp_reward = -1.1
        best_strategy = ""
        print("Player: {}, Dealer: {}".format(player_init_state + 1, dealer_init_state + 1))
        for strategy_idx, strategy in zip(range(11), ["Stick"] + [i for i in range(12, 22)]):
            player_end_state = end_state_list[player_init_state][strategy_idx]
            dealer_end_state = end_state_list[dealer_init_state][5]
            # print("Strategy: {}".format(strategy))
            # print(player_end_state)
            # print(dealer_end_state)
            reward = exp_reward(player_end_state, dealer_end_state)
            # print(reward)
            if reward > best_exp_reward:
                best_strategy = strategy
                best_exp_reward = reward
        print("Best: {}, Strategy: {}".format(best_exp_reward, best_strategy))
        tot_exp_reward += best_exp_reward / 100
print(tot_exp_reward)

# 0.1519, 0.1453, 0.1384, 0.1313, 0.0594, 0.0498, 0.3239

