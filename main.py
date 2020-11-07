import numpy as np
import pandas as pd
import time

from env_easy21 import Easy21
from Q_learning import QLearning, plot_reward_curve
from policy_iter import PolicyIteration


if __name__ == "__main__":
    env = Easy21()
    EPISODE_NUM = 1_000_000
    # print(env.trans_mat["(1, 1)"])
    # print(env.trans_mat["(14, 10)"][0])
    # print(env.trans_mat["(14, 10)"][1])
    """
    rewards, ave_rewards, names = [], [], []
    model = QLearning(env, learning_rate=0.05, epsilon=0.9, damping_factor=0.9999, 
                      final_epsilon=0.001, dynamic_epsilon=True)
    model.run(EPISODE_NUM)
    rewards.append(model.reward_trace)
    ave_rewards.append(model.average_reward_trace)
    names.append("DE, lr=0.05, e=0.9, r=0.9999")

    model = QLearning(env, learning_rate=0.05, epsilon=0.1, dynamic_epsilon=False)
    model.run(EPISODE_NUM)
    rewards.append(model.reward_trace)
    ave_rewards.append(model.average_reward_trace)
    names.append("E, lr=0.05, e=0.1")

    model = QLearning(env, learning_rate=0.05, epsilon=1.0, dynamic_epsilon=False)
    model.run(EPISODE_NUM)
    rewards.append(model.reward_trace)
    ave_rewards.append(model.average_reward_trace)
    names.append("E, lr=0.05, e=1.0")

    plot_reward_curve(rewards, ave_rewards, names)
    # model.show_Q_table()
    """
    #"""
    model = PolicyIteration(env)
    model.policy_iteration()
    # print(model.env.trans_mat)
    # print(model.value)
    # print(model.pi)
    model.test_policy()
    #"""
