import numpy as np
import time

from env_easy21 import Easy21
from policy_eval import *


class QLearning:
    def __init__(self, env: Easy21, learning_rate=0.05, reward_decay=1., episode_num=1_000_000, epsilon=0.1,
                 dynamic_epsilon=False, damping_factor=0.997, final_epsilon=0.001):
        self.env = env
        self.action_space = Easy21.action_space
        self.action_num = Easy21.action_num
        self.state_space = Easy21.state_space
        self.non_terminal_state_space = Easy21.non_terminal_state_space
        self.terminal_state_space = Easy21.terminal_state_space
        self.state_num = Easy21.state_num

        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.episode_num = episode_num
        self.epsilon = epsilon
        self.dynamic_epsilon = dynamic_epsilon
        self.r = damping_factor
        self.final_epsilon = final_epsilon

        self.reward_trace = []
        self.average_reward_trace = []
        self.rmse_trace = []

        if self.dynamic_epsilon:
            self.name = "LR={:.3f}_DE={:.3f}_r={:.4f}_{:d}".format(self.alpha, self.epsilon, self.r, self.episode_num)
        else:
            self.name = "LR={:.3f}_e={:.3f}_{:d}".format(self.alpha, self.epsilon, self.episode_num)

    def _update_q_table(self, state_, action_, reward_, next_state_):
        state_ = (state_[0], state_[1])
        next_state_, game_end = (next_state_[0], next_state_[1]), next_state_[2]
        if self.q_table.get(state_) is None: self.q_table[state_] = [0] * self.action_num
        if self.q_table.get(next_state_) is None: self.q_table[next_state_] = [0] * self.action_num

        q_target = reward_ if game_end == 1 else reward_ + self.gamma * max(self.q_table[next_state_])
        q_predict = self.q_table[state_][action_]

        # self.q_table[state_][action_] = q_predict + self.lr * (q_target - q_predict)
        self.q_table[state_][action_] += self.alpha * (q_target - q_predict)

    def choose_action(self, state_):
        """ Choose action of the given state np.argmax(self.q_table[state_]) according to Q-table. """
        state_ = (state_[0], state_[1])
        if self.q_table.get(state_) is None: self.q_table[state_] = [0] * self.action_num
        if np.random.rand() < self.epsilon:  # choose random action
            action = np.random.randint(self.action_num)
        else:  # choose the index of best action according to Q table
            action = np.argmax(self.q_table[state_])
        return action

    def choose_action_deterministic(self, state_):
        """ Choose action of the given state according to Q-table. """
        if self.q_table.get(state_) is None: self.q_table[state_] = [0] * self.action_num
        action = np.argmax(self.q_table[state_])
        return action

    def run_one_step(self, state):
        """ Run one step on the environment and return the reward. """
        action = self.choose_action(state)
        next_state, reward = self.env.step(state, action)
        self._update_q_table(state, action, reward, next_state)
        return next_state, reward

    def run_one_episode(self):
        """ Run one step on the environment and return the final reward. """
        state, reward = self.env.reset(), 0
        while state[2] == 0:  # game not ends
            state, reward = self.run_one_step(state)
        return reward

    def run(self, episode_num=None, section_num=10, print_info=True):
        episode_num = self.episode_num if episode_num is None else episode_num
        average_reward = 0
        average_reward_section = 0
        section_size = (episode_num // section_num)
        for i in range(episode_num):
            reward = self.run_one_episode()
            # self.reward_trace.append(reward)
            average_reward += (reward - average_reward) / (i + 1)
            self.average_reward_trace.append(average_reward)
            self.rmse_trace.append(rms_error_q(OPTIMAL_VALUE, self.q_table))

            if (i + 1) % section_size == 0:
                average_reward_section += (reward - average_reward_section) / section_size
                policy = {k: 0 if self.q_table.get(k, [0, 0])[0] > self.q_table.get(k, [0, 0])[1] else 1 
                          for k in self.non_terminal_state_space}
                print("Episode: {} / {}".format(i + 1, episode_num))
                if print_info:
                    print("\tSection: {} / {}".format(int((i + 1) / section_size), section_num))
                    print("\tAverage Reward: {:.6f}".format(self.average_reward_trace[-1]))
                    print("\tAverage Reward of Last Section: {:.6f}".format(average_reward_section))
                    print("\tQ-table RMSE: {:.6f}".format(self.rmse_trace[-1]))
                    print("\tDifferent policy num: {}".format(policy_dif(OPTIMAL_POLICY, policy)))
                average_reward_section = 0
            else:
                average_reward_section += (reward - average_reward_section) / ((i + 1) % section_size)

            if self.dynamic_epsilon:
                self.epsilon = max(self.epsilon * self.r, self.final_epsilon)

    def save_fig(self, fig_path):
        plot_Q(self.q_table, save_path=(fig_path + "Q_learning_" + self.name + ".png"))

    def save_result(self, log_path):
        with open(log_path + "Q_learning_" + self.name + ".txt", "w") as f:
            f.write(str(self.q_table))
        try:
            ART = np.array(self.average_reward_trace, dtype=np.float16)
            ART_name = log_path + "Q_learning_" + self.name + "_ave_reward_{}.txt".format(len(ART))
            np.savetxt(ART_name, ART, fmt="%.4f", delimiter=",")
            RMSE = np.array(self.rmse_trace, dtype=np.float16)
            RMSE_name = log_path + "Q_learning_" + self.name + "_rmse_{}.txt".format(len(RMSE))
            np.savetxt(RMSE_name, RMSE, fmt="%.4f", delimiter=",")
        except Exception as e:
            print("No experiment value! Saving average reward trace failed.")


if __name__ == '__main__':
    env = Easy21()
    log_path = "./log/"
    fig_path = "./fig/"
    OPTIMAL_POLICY = load_file_dangerous(log_path + "optimal_policy.txt")
    OPTIMAL_VALUE = load_file_dangerous(log_path + "optimal_value.txt")
    ave_rewards, rmses, names = [], [], []

    EPISODE_NUM = 100_000
    PARAM_DICT = {"DynamicEpsilon, LearningRate=0.005, Epsilon=0.9, DampFactor=0.9999": {"alpha": 0.005},
                  "DynamicEpsilon, LearningRate=0.01, Epsilon=0.9, DampFactor=0.9999": {"alpha": 0.01},
                  "DynamicEpsilon, LearningRate=0.05, Epsilon=0.9, DampFactor=0.9999": {"alpha": 0.05},
                  "DynamicEpsilon, LearningRate=0.1, Epsilon=0.9, DampFactor=0.9999": {"alpha": 0.1},
                  "DynamicEpsilon, LearningRate=0.5, Epsilon=0.9, DampFactor=0.9999": {"alpha": 0.5},

                  "Epsilon, LearningRate=0.01, Epsilon=0.001": {"de": False, "e": 0.001},
                  "Epsilon, LearningRate=0.01, Epsilon=0.01": {"de": False, "e": 0.01},
                  "Epsilon, LearningRate=0.01, Epsilon=0.1": {"de": False, "e": 0.1},
                  "Epsilon, LearningRate=0.01, Epsilon=1.0": {"de": False, "e": 1.},
                  }

    for name, param in PARAM_DICT.items():
        print("#" * 30 + "\n" + name + "\n" + "#" * 30)
        model = QLearning(env, episode_num=EPISODE_NUM,
                          learning_rate=param.get("alpha", 0.01), 
                          dynamic_epsilon=param.get("de", True),
                          epsilon=param.get("e", 0.9),
                          damping_factor=0.9999,
                          final_epsilon=0.001)
        model.run()
        ave_rewards.append(model.average_reward_trace)
        rmses.append(model.rmse_trace)
        names.append(name)
        model.save_fig(fig_path)
        model.save_result(log_path)

    plot_learning_curve(ave_rewards[:5], rmses[:5], names[:5])
    plot_learning_curve(ave_rewards[5:], rmses[5:], names[5:])

    result = []
    for i in range(len(ave_rewards)):
        result.append((names[i], ave_rewards[i][-1]))
    result.sort(key=lambda x: x[1])
    for name, reward in result:
        print(name)
        print("\tAverage Reward: {:.4f}".format(reward))
