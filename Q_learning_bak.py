import numpy as np
import pandas as pd
import time

from env_easy21 import Easy21


class QLearning:
    def __init__(self, env: Easy21, learning_rate=0.05, reward_decay=1., epsilon=0.1,
                 dynamic_epsilon=False, damping_factor=0.997, final_epsilon=0.001):
        self.env = env
        self.action_space = Easy21.action_space
        self.action_num = Easy21.action_num
        self.state_space = Easy21.state_space
        self.non_terminal_state_space = Easy21.non_terminal_state_space
        self.terminal_state_space = Easy21.terminal_state_space
        self.state_num = Easy21.state_num

        self.q_table = {}
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon
        self.dynamic_epsilon = dynamic_epsilon
        self.r = damping_factor
        self.final_epsilon = final_epsilon

        self.state = self.env.reset()
        self.reward_trace = []
        self.average_reward_trace = []

    def _extend_q_table(self, state_):
        """ Initialize the Q value of the state as all "0". """
        if state_ not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * self.action_num, index=self.q_table.columns, name=state_)
            )

    def _update_q_table(self, state_, action_, reward_, next_state_, game_end):
        """ 待补

        踩了几个雷,具体写一下:
            1. DataFrame 用 tuple 作 index 是可行的,但是用 df.loc[idx, col] 取出来的时候 idx 会出很多问题
            2. 用 df.loc[[idx], col] 取出来的 slice 是 DataFrame 而不是 Series, 哪怕 idx 只有一个也是 DataFrame, 需要类型转换
            3. 所以最后还是决定用 string 作 index, 写出来的代码可读性也高

        :param state_:
        :param action_:
        :param reward_:
        :param next_state_:
        :return:
        """
        self._extend_q_table(next_state_)  # initialize a record if no record about the state

        q_target = reward_ if game_end == 1 else reward_ + self.gamma * self.q_table.loc[next_state_, :].max()
        q_predict = self.q_table.at[state_, action_]
        # print(state_, action_, next_state_, q_target, q_predict, game_end)
        self.q_table.at[state_, action_] = q_predict + self.lr * (q_target - q_predict)

    def choose_action(self, state_):
        """ Choose action of the given state according to Q-table. """
        self._extend_q_table(state_)  # initialize a record if no record about the state

        if np.random.rand() < self.epsilon:  # choose random action
            action = np.random.randint(self.action_num)
        else:  # choose the INDEX of best action according to Q table
            action = int(self.q_table.loc[state_, :].argmax())
        return action

    def choose_action_deterministic(self, state_):
        """ Choose action of the given state according to Q-table. """
        self._extend_q_table(state_)  # initialize a record if no record about the state
        action = int(self.q_table.loc[state_, :].argmax())
        return action

    def run_one_step(self):
        """ Run one step on the environment and return the reward. """
        action = self.choose_action(str(self.state))
        next_state, reward = self.env.step(self.state, action)
        next_state, game_end = (next_state[0], next_state[1]), next_state[2]
        self._update_q_table(str(self.state), self.action_space[action], reward, str(next_state), game_end)
        self.state = next_state
        return reward, game_end

    def run_one_episode(self):
        """ Run one step on the environment and return the final reward. """
        reward, game_end = self.run_one_step()
        while game_end == 0:  # game not ends
            reward, game_end = self.run_one_step()
        self.state = self.env.reset()
        return reward

    def run(self, episode_num=10000):
        average_reward = 0
        exp_factor = 1 - 10 / episode_num if episode_num > 10 else 0
        for i in range(episode_num):
            if (i + 1) % (episode_num // 10) == 0:
                print("Episode: {}".format(i + 1))
                print("Average Reward: {}".format(self.average_reward_trace[-1]))
                print("Expected Reward for current policy: {:.5f}".format(self.expected_reward()))
                time.sleep(0.5)
            reward = self.run_one_episode()
            # print(self.q_table)
            self.reward_trace.append(reward)
            # #### Statistic average, incremental update #### #
            average_reward += (reward - average_reward) / (i + 1)
            # #### Exponential weighted average #### #
            # average_reward = average_reward * exp_factor + reward * (1 - exp_factor)
            self.average_reward_trace.append(average_reward)
            if self.dynamic_epsilon:
                self.epsilon = max(self.epsilon * self.r, self.final_epsilon)
        return average_reward

    def policy_evaluation(self, theta=0.001):
        delta = theta + 1
        value = {state: 0. for state in self.non_terminal_state_space}
        while delta > theta:
            delta = 0
            new_value = {state: 0. for state in self.non_terminal_state_space}
            for s in self.non_terminal_state_space:
                qsa = 0
                a = self.choose_action_deterministic(s)
                for prob, next_state, reward in self.env.trans_mat[s][a]:
                    next_state, game_end = (next_state[0], next_state[1]), next_state[2]
                    if game_end == 0:  # game not ends
                        qsa += prob * (reward + self.gamma * value[next_state])
                    else:
                        qsa += prob * reward
                new_value[s] += qsa
                delta = max(delta, abs(new_value[s] - value[s]))
            value = new_value
        return value

    def expected_reward(self, theta=0.001):
        value = self.policy_evaluation(theta)
        exp_r = 0.
        for init_s in [str((i, j)) for i in range(1, 11) for j in range(1, 11)]:
            # print(init_s, value[init_s])
            exp_r += value[init_s] / 100
        return exp_r
