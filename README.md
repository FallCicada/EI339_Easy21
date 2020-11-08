# EI339 Easy 21 Game
EI339 Artificial Intelligence Team Project

## Q-Learning

Codes for Q-learning are put in directory `Q_learning/`.

For simply reproducing the result in the report, please run `python main.py`.

### Code structure

* Main function: `main.py`

* Environment implementation of **Easy21** is in `env_easy21.py`.
* Extra tool function is put in `policy_eval.py` and `tools.py`.
* Algorithm implementation:
    * **Value iteration** code is in `value_iter.py`. This code is used for calculation theoretical optimal value function. Please run this code **first** if you'd like to seperately run each algorithm.
    * **Q-learning** code is in `Q_learning.py`.
    * **Policy iteration** code is in `policy_iter.py`.
    * **MCMC** (Markov Chain Monte Carlo) code is in `monte_carlo.py`. This code is used for result comparison.
* Result comparison plotting code is in `compare.py`. Please run it after running all algorithms' code.

### Logs

Running the codes will generate some log files, as well as some figures in the `Q_learning/log` and `Q_learning/fig`directory. You can clear the logs after the run of all codes.

### Parameters

* Main:
    * You can set episode number to run in parameter `episode_num` for each algorithm
    * For other pameters, please refer to each algorithms' parameter introduction.
* Value iteration
    * `reward_decay`: rewad decay factor $\gamma=1.0$. This experiment does not require change this.
    * `theta`: threshold to stop between two value update process in value iteration 
* Q-learning
    * `learning_rate`: learning rate $\alpha$.
    * `reward_decay`: reward decay factor $\gamma=1.0$..
    * `episode_num`: default number of episodes to run.
    * `epsilon`: the exploration factor $\epsilon$.
    * `dynamic_epsilon`: whether to turn on Dynamic Epsilon mode or not.
    * `damping_factor`: damping factor $r$ for exponential dynamic epsilon, suggest *None*.
    * `final_epsilon`: minimal epsilon for exponential dynamic epsilon, suggest *None*.
* Policy iteration
    * Same as Value iteration.
* MCMC
    * `learning_rate`: learning rate $\alpha$.
    * `reward_decay`: reward decay factor $\gamma=1.0$.

