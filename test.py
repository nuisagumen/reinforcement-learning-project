import math
import gym
import numpy as np
import matplotlib.pyplot as plt

HOLE_REWARD = -1
actions_titles = ['left', 'down', 'right', 'up']


class q_paradigm:
    """
    This class defines the main shared ideas behind the learning algorithms.
    It is not meant to be used directly, but rather as a base class for the other algorithms.
    """

    def __init__(self, name, alpha, gamma, state_space, num_actions):
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.state_space = state_space
        self.num_actions = num_actions
        self.q_values = np.full((state_space, num_actions), 0.0)

    # Not used
    def set_optimistic_Qs(self, value):
        """
        Set the q_values to a specific value. This is used for optimistic initialization.
        :param value:
        :return:
        """
        self.q_values = np.full((self.state_space, self.num_actions), float(value))

    def q_function(self, state, action, reward, _, next_state):
        pass

    def __call__(self, state, action, reward, _, next_state):
        self.q_function(state, action, reward, _, next_state)

    def get_Qs(self):
        """
        Return The Q values
        :return: The Q values
        """
        return self.q_values


class q_learning(q_paradigm):
    """
    Inherits from q_paradigm.
    This class implements the Q-Learning algorithm.
    """

    def __init__(self, name, alpha, gamma, state_space, num_actions):
        super().__init__(name, alpha, gamma, state_space, num_actions)

    def q_function(self, state, action, reward, _, next_state):
        """
        This method will calculate the q_value for a given state and action pair.
        Following the Q-Learning algorithm -> Q(s, a) = Q(s, a) + alpha * (R(s, a) + gamma * max(Q(s', a)) - Q(s, a))
        :param state: The current state of the agent
        :param action: The action selected in the current state
        :param reward: The reward received from the environment from taking the action in the current state
        :param _: Placeholder, not used as the next action is not needed for Q-Learning
        :param next_state: The resulting state from taking the action in the current state
        :return: The q_value for the given state and action pair (Q-Learning)
        """
        self.q_values[state][action] = float(
            self.q_values[state][action] + self.alpha * (reward + self.gamma * np.max(self.q_values[next_state])
                                                         - self.q_values[state][action]))


class sarsa(q_paradigm):
    """
    Inherits from q_paradigm.
    This class implements the SARSA algorithm.
    """

    def __init__(self, name, alpha, gamma, state_space, num_actions):
        super().__init__(name, alpha, gamma, state_space, num_actions)

    def q_function(self, state, action, reward, next_action, next_state):
        """
        This method will calculate the q_value for a given state and action pair.
        Following the SARSA algorithm -> Q(s, a) = Q(s, a) + alpha * (R(s, a) + gamma * Q(s', a') - Q(s, a))
        :param state: The current state of the agent
        :param action: The action selected in the current state
        :param reward: The reward received from the environment from taking the action in the current state
        :param next_action: The action selected in the next state
        :param next_state: The resulting state from taking the action in the current state
        :return: The q_value for the given state and action pair (SARSA)
        """
        self.q_values[state][action] = float(
            self.q_values[state][action] + self.alpha * (reward + self.gamma * self.q_values[next_state][next_action]
                                                         - self.q_values[state][action]))


class greedy_policy:
    """
    Class which implements the greedy exploration strategy or policy.
    """

    def __init__(self, name, is_optimistic=False, optimistic_value=0.0):
        self.name = name
        self.is_optimistic = is_optimistic
        self.optimistic_value = optimistic_value
        self.q_paradigm = None

    def __call__(self, state):
        """
        This method will return the action to take given a state.
        It will select the action with the highest Q value.
        :param state: The current state of the environment.
        :return: the index of the action to take.
        """
        Q = self.q_paradigm.get_Qs()[state]
        # Get the action with the maximum Q-value
        best_action = np.argmax(Q)
        best_actions = np.argwhere(Q == Q[best_action]).flatten()
        if len(best_actions) == 1:
            action = best_action
        else:
            # If there are multiple actions with the maximum Q-value, choose one randomly
            action = np.random.choice(best_actions)
        return action

    def set_q_paradigm(self, paradigm):
        """
        Set the q_paradigm to use for the greedy policy.
        :param paradigm: The q_paradigm to use. (Q-Learning or SARSA)
        :return: none
        """
        self.q_paradigm = paradigm
        if self.q_paradigm is not None and self.is_optimistic:
            self.q_paradigm.set_optimistic_Qs(self.optimistic_value)

    def update(self, state, action, reward, action1, state1):
        """
        Update the Q values for the greedy policy based of the q_paradigm being used.
        :param state: The current state of the agent
        :param action: The action selected in the current state
        :param reward: The reward received from the environment from taking the action in the current state
        :param action1: The action selected in the next state (only needed for SARSA)
        :param state1: The resulting state from taking the action in the current state
        :return: none, updates the Q values for the greedy policy
        """
        self.q_paradigm(state, action, reward, action1, state1)


class epsilon_greedy_policy:
    """
    Class which implements the epsilon-greedy exploration strategy or policy.
    """

    def __init__(self, name, epsilon, decay):
        self.name = name
        self.epsilon = epsilon  # The initial epsilon value
        self.decay = decay  # Decay rate for epsilon
        self.q_paradigm = None

    def update_epsilon(self):
        """
        Apply the decay rate to the epsilon value.
        :return: none, updates the field epsilon of the class.
        """
        self.epsilon *= self.decay

    def __call__(self, state):
        """
        This method will return the action to take given a state.
        It will select the action with the highest Q value or choose a random action with probability epsilon.
        :param state: The current state of the environment.
        :return: the index of the action to take.
        """
        if np.random.random() < self.epsilon or np.all(self.q_paradigm.get_Qs()[state] == 0):
            return np.random.randint(0, 4)
        else:
            Q = self.q_paradigm.get_Qs()
            return np.argmax(Q[state])

    def set_q_paradigm(self, paradigm):
        """
        Set the q_paradigm to use for the epsilon-greedy policy.
        :param paradigm: The q_paradigm to use. (Q-Learning or SARSA)
        :return: none, simply updates the field q_paradigm of the class.
        """
        self.q_paradigm = paradigm

    def update(self, state, action, reward, action1, state1):
        """
        Update the Q values for the epsilon-greedy policy based of the q_paradigm being used.
        Also updates the epsilon value by applying the decay rate.
        :param state: The current state of the agent
        :param action: The action selected in the current state
        :param reward: The reward received from the environment from taking the action in the current state
        :param action1: The action selected in the next state (only needed for SARSA)
        :param state1: The resulting state from taking the action in the current state
        :return: none, updates the Q values for the epsilon-greedy policy
        """
        self.q_paradigm(state, action, reward, action1, state1)
        self.update_epsilon()


class random_policy:
    """
    Class which implements the random exploration strategy or policy.
    Used for comparison purposes.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, state):
        """
        This method will return a random to take.
        :param state: The current state of the environment.
        :return: the index of the action to take.
        """
        return np.random.randint(0, 4)

    def set_q_paradigm(self, paradigm):
        pass

    def update(self, state, action, reward, action1, state1):
        pass


class aps:
    """
    Class which represents the Action Preference strategy.
    """

    def __init__(self, name, num_states, num_actions, alpha):
        self.time_step = 1
        self.num_states = num_states
        self.num_actions = num_actions
        self.name = name
        self.alpha = alpha
        self.probabilities_array = []
        self.rewards = np.zeros((num_states, num_actions))
        self.action_counts = np.full((num_states, num_actions), 1)
        self.action_preferences = np.zeros((num_states, num_actions))

    def get_preferences(self):
        """
        Getter for the action preferences.
        :return: The action preferences.
        """
        return self.action_preferences

    def __call__(self, state):
        """
        Returns the action to be chosen, based of the action preference strategy.
        Equation:
        πt(a) = e^(action_preference(a)) / Σ(e^(action_preference(a)))
        :return: the index of the hand
        """
        preferences_for_state = self.get_preferences()[state]
        action_indexes = list(range(self.num_actions))
        sum = np.sum(np.exp(preferences_for_state))
        self.probabilities_array.clear()
        for i in range(self.num_actions):
            self.probabilities_array.append(math.pow(math.e, preferences_for_state[i]) / sum)

        index = np.random.choice(action_indexes, 1, p=self.probabilities_array)
        return index[0]

    def print_prefs(self):
        """
        Print the preferences in a nice format
        """
        print("--------------------")
        for i in range(self.num_states):
            print(f'State: {i} -> {self.action_preferences[i]}')

    def update(self, state, action, reward, dis, dis_):
        """
        Updates the strategy data wit respect to action preferences.
        Equations for updating action preferences:
            Ht+1(s, a') = Ht(s, a') + step_size * (reward - average_reward) * (1 - πt(s, a'))
            Ht+1(s, a) = Ht(s, a) - step_size * (reward - average_reward) * πt(s, a)
        """
        preferences_for_state = self.get_preferences()[state]
        av_r = np.divide(self.rewards[state],
                         self.action_counts[state])  # now it is averaged over each hand, so becomes action specific

        for i in range(self.num_actions):
            if i == action:
                action_preference = preferences_for_state[i] + self.alpha * (reward - av_r[i]) * (
                        1 - self.probabilities_array[i])
                preferences_for_state[i] = action_preference
                self.rewards[state][i] += reward
            else:
                action_preference = preferences_for_state[i] - self.alpha * (reward - av_r[i]) * (
                    self.probabilities_array[i])
                preferences_for_state[i] = action_preference
        self.time_step += 1
        self.action_counts[state][action] += 1
        self.action_preferences[state] = preferences_for_state

    def set_q_paradigm(self, paradigm):
        """
        This method will set the Q paradigm for the policy.
        :param paradigm: The Q paradigm to use.
        """
        pass


class ucb:
    """
    Upper Confidence Bound
    """

    def __init__(self, name, c, decay, num_states, num_actions):
        self.name = name
        self.c = c
        self.decay = decay
        self.time_step = 0
        self.num_actions = num_actions
        self.action_counts = np.full((num_states, num_actions), 1)  # to avoid division by 0
        self.q_paradigm = None

    def calculate_uncertainty(self, state, action):
        """
        This method will return the uncertainty of an action.
        Equation: -> uncertainty = c * sqrt( ln(t) / action_count )
        :param state: The current state of the environment.
        :param action: The current action to calculate the uncertainty for.
        :return: The uncertainty of the action.
        """
        return float(self.c * math.sqrt(math.log(self.time_step + 1) / self.action_counts[state][action]))

    def __call__(self, state):
        """
        This method will return the action to take given a state and the Q table.
        It will select the action with the highest Q value given that state.
        :param state: The current state of the environment.
        :return: the index of the action to take.
        """
        Q = self.q_paradigm.get_Qs()[state]
        ucb = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if self.action_counts[state][i] > 0:
                uncertainty = self.calculate_uncertainty(state, i)
                ucb[i] = Q[i] + uncertainty

        action = np.argmax(ucb)
        return action

    def update(self, state, action, reward, action1, state1):
        """
        This method will update the necessary information for the UCB algorithm.
        :param state: the current state of the environment.
        :param action: the action taken in the current state.
        :param reward: the reward achieved by taking the action in the current state.
        :param action1: the action taken in the next state.
        :param state1: the resulting state from taking action in the current state.
        """
        self.q_paradigm(state, action, reward, action1, state1)
        self.time_step += 1
        self.action_counts[state][action] += 1
        self.c *= self.decay

    def set_q_paradigm(self, paradigm):
        """
        This method will set the Q paradigm for the policy.
        :param paradigm: The Q paradigm to use.
        """
        self.q_paradigm = paradigm


class agent:
    def __init__(self, num_actions, state_space, learning_rate, discount_factor, strategy, paradigm):
        self.num_actions = num_actions
        self.state_space = state_space
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.action_space = [i for i in range(num_actions)]
        self.policy = self.__init_policy(strategy)
        self.policy.set_q_paradigm(self.__init_paradigm(paradigm, num_actions))
        self.sum_reward = 0

    def __init_paradigm(self, paradigm, num_actions):
        if paradigm == 'sarsa':
            return sarsa(paradigm, 0.2, 0.8, self.state_space, num_actions)
        elif paradigm == 'q_learning':
            return q_learning(paradigm, 0.4, 0.8, self.state_space, num_actions)

    def __init_policy(self, strategy):
        """
        This method will initialize the policy based on the strategy.
        :param strategy: The name of the strategy to use.
        :return:
        """
        # hyperparameter for the strategies
        epsilon = 0.8  # for epsilon greedy
        epsilon_decay = 0.1  # for epsilon greedy
        initial_optimistic_value = 10  # for optimistic initial values
        exploration_constant = 10  # for UCB
        decay_rate_c = 0.95  # for decaying exploration constant

        if strategy == 'greedy':
            name = 'greedy'
            return greedy_policy(name)
        elif strategy == 'epsilon_greedy':
            return epsilon_greedy_policy('epsilon_greedy', epsilon, epsilon_decay)
        elif strategy == 'ucb':
            name = 'Upper Confidence Bound'
            return ucb(name, exploration_constant, decay_rate_c, self.state_space, self.num_actions)
        elif strategy == 'aps':
            name = 'Action Preferences'
            return aps(name, self.state_space, self.num_actions, self.alpha)
        elif strategy == 'optimistic':
            name = 'Optimistic Initial Values'
            return greedy_policy(name, True, initial_optimistic_value)
        elif strategy == 'random':
            name = 'random'
            return random_policy(name)

    def train(self, num_episodes, env):
        """
        This method will train the agent for the given number of episodes.
        :param num_episodes: The number of episodes to train the agent.
        :return: The plotting data for the average reward the agent achieved per episode.
        """
        x = []
        y = []
        for experiment in range(100):
            for episode in range(num_episodes):
                state = env.reset()[0]
                done = False
                action = self.policy(state)
                while not done:
                    updated_info = env.step(action)
                    state1 = updated_info[0]
                    reward = updated_info[1]
                    done = updated_info[2]

                    # When un-commented, the agent will learn from the environment when using APS.
                    # if done and reward == 0:
                    #     reward = -1

                    action1 = self.policy(state1)
                    self.policy.update(state, action, reward, action1, state1)
                    state = state1
                    action = action1
                    if reward == 1:
                        self.sum_reward += reward

            x.append(experiment)
            y.append(self.sum_reward / num_episodes)
            self.sum_reward = 0
            print("done with experiment" + str(experiment))

        return y


def exec_experiment():
    """
    This method will execute the experiment.
    :return: None, but will display the results of the experiment in the form of a graph.
    """
    # rgb_array
    map_size = 4
    env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)
    num_actions = 4
    state_space = map_size * map_size
    # hyper-parameters
    learning_rate = 0.1
    discount_factor = 0.9
    strategies = ['random', 'greedy', 'epsilon_greedy', 'aps', 'ucb']
    paradigms = ['sarsa', 'q_learning']
    num_episodes = 100

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    gamma_values = [0.7, 0.8, 0.9, 0.95, 0.99]

    for paradigm in paradigms:
        plot_for_paradigm = []
        for strategy in strategies:
            print("For " + paradigm + " and " + strategy)
            agent1 = agent(num_actions, state_space, learning_rate, discount_factor, strategy, paradigm)
            plot_data = agent1.train(num_episodes, env)
            plot_for_paradigm.append(plot_data)

        show_average_rewards(plot_for_paradigm, strategies, paradigm)
    # close the environment
    env.close()


def show_average_rewards(rewards, labels, paradigm):
    """
    This method will plot the average rewards for each strategy.
    :param paradigm: Name of the learning algorithm.
    :param rewards: The average rewards for each strategy.
    :param labels: The labels for the strategies.
    :return: None
    """
    plt.title(f'Learning Strategies with {paradigm}')
    plt.xlabel('Experiment Repetitions')
    plt.ylabel('Average Reward')
    if len(rewards) != len(labels):
        raise ValueError('The number of rewards and labels must be the same.')
    for i in range(len(rewards)):
        plt.plot(rewards[i], label=labels[i])

    plt.legend()
    plt.show()


def exec_trial(env):
    num_actions = map_size = 4
    state_space = map_size * map_size
    learning_rate = 0.1
    discount_factor = 0.9
    strategies = ['UCB', 'epsilon_greedy', 'aps', 'optimistic']
    paradigms = ['q_learning', 'sarsa']
    num_episodes = 1
    plot_notes = f'ucb, q_learning, action count = 0, with else inf condition in ucb.call '
    agent1 = agent(num_actions, state_space, learning_rate, discount_factor, strategies[3], paradigms[0])
    data = agent1.train(num_episodes, env)
    show_average_rewards([data], [strategies[3]], paradigms[0])


if __name__ == '__main__':
    # env = gym.make("FrozenLake-v1", render_mode='human', is_slippery=False)
    # env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)
    exec_experiment()
    # exec_trial(env)
