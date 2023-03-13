import numpy as np
import random
import hickle


class Agent:
    def __init__(self, n_states, n_actions,
                 alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_states = n_states
        self.n_actions = n_actions

        self.q_table = np.zeros([self.n_states, self.n_actions])

    def train(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]

        if done:
            next_value = 0
        else:
            next_value = np.amax(self.q_table[next_state])

        new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
        self.q_table[state, action] = new_value

    def make_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            best_action_value = np.amax(self.q_table[state])
            valid_action_list = [i for i, x in enumerate(self.q_table[state]) if x == best_action_value]
            action = random.choice(valid_action_list)

        return action

    def reset(self):
        self.q_table = np.zeros([self.n_states, self.n_actions])

    def save_data(self, n):
        if n == 1:
            hickle.dump(self.q_table, 'model/q_learning.hkl')
        else:
            hickle.dump(self.q_table, 'model/q_learning2.hkl')

    def load_data(self, n):
        if n == 1:
            self.q_table = hickle.load('model/q_learning.hkl')
        else:
            self.q_table = hickle.load('model/q_learning2.hkl')
