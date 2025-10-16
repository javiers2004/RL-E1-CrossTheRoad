# agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # puedes usar un diccionario para estados discretos

    def get_state_key(self, state):
        # Convierte el estado en algo hashable para la Q-table
        return tuple(state.flatten())

    def choose_action(self, state):
        key = self.get_state_key(state)
        if random.random() < self.epsilon or key not in self.q_table:
            return random.randint(0, self.n_actions-1)
        else:
            return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.n_actions)

        best_next = np.max(self.q_table[next_key])
        self.q_table[key][action] += self.lr * (reward + self.gamma * best_next * (not done) - self.q_table[key][action])
