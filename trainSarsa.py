# train_sarsa.py
import os
import time
import pickle
import numpy as np
import random
from env import CrossTheRoadVisionEnv
from torch.utils.tensorboard import SummaryWriter

QFILE = "sarsa_table.pkl"

class SarsaAgent:
    def __init__(self, n_actions, lr=0.01, gamma=0.99, epsilon=1.0, min_epsilon=0.05, decay=0.9999):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q_table = {}  # dict: state_key -> np.array(n_actions)

    def state_key(self, obs):
        return tuple(obs.flatten().tolist())

    def ensure(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)

    def choose(self, obs):
        key = self.state_key(obs)
        self.ensure(key)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[key]))

    def update(self, prev_obs, prev_action, reward, next_obs, next_action, done):
        k1 = self.state_key(prev_obs)
        k2 = self.state_key(next_obs)
        self.ensure(k1)
        self.ensure(k2)
        q = self.q_table[k1][prev_action]
        q_next = 0.0 if done else self.q_table[k2][next_action]
        target = reward + self.gamma * q_next
        self.q_table[k1][prev_action] += self.lr * (target - q)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon

def save_sarsa(agent, path=QFILE):
    with open(path, "wb") as f:
        pickle.dump(agent.q_table, f)

def load_sarsa(agent, path=QFILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            agent.q_table = pickle.load(f)
        print(f"Loaded SARSA table from {path}")
    else:
        print("No SARSA table file found; starting fresh.")

def train(episodes=1000000, max_steps=500, render_every=0):
    env = CrossTheRoadVisionEnv(height=14, width=12, vision=3,
                                car_spawn_prob=0.2, max_cars_per_lane=2, trail_prob=0.2)
    agent = SarsaAgent(env.action_space.n, lr=0.1, gamma=0.99,
                       epsilon=1.0, min_epsilon=0.05, decay=0.9995)

    load_sarsa(agent)
    writer = SummaryWriter(log_dir="runs/crossTheRoad_sarsa")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        action = agent.choose(obs)
        total_reward = 0.0

        for step in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.choose(next_obs)

            agent.update(obs, action, reward, next_obs, next_action, done)

            obs = next_obs
            action = next_action
            total_reward += reward

            if render_every and ep % render_every == 0:
                env.render()

            if done:
                break

        agent.decay_epsilon()

        writer.add_scalar("Reward/episode", total_reward, ep)
        writer.add_scalar("Epsilon/value", agent.epsilon, ep)

        if ep % 2000 == 0:
            save_sarsa(agent)
            avg50 = np.mean([total_reward])  
            print(f"Episode {ep}/{episodes}  reward={total_reward:.2f} eps={agent.epsilon:.4f}")

    save_sarsa(agent)
    env.close()
    writer.close()
    print("Training finished.")
    return agent

if __name__ == "__main__":
    agent = train(episodes=1000000, max_steps=500, render_every=0)
