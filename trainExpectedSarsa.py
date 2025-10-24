import os
import time
import pickle
import numpy as np
import random
from env import CrossTheRoadVisionEnv
from torch.utils.tensorboard import SummaryWriter
import tempfile
import shutil

QFILE = "expected_sarsa_table.pkl"


class ExpectedSarsaAgent:
    def __init__(self, n_actions, lr=0.01, gamma=0.99, epsilon=1.0, min_epsilon=0.05,
                 decay=0.9999): 
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q_table = {}  

    def state_key(self, obs):
        if obs.ndim > 1:
            obs = obs.flatten()
        return tuple(obs.tolist()) 

    def ensure(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)

    def choose(self, obs):
        key = self.state_key(obs)
        self.ensure(key)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[key]))

    def expected_value(self, values):
        max_value = np.max(values)
        n_actions = len(values)

        n_greedy = np.sum(values == max_value)

        non_greedy_prob = self.epsilon / n_actions
        greedy_prob = (1.0 - self.epsilon) / n_greedy + non_greedy_prob

        exp_val = 0.0
        for v in values:
            if v == max_value:
                exp_val += v * greedy_prob
            else:
                exp_val += v * non_greedy_prob
        return exp_val

    def update(self, prev_obs, action, reward, next_obs, done):
        k1 = self.state_key(prev_obs)
        k2 = self.state_key(next_obs)
        self.ensure(k1)
        self.ensure(k2)

        q = self.q_table[k1][action]

        q_next_values = self.q_table[k2]
        q_next = 0.0 if done else self.expected_value(q_next_values)

        target = reward + self.gamma * q_next
        self.q_table[k1][action] += self.lr * (target - q)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon


def save_expected_sarsa(agent, path=QFILE):
    try:
        # Use a temporary file to avoid partial writes
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(agent.q_table, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Ensure data is written to disk
        # Replace the original file with the temporary one
        shutil.move(tmp_file.name, path)
        print(f"Saved Expected SARSA table to {path}")
    except Exception as e:
        print(f"Error saving Expected SARSA table to {path}: {e}")


def load_expected_sarsa(agent, path=QFILE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                agent.q_table = pickle.load(f)
            print(f"Loaded Expected SARSA table from {path}. Size: {len(agent.q_table)}")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"Error loading Expected SARSA table from {path}: {e}. Starting with an empty Q-table.")
            agent.q_table = {}
    else:
        print("No Expected SARSA table file found; starting fresh.")
        agent.q_table = {}


def train(episodes=1000000, max_steps=500, render_every=0):  
    env = CrossTheRoadVisionEnv(height=14, width=12, vision=3,
                                car_spawn_prob=0.2, max_cars_per_lane=2, trail_prob=0.2)
    agent = ExpectedSarsaAgent(env.action_space.n)

    load_expected_sarsa(agent)
    writer = SummaryWriter(log_dir="runs")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0

        initial_q_size = len(agent.q_table)

        for step in range(max_steps):
            action = agent.choose(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            if render_every and ep % render_every == 0:
                env.render()

            if done:
                break

        agent.decay_epsilon()
        writer.add_scalar("Reward/episode", total_reward, ep)
        writer.add_scalar("Epsilon/value", agent.epsilon, ep)
        writer.add_scalar("QTable/size", len(agent.q_table), ep)

        if ep % 5000 == 0:  
            save_expected_sarsa(agent)
            print(
                f"Episode {ep}/{episodes} | R={total_reward:.2f} | Eps={agent.epsilon:.4f} | Q Size={len(agent.q_table)}")

    save_expected_sarsa(agent)
    env.close()
    writer.close()
    print("Training finished.")
    return agent


if __name__ == "__main__":
    agent = train(episodes=1000000, max_steps=500, render_every=0)