# train_qlearning.py
import os
import time
import pickle
import numpy as np
import random
from env import FroggerVisionEnv
from torch.utils.tensorboard import SummaryWriter

QFILE = "q_table.pkl"

class QLearningAgent:
    def __init__(self, n_actions, lr=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.05, decay=0.9995):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q_table = {}  # dict: state_key -> np.array(n_actions)

    def state_key(self, obs):
        # Convierte la observación a una tupla para usar como clave en Q-table
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

    def update(self, prev_obs, action, reward, next_obs, done):
        k1 = self.state_key(prev_obs)
        k2 = self.state_key(next_obs)
        self.ensure(k1)
        self.ensure(k2)
        q = self.q_table[k1][action]
        q_next = 0.0 if done else np.max(self.q_table[k2])  # Q-Learning usa max Q del siguiente estado
        target = reward + self.gamma * q_next
        self.q_table[k1][action] += self.lr * (target - q)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon

def save_q(agent, path=QFILE):
    with open(path, "wb") as f:
        pickle.dump(agent.q_table, f)

def load_q(agent, path=QFILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            agent.q_table = pickle.load(f)
        print(f"Loaded Q-table from {path}")
    else:
        print("No Q-table file found; starting fresh.")

def train(episodes=10000, max_steps=200, render_every=0):
    env = FroggerVisionEnv(height=14, width=12, vision=3,
                            car_spawn_prob=0.2, meteor_prob=0.2, trail_prob=0.2)
    agent = QLearningAgent(env.action_space.n, lr=0.1, gamma=0.99,
                           epsilon=1.0, min_epsilon=0.05, decay=0.9995)

    load_q(agent)
    writer = SummaryWriter(log_dir="runs/frogger_qlearning")

    rewards = []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0

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
        rewards.append(total_reward)

        # Log a TensorBoard
        writer.add_scalar("Reward/episode", total_reward, ep)
        writer.add_scalar("Epsilon/value", agent.epsilon, ep)

        # Impresión periódica
        if ep % 2000 == 0:
            avg = np.mean(rewards[-50:])
            print(f"Episode {ep}/{episodes}  avg50={avg:.3f} eps={agent.epsilon:.4f}")
            writer.add_scalar("Reward/avg50", avg, ep)

        # Autosave
        if ep % 500 == 0:
            save_q(agent)

    save_q(agent)
    env.close()
    writer.close()
    print("Training finished in {:.1f}s".format(time.time() - t0))
    return agent, rewards

if __name__ == "__main__":
    # Recomiendo muchos episodios para entrenamiento sólido
    agent, rewards = train(episodes=5000000, max_steps=200, render_every=5000000)
