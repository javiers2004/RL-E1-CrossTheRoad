# train_sarsa.py
import os
import time
import pickle
import numpy as np
import random
from env import CrossTheRoadVisionEnv  # our game
from torch.utils.tensorboard import SummaryWriter  # for graphs

QFILE = "sarsa_table.pkl"  # file to save our progress


class SarsaAgent:
    def __init__(self, n_actions, lr=0.01, gamma=0.99, epsilon=1.0, min_epsilon=0.05, decay=0.9999):
        self.n_actions = n_actions  # how many moves we can make
        self.lr = lr  # learning speed
        self.gamma = gamma  # importance of future reward
        self.epsilon = epsilon  # explore vs. exploit balance
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q_table = {}  # the table where we store (state, action) values

    def state_key(self, obs):
        return tuple(obs.flatten().tolist())  # turn the state array into a dictionary key

    def ensure(self, key):
        if key not in self.q_table:
            # start a new state with all q-values at zero
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)

    def choose(self, obs):
        key = self.state_key(obs)
        self.ensure(key)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # explore (random move)
        return int(np.argmax(self.q_table[key]))  # exploit (best move we know)

    def update(self, prev_obs, prev_action, reward, next_obs, next_action, done):
        # the key difference for SARSA: we need the *next action* to update
        k1 = self.state_key(prev_obs)
        k2 = self.state_key(next_obs)
        self.ensure(k1)
        self.ensure(k2)
        q = self.q_table[k1][prev_action]  # q-value for the move we just made

        # SARSA uses the Q-value for the specific *next action* (next_action) chosen
        q_next = 0.0 if done else self.q_table[k2][next_action]

        target = reward + self.gamma * q_next  # our new target q-value
        # update the q-value
        self.q_table[k1][prev_action] += self.lr * (target - q)

    def decay_epsilon(self):
        # reduce exploration
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon


def save_sarsa(agent, path=QFILE):
    # save the agent's table
    with open(path, "wb") as f:
        pickle.dump(agent.q_table, f)


def load_sarsa(agent, path=QFILE):
    # load the agent's table if it exists
    if os.path.exists(path):
        with open(path, "rb") as f:
            agent.q_table = pickle.load(f)
        print(f"Loaded SARSA table from {path}")
    else:
        print("No SARSA table file found; starting fresh.")


def train(episodes=1000000, max_steps=500, render_every=0):
    # setup the game and the agent
    env = CrossTheRoadVisionEnv(height=14, width=12, vision=3,
                                car_spawn_prob=0.2, max_cars_per_lane=2, trail_prob=0.2)
    agent = SarsaAgent(env.action_space.n, lr=0.1, gamma=0.99,
                       epsilon=1.0, min_epsilon=0.05, decay=0.9995)

    load_sarsa(agent)
    writer = SummaryWriter(log_dir="runs/crossTheRoad_sarsa")  # logging setup
    rewards = []
    # main training loop
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        action = agent.choose(obs)  # choose first action of the episode
        total_reward = 0.0

        for step in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)  # take the action
            done = terminated or truncated
            next_action = agent.choose(next_obs)  # choose the next action

            # the core SARSA update uses (S, A, R, S', A')
            agent.update(obs, action, reward, next_obs, next_action, done)

            # move to the next state/action pair
            obs = next_obs
            action = next_action
            total_reward += reward

            if render_every and ep % render_every == 0:
                env.render()  # show the game

            if done:
                break

        agent.decay_epsilon()  # reduce exploration
        rewards.append(total_reward)
        # log the data
        writer.add_scalar("Reward/episode", total_reward, ep)
        writer.add_scalar("Epsilon/value", agent.epsilon, ep)
        writer.add_scalar("QTable/size", len(agent.q_table), ep)

        if ep % 2000 == 0:
            save_sarsa(agent)  # autosave
            # the avg50 calculation here is simple, just printing the current reward
            avg50 = np.mean([total_reward])
            print(f"Episode {ep}/{episodes}  reward={total_reward:.2f} eps={agent.epsilon:.4f}")
            writer.add_scalar("Reward/avg50", avg50, ep)

    save_sarsa(agent)  # final save
    env.close()
    writer.close()
    print("Training finished.")
    return agent


if __name__ == "__main__":
    # run the training
    agent = train(episodes=1000000, max_steps=500, render_every=0)