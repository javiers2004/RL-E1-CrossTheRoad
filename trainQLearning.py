# train_qlearning.py
import os
import time
import pickle  # for saving/loading the q-table
import numpy as np
import random
from env import CrossTheRoadVisionEnv  # our environment
from torch.utils.tensorboard import SummaryWriter  # for logging progress

QFILE = "q_table.pkl"  # where we save our learned knowledge


class QLearningAgent:
    def __init__(self, n_actions, lr=0.01, gamma=0.99, epsilon=1.0, min_epsilon=0.05, decay=0.9999):
        self.n_actions = n_actions  # number of possible actions
        self.lr = lr  # learning rate (how big our steps are)
        self.gamma = gamma  # discount factor (how much we value future rewards)
        self.epsilon = epsilon  # probability of exploring
        self.min_epsilon = min_epsilon
        self.decay = decay  # how fast epsilon shrinks
        self.q_table = {}  # the main table: state -> action values

    def state_key(self, obs):
        return tuple(obs.flatten().tolist())  # turn the observation array into a saveable key

    def ensure(self, key):
        if key not in self.q_table:
            # if we see a new state, initialize its values to zero
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)

    def choose(self, obs):
        key = self.state_key(obs)
        self.ensure(key)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # explore (random action)
        return int(np.argmax(self.q_table[key]))  # exploit (best known action)

    def update(self, prev_obs, action, reward, next_obs, done):
        k1 = self.state_key(prev_obs)
        k2 = self.state_key(next_obs)
        self.ensure(k1)
        self.ensure(k2)
        q = self.q_table[k1][action]  # current estimate
        # Q-Learning uses the MAX Q-value from the next state
        q_next = 0.0 if done else np.max(self.q_table[k2])
        target = reward + self.gamma * q_next  # the updated estimate
        # update the Q-value with the TD error
        self.q_table[k1][action] += self.lr * (target - q)

    def decay_epsilon(self):
        # gradually reduce exploration over time
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon


def save_q(agent, path=QFILE):
    # simple save function
    with open(path, "wb") as f:
        pickle.dump(agent.q_table, f)


def load_q(agent, path=QFILE):
    # simple load function
    if os.path.exists(path):
        with open(path, "rb") as f:
            agent.q_table = pickle.load(f)
        print(f"Loaded Q-table from {path}")
    else:
        print("No Q-table file found; starting fresh.")


def train(episodes=1000000, max_steps=500, render_every=0):
    # setup the game and the agent
    env = CrossTheRoadVisionEnv(height=14, width=12, vision=3,
                                car_spawn_prob=0.2, max_cars_per_lane=2, trail_prob=0.2)
    agent = QLearningAgent(env.action_space.n, lr=0.1, gamma=0.99,
                           epsilon=1.0, min_epsilon=0.05, decay=0.9995)

    load_q(agent)
    writer = SummaryWriter(log_dir="runs/crossTheRoad_qlearning")  # logging setup

    rewards = []
    t0 = time.time()

    # loop through all episodes
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()  # start a new game
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.choose(obs)  # get action
            next_obs, reward, terminated, truncated, _ = env.step(action)  # take action
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)  # learn from the result
            obs = next_obs
            total_reward += reward

            if render_every and ep % render_every == 0:
                env.render()  # show the game

            if done:
                break

        agent.decay_epsilon()  # shrink epsilon after each episode
        rewards.append(total_reward)

        # Log data to TensorBoard
        writer.add_scalar("Reward/episode", total_reward, ep)
        writer.add_scalar("Epsilon/value", agent.epsilon, ep)
        writer.add_scalar("QTable/size", len(agent.q_table), ep)
        if ep % 2000 == 0:
            avg = np.mean(rewards[-50:])  # check average reward for the last 50 games
            print(f"Episode {ep}/{episodes}  avg50={avg:.3f} eps={agent.epsilon:.4f}")
            writer.add_scalar("Reward/avg50", avg, ep)

        # Autosave every 500 episodes
        if ep % 500 == 0:
            save_q(agent)

    save_q(agent)  # final save
    env.close()
    writer.close()
    print("Training finished in {:.1f}s".format(time.time() - t0))
    return agent, rewards


if __name__ == "__main__":
    # run the training
    agent, rewards = train(episodes=1000000, max_steps=500, render_every=0)