import os
import time
import pickle  # for saving and loading the q-table
import numpy as np
import random
from env import CrossTheRoadVisionEnv  # our game environment
from torch.utils.tensorboard import SummaryWriter  # to track training progress
import tempfile
import shutil

QFILE = "expected_sarsa_table.pkl"  # filename for saving the q-table


class ExpectedSarsaAgent:
    def __init__(self, n_actions, lr=0.01, gamma=0.99, epsilon=1.0, min_epsilon=0.05,
                 decay=0.9999):
        self.n_actions = n_actions  # how many actions the agent can take
        self.lr = lr  # learning rate (how fast we learn)
        self.gamma = gamma  # discount factor (how important future reward is)
        self.epsilon = epsilon  # exploration probability (trying new things)
        self.min_epsilon = min_epsilon  # minimum epsilon
        self.decay = decay  # how much epsilon decreases
        self.q_table = {}  # the q-table, where we store expected (state, action) values

    def state_key(self, obs):
        if obs.ndim > 1:
            obs = obs.flatten()
        return tuple(obs.tolist())  # convert the state to a tuple key for the q-table

    def ensure(self, key):
        if key not in self.q_table:
            # if the state is new, initialize its q-values to zero
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)

    def choose(self, obs):
        key = self.state_key(obs)
        self.ensure(key)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # choose random action (exploration)
        # choose the best action according to the q-table (exploitation)
        return int(np.argmax(self.q_table[key]))

    def expected_value(self, values):
        # calculates the expected value for the next state in expected sarsa
        max_value = np.max(values)
        n_actions = len(values)

        n_greedy = np.sum(values == max_value)  # how many actions are the best

        non_greedy_prob = self.epsilon / n_actions  # prob. of choosing a non-best action
        # prob. of choosing one of the best actions
        greedy_prob = (1.0 - self.epsilon) / n_greedy + non_greedy_prob

        exp_val = 0.0
        for v in values:
            if v == max_value:
                exp_val += v * greedy_prob  # value * prob. if it's one of the best
            else:
                exp_val += v * non_greedy_prob  # value * prob. if it's not one of the best
        return exp_val

    def update(self, prev_obs, action, reward, next_obs, done):
        k1 = self.state_key(prev_obs)
        k2 = self.state_key(next_obs)
        self.ensure(k1)
        self.ensure(k2)

        q = self.q_table[k1][action]  # current q-value

        q_next_values = self.q_table[k2]
        # expected value of the next state (0.0 if the game is over)
        q_next = 0.0 if done else self.expected_value(q_next_values)

        target = reward + self.gamma * q_next  # the value we want to aim for
        # update the q-table using the expected sarsa formula
        self.q_table[k1][action] += self.lr * (target - q)

    def decay_epsilon(self):
        # reduce epsilon over time to explore less
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon


def save_expected_sarsa(agent, path=QFILE):
    # save the q-table to a file so we don't lose progress
    try:
        # use a temporary file to ensure it saves correctly
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(agent.q_table, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            # move the temporary file to the final location
        shutil.move(tmp_file.name, path)
        print(f"Saved Expected SARSA table to {path}")
    except Exception as e:
        print(f"Error saving Expected SARSA table to {path}: {e}")


def load_expected_sarsa(agent, path=QFILE):
    # load the q-table from a file if it exists
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                agent.q_table = pickle.load(f)
            print(f"Loaded Expected SARSA table from {path}. Size: {len(agent.q_table)}")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"Error loading Expected SARSA table from {path}: {e}. Starting with an empty Q-table.")
            agent.q_table = {}  # start fresh if loading fails
    else:
        print("No Expected SARSA table file found; starting fresh.")
        agent.q_table = {}


def train(episodes=1000000, max_steps=500, render_every=0):
    # set up the environment
    env = CrossTheRoadVisionEnv(height=14, width=12, vision=3,
                                car_spawn_prob=0.2, max_cars_per_lane=2, trail_prob=0.2)
    agent = ExpectedSarsaAgent(env.action_space.n)

    load_expected_sarsa(agent)
    rewards = []
    writer = SummaryWriter(log_dir="runs/crossTheRoad_expected_sarsa")  # initialize tensorboard logging

    # main training loop
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()  # start a new game
        total_reward = 0.0

        initial_q_size = len(agent.q_table)

        for step in range(max_steps):
            action = agent.choose(obs)  # choose an action
            next_obs, reward, terminated, truncated, _ = env.step(action)  # take the step
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)  # learn from the experience
            obs = next_obs
            total_reward += reward

            if render_every and ep % render_every == 0:
                env.render()  # show the game visually

            if done:
                break

        agent.decay_epsilon()  # reduce exploration
        rewards.append(total_reward)
        # log data to tensorboard
        writer.add_scalar("Reward/episode", total_reward, ep)
        writer.add_scalar("Epsilon/value", agent.epsilon, ep)
        writer.add_scalar("QTable/size", len(agent.q_table), ep)

        if ep % 5000 == 0:  # save every 5000 episodes
            save_expected_sarsa(agent)
            avg = np.mean(rewards[-50:])
            print(
                f"Episode {ep}/{episodes} | R={total_reward:.2f} | Eps={agent.epsilon:.4f} | Q Size={len(agent.q_table)}")
            writer.add_scalar("Reward/avg50", avg, ep)

    save_expected_sarsa(agent)  # save one last time
    env.close()
    writer.close()
    print("Training finished.")
    return agent


if __name__ == "__main__":
    # start the training process
    agent = train(episodes=1000000, max_steps=500, render_every=0)