# play.py
import pickle
import time
import numpy as np
from env import CrossTheRoadVisionEnv

#QFILE = "q_table.pkl"
#QFILE = "sarsa_table.pkl"
QFILE = "expected_sarsa_table.pkl"

class PolicyPlayer:
    def __init__(self, qpath=QFILE):
        self.q = {}
        if qpath:
            try:
                with open(qpath, "rb") as f:
                    self.q = pickle.load(f)
                print("Loaded Q-table.")
            except Exception as e:
                print("Could not load Q-table:", e)
                self.q = {}

    def act(self, obs):
        key = tuple(obs.flatten().tolist())
        if key in self.q:
            return int(np.argmax(self.q[key]))
        # fallback random
        return np.random.randint(0, 4)

if __name__ == "__main__":
    env = CrossTheRoadVisionEnv(height=14, width=12, vision=3, car_spawn_prob=0.2, max_cars_per_lane=2, meteor_prob=0.2, trail_prob=0.2)
    player = PolicyPlayer()

    for ep in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = player.act(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env.render()
            time.sleep(0.05)
        print(f"Episode {ep+1} finished.")
    env.close()
