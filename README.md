# 🐔 Cross The Road — Reinforcement Learning

A **Frogger-style Reinforcement Learning project** where an agent learns to cross a busy road and a river using tabular RL algorithms. The environment is fully custom-built with **Gymnasium** and rendered with **Pygame**, and the agent is trained with three classic algorithms: **Q-Learning**, **SARSA**, and **Expected SARSA**.

## 🎮 The Environment

`CrossTheRoadVisionEnv` is a custom grid-world environment (default 14×12) where the agent must travel from the bottom to the goal row at the top while surviving multiple hazards:

- 🚗 **Car lanes** — cars spawn randomly in each lane with configurable probability and direction; collision is lethal
- 💨 **Car trails** — cars may leave a temporary danger trail behind them
- 🌊 **River rows** — falling into the water is lethal unless the agent lands on a floating **log**
- 🚦 **Traffic lights** — lane states are part of the observation

### Partial observability

Instead of seeing the whole grid, the agent only observes a local **vision window** (default 3×3) centered on itself, plus log positions and traffic light state. This keeps the state space tractable for tabular methods while making the problem realistically challenging.

**Actions:** `0=up`, `1=down`, `2=left`, `3=right`

## 🧠 Algorithms

Three tabular temporal-difference algorithms are implemented and can be compared:

| Script | Algorithm | Update target |
|--------|-----------|---------------|
| `trainQLearning.py` | Q-Learning (off-policy) | `max` Q-value of next state |
| `trainSarsa.py` | SARSA (on-policy) | Q-value of the action actually taken |
| `trainExpectedSarsa.py` | Expected SARSA | Expected Q-value under the ε-greedy policy |

All agents use an **ε-greedy exploration** strategy with exponential decay (ε: 1.0 → 0.05), learning rate `α = 0.01` and discount factor `γ = 0.99`. Training progress is logged to **TensorBoard**, and each learned Q-table is saved with `pickle` (`q_table.pkl`, `sarsa_table.pkl`, `expected_sarsa_table.pkl`).

## 📂 Project Structure

```
RL-E1-CrossTheRoad/
├── env.py                  # Custom Gymnasium environment (CrossTheRoadVisionEnv)
├── trainQLearning.py       # Q-Learning training loop
├── trainSarsa.py           # SARSA training loop
├── trainExpectedSarsa.py   # Expected SARSA training loop
├── play.py                 # Watch a trained agent play (Pygame rendering)
├── record.py               # Record a gameplay video (.mp4) of the trained agent
└── images/                 # Sprites: agent, cars, logs, water, explosions...
```

## 🚀 Getting Started

### Requirements

```bash
pip install gymnasium pygame numpy torch tensorboard opencv-python
```

### Train an agent

```bash
python trainQLearning.py        # or trainSarsa.py / trainExpectedSarsa.py
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

### Watch the trained agent

Set the desired Q-table file in `play.py` (`QFILE`), then:

```bash
python play.py
```

### Record a video

```bash
python record.py
```

Generates `partida_agente.mp4` with the agent playing using the selected policy.

## 🔧 Environment Customization

The environment accepts several parameters for experimentation:

```python
env = CrossTheRoadVisionEnv(
    height=14,              # grid rows
    width=12,               # grid columns
    vision=3,               # observation window size (odd, ≥3)
    car_spawn_prob=0.2,     # car spawn probability per lane per step
    max_cars_per_lane=2,    # traffic density cap
    trail_prob=0.2,         # probability of danger trails
    seed=None               # reproducibility
)
```

## 🛠️ Tech Stack

`Python` · `Gymnasium` · `Pygame` · `NumPy` · `TensorBoard (PyTorch)` · `OpenCV`
