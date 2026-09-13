# LunarLander-v3 — A2C Baseline

Minimal A2C agent using PyTorch, evaluated on a modified [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (stronger wind, lower gravity).

## Run

Dependencies (Python 3.10, Torch 2.9.1, gymnasium[box2d]) are pinned in `pyproject.toml` and `.python-version`, so `uv` sets everything up on first run:

```bash
uv run python eval.py
```

## Implementation

There are different files in this project

1. `agent.py` — defines the agent which picks the index from the policy logits at inference time.
2. `rl_model.py` (`collect_rollouts`) — sample an action from the policy during rollouts.
3. `rl_model.py` (`loss_a2c`) — compute discounted n-step returns (respecting `is_done`), the value and policy losses, and the entropy regularization term.

## Training

Agent has been trained over 50000 iterations. It reached 271.20 points, in the top 100 agents.

```bash
uv run python train.py --n_iterations=50000    # ~20 min
```



