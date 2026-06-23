# Market-Maker: DP vs RL vs Avellaneda–Stoikov

A study of optimal market making on a finite-horizon MDP. A market maker quotes a
one-unit ask and bid each step, earns the spread on fills, and carries inventory risk.
We solve the problem with **dynamic programming (exact ground truth)** and compare it
against **tabular Q-learning**, a hand-written **PPO**, and the **Avellaneda–Stoikov**
closed-form model (both gridded and continuous).

The goal is to measure how closely the learning agents recover the DP-optimal policy,
leading with **regret / value-gap** rather than exact policy match.

## Contents

| File | Description |
|---|---|
| [baseDP.py](baseDP.py) | DP ground truth and the shared market-making environment (marimo notebook) |
| [RL.py](RL.py) | Q-learning and PPO agents compared against DP and Avellaneda–Stoikov (marimo notebook) |
| [RL.md](RL.md) | Theory: Q-learning and PPO derivations |
| `*.ipynb` | Exported Jupyter versions of the notebooks |

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Running the marimo notebooks

The notebooks ([RL.py](RL.py), [baseDP.py](baseDP.py)) are [marimo](https://marimo.io)
apps — plain Python files that run as reactive, interactive notebooks.

Edit interactively in the browser:

```bash
uv run marimo edit RL.py
```

Run as a read-only app:

```bash
uv run marimo run RL.py
```

Or execute as a normal Python script:

```bash
uv run python RL.py
```
