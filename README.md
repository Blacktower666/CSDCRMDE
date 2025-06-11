# Hybrid Learning for Cold-Start-Aware Microservice Scheduling in Dynamic Edge Environments

This repository contains a simplified implementation of the algorithm proposed in our paper:

> **"Hybrid Learning for Cold-Start-Aware Microservice Scheduling in Dynamic Edge Environments"**

The method is based on **GRU-enhanced Soft Actor-Critic (SAC)** and features a two-stage training pipeline:
1. **Offline Imitation Learning** using expert trajectory data.
2. **Online Reinforcement Learning** using the SAC algorithm.

---

## 📦 Dependencies

Install required packages via pip or conda:

```bash
pip install torch numpy
```

Python built-in packages used:
- `random`
- `csv`
- `pickle`
- `argparse`
- `datetime`

---

## ⚙️ Configuration

You can set the environment parameters in:

```python
config.py
```

Key configurable items:
- `EDGE_NODE_NUM`: Number of edge nodes
- `max_tasks`, `min_tasks`: Number of microservice tasks per time slot
- `node_cpu_freq_max`: Maximum CPU frequency of nodes
- `epoch_imitation`: Number of imitation learning epochs
- `epoch`: Number of reinforcement learning episodes
- `e1`, `e2`: Weight coefficients for energy and delay in expert policy

---

## 🚀 Usage

### 1. Generate Expert Data (Optional)

If you haven’t generated expert trajectories yet, run:

```bash
python offline_data_collection.py
```

This will create an offline data file like:

```bash
offline_data_seq_n15_cpu650_task20-5_ep10.pkl
```

---

### 2. Train the GRU-SAC Model

Run the full training pipeline (both imitation + SAC):

```bash
python gru_sac_behavior_clone.py
```

The training consists of two phases:

#### 📍 Phase 1: Imitation Learning (Behavior Cloning)
- Trains the GRU-based actor to mimic expert policy
- Uses offline trajectory data saved in `.pkl` files

#### 📍 Phase 2: Online Reinforcement Learning (SAC)
- Trains actor and critic using online interaction with the environment
- Learns to balance scheduling delay and energy consumption

---

## 📁 Output

- Trained actor model from imitation learning is saved as:

```bash
imitation_gru_sac_ep{epoch_imitation}_{timestamp}.pth
```

- Training logs (CSV format) are saved under `log/`, e.g.:

```bash
log/gruBC_SAC_ep10_alpha1_n15_cpu650_task20-5_20250610.csv
```

Each line records:

```csv
Episode, Reward, Total Time, Total Energy, Completion Ratio, Download Time, Actor Loss, Critic Loss, Fail Times
```
