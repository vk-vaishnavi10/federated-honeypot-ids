# Federated Honeypot Networks for Rare-Attack Detection

Investigating whether a network of honeypots can collectively detect rare and
zero-day attacks that no single honeypot can, while keeping all captured data local.

## Setup
- Live Cowrie honeypot (SSH) for capture
- Real evaluation data: Kaggle Cowrie dataset (44,082 real internet-attacker sessions)
- 3–5 federated clients + FedAvg aggregation
- Per-session behavioral + command features

## Key findings (real data, n=5 seeds)
1. **Isolated honeypots detect 0% of rare attacks** they haven't seen (recall 0.000).
2. **Federated pooling recovers detection** (0.00 → 0.74–0.97) without sharing raw data.
3. **Simple augmentation (jitter/SMOTE) matches or beats GANs** at every sample size
   (3–300 per silo) — GAN complexity is unjustified for this task.
4. **Better aggregation alone (FedProx) doesn't help** — augmentation is required.
5. Real honeypot traffic is extremely imbalanced (miner: 2, recon: 11 in 44k sessions),
   motivating collective defense.

## Scripts
- `parse_multiclass.py` / `parse_rich.py` — Cowrie logs → feature vectors
- `federated.py` — federated IDS with FedAvg
- `zeroday.py` / `sweep.py` — zero-day propagation experiments
- `killer_seeds.py` — multi-seed rare-attack recovery
- `baselines.py` / `fair_privacy.py` — SMOTE / FedProx / jitter comparisons
- `crossover.py` — sample-size sweep for GAN vs jitter

## Note
Large data files (real_cowrie.json, real_data/) are gitignored — download the
Kaggle dataset separately.
