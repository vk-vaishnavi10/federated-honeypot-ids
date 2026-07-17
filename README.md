# Federated Threat Intelligence Across Geo-Distributed Honeypots

An empirical study of collaborative rare-attack detection across distributed
deployments — without sharing raw captures. Validated on honeypot session data
and network-flow data.

## Datasets
- **Dataset A** — single-deployment Cowrie capture (~44k real sessions)
- **Dataset B** — 10 physically-distinct SSH/Telnet honeypots, multiple global
  regions (~3.3M events / ~518k sessions)
- **Dataset C (generalization)** — CICIDS2017 network flows (~2.5M flows),
  simulated multi-site partitioning

## Key findings (real data, n=5 seeds, significance tested)
1. **Isolated deployments detect 0%** of an attack class they haven't observed.
2. **Federated aggregation recovers detection** (0.00 → 0.89–0.99 on honeypots),
   matching a centralized model (0.99 vs 1.00) while keeping captures local,
   p < 0.0001.
3. **Sentinel threshold** — reliable network-wide propagation of a rare attack
   requires several deployments to observe it; a single sentinel is averaged
   away by FedAvg.
   - Honeypot sessions: reliable by ~3 sentinels
   - CICIDS2017 flows: needs ~5–7 sentinels (3 → 0.65, 5 → 0.91, 7 → 0.98)
   - The threshold effect **generalizes across data types**; flow data needs
     more sentinels, suggesting session features carry richer attack signal.
4. **Cross-protocol transfer** — SSH-trained model detects Telnet attacks at 0.80.
5. **Negative result** — simple augmentation (jitter/SMOTE) matches or beats
   GAN-based synthesis across 3–300 samples/deployment; sophisticated
   aggregation offers no gain. Collaboration, not complexity, is decisive.

## Scripts
- `parse_multihoneypot.py` / `parse_rich.py` — Cowrie logs → features
- `q1_experiment.py` — isolated vs federated held-out recovery (honeypots)
- `q1_gaps.py` — multi-class recovery + cross-protocol
- `q1_final.py` — sentinel sweep + paradigm comparison (honeypots)
- `cicids_final.py` — sentinel threshold on CICIDS2017 flow data
- `baselines.py` / `fair_privacy.py` / `crossover.py` — GAN vs simple augmentation

## Notes
- Large data files are gitignored — download the datasets separately via kagglehub.
- CICIDS2017 "sites" are simulated partitions (stated honestly); Dataset B's 10
  honeypots are physically distinct deployments.
