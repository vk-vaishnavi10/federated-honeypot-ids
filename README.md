# Federated Threat Intelligence Across Geo-Distributed Honeypots

An empirical study of collaborative rare-attack detection across distributed
deployments without sharing raw captures. Validated on honeypot session data
and network-flow / IoT benchmarks.

## Datasets
- Dataset A: single-deployment Cowrie capture (~44k real sessions)
- Dataset B: 10 physically-distinct SSH/Telnet honeypots, multiple global regions (~518k sessions)
- Dataset C: CICIDS2017 network flows (~2.5M flows) - generalization test
- Dataset D: Bot-IoT (~3.6M IoT flow records) - generalization test

## Key findings (real data, n=5 seeds, statistically tested)
1. Isolated deployments detect 0% of an attack class they haven't observed.
2. Federated aggregation recovers detection (0.00 -> 0.89-0.99 on honeypots),
   matching a centralized model (0.99 vs 1.00) while keeping captures local
   (paired t-test p < 0.0001, large effect size).
3. Sentinel threshold: reliable propagation needs several deployments to observe
   the rare attack; a single sentinel is averaged away by FedAvg. Generalizes
   across data types but differs: honeypots ~3 sentinels, CICIDS2017 / Bot-IoT ~5-7.
4. Cross-protocol transfer: SSH-trained model detects Telnet attacks at 0.80.
5. Negative result: simple augmentation (jitter/SMOTE) matches or beats GAN
   synthesis across 3-300 samples/deployment; FedProx offers no gain.
   Collaboration, not algorithmic complexity, is decisive.

## Reproducing
Experiments write results to results/*.csv; figures are generated from those CSVs.
Run q1_final.py, cicids_final.py, botiot_experiment.py, baselines.py, then
make_figures.py / make_figures2.py / make_figure4.py.

## Notes
- Large data files are gitignored; download datasets via kagglehub.
- Dataset B honeypots are physically distinct; CICIDS2017 and Bot-IoT sites are
  simulated partitions (stated honestly in the paper).
- Empirical systems study using standard FedAvg; contribution is the findings
  and characterization, not a new aggregation algorithm.
