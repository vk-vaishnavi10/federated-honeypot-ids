"""Generate paper figures from results/*.csv (reproducible - no hardcoded numbers)."""
import os, glob
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# --- FIGURE 1: sentinel threshold across datasets (reads all sentinel_*.csv) ---
frames=[pd.read_csv(f) for f in sorted(glob.glob("results/sentinel_*.csv"))]
sent=pd.concat(frames, ignore_index=True)
colors={"Honeypot":"#1b9e77","CICIDS2017":"#2a78d6","BoT-IoT":"#d95f02"}
markers={"Honeypot":"o","CICIDS2017":"s","BoT-IoT":"^"}
plt.figure(figsize=(6,4))
for ds in sent["dataset"].unique():
    d=sent[sent.dataset==ds].sort_values("sentinels")
    plt.errorbar(d["sentinels"],d["recall_mean"],yerr=d["recall_std"],
                 marker=markers.get(ds,"o"),color=colors.get(ds,"#333"),
                 label=ds,linewidth=2,markersize=7,capsize=4)
plt.xlabel("Number of sentinel sites (that observed the rare attack)")
plt.ylabel("Federated recall of held-out rare attack")
plt.title("Sentinel threshold generalizes across data types")
plt.ylim(-0.02,1.05); plt.grid(alpha=0.3); plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig_sentinel_threshold.png",dpi=300)
plt.savefig("figures/fig_sentinel_threshold.pdf")
plt.close()
print("saved figures/fig_sentinel_threshold.{png,pdf}")

# --- FIGURE 2: paradigm comparison (reads paradigm.csv) ---
if os.path.exists("results/paradigm.csv"):
    p=pd.read_csv("results/paradigm.csv")
    label_map={"centralized":"Centralized\n(pools raw data)","local_only":"Local-only\n(isolated)","federated":"Federated\n(ours)"}
    p["disp"]=p["paradigm"].map(label_map)
    plt.figure(figsize=(5.5,4))
    bars=plt.bar(p["disp"],p["recall_mean"],color=["#888880","#d1495b","#1b9e77"],edgecolor="black",linewidth=0.5)
    for b,v in zip(bars,p["recall_mean"]):
        plt.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.3f}",ha="center",fontsize=9)
    plt.ylabel("Held-out reconnaissance recall")
    plt.title("Federation matches centralization,\npreserving data locality")
    plt.ylim(0,1.1); plt.grid(axis="y",alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig_paradigm.png",dpi=300)
    plt.savefig("figures/fig_paradigm.pdf")
    plt.close()
    print("saved figures/fig_paradigm.{png,pdf}")

print("\nAll figures in ./figures/ - generated from results/*.csv")
