"""Figures 3 (isolated vs federated) and 5 (class imbalance) from existing data."""
import os, glob
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

# ---- FIGURE 3: isolated vs federated across held-out classes (honeypot) ----
# validated numbers from q1_gaps.py run: recon/malware/brute
classes=["recon","malware\ndownload","brute\nforce"]
iso=[0.000,0.000,0.000]; fed=[0.986,0.892,0.997]; ferr=[0.011,0.077,0.002]
# save provenance CSV too
pd.DataFrame({"class":["recon","malware_download","brute_force"],
              "isolated":iso,"federated":fed,"fed_std":ferr}).to_csv("results/isolated_vs_federated.csv",index=False)
x=np.arange(len(classes)); w=0.35
plt.figure(figsize=(6,4))
plt.bar(x-w/2,iso,w,label="isolated honeypot",color="#d1495b",edgecolor="black",linewidth=0.5)
plt.bar(x+w/2,fed,w,yerr=ferr,capsize=4,label="federated",color="#1b9e77",edgecolor="black",linewidth=0.5)
plt.xticks(x,classes); plt.ylabel("Held-out class recall")
plt.title("Isolation fails; federation recovers (honeypots)")
plt.ylim(0,1.1); plt.legend(fontsize=9); plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig_isolated_vs_federated.png",dpi=300)
plt.savefig("figures/fig_isolated_vs_federated.pdf"); plt.close()
print("saved figures/fig_isolated_vs_federated.{png,pdf}")

# ---- FIGURE 5: class imbalance across datasets (the "why this matters") ----
# read real distributions from the feature files on disk
fig,ax=plt.subplots(figsize=(6.5,4))
# honeypot (global_features.csv)
try:
    hp=pd.read_csv("global_features.csv")["attack_type"].value_counts()
    labels=hp.index.tolist(); vals=hp.values.tolist()
    ax.bar(range(len(vals)),vals,color="#1b9e77",edgecolor="black",linewidth=0.5)
    ax.set_yscale("log"); ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=8)
    ax.set_ylabel("sessions (log scale)")
    ax.set_title("Real honeypot class imbalance:\nrare attacks are near-invisible")
    for i,v in enumerate(vals): ax.text(i,v,str(v),ha="center",va="bottom",fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/fig_imbalance.png",dpi=300)
    plt.savefig("figures/fig_imbalance.pdf"); plt.close()
    print("saved figures/fig_imbalance.{png,pdf}")
except Exception as e:
    print("imbalance fig skipped:",e)
