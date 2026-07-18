"""Figure 4: baseline comparison (the negative result) from results/baselines.csv"""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
b=pd.read_csv("results/baselines.csv")
plt.figure(figsize=(6.5,4))
colors=["#888880","#888880","#d1495b","#2a78d6","#1b9e77"][:len(b)]
bars=plt.bar(b["method"],b["recall_mean"],yerr=b["recall_std"],capsize=4,
             color=colors,edgecolor="black",linewidth=0.5)
for bar,v in zip(bars,b["recall_mean"]):
    plt.text(bar.get_x()+bar.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=8)
plt.ylabel("rare-attack recall (n=5)")
plt.title("Simple augmentation matches complex synthesis\n(GAN offers no advantage)")
plt.xticks(rotation=25,ha="right",fontsize=8)
plt.ylim(0,1.1); plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig_baselines.png",dpi=300)
plt.savefig("figures/fig_baselines.pdf"); plt.close()
print("saved figures/fig_baselines.{png,pdf}")
