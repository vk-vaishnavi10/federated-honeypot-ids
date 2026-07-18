"""Two final Q1 experiments:
 1. Sentinel sweep (vary # honeypots that see the rare class: 1,2,3,5)
    -> shows the result isn't cherry-picked to 3 sentinels.
 2. Paradigm baselines on the SAME held-out task:
    - centralized (all data pooled - the pre-FL baseline, violates privacy)
    - local-only (isolated honeypot)
    - federated (ours)
Real 10-honeypot data, n=5 seeds."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
warnings.filterwarnings("ignore")

df=pd.read_csv("global_features.csv")
df=df[df["attack_type"].isin(["benign","bruteforce","recon","malware_download"])].copy()
FEATURES=[c for c in df.columns if c not in ["honeypot","protocol","attack_type","label"]]
D=len(FEATURES); HELD=2  # recon

class IDS(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,6))
    def forward(s,x): return s.net(x)
def train(m,X,y,ep=40):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
    return m
def fedavg(ms,sz):
    tot=sum(sz); g=copy.deepcopy(ms[0].state_dict())
    for k in g: g[k]=sum(ms[i].state_dict()[k]*sz[i] for i in range(len(ms)))/tot
    return g

def setup(seed,n_sentinel):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    X=df[FEATURES].values.astype(np.float32); sc=StandardScaler().fit(X)
    hps=sorted(df["honeypot"].unique())
    sentinels=set(rng.choice(hps,n_sentinel,replace=False))
    held=df[df.label==HELD]; test=held.sample(min(1000,len(held)),random_state=seed); tidx=set(test.index)
    clients=[]
    for hp in hps:
        cd=df[(df.honeypot==hp)&(~df.index.isin(tidx))]
        if hp not in sentinels: cd=cd[cd.label!=HELD]
        if len(cd)>3000: cd=cd.sample(3000,random_state=seed)
        if len(cd)<10: continue
        clients.append((sc.transform(cd[FEATURES].values.astype(np.float32)).astype(np.float32),cd["label"].values,hp in sentinels))
    Xt=sc.transform(test[FEATURES].values.astype(np.float32)).astype(np.float32); yt=test["label"].values
    return clients,Xt,yt
def rr(g,Xt,yt):
    with torch.no_grad(): return recall_score(yt,g(torch.tensor(Xt)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)

# ===== 1. SENTINEL SWEEP =====
print("=== SENTINEL SWEEP: how many honeypots must see the rare attack? ===")
print(f"{'#sentinels':>11} | federated recall")
print("-"*34)
import os,csv
os.makedirs("results",exist_ok=True)
_sent_rows=[]
for ns in [1,2,3,5]:
    vals=[]
    for s in range(5):
        clients,Xt,yt=setup(s,ns)
        g=IDS()
        for _ in range(10):
            lms,sz=[],[]
            for (Xc,yc,_) in clients: lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
            g.load_state_dict(fedavg(lms,sz))
        vals.append(rr(g,Xt,yt))
    v=np.array(vals); print(f"{ns:>11} | {v.mean():.3f} ± {v.std():.3f}")
    _sent_rows.append({"dataset":"Honeypot","sentinels":ns,"recall_mean":round(float(v.mean()),4),"recall_std":round(float(v.std()),4)})

# ===== 2. PARADIGM BASELINES =====
with open("results/sentinel_honeypot.csv","w",newline="") as _f:
    _w=csv.DictWriter(_f,fieldnames=["dataset","sentinels","recall_mean","recall_std"]); _w.writeheader(); _w.writerows(_sent_rows)
print("saved results/sentinel_honeypot.csv")
print("\n=== PARADIGM COMPARISON (held-out recon, 3 sentinels) ===")
cen,loc,fed=[],[],[]
for s in range(5):
    clients,Xt,yt=setup(s,3)
    # centralized: pool ALL client data (violates privacy) - the classic pre-FL baseline
    Xall=np.vstack([c[0] for c in clients]); yall=np.concatenate([c[1] for c in clients])
    cen.append(rr(train(IDS(),Xall,yall,ep=60),Xt,yt))
    # local-only: one blind honeypot
    blind=[c for c in clients if not c[2]][0]
    loc.append(rr(train(IDS(),blind[0],blind[1],ep=60),Xt,yt))
    # federated (ours)
    g=IDS()
    for _ in range(10):
        lms,sz=[],[]
        for (Xc,yc,_) in clients: lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
        g.load_state_dict(fedavg(lms,sz))
    fed.append(rr(g,Xt,yt))
cen,loc,fed=np.array(cen),np.array(loc),np.array(fed)
print(f"{'paradigm':34s} recall | privacy")
print("-"*54)
print(f"{'centralized (pool all data)':34s} {cen.mean():.3f} | violated")
print(f"{'local-only (isolated honeypot)':34s} {loc.mean():.3f} | preserved (useless)")
print(f"{'federated (ours)':34s} {fed.mean():.3f} | preserved")

# --- save results for reproducible figures ---
import os, csv
os.makedirs("results", exist_ok=True)
with open("results/paradigm.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["paradigm","recall_mean","recall_std"])
    w.writerow(["centralized", round(float(cen.mean()),4), round(float(cen.std()),4)])
    w.writerow(["local_only",  round(float(loc.mean()),4), round(float(loc.std()),4)])
    w.writerow(["federated",   round(float(fed.mean()),4), round(float(fed.std()),4)])
print("saved results/paradigm.csv")
