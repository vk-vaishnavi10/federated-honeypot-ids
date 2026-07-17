"""Sweep every attack class as the held-out zero-day, across lambda values.
Produces the core results table: propagation threshold per attack type."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]
NAMES={1:"bruteforce",2:"recon",3:"malware_dl",4:"miner",5:"persistence"}

class IDS(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,6))
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

def run(X,y,held,lam,rounds=10):
    hmask=(y==held); Xtest,ytest=X[hmask],y[hmask]
    Xn,yn=X[~hmask],y[~hmask]
    idx=np.random.RandomState(0).permutation(len(Xn))
    parts=np.array_split(idx,3)
    Xh,yh=X[hmask],y[hmask]
    g=IDS()
    for r in range(rounds):
        lms,sz=[],[]
        for p in parts:
            lms.append(train(copy.deepcopy(g),Xn[p],yn[p])); sz.append(len(p))
        lms.append(train(copy.deepcopy(g),Xh,yh)); sz.append(int(len(Xh)*lam))
        g.load_state_dict(fedavg(lms,sz))
    with torch.no_grad():
        pred=g(torch.tensor(Xtest)).argmax(1).numpy()
    return recall_score(ytest,pred,labels=[held],average="macro",zero_division=0)

df=pd.read_csv("honeypot_features.csv")
X=df[FEATURES].values.astype(np.float32); y=df["label"].values
sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)

lams=[1,3,5,10,20]
print(f"{'attack type':14s} " + " ".join(f"λ={l:<4d}" for l in lams) + " | λ* (first ≥0.8)")
print("-"*72)
for cls in [1,2,3,4,5]:
    recs=[run(X,y,cls,l) for l in lams]
    star=next((l for l,r in zip(lams,recs) if r>=0.8), None)
    row=" ".join(f"{r:.2f} " for r in recs)
    print(f"{NAMES[cls]:14s} {row} | {star if star else '>20'}")
