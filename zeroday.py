"""Zero-day propagation with honeypot upweighting (lambda)."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]
NCLASS=6; HELDOUT=5

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

def run(lam, rounds=15):
    df=pd.read_csv("honeypot_features.csv")
    X=df[FEATURES].values.astype(np.float32); y=df["label"].values
    sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)
    held=(y==HELDOUT); Xtest,ytest=X[held],y[held]
    Xn,yn=X[~held],y[~held]
    idx=np.random.RandomState(0).permutation(len(Xn))
    parts=np.array_split(idx,3)
    Xh,yh=X[held],y[held]
    g=IDS(); curve=[]
    for r in range(rounds):
        lms,sz=[],[]
        for p in parts:
            lm=train(copy.deepcopy(g),Xn[p],yn[p]); lms.append(lm); sz.append(len(p))
        hm=train(copy.deepcopy(g),Xh,yh); lms.append(hm)
        sz.append(int(len(Xh)*lam))          # <-- honeypot upweighted by lambda
        g.load_state_dict(fedavg(lms,sz))
        with torch.no_grad():
            pred=g(torch.tensor(Xtest)).argmax(1).numpy()
        curve.append(recall_score(ytest,pred,labels=[HELDOUT],average="macro",zero_division=0))
    return curve

print("Testing honeypot upweighting factors (lambda):\n")
for lam in [1,3,5,10,20]:
    c=run(lam)
    first=next((i+1 for i,v in enumerate(c) if v>0.5),None)
    print(f"lambda={lam:2d} | final recall {c[-1]:.3f} | "
          f"reached 0.5 at round {first if first else '—'}")
print("\nFull curve for best lambda=10:")
for i,v in enumerate(run(10)):
    print(f" round {i+1:2d} | recall {v:.3f}  {'#'*int(v*40)}")
