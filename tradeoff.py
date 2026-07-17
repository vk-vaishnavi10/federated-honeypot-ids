"""The cost of honeypot upweighting: does high lambda hurt overall accuracy?
Measures held-out-class recall AND accuracy on the other 5 classes."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score

FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]
HELD=5  # persistence

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

df=pd.read_csv("honeypot_features.csv")
X=df[FEATURES].values.astype(np.float32); y=df["label"].values
sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)

hmask=(y==HELD); Xn,yn=X[~hmask],y[~hmask]
Xh,yh=X[hmask],y[hmask]
# holdout test for the KNOWN classes (from normal-client data)
ri=np.random.RandomState(1).permutation(len(Xn))
te=ri[:len(Xn)//5]; trn=ri[len(Xn)//5:]
Xk_test,yk_test=Xn[te],yn[te]         # known-class test
Xn_tr,yn_tr=Xn[trn],yn[trn]
parts=np.array_split(np.random.RandomState(0).permutation(len(Xn_tr)),3)

print(f"{'λ':>4} | zero-day recall | known-class acc | verdict")
print("-"*56)
for lam in [1,3,5,10,20,50]:
    g=IDS()
    for r in range(10):
        lms,sz=[],[]
        for p in parts:
            lms.append(train(copy.deepcopy(g),Xn_tr[p],yn_tr[p])); sz.append(len(p))
        lms.append(train(copy.deepcopy(g),Xh,yh)); sz.append(int(len(Xh)*lam))
        g.load_state_dict(fedavg(lms,sz))
    with torch.no_grad():
        zd=recall_score(y[hmask],g(torch.tensor(X[hmask])).argmax(1).numpy(),
                        labels=[HELD],average="macro",zero_division=0)
        ka=accuracy_score(yk_test,g(torch.tensor(Xk_test)).argmax(1).numpy())
    verdict="good" if (zd>0.8 and ka>0.85) else ("zero-day fails" if zd<=0.8 else "known-class hurt")
    print(f"{lam:>4} |     {zd:.3f}       |     {ka:.3f}      | {verdict}")
