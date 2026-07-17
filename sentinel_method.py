"""NOVEL METHOD: Class-conditional sentinel aggregation.
Problem: standard FedAvg drowns a SINGLE honeypot's rare-attack knowledge (recall 0.00).
Naive fix (upweight sentinel) causes catastrophic forgetting.
Our method: preserve the sentinel's rare-class output-layer knowledge while
averaging the rest normally. Goal: single-sentinel recall high + known-class preserved.
Real 10-honeypot data, n=5 seeds."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score
warnings.filterwarnings("ignore")

df=pd.read_csv("global_features.csv")
df=df[df["attack_type"].isin(["benign","bruteforce","recon","malware_download"])].copy()
FEATURES=[c for c in df.columns if c not in ["honeypot","protocol","attack_type","label"]]
D=len(FEATURES); HELD=2  # recon, given to ONE sentinel only

class IDS(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,6))
    def forward(s,x): return s.net(x)
def train(m,X,y,ep=40):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
    return m

def fedavg(states,sz):
    tot=sum(sz); g=copy.deepcopy(states[0])
    for k in g: g[k]=sum(states[i][k]*sz[i] for i in range(len(states)))/tot
    return g

def sentinel_agg(states, sz, sentinel_idx, held_class):
    """Class-conditional sentinel aggregation:
    Standard FedAvg for all params EXCEPT the final-layer row for the held class,
    which is taken preferentially from the sentinel (who alone has learned it)."""
    g=fedavg(states,sz)
    # final layer weight/bias are net.4.weight / net.4.bias (index 4 in Sequential)
    for k in g:
        if k.endswith("4.weight"):
            g[k][held_class]=states[sentinel_idx][k][held_class]   # preserve sentinel's rare-class row
        if k.endswith("4.bias"):
            g[k][held_class]=states[sentinel_idx][k][held_class]
    return g

def run(seed, method):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    X=df[FEATURES].values.astype(np.float32); sc=StandardScaler().fit(X)
    hps=sorted(df["honeypot"].unique())
    sentinel=rng.choice(hps)                       # ONE sentinel
    held=df[df.label==HELD]; test=held.sample(min(1000,len(held)),random_state=seed); tidx=set(test.index)
    # known-class test (from non-held classes)
    known=df[(df.label!=HELD)&(~df.index.isin(tidx))].sample(2000,random_state=seed)
    clients=[]; sent_i=None
    for j,hp in enumerate(hps):
        cd=df[(df.honeypot==hp)&(~df.index.isin(tidx))]
        if hp!=sentinel: cd=cd[cd.label!=HELD]      # only sentinel sees held class
        else: sent_i=len(clients)
        if len(cd)>3000: cd=cd.sample(3000,random_state=seed)
        if len(cd)<10: continue
        clients.append((sc.transform(cd[FEATURES].values.astype(np.float32)).astype(np.float32),cd["label"].values))
    Xt=sc.transform(test[FEATURES].values.astype(np.float32)).astype(np.float32); yt=test["label"].values
    Xk=sc.transform(known[FEATURES].values.astype(np.float32)).astype(np.float32); yk=known["label"].values
    g=IDS()
    for _ in range(12):
        states,sz=[],[]
        for (Xc,yc) in clients:
            m=train(copy.deepcopy(g),Xc,yc); states.append(m.state_dict()); sz.append(len(Xc))
        if method=="fedavg": g.load_state_dict(fedavg(states,sz))
        elif method=="upweight":
            sz2=list(sz); sz2[sent_i]*=10; g.load_state_dict(fedavg(states,sz2))
        elif method=="sentinel": g.load_state_dict(sentinel_agg(states,sz,sent_i,HELD))
    with torch.no_grad():
        zd=recall_score(yt,g(torch.tensor(Xt)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)
        ka=accuracy_score(yk,g(torch.tensor(Xk)).argmax(1).numpy())
    return zd,ka

print("SINGLE-sentinel setting (rare attack at exactly 1 of 10 honeypots)\n")
print(f"{'method':30s} zero-day recall | known-class acc")
print("-"*62)
for method,name in [("fedavg","FedAvg (standard)"),("upweight","naive upweight (λ=10)"),("sentinel","sentinel-aware (ours)")]:
    zds,kas=[],[]
    for s in range(5):
        z,k=run(s,method); zds.append(z); kas.append(k)
    zds,kas=np.array(zds),np.array(kas)
    print(f"{name:30s}    {zds.mean():.3f}±{zds.std():.3f}  |   {kas.mean():.3f}±{kas.std():.3f}")
