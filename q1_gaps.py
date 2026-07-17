"""Fill Q1 gaps: (1) multiple held-out classes, (2) cross-protocol generalization.
Real 10-honeypot data, n=5 seeds, t-tests."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from scipy import stats
warnings.filterwarnings("ignore")

df=pd.read_csv("global_features.csv")
FEATURES=[c for c in df.columns if c not in ["honeypot","protocol","attack_type","label"]]
D=len(FEATURES)

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

# ===== EXPERIMENT 1: multiple held-out classes =====
def heldout_exp(HELD, seed, classes):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    d=df[df["attack_type"].isin(classes)].copy()
    X=d[FEATURES].values.astype(np.float32); sc=StandardScaler().fit(X)
    hps=sorted(d["honeypot"].unique())
    sentinels=set(rng.choice(hps,3,replace=False))
    held_all=d[d.label==HELD]
    if len(held_all)<50: return None
    test=held_all.sample(min(1000,len(held_all)),random_state=seed); tidx=set(test.index)
    clients=[]
    for hp in hps:
        cd=d[(d.honeypot==hp)&(~d.index.isin(tidx))]
        if hp not in sentinels: cd=cd[cd.label!=HELD]
        if len(cd)>3000: cd=cd.sample(3000,random_state=seed)
        if len(cd)<10: continue
        clients.append((sc.transform(cd[FEATURES].values.astype(np.float32)).astype(np.float32),cd["label"].values,hp in sentinels))
    Xt=sc.transform(test[FEATURES].values.astype(np.float32)).astype(np.float32); yt=test["label"].values
    def rr(g):
        with torch.no_grad(): return recall_score(yt,g(torch.tensor(Xt)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)
    blind=[c for c in clients if not c[2]][0]
    solo=rr(train(IDS(),blind[0],blind[1],ep=60))
    g=IDS()
    for _ in range(10):
        lms,sz=[],[]
        for (Xc,yc,_) in clients: lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
        g.load_state_dict(fedavg(lms,sz))
    return solo,rr(g)

print("=== EXPERIMENT 1: held-out class recovery (isolated vs federated) ===")
classes=["benign","bruteforce","recon","malware_download"]
for HELD,name in [(2,"recon"),(3,"malware_download"),(1,"bruteforce")]:
    ss,ff=[],[]
    for s in range(5):
        r=heldout_exp(HELD,s,classes)
        if r: ss.append(r[0]); ff.append(r[1])
    if ss:
        ss,ff=np.array(ss),np.array(ff)
        t,p=stats.ttest_rel(ff,ss)
        print(f"{name:16s} isolated {ss.mean():.3f}±{ss.std():.3f} | federated {ff.mean():.3f}±{ff.std():.3f} | p={p:.4f}")

# ===== EXPERIMENT 2: cross-protocol (train SSH, test Telnet) =====
print("\n=== EXPERIMENT 2: cross-protocol generalization ===")
d=df[df["attack_type"].isin(classes)].copy()
ssh=d[d.protocol=="ssh"]; tel=d[d.protocol=="telnet"]
print(f"SSH sessions: {len(ssh)}, Telnet sessions: {len(tel)}")
if len(tel)>100:
    accs=[]
    for s in range(5):
        rng=np.random.RandomState(s); torch.manual_seed(s)
        Xtr=ssh[FEATURES].values.astype(np.float32); ytr=ssh["label"].values
        sc=StandardScaler().fit(Xtr)
        tr=ssh.sample(min(20000,len(ssh)),random_state=s)
        m=train(IDS(),sc.transform(tr[FEATURES].values.astype(np.float32)).astype(np.float32),tr["label"].values,ep=60)
        Xte=sc.transform(tel[FEATURES].values.astype(np.float32)).astype(np.float32)
        with torch.no_grad(): pred=m(torch.tensor(Xte)).argmax(1).numpy()
        from sklearn.metrics import accuracy_score
        accs.append(accuracy_score(tel["label"].values,pred))
    accs=np.array(accs)
    print(f"train SSH -> test Telnet accuracy: {accs.mean():.3f} ± {accs.std():.3f}")
