"""Q1 experiment on 10 REAL honeypots. Hold out a rare attack class from
most honeypots; only a few 'sentinel' honeypots see it. Measure whether
federated pooling recovers detection at the blind honeypots.
Real deployments, real non-IID, no synthetic difficulty. n=5 seeds + t-test."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from scipy import stats
warnings.filterwarnings("ignore")

df=pd.read_csv("global_features.csv")
# use classes with enough real samples
df=df[df["attack_type"].isin(["benign","bruteforce","recon","malware_download"])].copy()
FEATURES=[c for c in df.columns if c not in ["honeypot","protocol","attack_type","label"]]
D=len(FEATURES)
HELD=2  # recon = the rare attack only sentinels see
print(f"{D} features, holding out RECON (real, {(df.label==HELD).sum()} samples)\n")

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

def run(seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    d=df.copy()
    X=d[FEATURES].values.astype(np.float32)
    sc=StandardScaler().fit(X)
    hps=sorted(d["honeypot"].unique())
    # 3 sentinel honeypots keep recon; 7 blind honeypots have it removed
    sentinels=set(rng.choice(hps,3,replace=False))
    # held-out test = recon sessions pooled (measure if blind clients detect them)
    recon_all=d[d.label==HELD]
    recon_test=recon_all.sample(min(1000,len(recon_all)),random_state=seed)
    test_idx=set(recon_test.index)

    client_data=[]
    for hp in hps:
        cd=d[d.honeypot==hp]
        cd=cd[~cd.index.isin(test_idx)]
        if hp not in sentinels:
            cd=cd[cd.label!=HELD]   # blind: remove recon
        if len(cd)>3000: cd=cd.sample(3000,random_state=seed)
        Xc=sc.transform(cd[FEATURES].values.astype(np.float32)).astype(np.float32)
        client_data.append((Xc,cd["label"].values,hp in sentinels))

    Xtest=sc.transform(recon_test[FEATURES].values.astype(np.float32)).astype(np.float32)
    ytest=recon_test["label"].values
    def rr(g):
        with torch.no_grad():
            return recall_score(ytest,g(torch.tensor(Xtest)).argmax(1).numpy(),
                                labels=[HELD],average="macro",zero_division=0)

    # A: isolated blind honeypot (never saw recon)
    blind=[c for c in client_data if not c[2]][0]
    solo=rr(train(IDS(),blind[0],blind[1],ep=60))

    # B: federated across all 10 real honeypots
    g=IDS()
    for _ in range(10):
        lms,sz=[],[]
        for (Xc,yc,_) in client_data:
            lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
        g.load_state_dict(fedavg(lms,sz))
    fed=rr(g)
    return solo,fed

solos,feds=[],[]
for s in [0,1,2,3,4]:
    a,b=run(s); solos.append(a); feds.append(b)
    print(f"seed {s}: isolated={a:.3f}  federated={b:.3f}")

solos,feds=np.array(solos),np.array(feds)
print(f"\n{'setup':32s} recall (mean ± std)")
print("-"*52)
print(f"{'isolated blind honeypot':32s} {solos.mean():.3f} ± {solos.std():.3f}")
print(f"{'federated (10 real honeypots)':32s} {feds.mean():.3f} ± {feds.std():.3f}")
t,p=stats.ttest_rel(feds,solos)
print(f"\npaired t-test: t={t:.2f}, p={p:.4f}  ({'significant' if p<0.05 else 'NOT significant'})")
