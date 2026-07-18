"""Baseline comparison on the hard rare-attack task (n=5 seeds):
  - isolated, federated(FedAvg), federated+GAN (ours)
  - + SMOTE augmentation (competitor to GAN)
  - + FedProx aggregation (competitor federated optimizer)
Reports mean ± std for all. Honest: if SMOTE ties GAN, we say so."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

FEATURES=["login_attempts","login_success","login_failed","cmd_count","cmd_failed","downloads"]
RARE=3; PER_HP=3; NOISE=0.6

base=pd.read_csv("real_features.csv")
base=base[base["attack_type"].isin(["benign","bruteforce","malware_download"])].copy()

class IDS(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,6))
    def forward(s,x): return s.net(x)
class Gen(nn.Module):
    def __init__(s,z=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,6))
    def forward(s,x): return s.net(x)
class Disc(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(6,64),nn.LeakyReLU(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(s,x): return s.net(x)

def train_ids(m,X,y,ep=40,mu=0.0,gstate=None):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep):
        opt.zero_grad(); loss=ce(m(Xt),yt)
        if mu>0 and gstate is not None:  # FedProx proximal term
            prox=0.0
            for n,p in m.named_parameters(): prox=prox+((p-gstate[n])**2).sum()
            loss=loss+(mu/2)*prox
        loss.backward(); opt.step()
    return m
def train_gan(data,ep=400,z=16,b=8):
    G,D=Gen(z),Disc(); oG=torch.optim.Adam(G.parameters(),2e-4); oD=torch.optim.Adam(D.parameters(),2e-4)
    bce=nn.BCELoss(); dt=torch.tensor(data)
    for _ in range(ep):
        i=torch.randint(0,len(dt),(min(b,len(dt)),)); real=dt[i]
        f=G(torch.randn(len(i),z)).detach()
        lD=bce(D(real),torch.ones(len(i),1))+bce(D(f),torch.zeros(len(i),1)); oD.zero_grad(); lD.backward(); oD.step()
        g=G(torch.randn(len(i),z)); lG=bce(D(g),torch.ones(len(i),1)); oG.zero_grad(); lG.backward(); oG.step()
    return G
def fedavg(ms,sz):
    tot=sum(sz); g=copy.deepcopy(ms[0].state_dict())
    for k in g: g[k]=sum(ms[i].state_dict()[k]*sz[i] for i in range(len(ms)))/tot
    return g

def run(seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    benign=base[base.attack_type=="benign"].sample(6000,random_state=seed)
    df=pd.concat([benign,base[base.attack_type!="benign"]]).sample(frac=1,random_state=seed).reset_index(drop=True)
    X=df[FEATURES].values.astype(np.float32); y=df["label"].values
    X=StandardScaler().fit_transform(X).astype(np.float32)
    X=X+rng.normal(0,NOISE,X.shape).astype(np.float32)
    rare=X[y==RARE]; rng.shuffle(rare); rare_test=rare[1000:2000]
    common=X[y!=RARE]; yc=y[y!=RARE]
    ci=rng.permutation(len(common)); cparts=np.array_split(ci,5)
    hps=[]
    for k in range(5):
        r=rare[k*PER_HP:(k+1)*PER_HP]
        Xk=np.vstack([common[cparts[k]],r]); yk=np.concatenate([yc[cparts[k]],np.full(len(r),RARE)])
        hps.append((Xk,yk,r))
    def rr(g):
        with torch.no_grad():
            return recall_score(np.full(len(rare_test),RARE),
                g(torch.tensor(rare_test)).argmax(1).numpy(),labels=[RARE],average="macro",zero_division=0)
    out={}
    # isolated
    out["solo"]=rr(train_ids(IDS(),hps[0][0],hps[0][1],ep=80))
    # FedAvg
    g=IDS()
    for _ in range(10):
        lms,sz=[],[]
        for (Xk,yk,_) in hps: lms.append(train_ids(copy.deepcopy(g),Xk,yk)); sz.append(len(Xk))
        g.load_state_dict(fedavg(lms,sz))
    out["fed"]=rr(g)
    # FedProx
    g=IDS()
    for _ in range(10):
        gs={n:p.detach().clone() for n,p in g.named_parameters()}
        lms,sz=[],[]
        for (Xk,yk,_) in hps: lms.append(train_ids(copy.deepcopy(g),Xk,yk,mu=0.1,gstate=gs)); sz.append(len(Xk))
        g.load_state_dict(fedavg(lms,sz))
    out["fedprox"]=rr(g)
    # SMOTE (pool rare samples across honeypots, SMOTE-oversample, augment)
    allr=np.vstack([r for (_,_,r) in hps])
    # build a small set for SMOTE: rare + some common
    Xsm=np.vstack([allr, common[cparts[0]][:200]])
    ysm=np.concatenate([np.full(len(allr),1), np.zeros(200)])
    try:
        sm=SMOTE(k_neighbors=min(5,len(allr)-1),random_state=seed)
        Xres,yres=sm.fit_resample(Xsm,ysm)
        smote_rare=Xres[yres==1][len(allr):]   # the synthetic ones
    except Exception:
        smote_rare=allr
    g=IDS(); sy=np.full(len(smote_rare),RARE)
    for _ in range(10):
        lms,sz=[],[]
        for (Xk,yk,_) in hps:
            Xa=np.vstack([Xk,smote_rare]); ya=np.concatenate([yk,sy])
            lms.append(train_ids(copy.deepcopy(g),Xa,ya)); sz.append(len(Xa))
        g.load_state_dict(fedavg(lms,sz))
    out["smote"]=rr(g)
    # GAN (ours)
    g=IDS(); G=train_gan(allr)
    with torch.no_grad(): synth=G(torch.randn(300,16)).numpy().astype(np.float32)
    sy=np.full(len(synth),RARE)
    for _ in range(10):
        lms,sz=[],[]
        for (Xk,yk,_) in hps:
            Xa=np.vstack([Xk,synth]); ya=np.concatenate([yk,sy])
            lms.append(train_ids(copy.deepcopy(g),Xa,ya)); sz.append(len(Xa))
        g.load_state_dict(fedavg(lms,sz))
    out["gan"]=rr(g)
    return out

seeds=[0,1,2,3,4]; keys=["solo","fed","fedprox","smote","gan"]
acc={k:[] for k in keys}
for s in seeds:
    o=run(s)
    for k in keys: acc[k].append(o[k])
    print(f"seed {s}: "+" ".join(f"{k}={o[k]:.3f}" for k in keys))

names={"solo":"isolated honeypot","fed":"FedAvg","fedprox":"FedProx",
       "smote":"FedAvg + SMOTE","gan":"FedAvg + GAN (ours)"}
print(f"\n{'method':28s} mean ± std  (n=5)")
print("-"*50)
for k in keys:
    a=np.array(acc[k]); print(f"{names[k]:28s} {a.mean():.3f} ± {a.std():.3f}")

# --- save results for reproducible figure ---
import os as _os, csv as _csv
_os.makedirs("results", exist_ok=True)
with open("results/baselines.csv","w",newline="") as _f:
    _w=_csv.writer(_f); _w.writerow(["method","recall_mean","recall_std"])
    for _k in keys:
        _a=np.array(acc[_k]); _w.writerow([names[_k], round(float(_a.mean()),4), round(float(_a.std()),4)])
print("saved results/baselines.csv")
