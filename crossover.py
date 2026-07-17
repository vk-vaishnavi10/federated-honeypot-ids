"""Crossover: at what samples-per-honeypot does GAN start beating jitter?
Sweep PER_HP = 3,10,30,100,300 on rich features. n=3 seeds each."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
warnings.filterwarnings("ignore")

RARE=3; NOISE=0.4
base=pd.read_csv("rich_features.csv")
base=base[base["attack_type"].isin(["benign","bruteforce","malware_download"])].copy()
FEATURES=[c for c in base.columns if c not in ["attack_type","label"]]
D=len(FEATURES)

class IDS(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,6))
    def forward(s,x): return s.net(x)
class Gen(nn.Module):
    def __init__(s,z=32):
        super().__init__(); s.net=nn.Sequential(nn.Linear(z,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,x): return s.net(x)
class Disc(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,128),nn.LeakyReLU(0.2),nn.Dropout(0.3),nn.Linear(128,1),nn.Sigmoid())
    def forward(s,x): return s.net(x)

def train_ids(m,X,y,ep=50):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
    return m
def train_gan(data,ep=500,z=32,b=16):
    G,Dd=Gen(z),Disc(); oG=torch.optim.Adam(G.parameters(),2e-4); oD=torch.optim.Adam(Dd.parameters(),2e-4)
    bce=nn.BCELoss(); dt=torch.tensor(data)
    for _ in range(ep):
        i=torch.randint(0,len(dt),(min(b,len(dt)),)); real=dt[i]
        f=G(torch.randn(len(i),z)).detach()
        lD=bce(Dd(real),torch.ones(len(i),1))+bce(Dd(f),torch.zeros(len(i),1)); oD.zero_grad(); lD.backward(); oD.step()
        g=G(torch.randn(len(i),z)); lG=bce(Dd(g),torch.ones(len(i),1)); oG.zero_grad(); lG.backward(); oG.step()
    return G
def fedavg(ms,sz):
    tot=sum(sz); g=copy.deepcopy(ms[0].state_dict())
    for k in g: g[k]=sum(ms[i].state_dict()[k]*sz[i] for i in range(len(ms)))/tot
    return g
def jitter(r,n,seed):
    rng=np.random.RandomState(seed); idx=rng.randint(len(r),size=n)
    return (r[idx]+rng.normal(0,0.3,(n,r.shape[1]))).astype(np.float32)
def gan(r,n,seed):
    if len(r)<2: return np.repeat(r,n,axis=0)
    G=train_gan(r)
    with torch.no_grad(): return G(torch.randn(n,32)).numpy().astype(np.float32)

def run(per_hp,seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    benign=base[base.attack_type=="benign"].sample(6000,random_state=seed)
    df=pd.concat([benign,base[base.attack_type!="benign"]]).sample(frac=1,random_state=seed).reset_index(drop=True)
    X=df[FEATURES].values.astype(np.float32); y=df["label"].values
    X=StandardScaler().fit_transform(X).astype(np.float32)
    X=X+rng.normal(0,NOISE,X.shape).astype(np.float32)
    rare=X[y==RARE]; rng.shuffle(rare); rare_test=rare[2000:3000]
    common=X[y!=RARE]; yc=y[y!=RARE]
    ci=rng.permutation(len(common)); cparts=np.array_split(ci,5)
    hps=[]
    for k in range(5):
        r=rare[k*per_hp:(k+1)*per_hp]
        Xk=np.vstack([common[cparts[k]],r]); yk=np.concatenate([yc[cparts[k]],np.full(len(r),RARE)])
        hps.append((Xk,yk,r))
    def rr(g):
        with torch.no_grad():
            return recall_score(np.full(len(rare_test),RARE),
                g(torch.tensor(rare_test)).argmax(1).numpy(),labels=[RARE],average="macro",zero_division=0)
    def fed(fn):
        synth=np.vstack([fn(r,60,seed) for (_,_,r) in hps]); sy=np.full(len(synth),RARE)
        g=IDS()
        for _ in range(8):
            lms,sz=[],[]
            for (Xk,yk,_) in hps:
                Xa=np.vstack([Xk,synth]); ya=np.concatenate([yk,sy])
                lms.append(train_ids(copy.deepcopy(g),Xa,ya)); sz.append(len(Xa))
            g.load_state_dict(fedavg(lms,sz))
        return rr(g)
    return fed(jitter),fed(gan)

print(f"{'per_hp':>7} | {'jitter':>14} | {'GAN':>14} | winner")
print("-"*54)
for per_hp in [3,10,30,100,300]:
    js,gs=[],[]
    for s in [0,1,2]:
        j,g=run(per_hp,s); js.append(j); gs.append(g)
    jm,gm=np.mean(js),np.mean(gs)
    win="GAN" if gm>jm else "jitter"
    print(f"{per_hp:>7} | {jm:.3f} ± {np.std(js):.3f} | {gm:.3f} ± {np.std(gs):.3f} | {win}")
