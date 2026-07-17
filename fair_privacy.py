"""Fair head-to-head under IDENTICAL privacy model:
each honeypot synthesizes rare samples LOCALLY from its own 3 samples,
shares only synthetic. GAN vs SMOTE vs Gaussian-jitter, n=5 seeds."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

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

def train_ids(m,X,y,ep=40):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
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

def local_smote(r,n,seed):
    # r = one honeypot's 3 rare samples. SMOTE needs >=2 and k<len.
    if len(r)<2: return np.repeat(r,n,axis=0)
    try:
        Xtmp=np.vstack([r, r.mean(0,keepdims=True)+np.zeros((1,r.shape[1]),dtype=np.float32)])
        # trick: SMOTE between the few real points
        k=min(len(r)-1,2)
        pts=[]; rng=np.random.RandomState(seed)
        for _ in range(n):
            a,b=r[rng.randint(len(r))],r[rng.randint(len(r))]
            t=rng.rand(); pts.append(a+t*(b-a))
        return np.array(pts,dtype=np.float32)
    except Exception:
        return np.repeat(r,n,axis=0)

def local_jitter(r,n,seed):
    rng=np.random.RandomState(seed); idx=rng.randint(len(r),size=n)
    return (r[idx]+rng.normal(0,0.3,(n,r.shape[1]))).astype(np.float32)

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
    def fed_with_synth(synth_fn):
        # each honeypot synthesizes LOCALLY, shares only synthetic
        allsynth=[]
        for (_,_,r) in hps: allsynth.append(synth_fn(r,60,seed))
        synth=np.vstack(allsynth); sy=np.full(len(synth),RARE)
        g=IDS()
        for _ in range(10):
            lms,sz=[],[]
            for (Xk,yk,_) in hps:
                Xa=np.vstack([Xk,synth]); ya=np.concatenate([yk,sy])
                lms.append(train_ids(copy.deepcopy(g),Xa,ya)); sz.append(len(Xa))
            g.load_state_dict(fedavg(lms,sz))
        return rr(g)
    out={}
    out["smote_local"]=fed_with_synth(local_smote)
    out["jitter_local"]=fed_with_synth(local_jitter)
    # GAN local: each honeypot trains its own GAN on 3 samples
    def gan_synth(r,n,sd):
        if len(r)<2: return np.repeat(r,n,axis=0)
        G=train_gan(r,ep=300)
        with torch.no_grad(): return G(torch.randn(n,16)).numpy().astype(np.float32)
    out["gan_local"]=fed_with_synth(gan_synth)
    return out

seeds=[0,1,2,3,4]; keys=["smote_local","jitter_local","gan_local"]
acc={k:[] for k in keys}
for s in seeds:
    o=run(s)
    for k in keys: acc[k].append(o[k])
    print(f"seed {s}: "+" ".join(f"{k}={o[k]:.3f}" for k in keys))

names={"smote_local":"local SMOTE (privacy-safe)","jitter_local":"local jitter (privacy-safe)",
       "gan_local":"local GAN (ours, privacy-safe)"}
print(f"\n{'method (all share synthetic only)':34s} mean ± std  (n=5)")
print("-"*56)
for k in keys:
    a=np.array(acc[k]); print(f"{names[k]:34s} {a.mean():.3f} ± {a.std():.3f}")
