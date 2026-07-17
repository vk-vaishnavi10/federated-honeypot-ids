"""Killer experiment (hard regime): rare class is BOTH scarce (3/honeypot)
AND hard (feature noise + overlap). Tests whether GAN pooling recovers a
rare attack that a single honeypot genuinely cannot learn.
Reports the truth whether or not our method wins."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

FEATURES=["login_attempts","login_success","login_failed","cmd_count","cmd_failed","downloads"]
RARE=3; PER_HP=3; NOISE=0.6   # scarcity + noise level

df=pd.read_csv("real_features.csv")
df=df[df["attack_type"].isin(["benign","bruteforce","malware_download"])].copy()
benign=df[df.attack_type=="benign"].sample(6000,random_state=1)
df=pd.concat([benign,df[df.attack_type!="benign"]]).sample(frac=1,random_state=1).reset_index(drop=True)
X=df[FEATURES].values.astype(np.float32); y=df["label"].values
sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)

rng=np.random.RandomState(0)
# make it HARD: add noise to everything so classes overlap
X = X + rng.normal(0, NOISE, X.shape).astype(np.float32)

rare=X[y==RARE]; rng.shuffle(rare)
rare_test=rare[1000:2000]
common=X[y!=RARE]; yc=y[y!=RARE]

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
def rr(g):
    with torch.no_grad():
        return recall_score(np.full(len(rare_test),RARE),
            g(torch.tensor(rare_test)).argmax(1).numpy(),labels=[RARE],average="macro",zero_division=0)

ci=rng.permutation(len(common)); cparts=np.array_split(ci,5)
hps=[]
for k in range(5):
    r=rare[k*PER_HP:(k+1)*PER_HP]
    Xk=np.vstack([common[cparts[k]],r]); yk=np.concatenate([yc[cparts[k]],np.full(len(r),RARE)])
    hps.append((Xk,yk,r))

solo=train_ids(IDS(),hps[0][0],hps[0][1],ep=80); solo_rec=rr(solo)

g=IDS()
for _ in range(10):
    lms,sz=[],[]
    for (Xk,yk,_) in hps: lms.append(train_ids(copy.deepcopy(g),Xk,yk)); sz.append(len(Xk))
    g.load_state_dict(fedavg(lms,sz))
fed_rec=rr(g)

g=IDS(); synth_all=[]
allrare=np.vstack([r for (_,_,r) in hps])   # pool honeypots' rare samples for GAN
G=train_gan(allrare)
with torch.no_grad(): synth=G(torch.randn(300,16)).numpy().astype(np.float32)
sy=np.full(len(synth),RARE)
for _ in range(10):
    lms,sz=[],[]
    for (Xk,yk,_) in hps:
        Xa=np.vstack([Xk,synth]); ya=np.concatenate([yk,sy])
        lms.append(train_ids(copy.deepcopy(g),Xa,ya)); sz.append(len(Xa))
    g.load_state_dict(fedavg(lms,sz))
gan_rec=rr(g)

print(f"HARD regime: {PER_HP} rare samples/honeypot, feature noise σ={NOISE}\n")
print(f"{'setup':40s} rare-attack recall")
print("-"*60)
print(f"{'1. single honeypot (isolated)':40s}    {solo_rec:.3f}")
print(f"{'2. federated, no GAN':40s}    {fed_rec:.3f}")
print(f"{'3. federated + GAN pooling (ours)':40s}    {gan_rec:.3f}")
