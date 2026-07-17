"""CICIDS2017 hold-out with BALANCED per-site training (undersample dominants),
matching the setup that made the binary test hit 0.998. Hold out Brute Force."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from scipy import stats
warnings.filterwarnings("ignore")

PATH="/Users/vaishnavi/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv"
print("Loading CICIDS2017...")
df=pd.read_csv(PATH).replace([np.inf,-np.inf],np.nan).dropna()
classes=["Normal Traffic","DoS","DDoS","Port Scanning","Brute Force","Web Attacks","Bots"]
cmap={c:i for i,c in enumerate(classes)}
df=df[df["Attack Type"].isin(classes)].copy(); df["label"]=df["Attack Type"].map(cmap)
FEATURES=[c for c in df.columns if c not in ["Attack Type","label"]]
D=len(FEATURES); NCLASS=7; HELD=cmap["Brute Force"]
# balance the WHOLE dataset first: cap each class at 8000 so no class dominates
balanced=[]
for c in range(NCLASS):
    cd=df[df.label==c]
    balanced.append(cd.sample(min(8000,len(cd)),random_state=0))
df=pd.concat(balanced).reset_index(drop=True)
print("Balanced class counts:"); print(df["label"].value_counts().sort_index())
print(f"holding out BRUTE FORCE\n")

class IDS(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,NCLASS))
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
    X=df[FEATURES].values.astype(np.float32); sc=StandardScaler().fit(X)
    held=df[df.label==HELD]; test=held.sample(min(1500,len(held)),random_state=seed); tidx=set(test.index)
    held_tr=held[~held.index.isin(tidx)]
    rest=df[(df.label!=HELD)&(~df.index.isin(tidx))]
    idx=rng.permutation(len(rest)); sites=np.array_split(idx,10)
    bf=rng.permutation(len(held_tr)); bfp=np.array_split(bf,3)
    clients=[]
    for si in range(10):
        cd=rest.iloc[sites[si]]
        is_sent=si<3
        cc=pd.concat([cd,held_tr.iloc[bfp[si]]]) if is_sent else cd
        Xc=sc.transform(cc[FEATURES].values.astype(np.float32)).astype(np.float32)
        clients.append((Xc,cc["label"].values,is_sent))
    Xt=sc.transform(test[FEATURES].values.astype(np.float32)).astype(np.float32); yt=test["label"].values
    def rr(g):
        with torch.no_grad(): return recall_score(yt,g(torch.tensor(Xt)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)
    blind=[c for c in clients if not c[2]][0]
    solo=rr(train(IDS(),blind[0],blind[1],ep=50))
    g=IDS()
    for _ in range(10):
        lms,sz=[],[]
        for (Xc,yc,_) in clients: lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
        g.load_state_dict(fedavg(lms,sz))
    return solo,rr(g)

solos,feds=[],[]
for s in range(5):
    a,b=run(s); solos.append(a); feds.append(b); print(f"seed {s}: isolated={a:.3f}  federated={b:.3f}")
solos,feds=np.array(solos),np.array(feds)
print(f"\n{'setup':32s} BruteForce recall")
print("-"*52)
print(f"{'isolated site':32s} {solos.mean():.3f} ± {solos.std():.3f}")
print(f"{'federated (10 sites)':32s} {feds.mean():.3f} ± {feds.std():.3f}")
if feds.std()>0 or solos.std()>0:
    t,p=stats.ttest_rel(feds,solos); print(f"\npaired t-test: p={p:.4f}")
