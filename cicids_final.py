"""CICIDS2017: brute-force is 0.997 learnable multiclass, so weak federated
recovery = propagation problem. Test sentinel count (like honeypot experiment)."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from scipy import stats
warnings.filterwarnings("ignore")

PATH="/Users/vaishnavi/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv"
print("Loading...")
df=pd.read_csv(PATH).replace([np.inf,-np.inf],np.nan).dropna()
classes=["Normal Traffic","DoS","DDoS","Port Scanning","Brute Force","Web Attacks","Bots"]
cmap={c:i for i,c in enumerate(classes)}
df=df[df["Attack Type"].isin(classes)].copy(); df["label"]=df["Attack Type"].map(cmap)
FEATURES=[c for c in df.columns if c not in ["Attack Type","label"]]
D=len(FEATURES); NCLASS=7; HELD=cmap["Brute Force"]
bal=[df[df.label==c].sample(min(8000,len(df[df.label==c])),random_state=0) for c in range(NCLASS)]
df=pd.concat(bal).reset_index(drop=True)

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

def run(seed,n_sent,rounds):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    X=df[FEATURES].values.astype(np.float32); sc=StandardScaler().fit(X)
    held=df[df.label==HELD]; test=held.sample(1500,random_state=seed); tidx=set(test.index)
    held_tr=held[~held.index.isin(tidx)]
    rest=df[(df.label!=HELD)&(~df.index.isin(tidx))]
    idx=rng.permutation(len(rest)); sites=np.array_split(idx,10)
    bf=rng.permutation(len(held_tr)); bfp=np.array_split(bf,n_sent)
    clients=[]
    for si in range(10):
        cd=rest.iloc[sites[si]]
        is_sent=si<n_sent
        cc=pd.concat([cd,held_tr.iloc[bfp[si]]]) if is_sent else cd
        Xc=sc.transform(cc[FEATURES].values.astype(np.float32)).astype(np.float32)
        clients.append((Xc,cc["label"].values,is_sent))
    Xt=sc.transform(test[FEATURES].values.astype(np.float32)).astype(np.float32); yt=test["label"].values
    def rr(g):
        with torch.no_grad(): return recall_score(yt,g(torch.tensor(Xt)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)
    g=IDS()
    for _ in range(rounds):
        lms,sz=[],[]
        for (Xc,yc,_) in clients: lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
        g.load_state_dict(fedavg(lms,sz))
    return rr(g)

print("Sentinel sweep on CICIDS2017 (brute-force hold-out, 20 rounds):\n")
print(f"{'#sentinels':>11} | federated recall")
print("-"*34)
import os,csv
os.makedirs("results",exist_ok=True)
_rows=[]
for ns in [3,5,7]:
    vals=[run(s,ns,20) for s in range(5)]
    v=np.array(vals); print(f"{ns:>11} | {v.mean():.3f} ± {v.std():.3f}")
    _rows.append({"dataset":"CICIDS2017","sentinels":ns,"recall_mean":round(float(v.mean()),4),"recall_std":round(float(v.std()),4)})
_f=open("results/sentinel_cicids.csv","w",newline="");_w=csv.DictWriter(_f,fieldnames=["dataset","sentinels","recall_mean","recall_std"]);_w.writeheader();_w.writerows(_rows);_f.close();print("saved results/sentinel_cicids.csv")
