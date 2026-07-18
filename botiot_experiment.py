"""BoT-IoT generalization testbed. Hold out KEYLOGGING (73 real samples - the
rarest trainable attack) from most sites; give to sentinels. Sentinel sweep.
Imbalance handled from the start (balanced classes, sentinels retain rare)."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, glob, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
warnings.filterwarnings("ignore")

files=sorted(glob.glob('/Users/vaishnavi/.cache/kagglehub/datasets/vigneshvenkateswaran/bot-iot-5-data/versions/1/reduced_data_*.csv'))
print("Loading BoT-IoT...")
df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df=df.replace([np.inf,-np.inf],np.nan)

# use subcategory as label; keep classes with enough samples + keylogging
keep=["UDP","TCP","Service_Scan","OS_Fingerprint","HTTP","Keylogging"]
df=df[df["subcategory"].isin(keep)].copy()
cmap={c:i for i,c in enumerate(keep)}
df["label"]=df["subcategory"].map(cmap)
HELD=cmap["Keylogging"]

# numeric features only, drop labels/identifiers
drop=["category","subcategory","attack","label","pkSeqID","stime","ltime","saddr","daddr","proto","flgs","state"]
FEATURES=[c for c in df.columns if c not in drop and df[c].dtype in [np.float64,np.int64]]
df=df.dropna(subset=FEATURES)
D=len(FEATURES); NCLASS=len(keep)
print(f"{D} features, holding out KEYLOGGING ({(df.label==HELD).sum()} samples)")
print(f"(context: Data_Exfiltration had only 6 samples - cited as motivation)\n")

# balance: cap dominant classes so keylogging isn't drowned
bal=[df[df.label==c].sample(min(5000,len(df[df.label==c])),random_state=0) for c in range(NCLASS)]
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

def run(seed,n_sent,rounds=20):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    X=df[FEATURES].values.astype(np.float32); sc=StandardScaler().fit(X)
    held=df[df.label==HELD]
    n_test=max(10,len(held)//3)
    test=held.sample(n_test,random_state=seed); tidx=set(test.index)
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
    # also isolated blind
    blind=[c for c in clients if not c[2]][0]
    solo=rr(train(IDS(),blind[0],blind[1],ep=50))
    return solo,rr(g)

print("BoT-IoT sentinel sweep (Keylogging hold-out):\n")
print(f"{'#sentinels':>11} | isolated | federated")
print("-"*40)
import os,csv
os.makedirs("results",exist_ok=True)
_rows=[]
for ns in [3,5,7]:
    ss,ff=[],[]
    for s in range(5):
        a,b=run(s,ns); ss.append(a); ff.append(b)
    print(f"{ns:>11} | {np.mean(ss):.3f}    | {np.mean(ff):.3f} ± {np.std(ff):.3f}")
    _rows.append({"dataset":"BoT-IoT","sentinels":ns,"recall_mean":round(float(np.mean(ff)),4),"recall_std":round(float(np.std(ff)),4)})
_f=open("results/sentinel_botiot.csv","w",newline="");_w=csv.DictWriter(_f,fieldnames=["dataset","sentinels","recall_mean","recall_std"]);_w.writeheader();_w.writerows(_rows);_f.close();print("saved results/sentinel_botiot.csv")
