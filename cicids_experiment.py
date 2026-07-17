"""Generalization testbed: does isolated-fails -> federated-recovers hold on
CICIDS2017 network FLOW data (not just honeypot sessions)?
Simulate 10 sites, hold out a rare attack (Bots) from most, give to sentinels.
n=5 seeds, paired t-test. NOTE: sites are simulated partitions (stated honestly)."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score
from scipy import stats
warnings.filterwarnings("ignore")

PATH="/Users/vaishnavi/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv"
print("Loading CICIDS2017 (2.5M flows)...")
df=pd.read_csv(PATH)
df=df.replace([np.inf,-np.inf],np.nan).dropna()

# map classes to integers; hold out Bots (rarest)
classes=["Normal Traffic","DoS","DDoS","Port Scanning","Brute Force","Web Attacks","Bots"]
cmap={c:i for i,c in enumerate(classes)}
df=df[df["Attack Type"].isin(classes)].copy()
df["label"]=df["Attack Type"].map(cmap)
FEATURES=[c for c in df.columns if c not in ["Attack Type","label"]]
D=len(FEATURES); NCLASS=len(classes); HELD=cmap["Bots"]
print(f"{D} flow features, {len(df)} flows, holding out BOTS ({(df.label==HELD).sum()} samples)\n")

# subsample normal traffic so it doesn't dominate training time
normal=df[df.label==0].sample(200000,random_state=0)
df=pd.concat([normal,df[df.label!=0]]).reset_index(drop=True)

class IDS(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,NCLASS))
    def forward(s,x): return s.net(x)
def train(m,X,y,ep=30):
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
    held=df[df.label==HELD]; test=held.sample(min(500,len(held)),random_state=seed); tidx=set(test.index)
    pool=df[~df.index.isin(tidx)]
    idx=rng.permutation(len(pool)); sites=np.array_split(idx,10)  # 10 simulated sites
    sentinels={0,1,2}  # 3 sentinel sites keep Bots
    clients=[]
    for si,site in enumerate(sites):
        cd=pool.iloc[site]
        if si not in sentinels: cd=cd[cd.label!=HELD]  # blind sites: remove Bots
        if len(cd)>5000: cd=cd.sample(5000,random_state=seed)
        Xc=sc.transform(cd[FEATURES].values.astype(np.float32)).astype(np.float32)
        clients.append((Xc,cd["label"].values,si in sentinels))
    Xt=sc.transform(test[FEATURES].values.astype(np.float32)).astype(np.float32); yt=test["label"].values
    def rr(g):
        with torch.no_grad(): return recall_score(yt,g(torch.tensor(Xt)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)
    # isolated blind site
    blind=[c for c in clients if not c[2]][0]
    solo=rr(train(IDS(),blind[0],blind[1],ep=50))
    # federated
    g=IDS()
    for _ in range(10):
        lms,sz=[],[]
        for (Xc,yc,_) in clients: lms.append(train(copy.deepcopy(g),Xc,yc)); sz.append(len(Xc))
        g.load_state_dict(fedavg(lms,sz))
    return solo,rr(g)

solos,feds=[],[]
for s in range(5):
    a,b=run(s); solos.append(a); feds.append(b)
    print(f"seed {s}: isolated={a:.3f}  federated={b:.3f}")
solos,feds=np.array(solos),np.array(feds)
print(f"\n{'setup':32s} Bots recall (mean ± std)")
print("-"*54)
print(f"{'isolated site (CICIDS2017)':32s} {solos.mean():.3f} ± {solos.std():.3f}")
print(f"{'federated (10 sites)':32s} {feds.mean():.3f} ± {feds.std():.3f}")
t,p=stats.ttest_rel(feds,solos)
print(f"\npaired t-test: t={t:.2f}, p={p:.4f}  ({'significant' if p<0.05 else 'NOT significant'})")
