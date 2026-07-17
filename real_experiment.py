"""Real-data zero-day propagation + tradeoff. Hold out malware_download
(real, 7274 samples) from normal clients; only honeypot sees it.
Compare naive upweighting vs GAN augmentation."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score

# duration is all-zero in this real capture -> drop it, use 6 features
FEATURES=["login_attempts","login_success","login_failed","cmd_count","cmd_failed","downloads"]
LABELS={"benign":0,"bruteforce":1,"recon":2,"malware_download":3,"miner":4,"persistence":5}
HELD=3  # malware_download

df=pd.read_csv("real_features.csv")
# keep classes with enough samples: benign, bruteforce, malware
df=df[df["attack_type"].isin(["benign","bruteforce","malware_download"])].copy()
# balance benign down so it doesn't dominate (sample 7000)
benign=df[df.attack_type=="benign"].sample(7000,random_state=1)
other=df[df.attack_type!="benign"]
df=pd.concat([benign,other]).sample(frac=1,random_state=1).reset_index(drop=True)
print("Real experiment class counts:")
print(df["attack_type"].value_counts(),"\n")

X=df[FEATURES].values.astype(np.float32); y=df["label"].values
sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)

class IDS(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,6))
    def forward(s,x): return s.net(x)
class Gen(nn.Module):
    def __init__(s,z=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,6))
    def forward(s,x): return s.net(x)
class Disc(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(6,64),nn.LeakyReLU(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(s,x): return s.net(x)

def train_ids(m,X,y,ep=30):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
    return m
def train_gan(data,ep=300,z=16,b=32):
    G,D=Gen(z),Disc(); oG=torch.optim.Adam(G.parameters(),2e-4); oD=torch.optim.Adam(D.parameters(),2e-4)
    bce=nn.BCELoss(); dt=torch.tensor(data)
    for _ in range(ep):
        i=torch.randint(0,len(dt),(b,)); real=dt[i]; f=G(torch.randn(b,z)).detach()
        lD=bce(D(real),torch.ones(b,1))+bce(D(f),torch.zeros(b,1)); oD.zero_grad(); lD.backward(); oD.step()
        g=G(torch.randn(b,z)); lG=bce(D(g),torch.ones(b,1)); oG.zero_grad(); lG.backward(); oG.step()
    return G
def fedavg(ms,sz):
    tot=sum(sz); g=copy.deepcopy(ms[0].state_dict())
    for k in g: g[k]=sum(ms[i].state_dict()[k]*sz[i] for i in range(len(ms)))/tot
    return g

hmask=(y==HELD); Xn,yn=X[~hmask],y[~hmask]; Xh,yh=X[hmask],y[hmask]
ri=np.random.RandomState(1).permutation(len(Xn))
te=ri[:len(Xn)//5]; trn=ri[len(Xn)//5:]
Xk_test,yk_test=Xn[te],yn[te]
Xn_tr,yn_tr=Xn[trn],yn[trn]
# honeypot gets a subset of held-out (simulate one honeypot's limited capture)
Xh_tr=Xh[:500]; yh_tr=yh[:500]
Xh_test=Xh[500:1500]; yh_test=yh[500:1500]
parts=np.array_split(np.random.RandomState(0).permutation(len(Xn_tr)),3)

def evaluate(g):
    with torch.no_grad():
        zd=recall_score(yh_test,g(torch.tensor(Xh_test)).argmax(1).numpy(),labels=[HELD],average="macro",zero_division=0)
        ka=accuracy_score(yk_test,g(torch.tensor(Xk_test)).argmax(1).numpy())
    return zd,ka

# baseline: no honeypot help
g=IDS()
for r in range(8):
    lms,sz=[],[]
    for p in parts: lms.append(train_ids(copy.deepcopy(g),Xn_tr[p],yn_tr[p])); sz.append(len(p))
    g.load_state_dict(fedavg(lms,sz))
b_zd,b_ka=evaluate(g)

# A: naive upweight
g=IDS()
for r in range(8):
    lms,sz=[],[]
    for p in parts: lms.append(train_ids(copy.deepcopy(g),Xn_tr[p],yn_tr[p])); sz.append(len(p))
    lms.append(train_ids(copy.deepcopy(g),Xh_tr,yh_tr)); sz.append(int(len(Xh_tr)*5))
    g.load_state_dict(fedavg(lms,sz))
a_zd,a_ka=evaluate(g)

# B: GAN augment
G=train_gan(Xh_tr)
with torch.no_grad(): synth=G(torch.randn(300,16)).numpy().astype(np.float32)
sy=np.full(300,HELD)
g=IDS()
for r in range(8):
    lms,sz=[],[]
    for p in parts:
        Xa=np.vstack([Xn_tr[p],synth]); ya=np.concatenate([yn_tr[p],sy])
        lms.append(train_ids(copy.deepcopy(g),Xa,ya)); sz.append(len(Xa))
    lms.append(train_ids(copy.deepcopy(g),Xh_tr,yh_tr)); sz.append(len(Xh_tr))
    g.load_state_dict(fedavg(lms,sz))
c_zd,c_ka=evaluate(g)

print(f"{'method':30s} zero-day recall | known-class acc")
print("-"*64)
print(f"{'baseline (no honeypot)':30s}     {b_zd:.3f}      |    {b_ka:.3f}")
print(f"{'A: naive upweight (λ=5)':30s}     {a_zd:.3f}      |    {a_ka:.3f}")
print(f"{'B: GAN augmentation':30s}     {c_zd:.3f}      |    {c_ka:.3f}")
