"""Solve the tradeoff: honeypot shares GAN-synthetic novel-attack samples
to normal clients (privacy-preserving) instead of lopsided upweighting.
Compare: naive upweighting vs GAN augmentation."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score

FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]
HELD=5

class IDS(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,6))
    def forward(s,x): return s.net(x)

class Gen(nn.Module):
    def __init__(s,z=16):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,7))
    def forward(s,x): return s.net(x)
class Disc(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(7,64),nn.LeakyReLU(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(s,x): return s.net(x)

def train_ids(m,X,y,ep=40):
    opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
    return m

def train_gan(data,ep=300,z=16,b=16):
    G,D=Gen(z),Disc()
    oG=torch.optim.Adam(G.parameters(),2e-4); oD=torch.optim.Adam(D.parameters(),2e-4)
    bce=nn.BCELoss(); dt=torch.tensor(data)
    for _ in range(ep):
        i=torch.randint(0,len(dt),(b,)); real=dt[i]
        f=G(torch.randn(b,z)).detach()
        lD=bce(D(real),torch.ones(b,1))+bce(D(f),torch.zeros(b,1))
        oD.zero_grad(); lD.backward(); oD.step()
        g=G(torch.randn(b,z)); lG=bce(D(g),torch.ones(b,1))
        oG.zero_grad(); lG.backward(); oG.step()
    return G

def fedavg(ms,sz):
    tot=sum(sz); g=copy.deepcopy(ms[0].state_dict())
    for k in g: g[k]=sum(ms[i].state_dict()[k]*sz[i] for i in range(len(ms)))/tot
    return g

df=pd.read_csv("honeypot_features.csv")
X=df[FEATURES].values.astype(np.float32); y=df["label"].values
sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)
hmask=(y==HELD); Xn,yn=X[~hmask],y[~hmask]; Xh,yh=X[hmask],y[hmask]
ri=np.random.RandomState(1).permutation(len(Xn))
te=ri[:len(Xn)//5]; trn=ri[len(Xn)//5:]
Xk_test,yk_test=Xn[te],yn[te]
Xn_tr,yn_tr=Xn[trn],yn[trn]
parts=np.array_split(np.random.RandomState(0).permutation(len(Xn_tr)),3)

def evaluate(g):
    with torch.no_grad():
        zd=recall_score(y[hmask],g(torch.tensor(X[hmask])).argmax(1).numpy(),
                        labels=[HELD],average="macro",zero_division=0)
        ka=accuracy_score(yk_test,g(torch.tensor(Xk_test)).argmax(1).numpy())
    return zd,ka

# METHOD A: naive upweighting lambda=5
g=IDS()
for r in range(10):
    lms,sz=[],[]
    for p in parts: lms.append(train_ids(copy.deepcopy(g),Xn_tr[p],yn_tr[p])); sz.append(len(p))
    lms.append(train_ids(copy.deepcopy(g),Xh,yh)); sz.append(int(len(Xh)*5))
    g.load_state_dict(fedavg(lms,sz))
a_zd,a_ka=evaluate(g)

# METHOD B: GAN augmentation. Honeypot trains GAN on novel attack, shares SYNTHETIC
# samples to each normal client, who train on their data + synthetic novel attacks.
G=train_gan(Xh)
with torch.no_grad(): synth=G(torch.randn(40,16)).numpy().astype(np.float32)
synth_y=np.full(40,HELD)
g=IDS()
for r in range(10):
    lms,sz=[],[]
    for p in parts:
        Xaug=np.vstack([Xn_tr[p],synth]); yaug=np.concatenate([yn_tr[p],synth_y])
        lms.append(train_ids(copy.deepcopy(g),Xaug,yaug)); sz.append(len(Xaug))
    lms.append(train_ids(copy.deepcopy(g),Xh,yh)); sz.append(len(Xh))
    g.load_state_dict(fedavg(lms,sz))
b_zd,b_ka=evaluate(g)

print(f"{'method':28s} zero-day recall | known-class acc")
print("-"*62)
print(f"{'A: naive upweight (λ=5)':28s}     {a_zd:.3f}      |    {a_ka:.3f}")
print(f"{'B: GAN synthetic augment':28s}     {b_zd:.3f}      |    {b_ka:.3f}")
