"""6-class federated IDS with FedAvg, non-IID split, GAN augmentation toggle."""
import torch, torch.nn as nn, numpy as np, pandas as pd, copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]
NCLASS=6

class IDS(nn.Module):
    def __init__(self,inp=len(FEATURES),out=NCLASS):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inp,64),nn.ReLU(),
                               nn.Linear(64,32),nn.ReLU(),
                               nn.Linear(32,out))
    def forward(self,x): return self.net(x)

def local_train(model,X,y,epochs=40):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    ce=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad(); loss=ce(model(Xt),yt); loss.backward(); opt.step()
    return model

def fedavg(models,sizes):
    tot=sum(sizes); gs=copy.deepcopy(models[0].state_dict())
    for k in gs:
        gs[k]=sum(models[i].state_dict()[k]*sizes[i] for i in range(len(models)))/tot
    return gs

def evaluate(model,X,y):
    with torch.no_grad():
        pred=model(torch.tensor(X)).argmax(1).numpy()
    return accuracy_score(y,pred), f1_score(y,pred,average="macro")

def main(use_gan=False,rounds=15):
    df=pd.read_csv("honeypot_features.csv")
    X=df[FEATURES].values.astype(np.float32); y=df["label"].values
    sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)
    n=len(X); idx=np.random.RandomState(0).permutation(n)
    te,tr=idx[:n//5],idx[n//5:]
    Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
    # non-IID: sort by cmd_count so clients see different attack-type mixes
    o=np.argsort(Xtr[:,3]); Xtr,ytr=Xtr[o],ytr[o]
    parts=np.array_split(np.arange(len(Xtr)),3)
    g=IDS()
    for r in range(rounds):
        lms,sz=[],[]
        for p in parts:
            lm=copy.deepcopy(g); lm=local_train(lm,Xtr[p],ytr[p])
            lms.append(lm); sz.append(len(p))
        g.load_state_dict(fedavg(lms,sz))
        acc,f1=evaluate(g,Xte,yte)
        print(f"round {r+1:2d} | acc {acc:.4f} | macro-f1 {f1:.4f}")
    return evaluate(g,Xte,yte)

if __name__=="__main__":
    print("=== Baseline federated (no GAN) ===")
    a=main(False)
    print(f"\nFinal: acc {a[0]:.4f} | macro-f1 {a[1]:.4f}")
