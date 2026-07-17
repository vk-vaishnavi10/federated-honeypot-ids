import torch, torch.nn as nn, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix

FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]
df=pd.read_csv("honeypot_features.csv")
X=df[FEATURES].values.astype(np.float32); y=df["label"].values
sc=StandardScaler().fit(X); X=sc.transform(X).astype(np.float32)

# Can a model trained on ALL data detect persistence at all?
class M(nn.Module):
    def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,6))
    def forward(s,x): return s.n(x)
m=M(); opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
Xt,yt=torch.tensor(X),torch.tensor(y,dtype=torch.long)
for _ in range(200): opt.zero_grad(); ce(m(Xt),yt).backward(); opt.step()
pred=m(Xt).argmax(1).numpy()
print("Persistence (class 5) recall when model HAS seen it:",
      round(recall_score(y,pred,labels=[5],average='macro',zero_division=0),3))
print("\nWhat class 5 gets predicted as:")
mask=y==5
from collections import Counter
print(Counter(pred[mask]))

# mean feature values per class — are 5 and others separable?
print("\nMean features per class:")
print(df.groupby('label')[FEATURES].mean().round(1))
