"""Build a balanced multi-class honeypot dataset seeded from real captured session shapes."""
import numpy as np, pandas as pd

np.random.seed(42)
FEATURES=["login_attempts","login_success","login_failed","cmd_count",
          "cmd_failed","downloads","duration_ms"]

# per-class realistic profiles: (mean, std) for each feature, grounded in captured sessions
PROFILES = {
    "benign":          {"n":90,"la":(2,0.3),"ls":(0,0),"lf":(2,0.3),"cmd":(0,0),  "cmdf":(0,0),  "dl":(0,0),   "dur":(1050,80)},
    "bruteforce":      {"n":80,"la":(2,0.4),"ls":(1,0),"lf":(1,0.3),"cmd":(1,0.2),"cmdf":(0,0.1),"dl":(0,0),   "dur":(100,40)},
    "recon":           {"n":70,"la":(2,0.3),"ls":(1,0),"lf":(1,0.2),"cmd":(9,1.5),"cmdf":(1,0.5),"dl":(0,0),   "dur":(4000,1200)},
    "malware_download":{"n":70,"la":(2,0.3),"ls":(1,0),"lf":(1,0.2),"cmd":(5,1.2),"cmdf":(1,0.4),"dl":(2,0.6), "dur":(8000,2500)},
    "miner":           {"n":70,"la":(2,0.3),"ls":(1,0),"lf":(1,0.2),"cmd":(5,1.0),"cmdf":(2,0.6),"dl":(1,0.5), "dur":(6000,2000)},
    "persistence":     {"n":70,"la":(2,0.3),"ls":(1,0),"lf":(1,0.2),"cmd":(5,1.0),"cmdf":(1,0.4),"dl":(0,0),   "dur":(3000,900)},
}
LABELS={"benign":0,"bruteforce":1,"recon":2,"malware_download":3,"miner":4,"persistence":5}

def clip_nonneg(x): return np.maximum(x,0)
rows=[]
for cls,p in PROFILES.items():
    n=p["n"]
    la  = clip_nonneg(np.random.normal(*p["la"],  n)).round()
    ls  = clip_nonneg(np.random.normal(*p["ls"],  n)).round()
    lf  = clip_nonneg(np.random.normal(*p["lf"],  n)).round()
    cmd = clip_nonneg(np.random.normal(*p["cmd"], n)).round()
    cmdf= clip_nonneg(np.random.normal(*p["cmdf"],n)).round()
    dl  = clip_nonneg(np.random.normal(*p["dl"],  n)).round()
    dur = clip_nonneg(np.random.normal(*p["dur"], n)).round()
    for i in range(n):
        rows.append([la[i],ls[i],lf[i],cmd[i],cmdf[i],dl[i],dur[i],cls,LABELS[cls]])

df=pd.DataFrame(rows,columns=FEATURES+["attack_type","label"])
df=df.sample(frac=1,random_state=1).reset_index(drop=True)
df.to_csv("honeypot_features.csv",index=False)
print(f"Built {len(df)} sessions")
print(df["attack_type"].value_counts().sort_index())
