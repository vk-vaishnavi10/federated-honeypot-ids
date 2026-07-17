"""Rich feature parser: bag-of-commands + behavioral features per session.
Produces high-dimensional features with STRUCTURE that jitter/SMOTE break."""
import json, sys
from collections import defaultdict, Counter
import numpy as np, pandas as pd

infile  = sys.argv[1] if len(sys.argv)>1 else "real_cowrie.json"
outfile = sys.argv[2] if len(sys.argv)>2 else "rich_features.csv"

# command tokens we track (top real attacker commands)
CMD_VOCAB=["wget","curl","chmod","cat","ls","cd","echo","rm","cp","mv","tar",
           "wget_http","sh","bash","./","crontab","useradd","passwd","ssh",
           "uname","whoami","id","ps","netstat","free","w","last","history",
           "nproc","grep","kill","export","mkdir","touch","scp","tftp",
           "busybox","dd","mount","enable","system","shell","/bin/","http"]

S=defaultdict(lambda:{"ls":0,"lf":0,"cmds":[],"dl":0,"hassh":""})
with open(infile) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: e=json.loads(line)
        except: continue
        sid=e.get("session")
        if not sid: continue
        s=S[sid]; ev=e.get("eventid","")
        if ev=="cowrie.client.kex": s["hassh"]=e.get("hassh","")
        if ev=="cowrie.login.success": s["ls"]+=1
        if ev=="cowrie.login.failed": s["lf"]+=1
        if ev=="cowrie.command.input": s["cmds"].append(e.get("input","").lower())
        if ev=="cowrie.session.file_download": s["dl"]+=1

def cmd_vector(cmds):
    text=" ".join(cmds)
    return [text.count(tok) for tok in CMD_VOCAB]

def classify(s):
    txt=" ".join(s["cmds"])
    if s["ls"]==0: return "benign",0
    if s["dl"]>0 or "wget" in txt or "curl" in txt: return "malware_download",3
    if "xmrig" in txt or "crontab" in txt: return "miner",4
    if "useradd" in txt or "authorized_keys" in txt: return "persistence",5
    if len(s["cmds"])>=4 or "netstat" in txt or "/etc/passwd" in txt: return "recon",2
    return "bruteforce",1

rows=[]
for sid,s in S.items():
    cls,lab=classify(s)
    cv=cmd_vector(s["cmds"])
    row={"login_success":s["ls"],"login_failed":s["lf"],"n_cmds":len(s["cmds"]),
         "downloads":s["dl"],"hassh_hash":hash(s["hassh"])%1000/1000.0}
    for i,tok in enumerate(CMD_VOCAB): row[f"cmd_{tok}"]=cv[i]
    row["attack_type"]=cls; row["label"]=lab
    rows.append(row)

df=pd.DataFrame(rows)
df.to_csv(outfile,index=False)
print(f"Parsed {len(df)} sessions with {len([c for c in df.columns if c not in ['attack_type','label']])} features")
print(df["attack_type"].value_counts())
