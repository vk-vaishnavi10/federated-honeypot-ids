"""Parse the 10-honeypot global dataset, keeping each honeypot SEPARATE.
Each session tagged with its source honeypot = a real federated client."""
import json, os, glob
from collections import defaultdict, Counter
import pandas as pd

BASE=os.path.expanduser("~/.cache/kagglehub/datasets/nlaha11/global-ssh-and-telnet-honeypot-logs-cowrie/versions/1/logs")
CMD_VOCAB=["wget","curl","chmod","cat","ls","cd","echo","rm","cp","mv","tar",
           "sh","bash","./","crontab","useradd","passwd","ssh","uname","whoami",
           "id","ps","netstat","free","w","last","history","nproc","grep","kill",
           "export","mkdir","touch","scp","tftp","busybox","dd","mount","enable",
           "system","shell","/bin/","http","python","perl","nc","ping","apt","yum"]

def parse_honeypot(folder, hp_id):
    S=defaultdict(lambda:{"ls":0,"lf":0,"cmds":[],"dl":0,"hassh":"","proto":""})
    for fp in glob.glob(os.path.join(folder,"cowrie.json*")):
        with open(fp,errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: e=json.loads(line)
                except: continue
                sid=e.get("session")
                if not sid: continue
                s=S[sid]; ev=e.get("eventid","")
                if e.get("protocol"): s["proto"]=e["protocol"]
                if ev=="cowrie.client.kex": s["hassh"]=e.get("hassh","")
                if ev=="cowrie.login.success": s["ls"]+=1
                if ev=="cowrie.login.failed": s["lf"]+=1
                if ev=="cowrie.command.input": s["cmds"].append(e.get("input","").lower())
                if ev=="cowrie.session.file_download": s["dl"]+=1
    rows=[]
    for sid,s in S.items():
        txt=" ".join(s["cmds"])
        if s["ls"]==0: cls,lab="benign",0
        elif s["dl"]>0 or "wget" in txt or "curl" in txt or "tftp" in txt: cls,lab="malware_download",3
        elif "xmrig" in txt or "crontab" in txt or "miner" in txt: cls,lab="miner",4
        elif "useradd" in txt or "authorized_keys" in txt or "passwd" in txt: cls,lab="persistence",5
        elif len(s["cmds"])>=4 or "netstat" in txt or "/etc/passwd" in txt or "ps aux" in txt: cls,lab="recon",2
        else: cls,lab="bruteforce",1
        row={"honeypot":hp_id,"protocol":s["proto"],
             "login_success":s["ls"],"login_failed":s["lf"],
             "n_cmds":len(s["cmds"]),"downloads":s["dl"],
             "hassh_hash":hash(s["hassh"])%1000/1000.0}
        for tok in CMD_VOCAB: row[f"cmd_{tok}"]=txt.count(tok)
        row["attack_type"]=cls; row["label"]=lab
        rows.append(row)
    return rows

allrows=[]
folders=sorted(glob.glob(os.path.join(BASE,"cowrie-logs-*")))
for i,folder in enumerate(folders):
    hp=os.path.basename(folder).replace("cowrie-logs-","")
    r=parse_honeypot(folder,i)
    allrows.extend(r)
    print(f"honeypot {i} ({hp}): {len(r)} sessions")

df=pd.DataFrame(allrows)
df.to_csv("global_features.csv",index=False)
print(f"\nTotal: {len(df)} sessions, {len([c for c in df.columns if c not in ['honeypot','protocol','attack_type','label']])} features")
print("\nOverall class distribution:")
print(df["attack_type"].value_counts())
print("\nProtocol split:")
print(df["protocol"].value_counts())
