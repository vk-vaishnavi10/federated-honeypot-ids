"""Cowrie logs -> per-session features with ATTACK-TYPE labels (multi-class)."""
import json, sys
from collections import defaultdict
import csv

infile  = sys.argv[1] if len(sys.argv) > 1 else "cowrie_day2.json"
outfile = sys.argv[2] if len(sys.argv) > 2 else "honeypot_features.csv"

S = defaultdict(lambda: {"src_ip":"","hassh":"","ls":0,"lf":0,"cmd":0,
                         "cmdf":0,"dl":0,"dur":0,"inputs":[]})
with open(infile) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: e=json.loads(line)
        except: continue
        sid=e.get("session")
        if not sid: continue
        s=S[sid]; ev=e.get("eventid","")
        if e.get("src_ip"): s["src_ip"]=e["src_ip"]
        if ev=="cowrie.client.kex": s["hassh"]=e.get("hassh","")
        if ev=="cowrie.login.success": s["ls"]+=1
        if ev=="cowrie.login.failed":  s["lf"]+=1
        if ev=="cowrie.command.input":
            s["cmd"]+=1; s["inputs"].append(e.get("input",""))
        if ev=="cowrie.command.failed": s["cmdf"]+=1
        if ev=="cowrie.session.file_download": s["dl"]+=1
        if ev=="cowrie.session.closed": s["dur"]=e.get("duration_ms",0)

def classify(s):
    txt=" ".join(s["inputs"]).lower()
    if s["ls"]==0: return "benign"                       # failed login only
    if s["dl"]>0 or "wget" in txt or "curl" in txt: return "malware_download"
    if "xmrig" in txt or "crontab" in txt or "miner" in txt: return "miner"
    if "useradd" in txt or "authorized_keys" in txt or "passwd" in txt: return "persistence"
    if s["cmd"]>=4 or "netstat" in txt or "ps aux" in txt or "/etc/passwd" in txt: return "recon"
    return "bruteforce"

LABELS={"benign":0,"bruteforce":1,"recon":2,"malware_download":3,"miner":4,"persistence":5}
rows=[]
for sid,s in S.items():
    cls=classify(s)
    rows.append({"session":sid,"src_ip":s["src_ip"],"hassh":s["hassh"],
        "login_attempts":s["ls"]+s["lf"],"login_success":s["ls"],"login_failed":s["lf"],
        "cmd_count":s["cmd"],"cmd_failed":s["cmdf"],"downloads":s["dl"],
        "duration_ms":s["dur"],"attack_type":cls,"label":LABELS[cls]})

fields=["session","src_ip","hassh","login_attempts","login_success","login_failed",
        "cmd_count","cmd_failed","downloads","duration_ms","attack_type","label"]
with open(outfile,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

from collections import Counter
print(f"Parsed {len(rows)} sessions -> {outfile}")
for k,v in sorted(Counter(r['attack_type'] for r in rows).items()):
    print(f"  {k:18s} {v}")
