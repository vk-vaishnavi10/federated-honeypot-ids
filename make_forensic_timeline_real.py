"""Forensic timeline from a REAL session in the honeypot logs.
Finds an information-rich attack session (login + commands + ideally a download)
and plots its actual ordered events with real timestamps."""
import json, glob, os
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

# --- locate a log file: prefer the global multi-honeypot set, fall back to day2 ---
candidates = glob.glob(os.path.expanduser(
    "~/.cache/kagglehub/datasets/nlaha11/global-ssh-and-telnet-honeypot-logs-cowrie/versions/1/logs/*/cowrie.json*"))
if not candidates:
    candidates = ["cowrie_day2.json", "real_cowrie.json"]
    candidates = [c for c in candidates if os.path.exists(c)]

if not candidates:
    raise SystemExit("No log file found. Adjust the path in this script.")

# --- collect events per session; pick a session with a download + commands ---
from collections import defaultdict
sessions = defaultdict(list)
scanned = 0
for fp in candidates:
    try:
        with open(fp, errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: e=json.loads(line)
                except: continue
                sid=e.get("session")
                if not sid: continue
                sessions[sid].append(e)
        scanned += 1
    except Exception:
        continue
    # stop once we have plenty
    if len(sessions) > 5000: break

# score sessions: want login success + commands + download
def score(evs):
    types=[e.get("eventid","") for e in evs]
    s=0
    if "cowrie.login.success" in types: s+=2
    s += types.count("cowrie.command.input")
    if "cowrie.session.file_download" in types: s+=3
    return s

best_sid=max(sessions, key=lambda k: score(sessions[k]))
evs=sessions[best_sid]

# --- build ordered (time, label, detail) points ---
def ts(e):
    t=e.get("timestamp")
    try: return datetime.fromisoformat(t.replace("Z","+00:00"))
    except: return None

points=[]
for e in evs:
    ev=e.get("eventid","")
    t=ts(e)
    if ev=="cowrie.session.connect":
        points.append((t,"connect", e.get("src_ip","source IP")))
    elif ev=="cowrie.login.success":
        points.append((t,"login ok", f"user/pass"))
    elif ev=="cowrie.login.failed":
        points.append((t,"login fail", "credential try"))
    elif ev=="cowrie.command.input":
        cmd=(e.get("input","") or "")[:18]
        points.append((t,"command", cmd))
    elif ev=="cowrie.session.file_download":
        points.append((t,"download", "payload"))
    elif ev=="cowrie.session.closed":
        points.append((t,"close", f"dur {e.get('duration',0):.0f}s" if isinstance(e.get('duration'),(int,float)) else "end"))

points=[p for p in points if p[0] is not None]
points.sort(key=lambda p:p[0])
# limit to at most ~7 points for readability
if len(points)>7:
    # keep connect, first login, a few commands, download, close
    keep=[points[0]]+points[1:6]+[points[-1]]
    points=keep

t0=points[0][0]
rel=[(p[0]-t0).total_seconds() for p in points]

# --- plot ---
fig, ax = plt.subplots(figsize=(8.5, 2.8))
ax.set_xlim(-0.5, len(points)-0.5); ax.set_ylim(0,3); ax.axis("off")
ax.add_patch(FancyArrowPatch((-0.4,1.4),(len(points)-0.5,1.4),arrowstyle="-|>",
             mutation_scale=15,linewidth=1.4,color="#333"))
ax.text(len(points)-0.5,1.05,"time",fontsize=8,style="italic",ha="right")
colmap={"connect":"#cfe3d4","login ok":"#f2d0c9","login fail":"#eecfcf",
        "command":"#f2d0c9","download":"#e8b4b4","close":"#cfe3d4"}
for i,((t,lab,det),r) in enumerate(zip(points,rel)):
    ax.add_patch(Circle((i,1.4),0.13,facecolor=colmap.get(lab,"#ddd"),edgecolor="#333",linewidth=1.1,zorder=3))
    ax.text(i,2.05,lab,ha="center",va="center",fontsize=8,weight="bold")
    ax.text(i,0.68,f"{det}\n(+{r:.1f}s)",ha="center",va="center",fontsize=6.5,color="#555")

ax.set_title(f"Forensic evidence timeline of a real captured session (id {best_sid[:8]})",
             fontsize=9.5,weight="bold",pad=6)
plt.tight_layout()
plt.savefig("figures/fig_forensic_timeline.png",dpi=300,bbox_inches="tight")
plt.savefig("figures/fig_forensic_timeline.pdf",bbox_inches="tight")
plt.close()
print(f"saved figures/fig_forensic_timeline.{{png,pdf}} from session {best_sid}")
print(f"  events: {[p[1] for p in points]}")
