"""
IsThisSeatTaken — real-time GUI  v2
Run:  python is_this_seat_taken_gui_v2.py
      python is_this_seat_taken_gui_v2.py --log path/to/blackboard_0.txt --rows 2 --cols 4
Opens http://localhost:5050 automatically.
"""

import argparse, glob, json, os, re, threading, time, webbrowser
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from flask import Flask, Response, render_template_string

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--log", default=None)
parser.add_argument("--config", default=None)
parser.add_argument("--rows", type=int, default=None)
parser.add_argument("--cols", type=int, default=None)
parser.add_argument("--port", type=int, default=5050)
parser.add_argument("--agents", type=int, default=None)
parser.add_argument("--no-browser", action="store_true")
args, _ = parser.parse_known_args()

def _find_log(newer_than=None):
    matches = glob.glob("logs/**/blackboard_0.txt", recursive=True)
    if not matches:
        return "blackboard_0.txt"
    if newer_than is not None:
        fresh = [m for m in matches if os.path.getmtime(m) > newer_than]
        if fresh:
            return max(fresh, key=os.path.getmtime)
    return max(matches, key=os.path.getmtime)

def _find_config():
    matches = glob.glob("examples/configs/*.yaml") + glob.glob("configs/*.yaml") + glob.glob("**/*.yaml", recursive=True)
    for m in matches:
        if "seat" in m.lower():
            return m
    return matches[0] if matches else None

def _load_config(path):
    if path is None or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

cfg_path = args.config or _find_config()
cfg = _load_config(cfg_path)
env_cfg = cfg.get("environment", {})
net_cfg = cfg.get("communication_network", {})

_rows   = args.rows   or int(env_cfg.get("rows",   2))
_cols   = args.cols   or int(env_cfg.get("cols",   4))
_agents = args.agents or int(net_cfg.get("num_agents", 5))

args.rows   = _rows
args.cols   = _cols
args.agents = _agents

if args.log is None:
    args.log = _find_log()

print(f"[ist_gui] Config: {cfg_path or 'none'} → rows={args.rows} cols={args.cols} agents={args.agents}")
print(f"[ist_gui] Watching: {args.log}")

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
EVENT_HEADER = re.compile(
    r"\[Event #(\d+),\s*Iteration:\s*(\d+)\]\s*\[(\d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(\S+)\s*\((\w+)\)(.*)"
)

@dataclass
class Event:
    idx: int; iteration: int; time_str: str; phase: str
    agent: str; etype: str; content: str = ""
    action: Optional[str] = None; result_status: Optional[str] = None
    moved_from: Optional[str] = None; moved_to: Optional[str] = None
    current_seat: Optional[str] = None
    forced_agent: Optional[str] = None
    forced_from: Optional[str] = None; forced_to: Optional[str] = None
    settled_result: bool = False
    target: Optional[str] = None
    reason: Optional[str] = None

def _extract_inline(tail: str) -> str:
    tail = tail.strip()
    for prefix in ('Content:', 'Message: "', 'Message:'):
        if tail.startswith(prefix):
            return tail[len(prefix):].strip().rstrip('"')
    return ""

def _parse_details(raw: str):
    moved_from = moved_to = forced_agent = forced_from = forced_to = current_seat = None
    settled_result = False
    try:
        cleaned = raw.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        d = json.loads(cleaned)
        res = d.get("result", d)
        if isinstance(res, dict):
            a = res.get("action", "")
            settled_result = bool(res.get("settled", False))
            if a == "move":
                moved_from = res.get("from_seat")
                moved_to = res.get("to_seat")
            elif a in ("stay", "settle"):
                current_seat = res.get("current_seat")
            elif a in ("stand", "forced_stand"):
                moved_from = res.get("previous_seat")
            if res.get("target_reacted"):
                tr = res.get("target_result", {})
                if isinstance(tr, dict):
                    forced_agent = tr.get("agent")
                    forced_from = tr.get("from_seat") or tr.get("previous_seat")
                    forced_to = tr.get("to_seat") or tr.get("current_seat")
    except Exception:
        pass
    return moved_from, moved_to, forced_agent, forced_from, forced_to, current_seat, settled_result

def parse_log(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    events = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = EVENT_HEADER.search(line)
        if not m:
            i += 1; continue
        idx, iteration, time_str, phase, agent, etype, tail = (
            int(m.group(1)), int(m.group(2)), m.group(3),
            m.group(4).lower(), m.group(5), m.group(6).lower(), m.group(7)
        )
        content = _extract_inline(tail)
        action = None; result_status = None; reason = None
        details_buf = []; in_details = False
        moved_from = moved_to = forced_agent = forced_from = forced_to = current_seat = None
        settled_result = False
        target_agent = None
        j = i + 1
        while j < len(lines):
            nxt = lines[j].rstrip()
            if EVENT_HEADER.search(nxt) or nxt.startswith("===="):
                break
            s = nxt.strip()
            if s.startswith("Content:") and not content:
                content = s[len("Content:"):].strip()
            elif s.startswith('Message: "') and not content:
                content = s[len('Message: "'):].rstrip('"')
            elif s.startswith('Message:') and not content:
                content = s[len('Message:'):].strip().strip('"')
            elif s.startswith("Action_Type:"):
                action = s.split(":", 1)[1].strip()
            elif s.startswith("Action_Params:"):
                raw_p = s[len("Action_Params:"):].strip()
                try:
                    dp = json.loads(raw_p.replace("'",'"').replace("True","true").replace("False","false").replace("None","null"))
                    _tgt = dp.get("agent_id") or dp.get("target")
                    if _tgt: target_agent = _tgt
                    if not action:
                        action = dp.get("action")
                except Exception:
                    pass
            elif s.startswith("Result_Status:"):
                result_status = s.split(":", 1)[1].strip()
            elif s.startswith("Reason:") or s.startswith("reason:"):
                reason = s.split(":", 1)[1].strip().strip('"\'')
            elif s.startswith("Details:"):
                in_details = True
                details_buf.append(s[len("Details:"):].strip())
            elif in_details and s:
                # also try to fish reason out of details inline
                if '"reason"' in s or "'reason'" in s:
                    try:
                        partial = "{" + s.strip().lstrip("{").rstrip(",}") + "}"
                        pd = json.loads(partial.replace("'",'"').replace("None","null"))
                        if pd.get("reason") and not reason:
                            reason = pd["reason"]
                    except Exception:
                        pass
                details_buf.append(s)
            j += 1
        if details_buf:
            moved_from, moved_to, forced_agent, forced_from, forced_to, current_seat, settled_result = _parse_details(" ".join(details_buf))
            # also try to extract reason from the full details block
            if not reason:
                try:
                    full = " ".join(details_buf)
                    cleaned = full.replace("'",'"').replace("True","true").replace("False","false").replace("None","null")
                    d = json.loads(cleaned)
                    reason = d.get("reason") or (d.get("result", {}) or {}).get("reason")
                except Exception:
                    pass
        events.append(Event(
            idx=idx, iteration=iteration, time_str=time_str,
            phase=phase, agent=agent, etype=etype, content=content,
            action=action, result_status=result_status,
            moved_from=moved_from, moved_to=moved_to,
            current_seat=current_seat,
            forced_agent=forced_agent, forced_from=forced_from, forced_to=forced_to,
            settled_result=settled_result,
            target=target_agent, reason=reason,
        ))
        i = j
    return events

def infer_positions(events, num_agents):
    pos = {}
    settled = set()
    for e in events:
        if e.etype == "context" and "Initial seating:" in e.content:
            for part in e.content.replace("Initial seating:", "").split(","):
                part = part.strip()
                if "\u2192" in part:
                    a, s = part.split("\u2192", 1)
                    pos[a.strip()] = s.strip()
        if e.etype == "action_executed" and e.result_status == "success":
            if e.moved_to:
                pos[e.agent] = e.moved_to
                if e.settled_result:
                    settled.add(e.agent)
                else:
                    settled.discard(e.agent)
            if e.action in ("stand", "forced_stand") and e.moved_from:
                # agent vacated a seat — remove from pos
                if pos.get(e.agent) == e.moved_from:
                    pos.pop(e.agent, None)
                settled.discard(e.agent)
            if e.action == "settle":
                settled.add(e.agent)
            if e.action in ("move", "stand", "forced_stand") and not e.settled_result:
                settled.discard(e.agent)
            if e.forced_agent:
                settled.discard(e.forced_agent)
                if e.forced_to:
                    pos[e.forced_agent] = e.forced_to
                elif e.forced_from and pos.get(e.forced_agent) == e.forced_from:
                    pos.pop(e.forced_agent, None)
    return pos, settled

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SimState:
    def __init__(self):
        self.events: List[Event] = []
        self.positions: Dict[str,str] = {}
        self.settled: set = set()
        self.total_moves = 0
        self.current_iter = 0
        self.current_phase = "—"
        self.last_moved: Optional[str] = None

state = SimState()
state_lock = threading.Lock()
sse_clients: List = []
sse_lock = threading.Lock()
_last_mtime = -1.0
_sim_start_time = time.time()

def rebuild(events, num_agents):
    pos, settled = infer_positions(events, num_agents)
    moves = sum(1 for e in events if e.etype == "action_executed"
                and e.moved_to and e.result_status == "success")
    last_moved = None
    for e in reversed(events):
        if e.etype == "action_executed" and e.moved_to and e.result_status == "success":
            last_moved = e.agent; break
    return pos, settled, moves, last_moved

def watch_log():
    global _last_mtime
    while True:
        try:
            new = _find_log(newer_than=_sim_start_time)
            if new and new != args.log and os.path.exists(new):
                args.log = new
                _last_mtime = -1.0
                print(f"[ist_gui] Switched to: {args.log}")
            mtime = os.path.getmtime(args.log) if os.path.exists(args.log) else 0
            if mtime != _last_mtime:
                _last_mtime = mtime
                events = parse_log(args.log)
                pos, settled, moves, last_moved = rebuild(events, args.agents)
                with state_lock:
                    state.events = events
                    state.positions = pos
                    state.settled = settled
                    state.total_moves = moves
                    state.last_moved = last_moved
                    if events:
                        state.current_iter = events[-1].iteration
                        state.current_phase = events[-1].phase
                broadcast()
        except Exception as ex:
            print(f"[watcher] {ex}")
        time.sleep(0.8)

def _agent_colors(num_agents: int) -> Dict[str, str]:
    palette = [
        "#ef4444","#3b82f6","#22c55e","#f97316","#8b5cf6",
        "#ec4899","#14b8a6","#eab308","#6366f1","#84cc16",
        "#06b6d4","#f43f5e","#a855f7","#10b981","#fb923c",
    ]
    return {f"agent_{i}": palette[i % len(palette)] for i in range(num_agents)}

def build_payload():
    with state_lock:
        seats = []
        for r in range(1, args.rows+1):
            for c in range(1, args.cols+1):
                sid = f"seat_{r}_{c}"
                occupant = next((a for a,s in state.positions.items() if s==sid), None)
                is_settled = occupant in state.settled if occupant else False
                seats.append({
                    "id": sid, "row": r, "col": c,
                    "occupant": occupant,
                    "settled": is_settled,
                    "last_moved": occupant == state.last_moved and not is_settled,
                })
        chats = [{"idx":e.idx,"iter":e.iteration,"time":e.time_str,
                  "phase":e.phase,"agent":e.agent,"content":e.content}
                 for e in state.events if e.etype=="communication"]
        actions = []
        for e in state.events:
            if e.etype not in ("action_executed","action_failed"):
                continue
            actions.append({
                "idx": e.idx, "iter": e.iteration, "time": e.time_str,
                "phase": e.phase, "agent": e.agent,
                "action": e.action or e.etype,
                "status": e.result_status or "ok",
                "content": e.content,
                "moved_from": e.moved_from, "moved_to": e.moved_to,
                "current_seat": e.current_seat,
                "target": e.target,
                "reason": e.reason or "",
            })
        return {
            "seats": seats, "chats": chats, "actions": actions,
            "total_moves": state.total_moves, "iter": state.current_iter,
            "phase": state.current_phase, "total_events": len(state.events),
            "rows": args.rows, "cols": args.cols,
            "agent_colors": _agent_colors(args.agents),
            "settled_agents": list(state.settled),
            "standing_agents": [
                f"agent_{i}"
                for i in range(args.agents)
                if f"agent_{i}" not in state.positions
            ],
        }

def broadcast():
    data = f"data: {json.dumps(build_payload())}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try: q.append(data)
            except: dead.append(q)
        for q in dead: sse_clients.remove(q)

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IsThisSeatTaken — Flight</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#0b0f14;
  --bg2:#121820;
  --bg3:#182029;
  --surface:#1a222c;
  --surface2:#212b37;
  --border:#2c3946;
  --border2:#3a4a5a;
  --text:#e8edf2;
  --text2:#9caab8;
  --text3:#647485;
  --accent:#c8a96e;
  --accent2:#8b6f3e;
  --blue:#4a9eff;
  --blue-dim:#1a3d6b;
  --green:#4ade80;
  --green-dim:#14532d;
  --red:#f87171;
  --red-dim:#450a0a;
  --amber:#fbbf24;
  --amber-dim:#451a03;
  --purple:#a78bfa;
  --purple-dim:#2e1065;
  --sky:#5eb4ff;
  --r:8px;
  --r2:12px;
}

body{
  font-family:'Inter',sans-serif;
  background:var(--bg);
  color:var(--text);
  font-size:13px;
  line-height:1.5;
  height:100vh;
  overflow:hidden;
  display:flex;
  flex-direction:column;
}

/* ── Header ── */
header{
  background:var(--bg2);
  border-bottom:1px solid var(--border);
  padding:0 20px;
  height:48px;
  display:flex;
  align-items:center;
  gap:12px;
  flex-shrink:0;
}
.logo{
  font-size:14px;
  font-weight:600;
  color:var(--text);
  letter-spacing:-.01em;
  display:flex;
  align-items:center;
  gap:8px;
}
.logo-icon{
  width:24px;height:24px;
  background:var(--accent2);
  border-radius:6px;
  display:flex;align-items:center;justify-content:center;
  font-size:13px;
}
.hdr-sep{color:var(--border2);margin:0 2px}
.badge{
  font-size:10px;font-weight:600;
  padding:2px 8px;border-radius:20px;
  letter-spacing:.03em;text-transform:uppercase;
}
.bp{background:rgba(74,158,255,.15);color:#6cb3ff;border:1px solid rgba(74,158,255,.25)}
.be{background:rgba(74,222,128,.15);color:#6de297;border:1px solid rgba(74,222,128,.25)}
.bn{background:var(--surface2);color:var(--text3);border:1px solid var(--border)}
.hdr-log{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:var(--text3);
  max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.pulse{
  width:7px;height:7px;border-radius:50%;
  background:var(--green);
  margin-left:auto;
  box-shadow:0 0 0 0 rgba(74,222,128,.4);
  animation:ping 2s ease-out infinite;
}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(74,222,128,.4)}70%{box-shadow:0 0 0 7px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}

/* ── Layout ── */
.main{
  flex:1;
  position:relative;
  overflow:hidden;
  display:flex;
  min-height:0;
}
.stage{
  flex:1;
  overflow-y:auto;
  overflow-x:hidden;
  display:flex;
  justify-content:center;
}
.stage-inner{
  width:100%;max-width:640px;
  padding:56px 24px 40px;
  position:relative;
}

/* ── Drawer (toggleable Chat / Actions) ── */
.drawer-toggles{margin-left:auto;display:flex;gap:8px}
.toggle-btn{
  font-size:11px;font-weight:600;
  padding:5px 10px;border-radius:8px;
  border:1px solid var(--border);
  background:var(--surface);color:var(--text3);
  cursor:pointer;display:flex;align-items:center;gap:6px;
  transition:all .15s;
}
.toggle-btn:hover{border-color:var(--border2);color:var(--text2)}
.toggle-btn.on{
  background:rgba(74,158,255,.15);color:var(--blue);
  border-color:rgba(74,158,255,.35);
}
.toggle-btn .ph-count{margin:0}

.drawer{
  position:absolute;top:0;right:0;bottom:0;
  width:0;overflow:hidden;
  display:flex;
  background:var(--bg2);
  border-left:1px solid var(--border);
  transition:width .25s ease;
  z-index:30;
  box-shadow:-14px 0 32px rgba(0,0,0,.35);
}
.drawer.w1{width:340px}
.drawer.w2{width:660px}
.drawer-panel{
  width:330px;flex-shrink:0;
  display:none;flex-direction:column;min-height:0;
  border-right:1px solid var(--border);
}
.drawer-panel:last-child{border-right:none}
.drawer-panel.active{display:flex}

/* ── Panel headers ── */
.ph{
  padding:10px 16px;
  border-bottom:1px solid var(--border);
  background:var(--bg2);
  display:flex;align-items:center;gap:8px;
  flex-shrink:0;
}
.ph-label{
  font-size:11px;font-weight:600;
  color:var(--text3);text-transform:uppercase;letter-spacing:.08em;
}
.ph-count{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:1px 7px;
  font-size:10px;color:var(--text3);
  font-family:'JetBrains Mono',monospace;
}

/* ── Stats ── */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}
.stat{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:10px 8px;text-align:center;
}
.stat-v{
  font-size:22px;font-weight:600;color:var(--text);
  font-family:'JetBrains Mono',monospace;
  line-height:1;margin-bottom:3px;
}
.stat-l{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.06em}

/* ── Fuselage ── */
.fuselage{
  background:linear-gradient(180deg,#1c2530,#111820);
  border:1px solid var(--border2);
  border-radius:60px 60px 22px 22px;
  padding:18px 20px 24px;
  position:relative;
  overflow:hidden;
  box-shadow:0 20px 50px rgba(0,0,0,.35);
}
.fuselage::before{
  content:'';
  position:absolute;top:0;left:0;right:0;height:6px;
  background:repeating-linear-gradient(90deg,rgba(94,180,255,.4) 0 10px, transparent 10px 22px);
}
.nose,.tail{
  text-align:center;font-size:10px;letter-spacing:.18em;color:var(--text3);
  text-transform:uppercase;padding:2px 0;
}
.nose{margin-bottom:14px}
.tail{margin-top:14px;border-top:1px dashed var(--border);padding-top:10px}

#seat-grid{display:flex;flex-direction:column;gap:8px}

.plane-row{display:flex;align-items:center;gap:10px}
.plane-row.header-row{margin-bottom:2px}
.row-num{
  width:18px;flex-shrink:0;text-align:center;
  font-size:10px;color:var(--text3);font-family:'JetBrains Mono',monospace;
}
.row-seats{display:flex;gap:8px;flex:1;align-items:stretch}
.col-letter{
  flex:1;text-align:center;font-size:10px;color:var(--text3);
  text-transform:uppercase;letter-spacing:.04em;
}
.aisle-gap{
  width:26px;flex-shrink:0;position:relative;
}
.aisle-gap::before{
  content:'';position:absolute;top:0;bottom:0;left:50%;
  border-left:1px dashed var(--border2);
}

.seat{
  flex:1;min-width:0;
  border-radius:var(--r);border:1px solid var(--border);
  padding:8px 4px;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  min-height:64px;
  transition:all .3s ease;
  position:relative;
  overflow:hidden;
}
.seat.empty{background:var(--surface)}
.seat.occ{background:var(--blue-dim);border-color:rgba(74,158,255,.35)}
.seat.moved{background:rgba(74,222,128,.12);border-color:rgba(74,222,128,.4);animation:flash .6s ease}
.seat.settled{
  background:var(--surface);
  border-color:var(--accent2);
  border-width:1px;
}
.seat.settled::before{
  content:'';
  position:absolute;top:0;left:0;right:0;
  height:2px;
  background:linear-gradient(90deg,var(--accent2),var(--accent),var(--accent2));
  opacity:.8;
}
/* Diagonal hatch pattern for settled seats */
.seat.settled::after{
  content:'';
  position:absolute;inset:0;
  background-image:repeating-linear-gradient(
    45deg,
    transparent,
    transparent 4px,
    rgba(200,169,110,.06) 4px,
    rgba(200,169,110,.06) 5px
  );
  pointer-events:none;
}
.seat.k-window{box-shadow:inset 0 2px 0 rgba(94,180,255,.3)}
.seat.k-aisle{box-shadow:inset 0 -2px 0 rgba(255,255,255,.12)}

@keyframes flash{0%,100%{opacity:1}50%{opacity:.6}}

.seat-id{font-size:10px;color:var(--text3);margin-bottom:4px;font-family:'JetBrains Mono',monospace}
.agent-n{font-size:12px;font-weight:600;line-height:1}
.settled-tag{
  font-size:9px;color:var(--accent);
  margin-top:3px;opacity:.9;line-height:1;
}

/* ── Speech bubbles ── */
.bubble-layer{position:absolute;inset:0;pointer-events:none;z-index:6}
.bubble{
  position:absolute;transform:translate(-50%,-104%);
  max-width:170px;background:var(--surface2);border:1px solid var(--border2);
  border-radius:10px;padding:6px 9px;font-size:11px;line-height:1.4;color:var(--text);
  box-shadow:0 6px 18px rgba(0,0,0,.45);
  opacity:0;transition:opacity .25s ease, transform .25s ease;
  z-index:6;
}
.bubble.show{opacity:1;transform:translate(-50%,-112%)}
.bubble::after{
  content:'';position:absolute;left:50%;bottom:-6px;transform:translateX(-50%);
  border:6px solid transparent;border-top-color:var(--surface2);
}
.bubble .b-agent{font-weight:600;margin-bottom:2px;font-size:10px}

.legend{
  display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;
  padding-top:12px;border-top:1px solid var(--border);
}
.li{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text3)}
.ld{width:10px;height:10px;border-radius:2px;flex-shrink:0}

.standing-area{
  margin-top:12px;
  padding-top:12px;
  border-top:1px solid var(--border);
}
.standing-title{
  font-size:10px;
  color:var(--text3);
  text-transform:uppercase;
  letter-spacing:.06em;
  margin-bottom:7px;
}
.standing-list{display:flex;gap:6px;flex-wrap:wrap;min-height:24px}
.standing-chip{
  font-size:10px;
  font-weight:600;
  color:#fff;
  padding:3px 8px;
  border-radius:12px;
  border:1px solid rgba(255,255,255,.12);
}
.standing-empty{font-size:11px;color:var(--text3)}

/* ── Feeds ── */
.feed{
  flex:1;overflow-y:auto;padding:10px;
  display:flex;flex-direction:column;gap:5px;
  min-height:0;
}
.feed::-webkit-scrollbar{width:4px}
.feed::-webkit-scrollbar-track{background:transparent}
.feed::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

.msg{
  border-radius:var(--r);padding:9px 11px;
  font-size:12px;line-height:1.55;
  border-left:2px solid transparent;
  background:var(--surface);
  transition:background .2s;
}
.msg:hover{background:var(--surface2)}

.msg.chat{border-left-color:var(--blue-dim)}
.msg.act-ok{border-left-color:rgba(74,222,128,.4)}
.msg.act-fail{border-left-color:rgba(248,113,113,.4);background:rgba(248,113,113,.04)}
.msg.act-retry{border-left-color:rgba(251,191,36,.4);background:rgba(251,191,36,.03)}
.msg.act-settle{border-left-color:rgba(200,169,110,.6);background:rgba(200,169,110,.05)}
.msg.act-stand{border-left-color:rgba(160,157,152,.4)}
.msg.act-stay{border-left-color:rgba(74,158,255,.25)}

.msg-top{display:flex;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:wrap}
.chip{
  font-size:10px;font-weight:600;
  padding:2px 8px;border-radius:20px;
  color:#fff;letter-spacing:.02em;
}
.itag{font-size:10px;color:var(--text3);font-family:'JetBrains Mono',monospace}
.status-badge{
  font-size:10px;font-weight:500;
  padding:1px 7px;border-radius:10px;
  margin-left:auto;
}
.s-ok{color:var(--green);background:rgba(74,222,128,.12)}
.s-fail{color:var(--red);background:rgba(248,113,113,.12)}
.s-retry{color:var(--amber);background:rgba(251,191,36,.1)}

.mbody{color:var(--text2)}
.mbody b{color:var(--text);font-weight:500}
.mbody code{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--accent);
  background:var(--surface2);
  padding:1px 4px;border-radius:3px;
}

.reason{
  margin-top:4px;
  font-size:11px;color:var(--text3);
  font-style:italic;
  padding-top:4px;
  border-top:1px solid var(--border);
}

/* ── Filter buttons ── */
.filter-row{margin-left:auto;display:flex;gap:3px}
.fb{
  font-size:10px;padding:2px 9px;
  border-radius:6px;border:1px solid var(--border);
  background:var(--surface);cursor:pointer;color:var(--text3);
  transition:all .15s;
}
.fb:hover{border-color:var(--border2);color:var(--text2)}
.fb.on{
  background:rgba(74,158,255,.15);
  color:var(--blue);
  border-color:rgba(74,158,255,.35);
}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">✈️</div>
    IsThisSeatTaken · Flight
  </div>
  <span class="hdr-sep">·</span>
  <span class="badge bn" id="hdr-phase">—</span>
  <span style="color:var(--text3);font-size:12px">iter <b id="hdr-iter" style="color:var(--text);font-family:'JetBrains Mono',monospace">—</b></span>
  <span class="hdr-log" id="hdr-log">{{ log }}</span>
  <div class="drawer-toggles">
    <button class="toggle-btn" id="btn-chat" onclick="toggleDrawer('chat')">💬 Chat <span class="ph-count" id="chat-count">0</span></button>
    <button class="toggle-btn" id="btn-actions" onclick="toggleDrawer('actions')">⚡ Actions <span class="ph-count" id="act-count">0</span></button>
  </div>
  <div class="pulse"></div>
</header>
<div class="main">

  <!-- CENTER: the airplane -->
  <div class="stage">
    <div class="stage-inner">
      <div class="stats">
        <div class="stat"><div class="stat-v" id="s-iter">—</div><div class="stat-l">iter</div></div>
        <div class="stat"><div class="stat-v" id="s-moves">0</div><div class="stat-l">moves</div></div>
        <div class="stat"><div class="stat-v" id="s-evts">0</div><div class="stat-l">events</div></div>
      </div>
      <div class="fuselage">
        <div class="nose">✈ nose · boarding</div>
        <div id="seat-grid"></div>
        <div class="tail">tail ✈</div>
      </div>
      <div class="legend">
        <div class="li"><div class="ld" style="background:var(--blue-dim);border:1px solid rgba(74,158,255,.35)"></div>seated</div>
        <div class="li"><div class="ld" style="background:rgba(74,222,128,.12);border:1px solid rgba(74,222,128,.4)"></div>just moved</div>
        <div class="li">
          <div class="ld" style="border:1px solid var(--accent2);background:var(--surface);background-image:repeating-linear-gradient(45deg,transparent,transparent 4px,rgba(200,169,110,.15) 4px,rgba(200,169,110,.15) 5px)"></div>
          settled
        </div>
        <div class="li"><div class="ld" style="background:var(--surface);border:1px solid var(--border)"></div>empty</div>
        <div class="li"><span class="standing-chip" style="background:var(--text3);padding:1px 6px;font-size:8px">A0</span>not seated</div>
      </div>
      <div class="standing-area">
        <div class="standing-title">Standing</div>
        <div class="standing-list" id="standing-list"></div>
      </div>
    </div>
  </div>

  <!-- Toggleable drawer: Chat / Actions -->
  <div class="drawer" id="drawer">
    <div class="drawer-panel" id="panel-chat">
      <div class="ph"><span class="ph-label">Chat</span></div>
      <div class="feed" id="chat-feed"></div>
    </div>
    <div class="drawer-panel" id="panel-actions">
      <div class="ph">
        <span class="ph-label">Actions</span>
        <div class="filter-row">
          <button class="fb on" onclick="setF('all',this)">all</button>
          <button class="fb" onclick="setF('ok',this)">✓</button>
          <button class="fb" onclick="setF('fail',this)">✗</button>
          <button class="fb" onclick="setF('retry',this)">↻</button>
        </div>
      </div>
      <div class="feed" id="act-feed"></div>
    </div>
  </div>

</div>
<script>
const AC = {{ agent_colors | tojson }};
AC['SYSTEM'] = '#6b6864';
let actFilter = 'all', allActions = [], settledAgents = [];

function setF(f,btn){
  actFilter=f;
  document.querySelectorAll('.fb').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  renderActions();
}

/* ── Drawer toggles ── */
let chatOpen=false, actionsOpen=false;
function updateDrawer(){
  document.getElementById('panel-chat').classList.toggle('active', chatOpen);
  document.getElementById('panel-actions').classList.toggle('active', actionsOpen);
  const openCount=(chatOpen?1:0)+(actionsOpen?1:0);
  document.getElementById('drawer').className='drawer'+(openCount===1?' w1':openCount===2?' w2':'');
  document.getElementById('btn-chat').classList.toggle('on', chatOpen);
  document.getElementById('btn-actions').classList.toggle('on', actionsOpen);
}
function toggleDrawer(which){
  if(which==='chat') chatOpen=!chatOpen;
  if(which==='actions') actionsOpen=!actionsOpen;
  updateDrawer();
}

let seatByAgent = {};

function seatKind(col, cols){
  if(cols<=1) return 'window';
  const half = Math.floor(cols/2);
  if(col===1 || col===cols) return 'window';
  if(cols%2===0){
    if(col===half || col===half+1) return 'aisle';
  } else if(col===half+1){
    return 'aisle';
  }
  return 'middle';
}

function splitCols(cols){
  const half = Math.floor(cols/2);
  if(cols%2===0) return {left:half, gap:true};
  return {left:cols, gap:false};
}

function seatCellHtml(s, cols){
  let cls='seat ';
  if(!s.occupant){ cls+='empty'; }
  else if(s.settled){ cls+='settled'; }
  else if(s.last_moved){ cls+='moved'; }
  else { cls+='occ'; }
  cls += ' k-'+seatKind(s.col, cols);

  const col=AC[s.occupant]||'#888';
  const letter = String.fromCharCode(64+s.col);
  const label = `${s.row}${letter}`;

  let inner = `<span class="seat-id">${label}</span>`;
  if(s.occupant){
    inner += `<span class="agent-n" style="color:${col}">${s.occupant.replace('agent_','A')}</span>`;
    if(s.settled){
      inner += `<span class="settled-tag">✓ set</span>`;
    }
  } else {
    const icon = seatKind(s.col, cols)==='window' ? '○' : seatKind(s.col, cols)==='aisle' ? '›' : '—';
    inner += `<span style="font-size:9px;color:var(--text3)">${icon}</span>`;
  }
  return `<div class="${cls}" id="seat-${s.id}">${inner}</div>`;
}

function rowLettersHtml(cols){
  const {left,gap}=splitCols(cols);
  let cells='';
  for(let c=1;c<=left;c++) cells+=`<span class="col-letter">${String.fromCharCode(64+c)}</span>`;
  if(gap){
    cells+=`<div class="aisle-gap"></div>`;
    for(let c=left+1;c<=cols;c++) cells+=`<span class="col-letter">${String.fromCharCode(64+c)}</span>`;
  }
  return `<div class="plane-row header-row"><div class="row-num"></div><div class="row-seats">${cells}</div></div>`;
}

function renderPlaneRow(rowSeats, cols){
  const {left,gap}=splitCols(cols);
  let cells='';
  rowSeats.forEach((s,i)=>{
    if(gap && i===left) cells+=`<div class="aisle-gap"></div>`;
    cells+=seatCellHtml(s, cols);
  });
  const rowNum = rowSeats.length ? rowSeats[0].row : '';
  return `<div class="plane-row"><div class="row-num">${rowNum}</div><div class="row-seats">${cells}</div></div>`;
}

function renderGrid(seats,cols){
  seatByAgent = {};
  seats.forEach(s=>{ if(s.occupant) seatByAgent[s.occupant]=s.id; });

  let html = rowLettersHtml(cols);
  for(let i=0;i<seats.length;i+=cols){
    html += renderPlaneRow(seats.slice(i,i+cols), cols);
  }
  document.getElementById('seat-grid').innerHTML = html;
}

function renderStanding(agents){
  const el=document.getElementById('standing-list');
  if(!agents || agents.length===0){
    el.innerHTML='<span class="standing-empty">none</span>';
    return;
  }
  el.innerHTML=agents.map(a=>{
    const col=AC[a]||'#888';
    return `<span class="standing-chip" data-agent="${a}" style="background:${col}">${a.replace('agent_','A')}</span>`;
  }).join('');
}

/* ── Speech bubbles ── */
let lastChatIdx = -1, bubblesInitialized = false;
const bubbleTimers = {};

function ensureBubbleLayer(){
  const panel=document.querySelector('.stage-inner');
  let layer=document.getElementById('bubble-layer');
  if(!layer){
    layer=document.createElement('div');
    layer.id='bubble-layer';
    layer.className='bubble-layer';
    panel.appendChild(layer);
  }
  return layer;
}

function showBubble(agentId, text){
  const panel=document.querySelector('.stage-inner');
  const layer=ensureBubbleLayer();
  const seatEl=document.getElementById('seat-'+(seatByAgent[agentId]||''));
  const anchor=seatEl || document.querySelector(`.standing-chip[data-agent="${agentId}"]`) || document.getElementById('standing-list');
  if(!anchor) return;

  let bub=document.getElementById('bubble-'+agentId);
  if(!bub){
    bub=document.createElement('div');
    bub.id='bubble-'+agentId;
    bub.className='bubble';
    layer.appendChild(bub);
  }
  const color=AC[agentId]||'#888';
  const shown = (text||'').length>140 ? text.slice(0,140)+'…' : (text||'…');
  bub.innerHTML=`<div class="b-agent" style="color:${color}">${agentId.replace('agent_','A')}</div>${shown}`;

  const pr=panel.getBoundingClientRect();
  const ar=anchor.getBoundingClientRect();
  bub.style.left=(ar.left-pr.left+ar.width/2+panel.scrollLeft)+'px';
  bub.style.top=(ar.top-pr.top+panel.scrollTop)+'px';

  requestAnimationFrame(()=>bub.classList.add('show'));
  clearTimeout(bubbleTimers[agentId]);
  bubbleTimers[agentId]=setTimeout(()=>{
    bub.classList.remove('show');
    setTimeout(()=>{ bub.remove(); }, 300);
  }, 5000);
}

function processChats(chats){
  if(!bubblesInitialized){
    lastChatIdx = chats.reduce((m,c)=>Math.max(m,c.idx), -1);
    bubblesInitialized = true;
    return;
  }
  const fresh = chats.filter(c=>c.idx>lastChatIdx).sort((a,b)=>a.idx-b.idx);
  fresh.forEach(c=>showBubble(c.agent, c.content));
  if(fresh.length) lastChatIdx = fresh[fresh.length-1].idx;
}

function renderChats(chats){
  const f=document.getElementById('chat-feed');
  f.innerHTML=[...chats].reverse().map(e=>{
    const c=AC[e.agent]||'#9ca3af';
    const pb=e.phase==='planning'?'bp':'be';
    return `<div class="msg chat">
      <div class="msg-top">
        <span class="chip" style="background:${c}">${e.agent}</span>
        <span class="badge ${pb}">${e.phase}</span>
        <span class="itag">iter ${e.iter} · ${e.time}</span>
      </div>
      <div class="mbody">${e.content}</div>
    </div>`;
  }).join('');
  f.scrollTop=0;
  document.getElementById('chat-count').textContent=chats.length;
}

function actionLabel(e){
  const a = e.action||'';
  if(a==='move'){
    if(e.moved_from&&e.moved_to)
      return `move <code>${e.moved_from}</code> → <code>${e.moved_to}</code>`;
    if(e.moved_to)
      return `move → <code>${e.moved_to}</code>`;
    return `move <b style="color:var(--red)">?</b>`;
  }
  if(a==='request_move') return `request_move → <b>${e.target||'?'}</b>`;
  if(a==='complain')     return `complain → <b>${e.target||'?'}</b>`;
  if(a==='stay')         return `stay <span style="color:var(--text3)">at</span> <code>${e.current_seat||'seat'}</code>`;
  if(a==='stand')        return `stand <span style="color:var(--text3)">(vacate ${e.moved_from?'<code>'+e.moved_from+'</code>':'seat'})</span>`;
  if(a==='settle')       return `settle ✦ <code>${e.current_seat||'seat'}</code>`;
  return `<b>${a||'?'}</b>`;
}

function actionClass(e){
  const a=e.action||'';
  if(e.status==='success'){
    if(a==='settle') return 'act-settle';
    if(a==='stay')   return 'act-stay';
    if(a==='stand')  return 'act-stand';
    return 'act-ok';
  }
  if(e.status==='failed') return 'act-fail';
  if(e.status==='retry')  return 'act-retry';
  return 'act-ok';
}

function renderActions(){
  const f=document.getElementById('act-feed');
  let filtered=allActions;
  if(actFilter==='ok')    filtered=allActions.filter(e=>e.status==='success');
  if(actFilter==='fail')  filtered=allActions.filter(e=>e.status==='failed');
  if(actFilter==='retry') filtered=allActions.filter(e=>e.status==='retry');
  f.innerHTML=[...filtered].reverse().map(e=>{
    const c=AC[e.agent]||'#9ca3af';
    const pb=e.phase==='planning'?'bp':'be';
    const cls=actionClass(e);
    let sTag='';
    if(e.status==='success') sTag=`<span class="status-badge s-ok">ok</span>`;
    else if(e.status==='failed') sTag=`<span class="status-badge s-fail">failed</span>`;
    else if(e.status==='retry')  sTag=`<span class="status-badge s-retry">retry</span>`;

    // Show reason for non-success
    const showReason = (e.status==='failed'||e.status==='retry') && e.reason;
    const reasonHtml = showReason
      ? `<div class="reason">↳ ${e.reason}</div>`
      : '';

    return `<div class="msg ${cls}">
      <div class="msg-top">
        <span class="chip" style="background:${c}">${e.agent}</span>
        <span class="badge ${pb}">${e.phase}</span>
        <span class="itag">#${e.idx} · iter ${e.iter}</span>
        ${sTag}
      </div>
      <div class="mbody">${actionLabel(e)}</div>
      ${reasonHtml}
    </div>`;
  }).join('');
  f.scrollTop=0;
  document.getElementById('act-count').textContent=allActions.length;
}

function apply(d){
  if(d.agent_colors) Object.assign(AC, d.agent_colors);
  allActions=d.actions;
  settledAgents=d.settled_agents||[];
  document.getElementById('s-iter').textContent=d.iter||'—';
  document.getElementById('s-moves').textContent=d.total_moves;
  document.getElementById('s-evts').textContent=d.total_events;
  document.getElementById('hdr-iter').textContent=d.iter||'—';
  const ph=document.getElementById('hdr-phase');
  ph.textContent=d.phase||'—';
  ph.className='badge '+(d.phase==='planning'?'bp':d.phase==='execution'?'be':'bn');
  renderGrid(d.seats,d.cols);
  renderStanding(d.standing_agents||[]);
  processChats(d.chats||[]);
  renderChats(d.chats);
  renderActions();
}

const es=new EventSource('/stream');
es.onmessage=e=>{try{apply(JSON.parse(e.data))}catch(_){}};
fetch('/state').then(r=>r.json()).then(apply);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML, log=args.log, agent_colors=_agent_colors(args.agents))

@app.route("/state")
def state_ep():
    return json.dumps(build_payload()), 200, {"Content-Type":"application/json"}

@app.route("/stream")
def stream():
    q = []
    with sse_lock:
        sse_clients.append(q)
    q.append(f"data: {json.dumps(build_payload())}\n\n")
    def gen():
        try:
            while True:
                while q: yield q.pop(0)
                time.sleep(0.1)
        except GeneratorExit:
            with sse_lock:
                if q in sse_clients: sse_clients.remove(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

if __name__ == "__main__":
    threading.Thread(target=watch_log, daemon=True).start()
    print(f"[ist_gui] Watching: {args.log}")
    print(f"[ist_gui] http://localhost:{args.port}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
