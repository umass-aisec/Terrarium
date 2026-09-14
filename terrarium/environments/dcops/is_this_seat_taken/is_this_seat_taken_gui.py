"""
IsThisSeatTaken — real-time GUI  v2
Run:  python is_this_seat_taken_gui.py
      python is_this_seat_taken_gui.py --log path/to/blackboard_0.txt --rows 2 --cols 4
Opens http://localhost:5050 automatically.
"""

import argparse, glob, json, os, re, threading, time, webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, Response, render_template

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
    moved_from = moved_to = current_seat = None
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
            elif a == "stand":
                moved_from = res.get("previous_seat")
    except Exception:
        pass
    return moved_from, moved_to, current_seat, settled_result

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
        moved_from = moved_to = current_seat = None
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
            moved_from, moved_to, current_seat, settled_result = _parse_details(" ".join(details_buf))
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
            settled_result=settled_result,
            target=target_agent, reason=reason,
        ))
        i = j
    return events

def _load_initial_seating(log_path):
    """Starting positions from initial_seating.json next to the blackboard log.

    Returns None for runs recorded before the environment wrote that file; those
    carried the layout as an "Initial seating:" chat message instead.
    """
    path = Path(log_path).parent / "initial_seating.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def infer_positions(events, initial=None):
    pos = dict(initial or {})
    settled = set()
    for e in events:
        if initial is None and e.etype == "context" and "Initial seating:" in e.content:
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
            if e.action == "stand" and e.moved_from:
                # agent vacated a seat — remove from pos
                if pos.get(e.agent) == e.moved_from:
                    pos.pop(e.agent, None)
                settled.discard(e.agent)
            if e.action == "settle":
                settled.add(e.agent)
            if e.action in ("move", "stand") and not e.settled_result:
                settled.discard(e.agent)
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

def rebuild(events, initial=None):
    pos, settled = infer_positions(events, initial)
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
                initial = _load_initial_seating(args.log)
                pos, settled, moves, last_moved = rebuild(events, initial)
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
            "log": args.log,
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

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))

@app.route("/")
def index():
    return render_template("is_this_seat_taken.html", log=args.log, agent_colors=_agent_colors(args.agents))

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
