# Terrarium Dashboards

This folder contains the static dashboards plus helper scripts to build the data bundle they use.

## Files
- `build_data.py` — generates `dashboard_data.json` from the `logs/` directory.
- `dashboard_v2/index.html` — the live UI (polls `dashboard_data.json` and log files).
- `health_metric.py` — example Python hook to annotate runs with a custom health badge (`run.health`).
- `public/` — prebuilt dashboard + assets (legacy UI).

## Using a custom health metric
1) Edit `dashboards/health_metric.py` and implement `compute_health(run: dict) -> dict|bool|None`.
   - Recommended dict shape:
     ```python
     {
         "ok": bool,            # True = healthy, False = needs attention
         "label": "Your label", # badge text
         "reason": "details",   # tooltip text (optional)
         "score": 72.5,         # numeric (optional)
     }
     ```
   - Return `None` to skip; return `True`/`False` to set only `ok`.

2) Regenerate the data bundle with the hook applied:
   ```bash
   python dashboards/build_data.py \
     --logs-root logs \
     --health-metric dashboards/health_metric.py \
     --output dashboards/dashboard_v2/dashboard_data.json
   ```
   Notes:
   - If `--health-metric` is omitted and `dashboards/health_metric.py` exists, it is applied by default.
   - Point `--output` to the bundle your dashboard is loading (v2 tries `dashboard_v2/dashboard_data.json` first, then falls back to `public/dashboard_data.json`).

3) Serve the repo (so `/logs` and `dashboards/...` are reachable), e.g.:
   ```bash
   python -m http.server 8000
   ```
   Then open `http://localhost:8000/dashboards/dashboard_v2/index.html`.

4) For near–real-time updates, rerun `build_data.py` periodically (e.g., a small shell loop) while the dashboard auto-polls the JSON/logs.

## Where success_rate comes from
`build_data.py` inspects `scores_iteration_*.json` metadata per environment (e.g., MeetingScheduling) to compute a completion ratio. If absent, the UI falls back to event counts or your `run.health`. Customize the logic via your `compute_health` hook. 
