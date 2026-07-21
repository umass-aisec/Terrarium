#!/bin/bash
ROOT=$(git rev-parse --show-toplevel)

# Kill any leftover GUI from previous run
lsof -ti:5050 | xargs kill -9 2>/dev/null
sleep 1

# Start GUI first (it will auto-detect the new log once sim creates it)
python "$ROOT/terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py" \
    --no-browser &
GUI_PID=$!

# Open browser once
sleep 1
open "http://localhost:5050" 2>/dev/null || xdg-open "http://localhost:5050" 2>/dev/null

# Run sim
python "$ROOT/examples/base_main.py" --config "$ROOT/examples/configs/is_this_seat_taken.yaml" "$@"

kill $GUI_PID 2>/dev/null