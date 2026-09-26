# 2026-09-26 - Front camera finds the shuttle by motion, not YOLO

## What changed
- New shuttle source `"motion"` (`utils/shuttle_motion.py`): MOG2 background
  subtraction + a Kalman lock on the one shuttle in flight. No neural network,
  ~5-15 ms/frame, so it fits on the Pi next to pose. It is the new default in
  `config/settings.py`, but a Pi that already has saved settings keeps its old
  `shuttleSource` until it is switched (below).
- Flicker suppression: pixels that keep "moving" in the same place (fluorescent
  tubes, the white lines under them) are ignored.
- Needs the front camera **fixed**. A bumped or moving camera floods the
  detector with false motion until it relearns (a new drill resets it).
- The Settings page has no "Motion" option yet; switch it through the API.
  Leave the page's shuttle-source dropdown alone afterwards.
- No UI rebuild, no new packages.

## Commands
    cd /opt/aerosense && git pull && sudo setup/deploy/install.sh
    # switch the saved shuttle source to motion
    curl -s localhost:8000/api/settings \
      | python3 -c "import sys,json; s=json.load(sys.stdin); s['detection']['shuttleSource']='motion'; print(json.dumps(s))" \
      | curl -s -X PUT -H 'Content-Type: application/json' -d @- localhost:8000/api/settings >/dev/null
    curl -s localhost:8000/api/settings | grep -o '"shuttleSource":"[a-z]*"'

## Checks
- [ ] The last command prints `"shuttleSource":"motion"`
- [ ] Start a drill: the front view shows a shuttle marker following the
      shuttle in flight
- [ ] With nothing moving, no shuttle marker sits on the ceiling lights

## Rollback
Set `shuttleSource` back to `"local"` with the same command.
