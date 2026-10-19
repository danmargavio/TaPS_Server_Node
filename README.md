# TaPS ServerNode

Aggregates CameraNode streams, manages recording sessions, fuses AprilTag
poses into a robot track, and serves the web playback/fusion UI.

## Run (bench)

```
cd server_node          # CWD MUST be server_node/ - static/ resolves from here
python main.py server_node.yaml
```

Then open http://localhost:8090 (bench port; `web.port` in the yaml).

On the bench host (Windows) the SMB log share is unreachable; NT/SMB warnings
at startup are expected and benign.

## Configuration (`server_node.yaml`)

- `cameras[]` - one entry per CameraNode. Use the node's static IP:
  `host: "192.168.4.66"` (mDNS `.local` is unreliable from Windows and fails
  fetches silently). The camera is `CameraNode1` = Arducam UC-852 (OV9782)
  @ 1280x800 MJPG 100 fps - see `../camera_node/docs/SETUP_RPI.md`.
- `fusion.tags` - map of tracked tag36h11 ids (bench: 4, 14, 41). The node
  RECORDS every detection; this map only filters what ServerNode fuses.
- `fusion.tags.<id>.T_robot_tag` - tag mount on robot (currently identity +
  z 0.10 m placeholders until measured).
- `fusion.cameras[].T_field_cam` - camera extrinsics in field frame. The
  shipped value is a BENCH PLACEHOLDER: fused (x, y) are only trustworthy in
  relative terms until a real field survey is done.
- `fusion.window_ms` - fusion window (66 ms default; fine for 30-100 fps input).

## Endpoints

| Route | Description |
|-------|-------------|
| `/` | Web UI (live view, playback, fusion HUD) |
| `/api/cameras` | Per-camera connectivity, frame counts, `low_power` / `recording_state` JSON |
| `/api/cameras/{i}/stream` | MJPEG relay (INFINITE stream - never fetch without timeout) |
| `/api/cameras/{i}/lowpower` | POST `{"enable": bool}` - toggle the CameraNode's power mode (proxied to the node's `/lowpower`) |
| `/api/sessions` | Recorded sessions (downloaded .taps under `recordings/<session>/`) |
| `/api/session/{id}/frame` | Playback frame |
| `/api/session/{id}/fusion` | Fused robot track JSON |
| `/api/session/{id}/fusion.csv` | Track CSV: `frame,t_rel_s,ptp_ns,utc_iso,x_m,y_m,yaw_deg,n_obs,n_cameras,rms_m,observations` |
| `/api/session/{id}/play|pause|step|speed|seek|rotate` | Playback control POSTs |
| `/ws` | WebSocket telemetry |

## CameraNode power modes

CameraNodes **boot in low power mode** (camera device closed, AprilTag
idle; node web UI, PPS sync and NetworkTables stay up). ServerNode polls
each node's `/status` every 2 s and surfaces it:

- `/api/cameras` reports `low_power` (`true`/`false`, `null` = unknown or
  older node build) and `recording_state` (`1` = recording) per camera.
- The web UI shows an amber **LOW POWER** badge on such a camera; its stream
  carries only the node's placeholder frame until capture is enabled.
- Toggle from here with `POST /api/cameras/{i}/lowpower {"enable": false}`
  (proxied verbatim to the node). While low power the node refuses
  recording/calibration starts with HTTP 409 - disable low power first.

## Fusion semantics

- Per-tag positions come from each camera's .taps v0x03 pose records,
  time-windowed by GPS `ptp_ns` and chained through `T_field_cam` /
  `T_robot_tag`.
- Single camera + single tag: position valid, `yaw_deg = null` BY DESIGN
  (heading is not observable from one tag in one view).
- The HUD and CSV read the fusion track from disk after the session's files
  are present; sessions import by dropping .taps files into
  `recordings/<session_name>/`.

## Dependencies

Python 3.10+ with aiohttp, pyyaml, numpy, opencv-python
(`pip install -r requirements.txt`).