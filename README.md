# TaPS Consolidated System

Tracking and Playback System — consolidated CameraNode and ServerNode applications.

## Overview

TaPS provides high-precision robot tracking and multi-camera video recording/playback for FRC-style robot matches. The system consists of:

- **CameraNode** (C++) — Runs on Raspberry Pi devices, captures video from global shutter cameras, records to `.taps` format, streams MJPEG, and responds to NetworkTables triggers
- **ServerNode** (Python) — Runs on a Linux workstation, aggregates camera streams, manages recording sessions, receives log files, and provides a web-based playback interface

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SERVER NODE (Linux Workstation)               │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ CamNode1 │→│ CamNode2 │→│ CamNode3 │→│ CamNode4 │            │
│  │ (C++)    │  │ (C++)    │  │ (C++)    │  │ (C++)    │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│       │              │              │              │                 │
│       └──────────────┴──────────────┴──────────────┘                 │
│                          ↑ 2x2 composited MJPEG                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    ServerNode (Python)                        │   │
│  │  CameraAggregator │ SessionManager │ LogPoller │ Playback    │   │
│  │  WebServer (aiohttp) + WebSocket + HTML Frontend             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          ↑ SMB poll                                   │
│              ┌───────────────────────┐                               │
│              │ Robot Control PC      │                               │
│              │ (Windows SMB share)   │                               │
│              └───────────────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              NetworkTables (RoboRIO Client)                   │   │
│  │         Listens for matchStart/matchEnd boolean               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
TaPS_Consolidated/
├── common/                         # Shared Python code
│   ├── __init__.py
│   ├── log_parser.py              # FRC DriverStation log parser
│   └── config.py                  # YAML config loader
│
├── camera_node/                    # C++ CameraNode application
│   ├── CMakeLists.txt              # Build configuration
│   ├── src/                        # Source files (enhanced from existing)
│   ├── templates/                  # Web frontend templates
│   └── systemd/                    # systemd service files
│
├── server_node/                    # Python ServerNode application
│   ├── main.py                    # Entry point
│   ├── camera_aggregator.py       # MJPEG stream aggregation
│   ├── session_manager.py         # Recording session management
│   ├── log_poller.py              # SMB share polling
│   ├── log_sync.py                # Log file synchronization
│   ├── playback_engine.py         # .taps file playback
│   ├── web_server.py              # aiohttp web server
│   ├── static/index.html          # Web frontend
│   └── requirements.txt
│
├── shared_config/                  # Configuration templates
│   ├── camera_node_template.yaml
│   └── server_node_template.yaml
│
└── README.md                       # This file
```

## Quick Start

### Prerequisites

**CameraNode (Raspberry Pi):**
- Ubuntu for Raspberry Pi (headless)
- CMake 3.28+, GCC with C++20 support
- OpenCV 5.0+ with V4L2 support
- spdlog, libhttpserver
- libgpiod-dev (for GPIO/PPS)
- Python 3 with pyntcore package (for C++ ntcore bindings)

**ServerNode (Linux Workstation):**
- Ubuntu 22.04+
- Python 3.10+
- OpenCV 4.8+

### CameraNode Setup

1. Copy the template and edit for your camera:
```bash
cp shared_config/camera_node_template.yaml camera_node.yaml
```

2. Build:
```bash
cd camera_node
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

3. Install systemd service:
```bash
sudo cp systemd/camera_node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable camera_node
sudo systemctl start camera_node
```

### ServerNode Setup

1. Copy the template and edit for your setup:
```bash
cp shared_config/server_node_template.yaml server_node.yaml
```

2. Install Python dependencies:
```bash
cd server_node
pip install -r requirements.txt
pip install -e ../common  # If using as a package
```

3. Run:
```bash
python main.py server_node.yaml
```

4. Open a browser to `http://<server-ip>:8080`

## .taps File Format

Full specification: [`../common/taps_format.md`](../common/taps_format.md)

The `.taps` binary format stores video frames with GPS-aligned nanosecond
timestamps (PPS + chrony discipline; PTP optional). Version `0x02` is the
baseline; version `0x03` adds per-frame AprilTag detections (id, pose, and
quality/uncertainty metrics for fusion) plus camera intrinsics in the header.

```
Header:
  Magic:     'TaPS\x02' or 'TaPS\x03' (5 bytes)
  Encoder:   uint8 (0=JPEG, 1=RAW)
  Width:     uint64
  Height:    uint64
  FPS:       double
  ArgsLen:   uint32
  Args:      string (variable length)
  FrameCount:uint64
  [0x03 only] Fx,Fy,Cx,Cy: 4×double (intrinsics) · TagSizeM: double
            · CameraAlias: u32+string · TagFamily: u32+string

Frame (repeated):
  FrameIdx:  uint64
  PTP_Ns:    int64        (GPS-aligned nanoseconds; see spec)
  Size:      uint32
  [0x03 only] MetaSize:   uint32
  Data:      bytes (variable length)
  [0x03 only] Meta:       bytes (AprilTag records; see spec)
```

## Web Frontend

The ServerNode provides a web interface with:

- **Live Mode**: Real-time 2x2 grid view of all cameras
- **Playback Mode**: Frame-by-frame playback of recorded sessions with:
  - Play/Pause/Step controls
  - Speed control (0.25x–4x)
  - Per-stream rotation
  - Log message overlay synchronized with playback time
  - Session browser

## Configuration

### CameraNode (YAML)
```yaml
camera:
  device_id: 0          # /dev/video0
  width: 1920
  height: 1080
  fps: 60
  fourcc: "MJPG"

recording:
  output_dir: "/recordings"
  encoder: "jpeg"
  quality: 85

streaming:
  http_port: 8080
  compose_grid: true

pps:
  gpio_pin: 17

network_tables:
  server: "10.0.1.X"
  match_start_topic: "RoboRIO/matchStart"
  match_end_topic: "RoboRIO/matchEnd"

identity:
  name: "CameraNode1"
  index: 1
```

### ServerNode (YAML)
```yaml
cameras:
  - name: "CameraNode1"
    host: "10.0.1.11"
    http_port: 8080

network_tables:
  server: "10.0.1.X"
  match_start_topic: "RoboRIO/matchStart"
  match_end_topic: "RoboRIO/matchEnd"

smb_share:
  path: "\\\\10.0.2.X\\share\\logs"
  poll_interval: 30

web:
  host: "0.0.0.0"
  port: 8080
```

## API Reference

### CameraNode Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream` | GET | MJPEG single camera stream |
| `/grid-stream` | GET | MJPEG 2x2 composited grid |
| `/status` | GET | Current status JSON |
| `/recording` | POST | Start/stop recording (`{"action": "start/stop"}`) |
| `/events` | GET | SSE events (status, disk, cpu, mem) |
| `/files/` | GET | Browse recorded files |
| `/calibrate` | GET | Guided calibration wizard (focus test + capture + save) |
| `/calib/status` | GET | Calibration state / focus score / coverage JSON |
| `/calib/control` | POST | `{"action":"start|next|finish|abort|reset", ...}` |
| `/calib/stream` | GET | MJPEG calibration view with overlays |

### ServerNode Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/api/cameras` | GET | Camera status list |
| `/api/grid/stream` | GET | Composited grid JPEG |
| `/api/sessions` | GET | Available sessions |
| `/api/session/{id}/frame` | GET | Current playback frame |
| `/api/session/{id}/logs` | GET | Log entries at current time |
| `/api/session/{id}/play` | POST | Start playback |
| `/api/session/{id}/pause` | POST | Pause/resume |
| `/api/session/{id}/step` | POST | Frame step |
| `/api/session/{id}/speed` | POST | Set speed |
| `/api/session/{id}/rotate` | POST | Set rotation |
| `/api/session/{id}/seek` | POST | Seek to time |
| `/api/session/{id}/fusion` | GET | Pose-fusion track JSON (or `{"available": false, "reason": ...}`) |
| `/api/session/{id}/fusion/frame/{frame_idx}` | GET | Single fusion track entry (404 if the frame has no entry) |
| `/ws` | WS | Real-time WebSocket |

## Pose Fusion

Multi-camera AprilTag fusion (`pose_fusion.py`) reconstructs the robot's
planar pose `(x, y, yaw)` per reference-camera frame from the per-frame tag
detections stored in `.taps` **v0x03** recordings (see
[`../common/taps_format.md`](../common/taps_format.md)).

**Requirements**

- v0x03 recordings (AprilTag extras enabled on the CameraNodes) in the session directory
- A `fusion` section in `server_node.yaml` — see
  [`../shared_config/fusion_example.yaml`](../shared_config/fusion_example.yaml) —
  providing per-camera `T_field_cam` (4×4 row-major camera pose in the field frame,
  keyed by the `camera_alias` stored in each file header), per-tag `T_robot_tag`
  for the rigid tag mounts on the robot, `window_ms` (cross-camera PPS alignment
  window), and an optional `reference_camera` (default: longest recording)
- `numpy` (see `requirements.txt`)

**Endpoints** (read-only, computed lazily on first request, cached in memory):
`GET /api/session/{id}/fusion` returns `{"available": true, "reference_camera", "window_ms",
"cameras", "tag_ids", "n_frames", "track": [...]}` or `{"available": false, "reason": ...}`
(e.g. "no fusion config", v0x02-only recordings). Track entries:
`{"frame", "ptp_ns", "x", "y", "yaw_deg"|null, "n_obs", "n_cameras", "rms", "candidates": [...]}`.
`GET /api/session/{id}/fusion/frame/{frame_idx}` returns one entry or 404.
Results are also cached on disk as `fusion_track.json` inside the session directory and
recomputed automatically when any `.taps` file is newer.

**Caveats (v1 heuristics — not calibrated)**

- Observation weighting uses `σ = (reproj_error_rms_px / tag_px_diag) × distance` —
  an angular-pose-error × range model, not a true covariance; `decision_margin` and
  `hamming` are recorded but unused in the weights
- Tag mount heights are projected out (planar model); a single observation yields
  `x, y` with `yaw_deg: null`
- Frames with no accepted detections simply have no track entry (the frame endpoint 404s)

## Development Notes

### Reused Code
- `.taps` reader/writer from C++ CameraNode project
- Log parser from playback_server project
- NetworkTables trigger patterns from robot_tracker project
- HTTP server patterns from C++ http_server.h
- Web frontend inspired by C++ templates/index.html

### Future Work
- [ ] CameraNode: Implement PPS handler (pps_handler.h)
- [ ] CameraNode: Implement NetworkTables client (nt_client.h)
- [ ] CameraNode: Implement grid composition endpoint
- [ ] ServerNode: Full SMB polling implementation (requires smbprotocol setup)
- [ ] ServerNode: WebSocket session loading
- [ ] Testing: End-to-end integration tests
- [ ] Documentation: Deployment guides for CameraNode and ServerNode
