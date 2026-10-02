"""
Multi-camera AprilTag pose fusion for ServerNode.

Fuses per-frame AprilTag detections from several time-synchronized camera
recordings (.taps v0x03 — see common/taps_format.md) into a planar robot
pose track (x, y, yaw) in a fixed "field" coordinate frame.

Algorithm (v1, deterministic, numpy):

* The reference camera defines the output frame grid. For every reference
  frame with timestamp t, each other camera contributes its nearest-PTP
  frame when |Δt| <= window_ms (cameras share one GPS/PPS-disciplined
  nanosecond timeline, so alignment is a direct timestamp comparison).
* Detections that have ``pose_valid`` set AND a configured tag id become
  observations: ``p_field = R(field<-cam) @ translation + t(field<-cam)``
  in meters.
* Each observation gets weight ``w = 1/sigma^2`` using the v1 heuristic

      sigma = max(1e-4, (reproj_error_rms_px / max(tag_px_diag, 1.0))
                          * max(distance_m, 0.1))

  where ``distance_m = ||translation||`` (range in the camera frame).
  Rationale: the tag's RMS reprojection error is a pixel-domain estimate
  of corner uncertainty; dividing by the tag's apparent size in pixels
  converts it to an *angular* pose error, and multiplying by range turns
  the angular error into a metric position error at the tag. Larger
  (closer) tags and cleaner decodes therefore weigh more. This models
  angular pose error x range — it is a heuristic, NOT a calibrated
  covariance.

* The robot planar pose (x, y, yaw) is solved by weighted Gauss-Newton on
  the model ``p_pred_i = p + Rz(yaw) @ u_i``, where ``u_i`` is the
  configured robot-frame tag offset ``T_robot_tag`` projected onto the
  ground plane (its z is ignored). Initialization: p = weighted mean of
  (p_field_i - u_i), yaw = 0. Up to 10 iterations, step-norm tolerance
  1e-10, per-step clamps of 0.5 m / 0.5 rad. With a single observation
  the report is x,y from that candidate with yaw = None.

Heuristic caveats (v1): decision_margin and hamming are not used in the
weighting; tag-mount heights are flattened; occluded frames simply
produce no track entry.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # running as part of the server_node package (main.py / web_server.py)
    from server_node.playback_engine import TagDetection, TapsReader, VERSION_V3
except ImportError:  # direct script/test import from within server_node/
    from playback_engine import TagDetection, TapsReader, VERSION_V3

logger = logging.getLogger('server_node.pose_fusion')

CACHE_FILENAME = 'fusion_track.json'

_MAX_GN_ITERATIONS = 10
_STEP_TOL = 1e-10          # combined step-norm (m + rad) convergence tolerance
_MAX_STEP_M = 0.5          # per-iteration translation step clamp
_MAX_STEP_RAD = 0.5        # per-iteration yaw step clamp
_SIGMA_MIN_M = 1e-4        # sigma floor, meters
_DISTANCE_MIN_M = 0.1      # range floor used in the sigma heuristic


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraTrack:
    """Per-camera fusion input.

    frames: list indexed by dense frame_idx; each entry is either None
    (frame absent) or a ``(ptp_ns: int, tags: list[TagDetection])`` tuple.
    T_field_cam: 4x4 row-major camera pose in the field frame.
    """
    alias: str
    frames: list
    T_field_cam: np.ndarray

    def valid_frames(self) -> list:
        """[(frame_idx, (ptp_ns, tags)), ...] for present frames."""
        return [(i, f) for i, f in enumerate(self.frames) if f is not None]


@dataclass
class Observation:
    """One accepted tag detection expressed in the field frame."""
    camera: str
    tag_id: int
    p_field: np.ndarray   # (3,) tag origin in field frame, meters
    u: np.ndarray         # (2,) robot-frame tag offset projected to xy
    w: float              # 1 / sigma^2


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

def _mat4(values, what: str) -> np.ndarray:
    """Parse a 4x4 row-major matrix given as a flat 16-list (or nested 4x4)."""
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: expected 16 numbers (4x4 row-major)")
    if arr.size != 16:
        raise ValueError(f"{what}: expected 16 numbers (4x4 row-major), got {arr.size}")
    return arr.reshape(4, 4)


def parse_fusion_config(fusion_cfg: dict):
    """Validate the raw `fusion` YAML section.

    Returns (window_ms, reference_camera|None, cameras, tags) where cameras
    is a list of (alias, 4x4 np.ndarray) in config order and tags maps
    tag_id -> 4x4 T_robot_tag. Raises ValueError on anything unusable.
    """
    if not isinstance(fusion_cfg, dict):
        raise ValueError("fusion config must be a mapping")

    window_ms = float(fusion_cfg.get('window_ms', 30.0))
    if not (window_ms > 0):
        raise ValueError(f"window_ms must be > 0, got {window_ms}")

    reference_camera = fusion_cfg.get('reference_camera') or None

    cameras: list[tuple[str, np.ndarray]] = []
    for i, entry in enumerate(fusion_cfg.get('cameras') or []):
        if not isinstance(entry, dict):
            raise ValueError(f"fusion.cameras[{i}]: expected a mapping with alias/T_field_cam")
        alias = str(entry.get('alias', '')).strip()
        if not alias:
            raise ValueError(f"fusion.cameras[{i}]: missing 'alias'")
        cameras.append((alias, _mat4(entry.get('T_field_cam'),
                                     f"T_field_cam for camera '{alias}'")))
    if not cameras:
        raise ValueError("fusion config requires at least one camera")

    tags: dict[int, np.ndarray] = {}
    for key, val in (fusion_cfg.get('tags') or {}).items():
        try:
            tag_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"fusion.tags: bad tag id {key!r}")
        mat = val.get('T_robot_tag') if isinstance(val, dict) else val
        tags[tag_id] = _mat4(mat, f"T_robot_tag for tag {tag_id}")
    if not tags:
        raise ValueError("fusion config requires at least one tag (fusion.tags)")

    return window_ms, reference_camera, cameras, tags


# ---------------------------------------------------------------------------
#  Weighting + Gauss-Newton solver
# ---------------------------------------------------------------------------

def observation_weight(td: TagDetection) -> float:
    """Fuse weight w = 1/sigma^2 for one detection (heuristic; see module docstring)."""
    tr = np.asarray(td.translation, dtype=np.float64)
    distance = float(np.linalg.norm(tr))
    angular = float(td.reproj_error_rms_px) / max(float(td.tag_px_diag), 1.0)
    sigma = max(_SIGMA_MIN_M, angular * max(distance, _DISTANCE_MIN_M))
    return 1.0 / (sigma * sigma)


def solve_robot_pose(observations: list) -> dict:
    """Weighted Gauss-Newton solve of planar robot pose from observations.

    Model: p_pred_i = p + Rz(yaw) @ u_i  (u_i = robot-frame tag offset, xy).
    Returns a dict with keys x, y, yaw_deg (None for a single observation),
    rms (weighted RMS residual, meters) and candidates (per-observation
    detail dicts).
    """
    candidates = [{
        'camera': o.camera,
        'tag_id': o.tag_id,
        'p_field': [float(o.p_field[0]), float(o.p_field[1]), float(o.p_field[2])],
        'w': float(o.w),
    } for o in observations]

    P = np.array([[o.p_field[0], o.p_field[1]] for o in observations], dtype=np.float64)
    U = np.array([o.u for o in observations], dtype=np.float64)
    W = np.array([o.w for o in observations], dtype=np.float64)
    sw = float(W.sum())

    # Single observation: position from that candidate alone, yaw unobservable.
    if len(observations) == 1:
        p0 = P[0] - U[0]
        return {
            'x': float(p0[0]),
            'y': float(p0[1]),
            'yaw_deg': None,
            'rms': 0.0,
            'candidates': candidates,
        }

    # Init: p = weighted mean of (p_field_i - u_i), yaw = 0
    p_xy = (W[:, None] * (P - U)).sum(axis=0) / sw
    x, y = float(p_xy[0]), float(p_xy[1])
    yaw = 0.0

    n = len(observations)
    for _ in range(_MAX_GN_ITERATIONS):
        c, s = math.cos(yaw), math.sin(yaw)
        rot_x = c * U[:, 0] - s * U[:, 1]
        rot_y = s * U[:, 0] + c * U[:, 1]
        r = np.empty((n, 2), dtype=np.float64)
        r[:, 0] = P[:, 0] - (x + rot_x)
        r[:, 1] = P[:, 1] - (y + rot_y)

        # Jacobian of p_pred (stacked 2N x 3: d/dx, d/dy, d/dyaw)
        d_yaw = np.empty((n, 2), dtype=np.float64)
        d_yaw[:, 0] = -s * U[:, 0] - c * U[:, 1]
        d_yaw[:, 1] = c * U[:, 0] - s * U[:, 1]
        J = np.zeros((2 * n, 3), dtype=np.float64)
        J[0::2, 0] = 1.0
        J[1::2, 1] = 1.0
        J[0::2, 2] = d_yaw[:, 0]
        J[1::2, 2] = d_yaw[:, 1]

        Wv = np.repeat(W, 2)
        A = J.T @ (Wv[:, None] * J)
        b = J.T @ (Wv * r.reshape(-1))
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, b, rcond=None)[0]

        # Clamp step: norm(xy) <= 0.5 m, |dyaw| <= 0.5 rad
        step_xy = float(np.linalg.norm(delta[:2]))
        if step_xy > _MAX_STEP_M:
            delta[:2] *= _MAX_STEP_M / step_xy
        delta[2] = max(-_MAX_STEP_RAD, min(_MAX_STEP_RAD, float(delta[2])))

        x += float(delta[0])
        y += float(delta[1])
        yaw += float(delta[2])

        if math.hypot(float(np.linalg.norm(delta[:2])), abs(float(delta[2]))) < _STEP_TOL:
            break

    # Weighted RMS residual at the solution
    c, s = math.cos(yaw), math.sin(yaw)
    rx = P[:, 0] - (x + c * U[:, 0] - s * U[:, 1])
    ry = P[:, 1] - (y + s * U[:, 0] + c * U[:, 1])
    rms = math.sqrt(float((W * (rx * rx + ry * ry)).sum()) / sw)

    return {
        'x': float(x),
        'y': float(y),
        'yaw_deg': float(math.degrees(yaw)),
        'rms': float(rms),
        'candidates': candidates,
    }


# ---------------------------------------------------------------------------
#  FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """Builds a (x, y, yaw) robot pose track from per-camera tag detections.

    Constructed either directly from CameraTrack inputs or via
    FusionEngine.from_session(), which reads a session directory of .taps
    files and matches them against a parsed `fusion` config section.
    """

    def __init__(self, cameras: list, tags: dict,
                 window_ms: float = 30.0,
                 reference_camera: Optional[str] = None):
        if not cameras:
            raise ValueError("FusionEngine requires at least one CameraTrack")
        if not tags:
            raise ValueError("FusionEngine requires at least one configured tag")

        self.cameras = list(cameras)
        self.tags = {int(k): np.asarray(v, dtype=np.float64) for k, v in tags.items()}
        self.window_ms = float(window_ms)

        by_alias = {c.alias: c for c in self.cameras}
        if reference_camera and reference_camera in by_alias:
            ref = by_alias[reference_camera]
        else:
            if reference_camera:
                logger.warning(f"Fusion reference_camera {reference_camera!r} not among "
                               f"matched cameras; defaulting to longest recording")
            # Longest recording = most frames; deterministic alias tie-break
            ref = min(self.cameras, key=lambda c: (-len(c.valid_frames()), c.alias))
        self.reference_camera = ref.alias

        # Robot-frame tag offsets projected to the ground plane (z ignored)
        self._tag_u = {tid: t[:2, 3].copy() for tid, t in self.tags.items()}

        self.track = self._build_track()

    # -- track construction --------------------------------------------------

    def _observations_from(self, cam: CameraTrack, frame) -> list:
        """Accepted observations (pose_valid + configured id) from one camera frame."""
        out = []
        _, detections = frame
        if not detections:
            return out
        R = cam.T_field_cam[:3, :3]
        tv = cam.T_field_cam[:3, 3]
        for td in detections:
            if not td.pose_valid:
                continue
            u = self._tag_u.get(int(td.tag_id))
            if u is None:
                continue
            p_field = R @ np.asarray(td.translation, dtype=np.float64) + tv
            out.append(Observation(
                camera=cam.alias,
                tag_id=int(td.tag_id),
                p_field=p_field,
                u=u,
                w=observation_weight(td),
            ))
        return out

    def _build_track(self) -> list:
        ref = None
        for c in self.cameras:
            if c.alias == self.reference_camera:
                ref = c
                break

        window_ns = self.window_ms * 1e6

        # Pre-extract non-reference cameras' timestamps for nearest-PTP lookup
        prepared = []
        for c in self.cameras:
            if c is ref:
                continue
            valid = c.valid_frames()
            ptp = np.array([f[1][0] for f in valid], dtype=np.int64)
            prepared.append((c, valid, ptp))

        track = []
        for k, frame in enumerate(ref.frames):
            if frame is None:
                continue
            t = frame[0]

            observations = self._observations_from(ref, frame)
            for cam, valid, ptp in prepared:
                if ptp.size == 0:
                    continue
                # Nearest frame by PPS-disciplined timestamp
                j = int(np.searchsorted(ptp, t))
                best_j, best_dt = None, None
                for cand in (j - 1, j):
                    if 0 <= cand < ptp.size:
                        dt = abs(int(ptp[cand]) - int(t))
                        if best_dt is None or dt < best_dt:
                            best_dt, best_j = dt, cand
                if best_j is None or best_dt > window_ns:
                    continue
                observations.extend(self._observations_from(cam, valid[best_j][1]))

            if not observations:
                continue

            solve = solve_robot_pose(observations)
            track.append({
                'frame': k,
                'ptp_ns': int(t),
                'x': solve['x'],
                'y': solve['y'],
                'yaw_deg': solve['yaw_deg'],
                'n_obs': len(observations),
                'n_cameras': len({o.camera for o in observations}),
                'rms': solve['rms'],
                'candidates': solve['candidates'],
            })
        return track

    # -- output ----------------------------------------------------------------

    def to_json_dict(self) -> dict:
        """JSON-serializable track with config metadata."""
        return {
            'available': True,
            'reference_camera': self.reference_camera,
            'window_ms': self.window_ms,
            'cameras': [
                {
                    'alias': c.alias,
                    'T_field_cam': [float(v) for v in c.T_field_cam.reshape(-1)],
                }
                for c in self.cameras
            ],
            'tag_ids': sorted(int(t) for t in self.tags),
            'n_frames': len(self.track),
            'track': self.track,
        }

    # -- session loading ---------------------------------------------------------

    @classmethod
    def from_session(cls, session_dir, fusion_config) -> Optional['FusionEngine']:
        """Build an engine from a session directory of .taps files.

        Returns None when fusion cannot run: no/invalid config, no .taps
        files, no v0x03 recordings matching configured camera aliases, or
        no valid detections of configured tags anywhere in the session.
        """
        if not fusion_config:
            return None
        # Accept either the raw document or the 'fusion' section itself
        if isinstance(fusion_config, dict) and isinstance(fusion_config.get('fusion'), dict):
            fusion_config = fusion_config['fusion']

        try:
            window_ms, ref_name, cameras_cfg, tags = parse_fusion_config(fusion_config)
        except (ValueError, TypeError) as e:
            logger.warning(f"Fusion config invalid: {e}")
            return None

        session_dir = Path(session_dir)
        files = sorted(session_dir.glob('*.taps'))
        if not files:
            logger.info(f"Fusion: no .taps files in {session_dir}")
            return None

        cfg_T = dict(cameras_cfg)
        by_alias: dict[str, CameraTrack] = {}
        total_detections = 0

        for fpath in files:
            try:
                reader = TapsReader(str(fpath))
            except Exception as e:
                logger.warning(f"Fusion: cannot open {fpath.name}: {e}")
                continue
            try:
                if reader.version != VERSION_V3:
                    logger.info(f"Fusion: skipping v0x{reader.version:02X} file {fpath.name}")
                    continue

                header_alias = reader.header.camera_alias or ''
                alias = header_alias if header_alias in cfg_T else None
                if alias is None and header_alias:
                    logger.warning(f"Fusion: {fpath.name} alias {header_alias!r} not in "
                                   f"fusion config cameras; skipping")
                    continue
                if alias is None:
                    # v0x03 file without alias: fall back to filename matching
                    for cfg_alias in cfg_T:
                        if cfg_alias in fpath.name:
                            alias = cfg_alias
                            break
                if alias is None:
                    logger.warning(f"Fusion: cannot map {fpath.name} to a configured "
                                   f"camera alias; skipping")
                    continue
                if alias in by_alias:
                    logger.warning(f"Fusion: duplicate file for camera {alias}: "
                                   f"{fpath.name}; skipping")
                    continue

                frames: list = []
                while True:
                    frame = reader.read_next_frame()
                    if frame is None:
                        break
                    while len(frames) <= frame.frame_idx:
                        frames.append(None)
                    frames[frame.frame_idx] = (int(frame.ptp_ns), list(frame.tags))
                    total_detections += sum(
                        1 for td in frame.tags
                        if td.pose_valid and int(td.tag_id) in tags
                    )

                by_alias[alias] = CameraTrack(alias=alias, frames=frames,
                                              T_field_cam=cfg_T[alias])
            finally:
                reader.close()

        # Keep config order for deterministic output
        ordered = [by_alias[a] for (a, _T) in cameras_cfg if a in by_alias]
        if not ordered:
            logger.info("Fusion: no v0x03 recordings matched the configured camera aliases")
            return None
        if total_detections == 0:
            logger.info("Fusion: no valid detections of configured tags in session "
                        f"{session_dir.name}")
            return None

        try:
            return cls(ordered, tags, window_ms=window_ms, reference_camera=ref_name)
        except Exception as e:
            logger.error(f"Fusion build failed for {session_dir}: {e}")
            return None


# ---------------------------------------------------------------------------
#  On-disk track cache (session_dir/fusion_track.json)
# ---------------------------------------------------------------------------

def _latest_taps_mtime(session_dir) -> float:
    latest = 0.0
    for f in Path(session_dir).glob('*.taps'):
        try:
            latest = max(latest, f.stat().st_mtime)
        except OSError:
            continue
    return latest


def load_cached_track(session_dir) -> Optional[dict]:
    """Load the cached fusion track, or None if absent/stale/invalid.

    Invalidation rule: any .taps file newer than the cache file.
    """
    path = Path(session_dir) / CACHE_FILENAME
    if not path.exists():
        return None
    try:
        if path.stat().st_mtime < _latest_taps_mtime(session_dir):
            logger.debug(f"Fusion cache stale (a .taps file is newer): {path}")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Discarding unreadable fusion cache {path}: {e}")
        return None
    if not isinstance(data, dict) or not data.get('available'):
        return None
    return data


def save_track_cache(session_dir, track: dict) -> None:
    """Atomically write the fusion track JSON into the session directory."""
    path = Path(session_dir) / CACHE_FILENAME
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(track, f)
    os.replace(tmp, path)


def get_fusion_track(session_dir, fusion_config, use_cache: bool = True) -> Optional[dict]:
    """Full fusion track JSON for a session, using/refreshing the disk cache.

    Returns None when fusion is unavailable (no config, no v0x03 tag data).
    Unavailable results are never cached — config or recordings may change.
    """
    if not fusion_config:
        return None
    if use_cache:
        cached = load_cached_track(session_dir)
        if cached is not None:
            return cached

    engine = FusionEngine.from_session(session_dir, fusion_config)
    if engine is None:
        return None

    data = engine.to_json_dict()
    try:
        save_track_cache(session_dir, data)
    except OSError as e:
        logger.warning(f"Could not write fusion cache in {session_dir}: {e}")
    return data
