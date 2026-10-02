"""
Tests for .taps v0x02/v0x03 parsing (playback_engine.TapsReader) and
multi-camera pose fusion (pose_fusion).

Standalone:  python server_node/test_taps_v3.py
pytest:      pytest server_node/test_taps_v3.py
"""

import contextlib
import itertools
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playback_engine import TagDetection, TapsReader  # noqa: E402
from pose_fusion import (CameraTrack, FusionEngine,  # noqa: E402
                         get_fusion_track, load_cached_track)

IDENTITY9 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# Keep scratch dirs inside the workspace (system temp may be sandboxed away).
# Plain mkdir + rmtree instead of tempfile.TemporaryDirectory: the latter's
# restrictive DACL trips some sandbox setups.
TEST_TMP_ROOT = Path(__file__).resolve().parent / '.test_tmp'
TEST_TMP_ROOT.mkdir(exist_ok=True)
_tmp_counter = itertools.count()


@contextlib.contextmanager
def tmpdir():
    d = TEST_TMP_ROOT / f"t{os.getpid()}_{next(_tmp_counter)}"
    d.mkdir()
    try:
        yield str(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
#  Synthetic .taps builders (packed per common/taps_format.md)
# ---------------------------------------------------------------------------

def pack_tag_record(tag_id=7, decision_margin=8.5, hamming=0, pose_valid=1,
                    translation=(0.25, 0.5, 1.5), rotation=IDENTITY9,
                    reproj_error_rms_px=0.5, tag_px_diag=128.0):
    rec = struct.pack('<HfBB3f9fff', tag_id, decision_margin, hamming, pose_valid,
                      *translation, *rotation, reproj_error_rms_px, tag_px_diag)
    assert len(rec) == 64
    return rec


def build_taps_file(path, frames, version=3, encoder=0, width=1280, height=720,
                    fps=60.0, args='quality:85', fx=1000.5, fy=1001.25,
                    cx=640.0, cy=360.0, tag_size_m=0.16, alias='CameraNode1',
                    family='tag36h11'):
    """Write a .taps file. frames: list of (frame_idx, ptp_ns, image_bytes, meta_bytes)."""
    args_bytes = args.encode('utf-8')
    out = bytearray()
    out += b'TaPS\x02' if version == 2 else b'TaPS\x03'
    out += struct.pack('<BQQdI', encoder, width, height, fps, len(args_bytes))
    out += args_bytes
    out += struct.pack('<Q', len(frames))
    if version == 3:
        out += struct.pack('<4d', fx, fy, cx, cy)
        out += struct.pack('<d', tag_size_m)
        alias_bytes = alias.encode('utf-8')
        family_bytes = family.encode('utf-8')
        out += struct.pack('<I', len(alias_bytes)) + alias_bytes
        out += struct.pack('<I', len(family_bytes)) + family_bytes

    for (idx, ptp_ns, image, meta) in frames:
        if version == 3:
            out += struct.pack('<QqII', idx, ptp_ns, len(image), len(meta))
        else:
            out += struct.pack('<QqI', idx, ptp_ns, len(image))
        out += image
        out += meta

    Path(path).write_bytes(bytes(out))
    return path


def make_meta(records):
    return bytes([len(records)]) + b''.join(records)


# ---------------------------------------------------------------------------
#  TapsReader — v0x02 (regression: args_length offset bug)
# ---------------------------------------------------------------------------

def test_v2_roundtrip():
    with tmpdir() as d:
        f0 = b'\xff\xd8' + b'A' * 37          # fake JPEG payload
        f1 = b'\xff\xd8' + b'B' * 91
        path = build_taps_file(Path(d) / 'old.taps',
                               [(0, 1_000_000_000, f0, b''),
                                (1, 1_016_666_667, f1, b'')],
                               version=2)
        r = TapsReader(str(path))
        try:
            assert r.version == 2
            h = r.header
            assert h.version == 2
            assert h.encoder_type == 0
            assert h.width == 1280 and h.height == 720
            assert h.target_fps == 60.0
            assert h.encoder_args == 'quality:85'   # u32 args_length, not u64
            assert h.frame_count == 2
            assert h.camera_alias is None and h.tag_family is None
            assert h.fx == 0.0 and h.tag_size_m == 0.0

            fr0 = r.read_frame(0)
            assert fr0 is not None and fr0.data == f0 and fr0.ptp_ns == 1_000_000_000
            assert fr0.tags == []
            fr1 = r.read_frame(1)
            assert fr1 is not None and fr1.data == f1
            assert r.seek_to_frame(1) is True
            assert r.seek_to_frame(2) is False
            assert r.read_frame(2) is None
        finally:
            r.close()


def test_bad_magic_rejected():
    with tmpdir() as d:
        path = Path(d) / 'junk.taps'
        path.write_bytes(b'TaPS\x04' + b'\x00' * 100)
        try:
            TapsReader(str(path))
            assert False, "expected ValueError for unknown version byte"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
#  TapsReader — v0x03 header extras + tag meta
# ---------------------------------------------------------------------------

def test_v3_header_and_frames():
    with tmpdir() as d:
        rec = pack_tag_record(tag_id=7, decision_margin=8.5, hamming=1, pose_valid=1,
                              translation=(0.25, 0.5, 1.5), reproj_error_rms_px=0.5,
                              tag_px_diag=128.0)
        f0 = b'\xff\xd8' + b'X' * 50
        f1 = b'\xff\xd8' + b'Y' * 21
        path = build_taps_file(Path(d) / 'new.taps',
                               [(0, 2_000_000_000, f0, b''),          # meta_size = 0
                                (1, 2_016_666_667, f1, make_meta([rec]))],
                               version=3)
        r = TapsReader(str(path))
        try:
            assert r.version == 3
            h = r.header
            assert h.encoder_args == 'quality:85' and h.frame_count == 2
            assert h.fx == 1000.5 and h.fy == 1001.25
            assert h.cx == 640.0 and h.cy == 360.0
            assert h.tag_size_m == 0.16
            assert h.camera_alias == 'CameraNode1'
            assert h.tag_family == 'tag36h11'

            fr0 = r.read_frame(0)
            assert fr0 is not None and fr0.data == f0
            assert fr0.tags == []                      # meta_size=0 → no detections

            fr1 = r.read_frame(1)
            assert fr1 is not None and fr1.data == f1
            assert fr1.ptp_ns == 2_016_666_667
            assert len(fr1.tags) == 1
            t = fr1.tags[0]
            assert t.tag_id == 7
            assert t.decision_margin == 8.5
            assert t.hamming == 1
            assert t.pose_valid is True
            assert t.translation == (0.25, 0.5, 1.5)   # exact in f32
            assert t.rotation == IDENTITY9
            assert t.reproj_error_rms_px == 0.5
            assert t.tag_px_diag == 128.0

            assert r.seek_to_frame(1) is True
            assert r.seek_to_frame(99) is False
        finally:
            r.close()


def test_v3_sequential_read_and_two_tags():
    with tmpdir() as d:
        rec7 = pack_tag_record(tag_id=7, translation=(0.1, 0.2, 1.25))
        rec8 = pack_tag_record(tag_id=8, translation=(-0.5, 0.25, 2.0),
                               reproj_error_rms_px=0.25, tag_px_diag=64.0,
                               hamming=2, pose_valid=0)
        f0 = b'a' * 8
        f1 = b'b' * 8
        path = build_taps_file(Path(d) / 'two.taps',
                               [(0, 10, f0, make_meta([rec7])),
                                (1, 20, f1, make_meta([rec7, rec8]))],
                               version=3)
        r = TapsReader(str(path))
        try:
            fr0 = r.read_next_frame()                   # sequential from start of frames
            assert fr0 is not None and fr0.frame_idx == 0 and fr0.data == f0
            assert len(fr0.tags) == 1 and fr0.tags[0].tag_id == 7
            fr1 = r.read_next_frame()
            assert fr1 is not None and fr1.frame_idx == 1 and fr1.data == f1
            assert len(fr1.tags) == 2
            assert fr1.tags[1].tag_id == 8 and fr1.tags[1].pose_valid is False
            assert fr1.tags[1].hamming == 2
            assert fr1.tags[1].translation == (-0.5, 0.25, 2.0)
            assert r.read_next_frame() is None          # EOF
        finally:
            r.close()


def test_v3_empty_encoder_args():
    with tmpdir() as d:
        path = build_taps_file(Path(d) / 'noargs.taps',
                               [(0, 5, b'z' * 4, b'')], version=3, args='')
        r = TapsReader(str(path))
        try:
            assert r.header.encoder_args == ''
            assert r.header.frame_count == 1
            fr = r.read_frame(0)
            assert fr is not None and fr.data == b'z' * 4
        finally:
            r.close()


# ---------------------------------------------------------------------------
#  Pose fusion — Gauss-Newton recovery (exact geometry, no files)
# ---------------------------------------------------------------------------

TRUE_X, TRUE_Y, TRUE_YAW_DEG = 2.0, 1.5, 30.0
U7 = (0.10, 0.20)
U8 = (-0.30, 0.40)
T_B = [1.0, 0.0, 0.0, 3.0,
       0.0, 1.0, 0.0, 3.0,
       0.0, 0.0, 1.0, 0.0,
       0.0, 0.0, 0.0, 1.0]          # camera B is translated (3,3) in field frame


def _field_pos(u):
    c, s = math.cos(math.radians(TRUE_YAW_DEG)), math.sin(math.radians(TRUE_YAW_DEG))
    return (TRUE_X + c * u[0] - s * u[1], TRUE_Y + s * u[0] + c * u[1])


def _tag_detection(tag_id, translation):
    return TagDetection(tag_id=tag_id, decision_margin=8.0, hamming=0, pose_valid=True,
                        translation=translation, rotation=IDENTITY9,
                        reproj_error_rms_px=0.4, tag_px_diag=100.0)


def _tag_matrix(u):
    T = np.eye(4)
    T[:2, 3] = u
    return T


def _camera_tracks(ptp_offset_ns=5_000_000):
    frames_a, frames_b = [], []
    for k in range(3):
        t = 1_000_000_000 + k * 16_666_667
        p7 = _field_pos(U7)
        p8 = _field_pos(U8)
        # Camera A: T_field_cam = identity → camera-frame translation == field pos
        frames_a.append((t, [_tag_detection(7, (p7[0], p7[1], 1.8))]))
        # Camera B: translation-only extrinsics (3,3,0) → tr = p_field - (3,3,0)
        frames_b.append((t + ptp_offset_ns, [_tag_detection(8, (p8[0] - 3.0, p8[1] - 3.0, 2.2))]))
    cam_a = CameraTrack('CamA', frames_a, np.eye(4))
    cam_b = CameraTrack('CamB', frames_b, np.array(T_B).reshape(4, 4))
    return cam_a, cam_b


def test_fusion_two_cameras_exact_pose():
    cam_a, cam_b = _camera_tracks()
    eng = FusionEngine([cam_a, cam_b], {7: _tag_matrix(U7), 8: _tag_matrix(U8)},
                       window_ms=30.0, reference_camera='CamA')
    assert eng.reference_camera == 'CamA'
    assert len(eng.track) == 3
    e0 = eng.track[0]
    assert e0['frame'] == 0 and e0['n_obs'] == 2 and e0['n_cameras'] == 2
    assert abs(e0['x'] - TRUE_X) < 1e-6
    assert abs(e0['y'] - TRUE_Y) < 1e-6
    assert abs(e0['yaw_deg'] - TRUE_YAW_DEG) < 1e-6
    assert e0['rms'] < 1e-9
    assert {c['camera'] for c in e0['candidates']} == {'CamA', 'CamB'}
    # JSON-serializable
    json.dumps(eng.to_json_dict())


def test_fusion_window_rejection_and_single_observation():
    # Camera B offset 100 ms > 30 ms window → rejected; single observation remains
    cam_a, cam_b = _camera_tracks(ptp_offset_ns=100_000_000)
    eng = FusionEngine([cam_a, cam_b], {7: _tag_matrix(U7), 8: _tag_matrix(U8)},
                       window_ms=30.0, reference_camera='CamA')
    e0 = eng.track[0]
    assert e0['n_obs'] == 1 and e0['n_cameras'] == 1
    assert e0['yaw_deg'] is None                     # unobservable from one tag
    p7 = _field_pos(U7)
    assert abs(e0['x'] - (p7[0] - U7[0])) < 1e-9     # x,y from the single candidate
    assert abs(e0['y'] - (p7[1] - U7[1])) < 1e-9


# ---------------------------------------------------------------------------
#  FusionEngine.from_session + fusion_track.json caching
# ---------------------------------------------------------------------------

FUSION_CFG = {
    'window_ms': 30,
    'reference_camera': 'CameraNode1',
    'cameras': [
        {'alias': 'CameraNode1',
         'T_field_cam': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        {'alias': 'CameraNode2', 'T_field_cam': T_B},
    ],
    'tags': {
        7: {'T_robot_tag': [1, 0, 0, U7[0], 0, 1, 0, U7[1],
                            0, 0, 1, 0.05, 0, 0, 0, 1]},
        8: {'T_robot_tag': [1, 0, 0, U8[0], 0, 1, 0, U8[1],
                            0, 0, 1, 0.05, 0, 0, 0, 1]},
    },
}


def _write_session(d):
    frames_a, frames_b = [], []
    for k in range(3):
        t = 1_000_000_000 + k * 16_666_667
        p7, p8 = _field_pos(U7), _field_pos(U8)
        frames_a.append((k, t, b'\xff\xd8' + b'a' * (17 + k),
                         make_meta([pack_tag_record(
                             tag_id=7, translation=(p7[0], p7[1], 1.8),
                             reproj_error_rms_px=0.4, tag_px_diag=100.0)])))
        frames_b.append((k, t + 5_000_000, b'\xff\xd8' + b'b' * (23 + k),
                         make_meta([pack_tag_record(
                             tag_id=8, translation=(p8[0] - 3.0, p8[1] - 3.0, 2.2),
                             reproj_error_rms_px=0.4, tag_px_diag=100.0)])))
    build_taps_file(Path(d) / 'CameraNode1_rec.taps', frames_a, version=3, alias='CameraNode1')
    build_taps_file(Path(d) / 'CameraNode2_rec.taps', frames_b, version=3, alias='CameraNode2')


def test_fusion_from_session_and_cache():
    with tmpdir() as d:
        _write_session(d)
        eng = FusionEngine.from_session(d, FUSION_CFG)
        assert eng is not None
        assert eng.reference_camera == 'CameraNode1'
        assert len(eng.track) == 3
        e0 = eng.track[0]
        assert e0['n_obs'] == 2 and e0['n_cameras'] == 2
        # f32 storage in TagRecord limits recovery precision
        assert abs(e0['x'] - TRUE_X) < 1e-4
        assert abs(e0['y'] - TRUE_Y) < 1e-4
        assert abs(e0['yaw_deg'] - TRUE_YAW_DEG) < 1e-3

        cache = Path(d) / 'fusion_track.json'
        assert not cache.exists()  # from_session is side-effect free

        # get_fusion_track builds + writes the disk cache
        cached = get_fusion_track(d, FUSION_CFG)
        assert cache.exists()
        assert cached is not None and cached['available'] is True
        assert cached == json.loads(json.dumps(eng.to_json_dict()))

        # Second read comes from the fresh cache (still identical content)
        assert get_fusion_track(d, FUSION_CFG) == cached

        # Any .taps newer than the cache invalidates it: normalize the cache
        # timestamp to a known fresh value, then bump the recording past it.
        taps_file = Path(d) / 'CameraNode1_rec.taps'
        fresh = taps_file.stat().st_mtime + 3600
        os.utime(cache, (fresh, fresh))
        os.utime(taps_file, (fresh + 60, fresh + 60))
        assert load_cached_track(d) is None

        # No config → None
        assert FusionEngine.from_session(d, None) is None


def test_fusion_v2_only_session_unavailable():
    with tmpdir() as d:
        build_taps_file(Path(d) / 'v2only.taps',
                        [(0, 5, b'q' * 4, b''), (1, 6, b'r' * 4, b'')],
                        version=2)
        assert FusionEngine.from_session(d, FUSION_CFG) is None
        assert get_fusion_track(d, FUSION_CFG) is None


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
