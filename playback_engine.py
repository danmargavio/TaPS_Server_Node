"""
Playback engine for ServerNode.

Supports both .taps binary format (with PTP timestamps) and standard video
files (.mp4, .avi, .mkv, .mov) for playback. Provides synchronized playback
with frame advance, speed control, rotation, and log overlay.
"""

import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger('server_node.playback_engine')

# Supported video formats
TAPS_EXTENSION = '.taps'
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.m4v', '.webm'}


# .taps file constants
TAPS_MAGIC = b'TaPS\x02'
ENCODER_JPEG = 0
ENCODER_RAW = 1


# ---------------------------------------------------------------------------
#  TapsReader — reads .taps binary format
# ---------------------------------------------------------------------------

@dataclass
class TapsFileHeader:
    """Parsed .taps file header."""
    encoder_type: int
    width: int
    height: int
    target_fps: float
    encoder_args: str
    frame_count: int


@dataclass
class TapsFrame:
    """A single frame from a .taps file."""
    frame_idx: int
    ptp_ns: int
    data: bytes


class TapsReader:
    """Reader for .taps video files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file: Optional[open] = None
        self.header: Optional[TapsFileHeader] = None
        self._first_frame_offset: int = 0
        self._load_header()

    def _load_header(self):
        """Read and validate the .taps file header."""
        try:
            self._file = open(self.file_path, 'rb')
            raw = self._file.read(32)  # Read enough for header

            if len(raw) < 5 or raw[:5] != TAPS_MAGIC:
                raise ValueError(f"Invalid .taps file: bad magic")

            # Parse header fields (packed struct)
            # After magic (5 bytes): encoder_type(1), width(8), height(8), fps(8),
            # args_length(4), args(variable), frame_count(8)
            offset = 5
            encoder_type = struct.unpack_from('B', raw, offset)[0]
            offset += 1

            if len(raw) < offset + 32:
                # Need more data
                extra = self._file.read(64)
                raw += extra

            width = struct.unpack_from('Q', raw, offset)[0]
            offset += 8
            height = struct.unpack_from('Q', raw, offset)[0]
            offset += 8
            target_fps = struct.unpack_from('d', raw, offset)[0]
            offset += 8

            args_length = struct.unpack_from('I', raw, offset)[0]
            offset += 8

            # Read encoder args
            encoder_args = ''
            if args_length > 0 and len(raw) >= offset + args_length:
                encoder_args = raw[offset:offset + args_length].decode('utf-8', errors='replace')
                offset += args_length

            frame_count = struct.unpack_from('Q', raw, offset)[0]

            self.header = TapsFileHeader(
                encoder_type=encoder_type,
                width=int(width),
                height=int(height),
                target_fps=float(target_fps),
                encoder_args=encoder_args,
                frame_count=int(frame_count),
            )

            # Store first frame offset
            self._first_frame_offset = self._file.tell()

        except Exception as e:
            if self._file:
                self._file.close()
            raise ValueError(f"Failed to read .taps header {self.file_path}: {e}")

    def read_frame(self, frame_idx: int) -> Optional[TapsFrame]:
        """Read a specific frame by index."""
        if not self._file:
            return None

        try:
            # Seek to first frame
            self._file.seek(self._first_frame_offset)

            # Linear scan to target frame
            for i in range(frame_idx + 1):
                # Read frame header: frame_idx(8), ptp_ns(8), frame_size(4) = 20 bytes
                header_data = self._file.read(20)
                if len(header_data) < 20:
                    return None

                f_idx, ptp_ns, f_size = struct.unpack_from('QqI', header_data, 0)

                if f_idx == frame_idx:
                    # Read frame data
                    data = self._file.read(f_size)
                    if len(data) < f_size:
                        return None
                    return TapsFrame(frame_idx=f_idx, ptp_ns=ptp_ns, data=data)

                # Skip frame data
                self._file.seek(f_size, 1)

        except Exception as e:
            logger.error(f"Failed to read frame {frame_idx}: {e}")

        return None

    def read_next_frame(self) -> Optional[TapsFrame]:
        """Read the next frame sequentially."""
        if not self._file:
            return None

        try:
            header_data = self._file.read(20)
            if len(header_data) < 20:
                return None

            f_idx, ptp_ns, f_size = struct.unpack_from('QqI', header_data, 0)

            data = self._file.read(f_size)
            if len(data) < f_size:
                return None

            return TapsFrame(frame_idx=f_idx, ptp_ns=ptp_ns, data=data)

        except Exception as e:
            logger.error(f"Failed to read next frame: {e}")
            return None

    def seek_to_frame(self, frame_idx: int) -> bool:
        """Seek to a specific frame index."""
        if not self._file:
            return False

        try:
            self._file.seek(self._first_frame_offset)

            for i in range(frame_idx + 1):
                header_data = self._file.read(20)
                if len(header_data) < 20:
                    return False

                f_idx, ptp_ns, f_size = struct.unpack_from('QqI', header_data, 0)

                if f_idx == frame_idx:
                    return True

                self._file.seek(f_size, 1)

        except Exception as e:
            logger.error(f"Failed to seek to frame {frame_idx}: {e}")

        return False

    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
#  VideoFileReader — reads standard video files via OpenCV
# ---------------------------------------------------------------------------

class VideoFileReader:
    """Reader for standard video files (.mp4, .avi, .mkv, etc.)."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0
        self._loaded = False
        self._load()

    def _load(self):
        """Open and probe the video file."""
        try:
            self._cap = cv2.VideoCapture(self.file_path)
            if not self._cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.file_path}")

            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self._loaded = True
        except Exception as e:
            if self._cap:
                self._cap.release()
                self._cap = None
            raise ValueError(f"Failed to open video {self.file_path}: {e}")

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame by index."""
        if not self._cap or not self._loaded:
            return None

        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
            return None
        except Exception as e:
            logger.error(f"Failed to read frame {frame_idx}: {e}")
            return None

    def read_next_frame(self) -> Optional[np.ndarray]:
        """Read the next frame sequentially."""
        if not self._cap or not self._loaded:
            return None

        ret, frame = self._cap.read()
        if ret and frame is not None:
            return frame
        return None

    def seek_to_frame(self, frame_idx: int) -> bool:
        """Seek to a specific frame index."""
        if not self._cap or not self._loaded:
            return False

        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            return True
        except Exception:
            return False

    def close(self):
        """Close the video file."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._loaded = False

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
#  Unified stream abstraction
# ---------------------------------------------------------------------------

StreamReader = Union[TapsReader, VideoFileReader]


# ---------------------------------------------------------------------------
#  PlaybackEngine — manages playback of recorded sessions
# ---------------------------------------------------------------------------

class PlaybackEngine:
    """Manages playback of recorded sessions (.taps or video files)."""

    def __init__(self, recording_dir: str):
        self.recording_dir = Path(recording_dir)
        self._streams: list[tuple[str, StreamReader]] = []  # (path, reader)
        self._current_frame_idx: int = 0
        self._current_time: float = 0.0
        self._playing: bool = False
        self._paused: bool = False
        self._speed: float = 1.0
        self._durations: list[float] = []  # Duration per stream
        self._fps: float = 30.0  # Common FPS for timing
        self._rotations: list[int] = [0, 0, 0, 0]  # Per-stream rotation
        self._formats: list[str] = []  # 'taps' or 'video' per stream

    def load_videos(self, file_paths: list[str]) -> bool:
        """
        Load video files for playback.

        Supports .taps files (with PTP timestamps) and standard video files
        (.mp4, .avi, .mkv, .mov).

        Args:
            file_paths: Paths to video files (up to 4).

        Returns:
            True if all files loaded successfully.
        """
        self._streams.clear()
        self._rotations = [0] * len(file_paths)
        self._formats = []

        if len(file_paths) > 4:
            file_paths = file_paths[:4]

        max_duration = 0.0
        common_fps = 30.0
        first_fps_set = False

        for path in file_paths:
            try:
                ext = Path(path).suffix.lower()
                if ext == TAPS_EXTENSION:
                    reader: StreamReader = TapsReader(path)
                    self._formats.append('taps')

                    fps = reader.header.target_fps if reader.header else 30.0
                    duration = reader.header.frame_count / fps if reader.header else 0.0
                    # Use FIRST video's FPS for all timing/frame counting
                    if not first_fps_set:
                        common_fps = fps
                        first_fps_set = True
                    max_duration = max(max_duration, duration)

                    logger.info(f"Loaded {path}: {reader.header.frame_count} frames, "
                               f"{reader.header.width}x{reader.header.height} @ {fps:.1f}fps, "
                               f"{duration:.1f}s (.taps)")

                elif ext in VIDEO_EXTENSIONS:
                    reader = VideoFileReader(path)
                    self._formats.append('video')

                    fps = reader.fps
                    duration = reader.total_frames / fps if fps > 0 else 0.0
                    # Use FIRST video's FPS for all timing/frame counting
                    if not first_fps_set:
                        common_fps = fps
                        first_fps_set = True
                    max_duration = max(max_duration, duration)

                    logger.info(f"Loaded {path}: {reader.total_frames} frames, "
                               f"{reader.width}x{reader.height} @ {fps:.1f}fps, "
                               f"{duration:.1f}s ({ext})")

                else:
                    logger.warning(f"Unsupported format: {ext} — skipping {path}")
                    continue

                self._streams.append((path, reader))

            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                # Clean up already loaded readers
                for _, reader in self._streams:
                    reader.close()
                self._streams.clear()
                self._formats.clear()
                return False

        if not self._streams:
            return False

        self._fps = common_fps
        self._durations = [max_duration] * len(self._streams)
        self._reset()

        return True

    def _reset(self):
        """Reset playback to beginning."""
        self._current_frame_idx = 0
        self._current_time = 0.0
        self._paused = False
        self._playing = False

        # Reset all readers to beginning
        for _, reader in self._streams:
            reader.seek_to_frame(0)

    def start(self):
        """Start playback."""
        if not self._streams:
            return
        self._playing = True
        self._paused = False

    def stop(self):
        """Stop playback."""
        self._playing = False
        self._paused = True
        self._reset()

    def pause(self):
        """Pause playback."""
        self._paused = not self._paused

    def step_forward(self) -> Optional[np.ndarray]:
        """Step one frame forward. Returns composited grid frame."""
        if not self._streams:
            return None

        frame = self._read_frame_at_index(self._current_frame_idx)
        if frame is not None:
            self._current_frame_idx += 1
            self._current_time = self._current_frame_idx / self._fps if self._fps > 0 else 0
        return frame

    def step_backward(self) -> Optional[np.ndarray]:
        """Step one frame backward. Returns composited grid frame."""
        if not self._streams or self._current_frame_idx <= 0:
            return None

        self._current_frame_idx -= 1
        self._current_time = self._current_frame_idx / self._fps if self._fps > 0 else 0
        frame = self._read_frame_at_index(self._current_frame_idx)
        return frame

    def seek(self, seconds: float):
        """Seek to a specific time in seconds."""
        if not self._streams or self._fps <= 0:
            return

        target_frame = int(seconds * self._fps)
        max_frame = self.get_total_frames()
        target_frame = max(0, min(target_frame, max_frame))

        self._current_frame_idx = target_frame
        self._current_time = target_frame / self._fps

        # Seek all readers
        for _, reader in self._streams:
            reader.seek_to_frame(target_frame)

    def set_speed(self, speed: float):
        """Set playback speed (0.1 = slow, 1.0 = normal, 4.0 = fast)."""
        self._speed = max(0.1, min(speed, 10.0))

    def set_rotation(self, stream_index: int, degrees: int):
        """Set rotation for a specific stream."""
        if 0 <= stream_index < len(self._rotations):
            self._rotations[stream_index] = ((degrees % 360) + 360) % 360

    def get_current_time(self) -> float:
        """Get current playback time in seconds."""
        return self._current_time

    def get_current_frame(self) -> int:
        """Get current frame index."""
        return self._current_frame_idx

    def get_total_frames(self) -> int:
        """Get total frame count from first stream."""
        if self._streams:
            reader = self._streams[0][1]
            if isinstance(reader, TapsReader) and reader.header:
                return reader.header.frame_count
            elif isinstance(reader, VideoFileReader):
                return reader.total_frames
        return 0

    def get_duration(self) -> float:
        """Get total duration in seconds."""
        if self._current_frame_idx >= self.get_total_frames():
            return self._current_time
        return self.get_total_frames() / self._fps if self._fps > 0 else 0

    def get_stream_type(self, index: int) -> str:
        """Get the format type of a stream ('taps' or 'video')."""
        if 0 <= index < len(self._formats):
            return self._formats[index]
        return 'unknown'

    def _read_frame_at_index(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read frames from all streams and compose a 2x2 grid."""
        if not self._streams:
            return None

        tile_w = 640
        tile_h = 480
        grid = np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)

        for idx, (path, reader) in enumerate(self._streams):
            is_taps = self._formats[idx] == 'taps'

            if is_taps and isinstance(reader, TapsReader):
                # .taps format — read JPEG data, decode
                frame_data = reader.read_frame(frame_idx)
                if frame_data is None:
                    placeholder = self._generate_placeholder(tile_w, tile_h, idx)
                    col = idx % 2
                    row = idx // 2
                    grid[row * tile_h:(row + 1) * tile_h,
                         col * tile_w:(col + 1) * tile_w] = placeholder
                    continue

                try:
                    nparr = np.frombuffer(frame_data.data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception as e:
                    logger.error(f"Failed to decode frame from stream {idx}: {e}")
                    continue

            elif isinstance(reader, VideoFileReader):
                # Standard video — reader returns np.ndarray directly
                frame = reader.read_frame(frame_idx)
                if frame is None:
                    placeholder = self._generate_placeholder(tile_w, tile_h, idx)
                    col = idx % 2
                    row = idx // 2
                    grid[row * tile_h:(row + 1) * tile_h,
                         col * tile_w:(col + 1) * tile_w] = placeholder
                    continue

            else:
                # Unknown reader type
                continue

            # Apply rotation
            rotation = self._rotations[idx]
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Scale to tile size
            fh, fw = frame.shape[:2]
            if fw != tile_w or fh != tile_h:
                frame = cv2.resize(frame, (tile_w, tile_h))

            # Place in grid
            col = idx % 2
            row = idx // 2
            grid[row * tile_h:(row + 1) * tile_h,
                 col * tile_w:(col + 1) * tile_w] = frame

        return grid

    @staticmethod
    def _generate_placeholder(width: int, height: int, stream_index: int) -> np.ndarray:
        """Generate a placeholder frame for an offline stream."""
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Camera {stream_index + 1}\nOffline",
                    (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        return placeholder

    def get_frame_as_jpeg(self, frame: Optional[np.ndarray], quality: int = 80) -> Optional[bytes]:
        """Encode a frame as JPEG bytes."""
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def enumerate_sessions(self) -> list[dict]:
        """
        Enumerate available recording sessions in the output directory.

        Finds directories containing .taps files or standard video files
        (.mp4, .avi, .mkv, .mov).
        """
        sessions = []
        if not self.recording_dir.exists():
            return sessions

        for session_dir in sorted(self.recording_dir.iterdir()):
            if session_dir.is_dir():
                # Find .taps files
                taps_files = sorted(session_dir.glob('*.taps'))
                # Find standard video files
                video_files = sorted([
                    f for f in session_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
                ])

                # Only include if there are video files of any kind
                all_files = taps_files + video_files
                if all_files:
                    sessions.append({
                        'id': session_dir.name,
                        'path': str(session_dir),
                        'camera_count': len(all_files),
                        'taps_files': [str(f) for f in taps_files],
                        'video_files': [str(f) for f in video_files],
                        'all_files': [str(f) for f in all_files],
                    })

        return sessions

    def close(self):
        """Close all stream readers."""
        for _, reader in self._streams:
            reader.close()
        self._streams.clear()
        self._formats.clear()
