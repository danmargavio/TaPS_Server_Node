"""
Camera aggregator for ServerNode.

Fetches MJPEG frames from CameraNodes and caches the latest frame per camera.
Provides individual camera frames and composed 2x2 grid.
"""

import asyncio
import logging
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

# Add parent directory to path so 'common' package is findable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from aiohttp import ClientSession, ClientTimeout

from common.config import CameraConfig

logger = logging.getLogger('server_node.camera_aggregator')


class CameraFrameCache:
    """Thread-safe cache for a single camera's latest frame."""

    def __init__(self, camera_config: CameraConfig):
        self.camera = camera_config
        self._frame: Optional[np.ndarray] = None
        self._jpeg_bytes: Optional[bytes] = None
        self._last_update: float = 0.0
        self._connected: bool = False
        self._error: str = ''
        self._frame_count: int = 0

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self._frame

    @property
    def jpeg_bytes(self) -> Optional[bytes]:
        return self._jpeg_bytes

    @property
    def last_update(self) -> float:
        return self._last_update

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> str:
        return self._error

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def update_frame(self, jpeg_data: bytes) -> bool:
        """Decode and cache a new JPEG frame."""
        try:
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._frame = frame
                self._jpeg_bytes = jpeg_data
                self._last_update = time.time()
                self._connected = True
                self._error = ''
                self._frame_count += 1
                return True
        except Exception as e:
            self._error = str(e)
            logger.debug(f"Failed to decode frame from {self.camera.name}: {e}")
        return False

    def mark_disconnected(self, error: str = ''):
        """Mark camera as disconnected."""
        self._connected = False
        if error:
            self._error = error

    def generate_placeholder(self, size: tuple = (640, 480)) -> bytes:
        """Generate a placeholder frame indicating offline status."""
        h, w = size
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"{self.camera.name}\nOFFLINE",
                    (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()


class CameraAggregator:
    """Aggregates MJPEG streams from multiple CameraNodes."""

    def __init__(self, camera_configs: list[CameraConfig]):
        self.camera_configs = camera_configs
        self.caches: dict[int, CameraFrameCache] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._session: Optional[ClientSession] = None
        self._grid_size = (1280, 960)  # Default grid preview size

        for i, config in enumerate(camera_configs):
            self.caches[i] = CameraFrameCache(config)

    async def start(self):
        """Start background tasks to fetch frames from all cameras."""
        self._running = True
        self._session = ClientSession(
            timeout=ClientTimeout(total=5, connect=2)
        )

        for i, cache in self.caches.items():
            task = asyncio.create_task(self._fetch_loop(i, cache))
            self._tasks.append(task)

        logger.info(f"Camera aggregator started for {len(self.camera_configs)} cameras")

    async def stop(self):
        """Stop all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()
        self._tasks.clear()
        logger.info("Camera aggregator stopped")

    async def _fetch_loop(self, index: int, cache: CameraFrameCache):
        """Background loop to fetch latest frame from a camera."""
        stream_url = cache.camera.stream_url
        logger.info(f"Starting fetch loop for {cache.camera.name} at {stream_url}")

        while self._running:
            try:
                async with self._session.get(stream_url, timeout=3.0) as response:
                    if response.status != 200:
                        cache.mark_disconnected(f"HTTP {response.status}")
                        await asyncio.sleep(1)
                        continue

                    # MJPEG stream: read until we get a complete frame
                    # The stream is multipart/x-mixed-replace with boundary "--frame"
                    buffer = b''
                    while self._running:
                        chunk = await response.content.readuntil(b'\r\n\r\n')
                        if not chunk:
                            break
                        buffer += chunk

                        # Check if we have a complete frame
                        if b'\r\n--frame\r\n' in buffer or (
                            b'Content-Type: image/jpeg\r\n' in buffer and
                            b'\r\n' in buffer[buffer.rfind(b'\r\n'):]
                        ):
                            # Extract JPEG data between boundaries
                            jpeg_start = buffer.find(b'\r\n\r\n')
                            if jpeg_start > 0:
                                jpeg_data = buffer[jpeg_start + 4:]
                                # Find next boundary
                                next_boundary = jpeg_data.find(b'\r\n--frame')
                                if next_boundary > 0:
                                    jpeg_data = jpeg_data[:next_boundary]
                                    # Remove trailing \r\n
                                    if jpeg_data.endswith(b'\r\n'):
                                        jpeg_data = jpeg_data[:-2]
                                    cache.update_frame(jpeg_data)
                                    buffer = jpeg_data[next_boundary:]
                                else:
                                    # Frame goes until end of buffer, wait for more
                                    break

            except (asyncio.TimeoutError, ClientError, Exception) as e:
                cache.mark_disconnected(str(e)[:100])
                await asyncio.sleep(1)

    async def get_frame(self, camera_index: int) -> Optional[bytes]:
        """Get the latest cached frame for a camera."""
        cache = self.caches.get(camera_index)
        if cache:
            return cache.jpeg_bytes
        return None

    async def get_grid_frame(self) -> Optional[bytes]:
        """Get a composed 2x2 grid from cached frames."""
        frames = []
        for i in range(4):
            cache = self.caches.get(i)
            if cache and cache.frame is not None:
                frames.append(cache.frame)
            else:
                # Use placeholder if camera is disconnected
                frames.append(None)

        if not any(f is not None for f in frames):
            # All cameras offline, generate all placeholders
            frames = [
                cv2.imdecode(np.frombuffer(c.generate_placeholder(), np.uint8), cv2.IMREAD_COLOR)
                for c in self.caches.values()
            ]

        # Compose 2x2 grid
        grid_w, grid_h = self._grid_size
        tile_w = grid_w // 2
        tile_h = grid_h // 2
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            if frame is None:
                # Placeholder
                cache = self.caches[idx]
                placeholder = cache.generate_placeholder((tile_w, tile_h))
                frame = cv2.imdecode(np.frombuffer(placeholder, np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                # Scale to tile size
                fh, fw = frame.shape[:2]
                if fw != tile_w or fh != tile_h:
                    frame = cv2.resize(frame, (tile_w, tile_h))

                # Calculate position in grid
                col = idx % 2
                row = idx // 2
                y_start = row * tile_h
                x_start = col * tile_w
                grid[y_start:y_start + tile_h, x_start:x_start + tile_w] = frame

        _, buf = cv2.imencode('.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def get_status(self) -> list[dict]:
        """Get status information for all cameras."""
        status = []
        for i, cache in self.caches.items():
            status.append({
                'index': i,
                'name': cache.camera.name,
                'host': cache.camera.host,
                'connected': cache.connected,
                'frame_count': cache.frame_count,
                'last_update': cache.last_update,
                'error': cache.error,
            })
        return status


class ClientError(Exception):
    """Custom client error for aiohttp errors."""
    pass
