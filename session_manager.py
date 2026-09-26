"""
Session manager for ServerNode.

Manages recording sessions triggered by NetworkTables matchStart/matchEnd signals.
Sends recording control commands to CameraNodes via HTTP.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
from ntcore import NetworkTable

logger = logging.getLogger('server_node.session_manager')


@dataclass
class RecordingSession:
    """Represents a single recording session."""
    id: str
    start_time: float
    camera_hosts: list[str]
    log_file: Optional[str] = None
    end_time: Optional[float] = None
    status: str = 'recording'  # recording, complete, error

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def taps_path(self) -> Path:
        session_dir = Path(self.camera_hosts[0].replace(':', '_')) / str(self.start_time)
        return session_dir / f"session_{self.id}.taps"


class SessionManager:
    """Manages recording sessions triggered by NetworkTables."""

    def __init__(self, camera_hosts: list[str], nt_server: str,
                 nt_match_start_topic: str = "RoboRIO/matchStart",
                 nt_match_end_topic: str = "RoboRIO/matchEnd"):
        self.camera_hosts = camera_hosts
        self.nt_server = nt_server
        self.nt_match_start_topic = nt_match_start_topic
        self.nt_match_end_topic = nt_match_end_topic
        self._active_session: Optional[RecordingSession] = None
        self._sessions: list[RecordingSession] = []
        self._running = False
        self._nt_table = None
        self._session_id_counter = 0

    def start(self):
        """Start NetworkTables listener and session management."""
        try:
            NetworkTables.initialize(server=self.nt_server)
            self._nt_table = NetworkTables.getTable("RoboRIO")
            logger.info(f"NetworkTables connected to {self.nt_server}")

            # Set up listeners
            self._nt_table.addBooleanCell(self.nt_match_start_topic).addListener(
                self._on_match_start, immediate=True
            )
            self._nt_table.addBooleanCell(self.nt_match_end_topic).addListener(
                self._on_match_end, immediate=True
            )
            logger.info("NT listeners registered")

        except Exception as e:
            logger.warning(f"NetworkTables init failed: {e}. Will continue without NT triggers.")
            self._nt_table = None

        self._running = True

    def stop(self):
        """Stop session management."""
        self._running = False
        try:
            if self._nt_table:
                NetworkTables.shutdown()
        except Exception as e:
            logger.warning(f"NT shutdown error: {e}")
        logger.info("Session manager stopped")

    def _on_match_start(self, topic: str, value, is_new: bool):
        """Handle match start trigger from NetworkTables."""
        if value and is_new:
            logger.info("Match start triggered via NetworkTables")
            self._start_recording()

    def _on_match_end(self, topic: str, value, is_new: bool):
        """Handle match end trigger from NetworkTables."""
        if value and is_new:
            logger.info("Match end triggered via NetworkTables")
            self._stop_recording()

    def _start_recording(self):
        """Start recording on all CameraNodes."""
        if self._active_session:
            logger.warning("Recording already in progress")
            return

        self._session_id_counter += 1
        session_id = f"match_{self._session_id_counter}"

        self._active_session = RecordingSession(
            id=session_id,
            start_time=time.time(),
            camera_hosts=self.camera_hosts,
        )
        self._sessions.append(self._active_session)

        # Send recording start to all CameraNodes
        asyncio.create_task(self._broadcast_recording(True))
        logger.info(f"Recording session {session_id} started")

    def _stop_recording(self):
        """Stop recording on all CameraNodes."""
        if not self._active_session:
            logger.warning("No active recording to stop")
            return

        self._active_session.end_time = time.time()
        self._active_session.status = 'complete'

        # Send recording stop to all CameraNodes
        asyncio.create_task(self._broadcast_recording(False))
        logger.info(f"Recording session {self._active_session.id} stopped")
        self._active_session = None

    async def _broadcast_recording(self, start: bool):
        """Send recording command to all CameraNodes."""
        action = "start" if start else "stop"
        async with aiohttp.ClientSession() as session:
            for host in self.camera_hosts:
                try:
                    url = f"http://{host}/recording"
                    data = json.dumps({"action": action}).encode()
                    headers = {"Content-Type": "application/json"}
                    async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            logger.info(f"Recording {action} sent to {host}")
                        else:
                            logger.warning(f"Recording {action} to {host} returned {resp.status}")
                except Exception as e:
                    logger.error(f"Failed to send recording {action} to {host}: {e}")

    def get_active_session(self) -> Optional[RecordingSession]:
        """Get the currently active recording session."""
        return self._active_session

    def list_sessions(self) -> list[dict]:
        """List all recording sessions with metadata."""
        result = []
        for session in self._sessions:
            result.append({
                'id': session.id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'duration': session.duration,
                'status': session.status,
                'camera_hosts': session.camera_hosts,
            })
        return result
