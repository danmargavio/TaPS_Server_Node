"""
Marker management for ServerNode.

Handles creating, loading, and saving session markers (START, END, NOTE).
Markers are stored as JSON files in the session directory.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger('server_node.marker_manager')

MARKER_FILE = 'markers.json'


@dataclass
class Marker:
    """Represents a single marker in a session."""
    frame: int
    time: float
    type: str  # 'START', 'END', or 'NOTE'
    text: str = ''


class MarkerManager:
    """Manages markers for a session."""

    def __init__(self, session_path: str):
        self.session_path = Path(session_path)
        self._markers: list[Marker] = []
        self._loaded = False

    def load_markers(self) -> int:
        """Load markers from the session's markers.json file."""
        marker_file = self.session_path / MARKER_FILE
        if not marker_file.exists():
            logger.info(f"No markers file found at {marker_file}")
            return 0

        try:
            with open(marker_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._markers = []
            for item in data.get('markers', []):
                self._markers.append(Marker(
                    frame=item['frame'],
                    time=item['time'],
                    type=item['type'],
                    text=item.get('text', ''),
                ))

            self._loaded = True
            logger.info(f"Loaded {len(self._markers)} markers from {marker_file}")
            return len(self._markers)
        except Exception as e:
            logger.error(f"Failed to load markers: {e}")
            self._markers = []
            return 0

    def save_markers(self) -> bool:
        """Save markers to the session's markers.json file."""
        try:
            marker_file = self.session_path / MARKER_FILE
            data = {
                'version': 1,
                'markers': [asdict(m) for m in self._markers],
            }

            with open(marker_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self._markers)} markers to {marker_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save markers: {e}")
            return False

    def add_marker(self, frame: int, time: float, marker_type: str, text: str = '') -> bool:
        """
        Add a new marker.

        Args:
            frame: Frame number (from video #1).
            time: Playback time in seconds.
            marker_type: 'START', 'END', or 'NOTE'.
            text: Optional marker text.

        Returns:
            True if marker was added successfully.
        """
        if marker_type not in ('START', 'END', 'NOTE'):
            logger.error(f"Invalid marker type: {marker_type}")
            return False

        # Enforce single START/END
        if marker_type in ('START', 'END'):
            existing = [m for m in self._markers if m.type == marker_type]
            if existing:
                # Replace existing marker
                self._markers.remove(existing[0])
                logger.info(f"Replaced existing {marker_type} marker")

        marker = Marker(frame=frame, time=time, type=marker_type, text=text)
        self._markers.append(marker)

        # Sort by frame number
        self._markers.sort(key=lambda m: m.frame)

        # Auto-save
        self.save_markers()
        return True

    def get_markers_at_frame(self, frame: int, tolerance: int = 5) -> list[Marker]:
        """Get markers within tolerance frames of the given frame."""
        return [
            m for m in self._markers
            if abs(m.frame - frame) <= tolerance
        ]

    def get_marker_range(self) -> Optional[tuple[Marker, Marker]]:
        """Get START and END markers if both exist."""
        start_markers = [m for m in self._markers if m.type == 'START']
        end_markers = [m for m in self._markers if m.type == 'END']

        if start_markers and end_markers:
            return (start_markers[0], end_markers[0])
        return None

    def get_all_markers(self) -> list[dict]:
        """Get all markers as dicts."""
        return [asdict(m) for m in self._markers]

    @property
    def marker_count(self) -> int:
        """Number of loaded markers."""
        return len(self._markers)
