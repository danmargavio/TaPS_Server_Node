"""
Log synchronization for ServerNode.

Parses log files and provides log entries synchronized with playback time.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so 'common' package is findable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.log_parser import LogEntry, parse_log_file, get_entries_at_time

logger = logging.getLogger('server_node.log_sync')


class LogSync:
    """Synchronizes log entries with playback time."""

    def __init__(self):
        self._entries: list[LogEntry] = []
        self._source_path: Optional[str] = None

    def load_log_file(self, log_path: str) -> int:
        """
        Load and parse a log file.

        Args:
            log_path: Path to the log file.

        Returns:
            Number of entries parsed.
        """
        try:
            entries = parse_log_file(log_path)
            self._entries = entries
            self._source_path = log_path
            logger.info(f"Loaded {len(entries)} log entries from {log_path}")
            return len(entries)
        except Exception as e:
            logger.error(f"Failed to load log file {log_path}: {e}")
            self._entries = []
            return 0

    def get_entries_at_time(self, playback_time: float, tolerance: float = 0.1) -> list[str]:
        """
        Get log messages that should be displayed at a given playback time.

        Args:
            playback_time: Current playback position in seconds.
            tolerance: Time window in seconds.

        Returns:
            List of message strings to display.
        """
        return get_entries_at_time(self._entries, playback_time, tolerance)

    def clear(self):
        """Clear all loaded log entries."""
        self._entries.clear()
        self._source_path = None

    @property
    def entry_count(self) -> int:
        """Number of loaded log entries."""
        return len(self._entries)

    @property
    def source_path(self) -> Optional[str]:
        """Path of the currently loaded log file."""
        return self._source_path

    @property
    def entries(self) -> list[LogEntry]:
        """Access all loaded entries (read-only)."""
        return self._entries
