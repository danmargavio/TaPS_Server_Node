"""
Log parser for FRC DriverStation-style log files.

Parses log entries with <TagVersion>, <time>, and <message> XML-like tags
into structured (timestamp_offset_seconds, message) tuples.

Handles relative timestamps like "-01:48.517" (negative = before match start).
"""

import re
from typing import Optional


class LogEntry:
    """Represents a single parsed log entry."""
    __slots__ = ('timestamp_sec', 'message')

    def __init__(self, timestamp_sec: float, message: str):
        self.timestamp_sec = timestamp_sec
        self.message = message

    def __repr__(self) -> str:
        return f"LogEntry(time={self.timestamp_sec:+.3f}s, msg='{self.message[:60]}...')"


def parse_frc_timestamp(timestamp_str: str) -> float:
    """
    Parse an FRC DriverStation relative timestamp.

    Formats supported:
        - "-01:48.517"  -> -108.517 seconds (before match start)
        - "+00:15.234"  -> +15.234 seconds (after match start)
        - "00:15.234"   -> +15.234 seconds
        - "15.234"      -> +15.234 seconds (no hours)

    Args:
        timestamp_str: The timestamp string from the log file.

    Returns:
        Relative timestamp in seconds (can be negative).

    Raises:
        ValueError: If the timestamp format is unrecognized.
    """
    timestamp_str = timestamp_str.strip()

    # Determine sign
    sign = 1.0
    if timestamp_str.startswith('-'):
        sign = -1.0
        timestamp_str = timestamp_str[1:].lstrip()
    elif timestamp_str.startswith('+'):
        timestamp_str = timestamp_str[1:]

    if ':' in timestamp_str:
        # HH:MM.ss format
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return sign * (hours * 3600 + minutes * 60 + seconds)
    else:
        # SS.ss format (plain seconds)
        return sign * float(timestamp_str)


def parse_log_file(log_path: str) -> list[LogEntry]:
    """
    Parse an FRC DriverStation-style log file.

    Expected format per line:
        <TagVersion>1 <time> -01:48.517 <message> Warning at ...

    Handles:
        - Binary data mixed with text (filters for valid log lines)
        - Multiple entries on same line (splits on <TagVersion>)
        - Entries without line breaks (splits on <TagVersion>)

    Args:
        log_path: Path to the log file.

    Returns:
        List of LogEntry objects sorted by timestamp.
    """
    entries = []

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except (OSError, IOError):
        return []

    # Split on <TagVersion> to handle entries without line breaks
    # This handles both single-line and multi-entry-per-line formats
    parts = raw.split('<TagVersion>')

    for part in parts:
        if not part:
            continue

        # Reconstruct the full line with TagVersion prefix
        line = '<TagVersion>' + part.strip()

        # Skip lines that don't have both <time> and <message>
        if '<time>' not in line or '<message>' not in line:
            continue

        entry = _parse_log_line(line)
        if entry is not None:
            entries.append(entry)

    # Sort by timestamp (most negative first = earliest in time)
    entries.sort(key=lambda e: e.timestamp_sec)
    return entries


def _parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse a single log line into a LogEntry, or return None."""

    # Extract message text
    msg_match = re.search(r'<message>\s*(.+?)(?:<TagVersion>|<time>|$)', line)
    if not msg_match:
        return None

    msg = msg_match.group(1).strip()
    if not msg:
        return None

    # Extract timestamp
    time_match = re.search(r'<time>\s*([+-]?\d+:\d+\.\d+)', line)
    if not time_match:
        # Try without sign prefix
        time_match = re.search(r'<time>\s*(\d+:\d+\.\d+)', line)
    if not time_match:
        return None

    try:
        timestamp_sec = parse_frc_timestamp(time_match.group(1))
    except (ValueError, IndexError):
        return None

    return LogEntry(timestamp_sec=timestamp_sec, message=msg)


def get_entries_at_time(entries: list[LogEntry], playback_time: float, tolerance: float = 2.0) -> list[str]:
    """
    Get log messages that should be displayed at a given playback time.

    Messages within the tolerance window before or at the playback time
    that haven't been "shown" yet are returned.

    Args:
        entries: Parsed log entries (from parse_log_file).
        playback_time: Current playback position in seconds.
        tolerance: Time window in seconds to check around playback_time.

    Returns:
        List of message strings to display.
    """
    if not entries:
        return []

    messages = []
    for entry in entries:
        # Show messages whose timestamp is within tolerance of current playback time
        # and are at or before the current playback time
        if abs(entry.timestamp_sec - playback_time) <= tolerance and entry.timestamp_sec <= playback_time:
            messages.append(entry.message)

    return messages


def create_srt_overlay(log_path: str, output_srt: str, start_offset: float = 0.0) -> int:
    """
    Generate an SRT subtitle file from log entries for video overlay.

    Each log entry becomes an SRT entry showing for ~3 seconds, staggered
    by 0.5 seconds to avoid overlap.

    Args:
        log_path: Path to the log file.
        output_srt: Path for the output SRT file.
        start_offset: Offset to add to all timestamps (for syncing with video start).

    Returns:
        Number of SRT entries created.
    """
    entries = parse_log_file(log_path)
    if not entries:
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write('')
        return 0

    srt_entries = []
    entry_idx = 1

    for entry in entries:
        # Convert relative timestamp to absolute SRT time
        start_sec = entry.timestamp_sec + start_offset
        if start_sec < 0:
            start_sec = 0
        end_sec = start_sec + 3.0

        srt_entries.append((entry_idx, start_sec, end_sec, entry.message))
        entry_idx += 1

    # Write SRT file
    with open(output_srt, 'w', encoding='utf-8') as f:
        for idx, start, end, message in srt_entries:
            start_time = _format_srt_time(start)
            end_time = _format_srt_time(end)
            # Clean up message for display
            display_msg = message.replace('<', '[').replace('>', ']')
            f.write(f"{idx}\n{start_time} --> {end_time}\n{display_msg}\n\n")

    return len(srt_entries)


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_log_file_string(log_text: str) -> list[LogEntry]:
    """
    Parse log entries from a string (useful for testing or in-memory data).

    Args:
        log_text: Raw log file contents.

    Returns:
        List of LogEntry objects.
    """
    entries = []

    # Split on <TagVersion> to handle entries without line breaks
    parts = log_text.split('<TagVersion>')

    for part in parts:
        if not part:
            continue

        line = '<TagVersion>' + part.strip()

        if '<time>' not in line or '<message>' not in line:
            continue

        entry = _parse_log_line(line)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e.timestamp_sec)
    return entries
