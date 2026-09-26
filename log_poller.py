"""
Log poller for ServerNode.

Periodically polls the Robot Control Computer's SMB share for new or
changed log files. Copies them to local storage for parsing.
"""

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger('server_node.log_poller')


class LogPoller:
    """Polls an SMB share for new/changed log files."""

    def __init__(self, smb_path: str, local_dir: str, poll_interval: int = 30):
        self.smb_path = smb_path
        self.local_dir = Path(local_dir)
        self.poll_interval = poll_interval
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._seen_files: dict[str, str] = {}  # hash -> local_path

        # Ensure local directory exists
        self.local_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the background polling loop."""
        self._running = True
        task = asyncio.create_task(self._poll_loop())
        self._tasks.append(task)
        logger.info(f"Log poller started (SMB: {self.smb_path}, interval: {self.poll_interval}s)")

    async def stop(self):
        """Stop the polling loop."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info("Log poller stopped")

    async def _poll_loop(self):
        """Background loop that polls the SMB share periodically."""
        while self._running:
            try:
                new_files = await self._poll()
                for file_path in new_files:
                    logger.info(f"New log file: {file_path}")
            except Exception as e:
                logger.error(f"Poll error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _poll(self) -> list[str]:
        """
        Poll the SMB share for new/changed files.

        Uses smbprotocol library if available, falls back to a simpler
        UNC path listing if not.

        Returns:
            List of local paths for newly downloaded files.
        """
        new_files = []

        try:
            import smbprotocol.connection
            import smbprotocol.session
            import smbprotocol.open
            import smbprotocol.tree

            new_files = await self._poll_smb()
        except ImportError:
            logger.warning("smbprotocol not available, trying UNC path listing")
            try:
                new_files = await self._poll_unc()
            except Exception as e:
                logger.error(f"UNC poll failed: {e}")

        return new_files

    async def _poll_smb(self) -> list[str]:
        """Poll using smbprotocol library."""
        new_files = []
        # Note: Full SMB implementation requires smbprotocol library setup
        # This is a simplified version - full implementation would use:
        # - SMBConnection or AsyncSMBConnection
        # - Connect to share
        # - List files
        # - Compare hashes
        # - Download new files

        # Placeholder: In production, implement full SMB polling:
        # from smbprotocol.connection import Connection
        # from smbprotocol.session import Session
        # from smbprotocol.open import Open
        # Connect, list share, compare, download

        logger.info(f"SMB poll of {self.smb_path} (placeholder - smbprotocol setup required)")
        return new_files

    async def _poll_unc(self) -> list[str]:
        """Poll using Windows UNC path (works on Linux with smbclient or Python)."""
        new_files = []

        # Use Python's pathlib with UNC paths
        try:
            smb_path = Path(self.smb_path)
            if not smb_path.exists():
                logger.warning(f"SMB path not accessible: {self.smb_path}")
                return new_files

            # List files in the share
            for file_path in smb_path.glob('*.txt'):
                if file_path.is_file():
                    file_hash = self._hash_file_unc(file_path)
                    if file_hash and file_hash not in self._seen_files:
                        # New file - copy to local storage
                        local_path = self.local_dir / file_path.name
                        try:
                            shutil.copy2(file_path, local_path)
                            self._seen_files[file_hash] = str(local_path)
                            new_files.append(str(local_path))
                        except Exception as e:
                            logger.error(f"Failed to copy {file_path}: {e}")

        except Exception as e:
            logger.error(f"UNC poll error: {e}")

        return new_files

    def _hash_file_unc(self, path: Path) -> Optional[str]:
        """Calculate a simple hash of a file for change detection."""
        try:
            h = hashlib.md5()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def get_local_files(self) -> list[str]:
        """Get list of locally cached log file paths."""
        return list(self._seen_files.values())

    def clear_seen(self):
        """Clear the seen files cache (for testing or manual reset)."""
        self._seen_files.clear()
