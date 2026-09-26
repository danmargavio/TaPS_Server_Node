"""
TaPS ServerNode - Main entry point.

Wires together all components:
- Camera aggregator (MJPEG from CameraNodes)
- Session manager (NetworkTables-triggered recording)
- Log poller (SMB share polling for log files)
- Playback engine (.taps file playback)
- Web server (aiohttp frontend)
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path so 'common' package is findable
# Also add current directory so local server_node modules import correctly
# (handles both: cd consolidated && python server_node/main.py  AND  cd server_node && python main.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # consolidated/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # server_node/

from aiohttp import web

from common.config import ServerNodeConfig
from server_node.camera_aggregator import CameraAggregator
from server_node.session_manager import SessionManager
from server_node.log_poller import LogPoller
from server_node.log_sync import LogSync
from server_node.playback_engine import PlaybackEngine
from server_node.web_server import WebServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('server_node')


class TaPSServerNode:
    """Main ServerNode application, coordinating all components."""

    def __init__(self, config_path: str):
        self.config = ServerNodeConfig.from_yaml(config_path)
        self.camera_aggregator = CameraAggregator(self.config.cameras)
        self.session_manager = SessionManager(
            camera_hosts=[c.recording_url for c in self.config.cameras],
            nt_server=self.config.nt_server,
            nt_match_start_topic=self.config.nt_match_start_topic,
            nt_match_end_topic=self.config.nt_match_end_topic,
        )
        self.log_poller = LogPoller(
            smb_path=self.config.smb_path,
            local_dir=Path(self.config.output_dir) / 'logs',
            poll_interval=self.config.smb_poll_interval,
        )
        self.log_sync = LogSync()
        self.playback_engine = PlaybackEngine(self.config.output_dir)
        self.web_server = WebServer(
            host=self.config.web_host,
            port=self.config.web_port,
            camera_aggregator=self.camera_aggregator,
            session_manager=self.session_manager,
            log_poller=self.log_poller,
            log_sync=self.log_sync,
            playback_engine=self.playback_engine,
            config=self.config,
        )

    async def start(self):
        """Start all components."""
        logger.info("Starting TaPS ServerNode...")

        # Start camera aggregator
        await self.camera_aggregator.start()
        logger.info("Camera aggregator started")

        # Start session manager (NetworkTables listener)
        self.session_manager.start()
        logger.info("Session manager started")

        # Start log poller
        await self.log_poller.start()
        logger.info("Log poller started")

        # Start web server
        await self.web_server.start()
        logger.info(f"Web server started on http://{self.config.web_host}:{self.config.web_port}")

    async def stop(self):
        """Stop all components gracefully."""
        logger.info("Shutting down TaPS ServerNode...")

        await self.web_server.stop()
        self.session_manager.stop()
        await self.log_poller.stop()
        await self.camera_aggregator.stop()

        logger.info("TaPS ServerNode stopped")


def main():
    """Entry point for the ServerNode application."""
    # Determine config path
    config_path = 'server_node.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Copy shared_config/server_node_template.yaml to server_node.yaml and edit")
        sys.exit(1)

    node = TaPSServerNode(str(config_file.resolve()))

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        logger.info("Received shutdown signal")
        loop.create_task(node.stop())

    # add_signal_handler is not supported on Windows
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Windows: Ctrl+C handled by event loop automatically
        logger.info("Signal handlers not available on this platform")

    try:
        loop.run_until_complete(node.start())
        # Keep running until shutdown signal
        loop.run_forever()
    finally:
        loop.run_until_complete(node.stop())
        loop.close()


if __name__ == '__main__':
    main()
