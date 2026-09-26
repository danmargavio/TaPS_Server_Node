"""
Configuration loader for TaPS applications.

Loads YAML configuration files with schema validation.
Provides type-safe access to configuration values with defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# CameraNode configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraNodeConfig:
    """Configuration for a single CameraNode instance."""

    # Camera settings
    camera_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 60
    fourcc: str = "MJPG"

    # Recording settings
    output_dir: str = "/recordings"
    encoder: str = "jpeg"
    quality: int = 85
    encoder_threads: int = 4

    # Streaming settings
    http_port: int = 8080
    preview_width: int = 640
    preview_height: int = 480
    preview_fps: int = 15
    jpeg_quality: int = 85
    compose_grid: bool = True

    # PPS / GPIO settings
    pps_gpio_pin: int = -1  # -1 = disabled

    # NetworkTables settings
    nt_server: str = "10.0.1.X"
    nt_match_start_topic: str = "RoboRIO/matchStart"
    nt_match_end_topic: str = "RoboRIO/matchEnd"

    # Identity
    name: str = "CameraNode1"
    index: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> 'CameraNodeConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'CameraNodeConfig':
        """Create configuration from a dictionary."""
        camera = data.get('camera', {})
        recording = data.get('recording', {})
        streaming = data.get('streaming', {})
        pps = data.get('pps', {})
        nt = data.get('network_tables', {})
        identity = data.get('identity', {})

        return cls(
            camera_id=camera.get('device_id', cls.camera_id),
            width=camera.get('width', cls.width),
            height=camera.get('height', cls.height),
            fps=camera.get('fps', cls.fps),
            fourcc=camera.get('fourcc', cls.fourcc),
            output_dir=recording.get('output_dir', cls.output_dir),
            encoder=recording.get('encoder', cls.encoder),
            quality=recording.get('quality', cls.quality),
            encoder_threads=recording.get('encoder_threads', cls.encoder_threads),
            http_port=streaming.get('http_port', cls.http_port),
            preview_width=streaming.get('preview_width', cls.preview_width),
            preview_height=streaming.get('preview_height', cls.preview_height),
            preview_fps=streaming.get('preview_fps', cls.preview_fps),
            jpeg_quality=streaming.get('jpeg_quality', cls.jpeg_quality),
            compose_grid=streaming.get('compose_grid', cls.compose_grid),
            pps_gpio_pin=pps.get('gpio_pin', cls.pps_gpio_pin),
            nt_server=nt.get('server', cls.nt_server),
            nt_match_start_topic=nt.get('match_start_topic', cls.nt_match_start_topic),
            nt_match_end_topic=nt.get('match_end_topic', cls.nt_match_end_topic),
            name=identity.get('name', cls.name),
            index=identity.get('index', cls.index),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            'camera': {
                'device_id': self.camera_id,
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'fourcc': self.fourcc,
            },
            'recording': {
                'output_dir': self.output_dir,
                'encoder': self.encoder,
                'quality': self.quality,
                'encoder_threads': self.encoder_threads,
            },
            'streaming': {
                'http_port': self.http_port,
                'preview_width': self.preview_width,
                'preview_height': self.preview_height,
                'preview_fps': self.preview_fps,
                'jpeg_quality': self.jpeg_quality,
                'compose_grid': self.compose_grid,
            },
            'pps': {
                'gpio_pin': self.pps_gpio_pin,
            },
            'network_tables': {
                'server': self.nt_server,
                'match_start_topic': self.nt_match_start_topic,
                'match_end_topic': self.nt_match_end_topic,
            },
            'identity': {
                'name': self.name,
                'index': self.index,
            },
        }

    def to_cmake_args(self) -> list[str]:
        """Generate command-line arguments for the C++ CameraNode binary."""
        args = [
            f'--camera={self.camera_id}',
            f'--fps={self.fps}',
            f'--width={self.width}',
            f'--height={self.height}',
            f'--fourcc={self.fourcc}',
            f'--output={self.output_dir}',
            f'--encoder={self.encoder}',
            f'--encoder-args=quality:{self.quality}',
            f'--encoder-threads={self.encoder_threads}',
            f'--http-port={self.http_port}',
            f'--stream-quality={self.jpeg_quality}',
            f'--stream-width={self.preview_width}',
            f'--stream-height={self.preview_height}',
            f'--stream-fps={self.preview_fps}',
        ]
        if self.pps_gpio_pin >= 0:
            args.append(f'--pps-gpio={self.pps_gpio_pin}')
        if self.nt_server and self.nt_server != "10.0.1.X":
            args.extend([
                f'--nt-server={self.nt_server}',
                f'--nt-match-start={self.nt_match_start_topic}',
                f'--nt-match-end={self.nt_match_end_topic}',
                f'--nt-local={self.name}',
            ])
        return args


# ---------------------------------------------------------------------------
# CameraConfig for ServerNode
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Configuration for a single camera source on the ServerNode."""
    name: str = "CameraNode1"
    host: str = "10.0.1.11"
    http_port: int = 8080

    @property
    def stream_url(self) -> str:
        return f"http://{self.host}:{self.http_port}/stream"

    @property
    def grid_url(self) -> str:
        return f"http://{self.host}:{self.http_port}/grid-stream"

    @property
    def recording_url(self) -> str:
        return f"http://{self.host}:{self.http_port}/recording"


# ---------------------------------------------------------------------------
# ServerNode configuration
# ---------------------------------------------------------------------------

@dataclass
class ServerNodeConfig:
    """Configuration for the ServerNode instance."""

    # Camera sources
    cameras: list[CameraConfig] = field(default_factory=list)

    # NetworkTables settings
    nt_server: str = "10.0.1.X"
    nt_match_start_topic: str = "RoboRIO/matchStart"
    nt_match_end_topic: str = "RoboRIO/matchEnd"

    # SMB share settings
    smb_path: str = "\\\\10.0.2.X\\share\\logs"
    smb_username: str = ""
    smb_password: str = ""
    smb_poll_interval: int = 30

    # Recording settings
    output_dir: str = "/recordings"

    # Web server settings
    web_host: str = "0.0.0.0"
    web_port: int = 8080

    @classmethod
    def from_yaml(cls, path: str) -> 'ServerNodeConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'ServerNodeConfig':
        """Create configuration from a dictionary."""
        cameras_data = data.get('cameras', [])
        cameras = [
            CameraConfig(
                name=c.get('name', 'CameraNode1'),
                host=c.get('host', '10.0.1.11'),
                http_port=c.get('http_port', 8080),
            )
            for c in cameras_data
        ]

        nt = data.get('network_tables', {})
        smb = data.get('smb_share', {})
        web = data.get('web', {})

        return cls(
            cameras=cameras,
            nt_server=nt.get('server', cls.nt_server),
            nt_match_start_topic=nt.get('match_start_topic', cls.nt_match_start_topic),
            nt_match_end_topic=nt.get('match_end_topic', cls.nt_match_end_topic),
            smb_path=smb.get('path', cls.smb_path),
            smb_username=smb.get('username', cls.smb_username),
            smb_password=smb.get('password', cls.smb_password),
            smb_poll_interval=smb.get('poll_interval', cls.smb_poll_interval),
            output_dir=data.get('recording', {}).get('output_dir', cls.output_dir),
            web_host=web.get('host', cls.web_host),
            web_port=web.get('port', cls.web_port),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            'cameras': [
                {'name': c.name, 'host': c.host, 'http_port': c.http_port}
                for c in self.cameras
            ],
            'network_tables': {
                'server': self.nt_server,
                'match_start_topic': self.nt_match_start_topic,
                'match_end_topic': self.nt_match_end_topic,
            },
            'smb_share': {
                'path': self.smb_path,
                'username': self.smb_username,
                'password': self.smb_password,
                'poll_interval': self.smb_poll_interval,
            },
            'recording': {
                'output_dir': self.output_dir,
            },
            'web': {
                'host': self.web_host,
                'port': self.web_port,
            },
        }

    def get_camera_by_index(self, index: int) -> Optional[CameraConfig]:
        """Get camera configuration by zero-based index."""
        if 0 <= index < len(self.cameras):
            return self.cameras[index]
        return None

    def get_camera_by_name(self, name: str) -> Optional[CameraConfig]:
        """Get camera configuration by name."""
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_camera_node_config(path: str) -> CameraNodeConfig:
    """Load and validate CameraNode configuration from a YAML file."""
    config = CameraNodeConfig.from_yaml(path)
    _validate_camera_node_config(config)
    return config


def load_server_node_config(path: str) -> ServerNodeConfig:
    """Load and validate ServerNode configuration from a YAML file."""
    config = ServerNodeConfig.from_yaml(path)
    _validate_server_node_config(config)
    return config


def _validate_camera_node_config(config: CameraNodeConfig) -> None:
    """Validate CameraNode configuration values."""
    if config.camera_id < 0:
        raise ValueError(f"camera_id must be >= 0, got {config.camera_id}")
    if config.width < 64 or config.width > 7680:
        raise ValueError(f"width must be 64-7680, got {config.width}")
    if config.height < 48 or config.height > 4320:
        raise ValueError(f"height must be 48-4320, got {config.height}")
    if config.fps < 1 or config.fps > 1000:
        raise ValueError(f"fps must be 1-1000, got {config.fps}")
    if config.fourcc not in ("MJPG", "YUYV", "H264", "NV12", "YUV420P"):
        raise ValueError(f"Unsupported fourcc: {config.fourcc}")
    if config.http_port < 1 or config.http_port > 65535:
        raise ValueError(f"http_port must be 1-65535, got {config.http_port}")
    if config.pps_gpio_pin < -1 or config.pps_gpio_pin > 1023:
        raise ValueError(f"pps_gpio_pin must be -1 or 0-1023, got {config.pps_gpio_pin}")


def _validate_server_node_config(config: ServerNodeConfig) -> None:
    """Validate ServerNode configuration values."""
    if not config.cameras:
        raise ValueError("At least one camera must be configured")
    if len(config.cameras) > 4:
        raise ValueError("Maximum 4 cameras supported")
    if config.web_port < 1 or config.web_port > 65535:
        raise ValueError(f"web_port must be 1-65535, got {config.web_port}")
    if config.smb_poll_interval < 5:
        raise ValueError(f"smb_poll_interval must be >= 5, got {config.smb_poll_interval}")

    # Check for duplicate camera names
    names = [c.name for c in config.cameras]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate camera names: {set(n for n in names if names.count(n) > 1)}")
