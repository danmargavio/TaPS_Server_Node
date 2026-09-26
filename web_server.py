"""
Web server for ServerNode.

aiohttp server providing REST API, WebSocket, and static frontend.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so 'common' package is findable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marker_manager import MarkerManager

from aiohttp import web, WSMsgType

from common.config import ServerNodeConfig
from server_node.camera_aggregator import CameraAggregator
from server_node.session_manager import SessionManager
from server_node.log_poller import LogPoller
from server_node.log_sync import LogSync
from server_node.playback_engine import PlaybackEngine

logger = logging.getLogger('server_node.web_server')


class WebServer:
    """HTTP/WebSocket server for TaPS ServerNode."""

    def __init__(self, host: str, port: int,
                 camera_aggregator: CameraAggregator,
                 session_manager: SessionManager,
                 log_poller: LogPoller,
                 log_sync: LogSync,
                 playback_engine: PlaybackEngine,
                 config: ServerNodeConfig):
        self.host = host
        self.port = port
        self.camera_aggregator = camera_aggregator
        self.session_manager = session_manager
        self.log_poller = log_poller
        self.log_sync = log_sync
        self.playback_engine = playback_engine
        self.config = config

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # WebSocket clients for real-time updates
        self._ws_clients: set = set()

        # Marker managers (per session)
        self._marker_managers: dict[str, MarkerManager] = {}
        self._update_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the HTTP server."""
        self._app = web.Application()
        self._setup_routes()
        self._setup_middlewares()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        # Start WebSocket broadcast task
        self._update_task = asyncio.create_task(self._broadcast_updates())

        logger.info(f"Web server listening on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP server."""
        if self._update_task:
            self._update_task.cancel()

        # Close all WebSocket connections
        for ws in self._ws_clients.copy():
            await ws.close()
        self._ws_clients.clear()

        if self._runner:
            await self._runner.cleanup()

        logger.info("Web server stopped")

    def _setup_routes(self):
        """Register all HTTP routes."""
        app = self._app

        # Static files
        app.router.add_get('/', self.handle_index)
        app.router.add_static('/static', 'static', name='static')

        # API - Cameras
        app.router.add_get('/api/cameras', self.handle_cameras)
        app.router.add_get('/api/cameras/{index}/stream', self.handle_camera_stream)

        # API - Grid
        app.router.add_get('/api/grid/stream', self.handle_grid_stream)

        # API - Sessions
        app.router.add_get('/api/sessions', self.handle_sessions_list)
        app.router.add_get('/api/session/{session_id}/metadata', self.handle_session_metadata)
        app.router.add_get('/api/session/{session_id}/frame', self.handle_session_frame)
        app.router.add_get('/api/session/{session_id}/current-frame', self.handle_current_frame)
        app.router.add_get('/api/session/{session_id}/logs', self.handle_session_logs)
        app.router.add_get('/api/session/{session_id}/markers', self.handle_session_markers)
        app.router.add_post('/api/session/{session_id}/markers', self.handle_create_marker)

        # API - Playback controls
        app.router.add_post('/api/session/{session_id}/play', self.handle_play)
        app.router.add_post('/api/session/{session_id}/pause', self.handle_pause)
        app.router.add_post('/api/session/{session_id}/step', self.handle_step)
        app.router.add_post('/api/session/{session_id}/speed', self.handle_speed)
        app.router.add_post('/api/session/{session_id}/rotate', self.handle_rotate)
        app.router.add_post('/api/session/{session_id}/seek', self.handle_seek)

        # WebSocket
        app.router.add_get('/ws', self.handle_websocket)

    def _setup_middlewares(self):
        """Setup middleware (CORS, error handling, etc.)."""
        # CORS middleware
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        self._app.middlewares.append(cors_middleware)

    # ------------------------------------------------------------------
    #  Route handlers
    # ------------------------------------------------------------------

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve the main HTML page."""
        try:
            with open('static/index.html', 'r', encoding='utf-8') as f:
                html = f.read()
            return web.Response(text=html, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text='Frontend not found. Place static/index.html in the server_node directory.',
                               status=500)

    async def handle_cameras(self, request: web.Request) -> web.Response:
        """List cameras and their status."""
        status = self.camera_aggregator.get_status()
        return web.json_response({'cameras': status})

    async def handle_camera_stream(self, request: web.Request) -> web.Response:
        """Proxy a single camera MJPEG stream."""
        try:
            index = int(request.match_info['index'])
            frame = await self.camera_aggregator.get_frame(index)
            if frame is None:
                return web.Response(status=404, text='No stream available')
            return web.Response(body=frame, content_type='image/jpeg')
        except (ValueError, KeyError):
            return web.Response(status=400, text='Invalid camera index')

    async def handle_grid_stream(self, request: web.Request) -> web.Response:
        """Proxy the composed 2x2 grid MJPEG stream."""
        frame = await self.camera_aggregator.get_grid_frame()
        if frame is None:
            return web.Response(status=404, text='No grid available')
        return web.Response(body=frame, content_type='image/jpeg')

    async def handle_sessions_list(self, request: web.Request) -> web.Response:
        """List available recording sessions."""
        sessions = self.playback_engine.enumerate_sessions()
        # Add active session
        active = self.session_manager.get_active_session()
        if active:
            sessions.append({
                'id': active.id,
                'path': '',
                'camera_count': len(active.camera_hosts),
                'taps_files': [],
                'status': active.status,
                'active': True,
            })
        return web.json_response({'sessions': sessions})

    async def handle_session_metadata(self, request: web.Request) -> web.Response:
        """Get session metadata."""
        session_id = request.match_info['session_id']
        # Find session by ID
        sessions = self.playback_engine.enumerate_sessions()
        for s in sessions:
            if s['id'] == session_id:
                return web.json_response({
                    'id': s['id'],
                    'camera_count': s['camera_count'],
                    'taps_files': s.get('taps_files', []),
                    'video_files': s.get('video_files', []),
                    'all_files': s.get('all_files', []),
                })
        return web.json_response({'error': 'Session not found'}, status=404)

    async def handle_session_frame(self, request: web.Request) -> web.Response:
        """Get current playback frame as JPEG (advances frame)."""
        quality = int(request.query.get('quality', 80))
        frame = self.playback_engine.step_forward()
        if frame is None:
            return web.Response(status=404, text='No frame available')
        jpeg = self.playback_engine.get_frame_as_jpeg(frame, quality)
        if jpeg is None:
            return web.Response(status=500, text='Frame encode failed')
        return web.Response(body=jpeg, content_type='image/jpeg')

    async def handle_current_frame(self, request: web.Request) -> web.Response:
        """Get current playback frame as JPEG WITHOUT advancing."""
        quality = int(request.query.get('quality', 80))
        frame = self.playback_engine._read_frame_at_index(self.playback_engine._current_frame_idx)
        if frame is None:
            return web.Response(status=404, text='No frame available')
        jpeg = self.playback_engine.get_frame_as_jpeg(frame, quality)
        if jpeg is None:
            return web.Response(status=500, text='Frame encode failed')
        return web.Response(body=jpeg, content_type='image/jpeg')

    async def handle_session_logs(self, request: web.Request) -> web.Response:
        """Get log entries at current playback time."""
        logs = self.log_sync.get_entries_at_time(
            self.playback_engine.get_current_time()
        )
        # If no entries found with time sync, show all entries
        if not logs and self.log_sync.entry_count > 0:
            logs = [e.message for e in self.log_sync.entries]
        return web.json_response({'logs': logs, 'time': self.playback_engine.get_current_time()})

    async def handle_session_markers(self, request: web.Request) -> web.Response:
        """Get all markers for a session."""
        session_id = request.match_info['session_id']
        manager = self._get_marker_manager(session_id)
        if manager:
            return web.json_response({'markers': manager.get_all_markers()})
        return web.json_response({'markers': []})

    async def handle_create_marker(self, request: web.Request) -> web.Response:
        """Create a new marker for a session."""
        session_id = request.match_info['session_id']
        try:
            data = await request.json()
            frame = int(data.get('frame', 0))
            time = float(data.get('time', 0.0))
            marker_type = data.get('type', 'NOTE')
            text = data.get('text', '')

            manager = self._get_marker_manager(session_id)
            if not manager:
                return web.json_response({'error': 'Session not loaded'}, status=400)

            success = manager.add_marker(frame, time, marker_type, text)
            if success:
                return web.json_response({'status': 'ok', 'markers': manager.get_all_markers()})
            return web.json_response({'error': 'Failed to create marker'}, status=500)
        except Exception as e:
            logger.error(f"Error creating marker: {e}")
            return web.json_response({'error': str(e)}, status=500)

    def _get_marker_manager(self, session_id: str) -> Optional[MarkerManager]:
        """Get or create a MarkerManager for a session."""
        if session_id not in self._marker_managers:
            # Find session path
            sessions = self.playback_engine.enumerate_sessions()
            for s in sessions:
                if s['id'] == session_id:
                    self._marker_managers[session_id] = MarkerManager(s['path'])
                    self._marker_managers[session_id].load_markers()
                    break

        return self._marker_managers.get(session_id)

    async def handle_play(self, request: web.Request) -> web.Response:
        """Start playback."""
        self.playback_engine.start()
        return web.json_response({'status': 'playing', 'time': self.playback_engine.get_current_time()})

    async def handle_pause(self, request: web.Request) -> web.Response:
        """Pause/resume playback."""
        self.playback_engine.pause()
        return web.json_response({'status': 'paused' if self.playback_engine._paused else 'playing'})

    async def handle_step(self, request: web.Request) -> web.Response:
        """Step forward or backward."""
        try:
            data = await request.json()
            direction = data.get('direction', 'forward')
            if direction == 'backward':
                frame = self.playback_engine.step_backward()
            else:
                frame = self.playback_engine.step_forward()
            return web.json_response({
                'status': 'stepped',
                'frame': self.playback_engine.get_current_frame(),
                'time': self.playback_engine.get_current_time(),
            })
        except Exception as e:
            logger.error(f"Step error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def handle_speed(self, request: web.Response) -> web.Response:
        """Set playback speed."""
        try:
            data = await request.json()
            speed = float(data.get('speed', 1.0))
            self.playback_engine.set_speed(speed)
            return web.json_response({'status': 'speed set', 'speed': speed})
        except (json.JSONDecodeError, ValueError):
            return web.json_response({'error': 'Invalid speed'}, status=400)

    async def handle_rotate(self, request: web.Response) -> web.Response:
        """Set stream rotation."""
        try:
            data = await request.json()
            stream = int(data.get('stream', 0))
            degrees = int(data.get('degrees', 90))
            self.playback_engine.set_rotation(stream, degrees)
            return web.json_response({'status': 'rotation set'})
        except (json.JSONDecodeError, ValueError):
            return web.json_response({'error': 'Invalid rotation'}, status=400)

    async def handle_seek(self, request: web.Response) -> web.Response:
        """Seek to a specific time."""
        try:
            data = await request.json()
            seconds = float(data.get('time', 0))
            self.playback_engine.seek(seconds)
            return web.json_response({'status': 'seeked', 'time': self.playback_engine.get_current_time()})
        except (json.JSONDecodeError, ValueError):
            return web.json_response({'error': 'Invalid time'}, status=400)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint for real-time updates."""
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        self._ws_clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # Handle client commands via WebSocket
                        action = data.get('action')
                        if action == 'load_session':
                            session_id = data.get('session_id')
                            if session_id:
                                try:
                                    # Find and load session
                                    sessions = self.playback_engine.enumerate_sessions()
                                    session_found = False
                                    for s in sessions:
                                        if s['id'] == session_id:
                                            session_found = True
                                            # Load all video files (.taps + .mp4 + etc.)
                                            all_files = s.get('all_files')
                                            if all_files and len(all_files) > 0:
                                                loaded = self.playback_engine.load_videos(all_files)
                                                # Load associated log file
                                                log_files = list(Path(s['path']).glob('*.txt'))
                                                if log_files:
                                                    self.log_sync.load_log_file(str(log_files[0]))
                                                await ws.send_json({
                                                    'type': 'session_loaded',
                                                    'session_id': session_id,
                                                    'files': len(all_files),
                                                })
                                                logger.info(f"Session {session_id} loaded: {len(all_files)} files")
                                            else:
                                                await ws.send_json({
                                                    'type': 'session_error',
                                                    'session_id': session_id,
                                                    'error': 'No video files found in session',
                                                })
                                                logger.warning(f"Session {session_id} has no video files")
                                            break
                                    if not session_found:
                                        await ws.send_json({
                                            'type': 'session_error',
                                            'session_id': session_id,
                                            'error': 'Session not found',
                                        })
                                        logger.warning(f"Session {session_id} not found")
                                except Exception as e:
                                    logger.error(f"Error loading session {session_id}: {e}")
                                    await ws.send_json({
                                        'type': 'session_error',
                                        'session_id': session_id,
                                        'error': str(e),
                                    })
                    except json.JSONDecodeError:
                        pass

                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._ws_clients.discard(ws)

        return ws

    async def _broadcast_updates(self):
        """Periodically broadcast state updates to all WebSocket clients."""
        try:
            while True:
                state = {
                    'type': 'state_update',
                    'time': self.playback_engine.get_current_time(),
                    'frame': self.playback_engine.get_current_frame(),
                    'playing': self.playback_engine._playing,
                    'paused': self.playback_engine._paused,
                    'cameras': self.camera_aggregator.get_status(),
                }

                # Broadcast to all connected clients
                dead_clients = set()
                for ws in self._ws_clients:
                    try:
                        await ws.send_json(state)
                    except Exception:
                        dead_clients.add(ws)

                self._ws_clients -= dead_clients
                await asyncio.sleep(0.1)  # 10 Hz update rate
        except asyncio.CancelledError:
            pass
