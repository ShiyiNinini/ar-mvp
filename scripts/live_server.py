#!/usr/bin/env python3
"""Serve HandTrack and bridge the existing FYP tracker to local browser frames."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, data):
    staging = path.with_suffix('.tmp')
    staging.write_bytes(data)
    staging.replace(path)


def worker(args):
    # Reuse the FYP model, calibration validation, inference and overlay functions.
    os.chdir(args.fyp)
    sys.path.insert(0, str(args.fyp))
    import cv2
    import run_yolo_obb_umetrack_realtime as tracker
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    config = tracker.build_parser().parse_args([
        '--calibration', str(args.calibration), '--yolo-model', str(args.yolo_model)])
    config.yolo_model = str(args.yolo_model)
    if not Path(config.yolo_model).is_file():
        raise FileNotFoundError(config.yolo_model)
    calibration = tracker.load_stereo_calibration(config.calibration)
    capture = tracker.RealSenseStereoCapture(stream='infrared', width=848, height=800,
                                             fps=config.realsense_fps)
    try:
        tracker._validate_calibration(calibration, capture.calibration)
        runtime = tracker._make_runtime(config, calibration)
        started = time.monotonic()
        for index, frame in enumerate(capture.frames()):
            if stop.is_set():
                break
            if frame.left.shape != (800, 848) or frame.right.shape != (800, 848):
                raise ValueError('Expected native 848x800 T265 stereo frames')
            result = tracker.process_frame(runtime, index, frame)
            fps = (index + 1) / max(time.monotonic() - started, .001)
            preview = tracker._preview(frame, runtime, result, fps)
            ok, jpg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise RuntimeError('Could not encode camera frame')
            atomic(args.output / 'frame.jpg', jpg.tobytes())
            atomic(args.output / 'status.json', json.dumps({
                'state': 'live', 'fps': round(fps, 1),
                'hands': len(result.tracked_hands), 'updated': time.time(),
                'model': str(args.yolo_model),
            }).encode())
    finally:
        capture.close()


class Bridge:
    def __init__(self, args, output):
        self.args, self.output = args, output
        self.process = None
        self.lock = threading.Lock()
        self.log = None
        self.started = 0

    def status(self):
        if self.process is None:
            return {'state': 'idle'}
        if self.process.poll() is not None:
            lines = (self.output / 'worker.log').read_text(errors='replace').splitlines()
            return {'state': 'error', 'message': next((s for s in reversed(lines) if s.strip()),
                                                       'Camera process stopped')[-500:]}
        path = self.output / 'status.json'
        if path.exists():
            info = json.loads(path.read_text())
            if time.time() - info['updated'] > 10:
                return {'state': 'error', 'message': 'Camera frames stopped. Disconnect and reconnect the T265.'}
            return info
        if time.monotonic() - self.started > 120:
            return {'state': 'error', 'message': 'Camera startup timed out. Disconnect and check the T265 connection.'}
        return {'state': 'starting'}

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.status()
            for name in ('frame.jpg', 'status.json'):
                (self.output / name).unlink(missing_ok=True)
            if self.log:
                self.log.close()
            self.log = (self.output / 'worker.log').open('w')
            env = dict(os.environ, MPLCONFIGDIR=str(self.output / 'mpl'),
                       XDG_CACHE_HOME=str(self.output / 'cache'))
            self.started = time.monotonic()
            self.process = subprocess.Popen([
                str(self.args.python), str(Path(__file__).resolve()), '--worker',
                '--fyp', str(self.args.fyp), '--calibration', str(self.args.calibration),
                '--yolo-model', str(self.args.yolo_model), '--output', str(self.output),
            ], stdout=self.log, stderr=subprocess.STDOUT, env=env)
            return self.status()

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
            if self.log:
                self.log.close()
                self.log = None
            return {'state': 'idle'}


def handler_for(bridge):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def allowed(self):
            host = self.headers.get('Host', '')
            valid = {f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return host in valid and (not origin or origin == f'http://{host}')

        def respond(self, data, kind='application/json', code=200):
            if isinstance(data, dict):
                data = json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.allowed():
                return self.respond({'message': 'Local access only'}, code=403)
            path = urlsplit(self.path).path
            if path == '/api/status':
                return self.respond(bridge.status())
            if path == '/api/frame':
                frame = bridge.output / 'frame.jpg'
                if bridge.status()['state'] != 'live' or not frame.exists():
                    return self.respond({'message': 'No live frame'}, code=503)
                return self.respond(frame.read_bytes(), 'image/jpeg')
            if path.startswith('/api/'):
                return self.respond({'message': 'Unknown endpoint'}, code=404)
            return super().do_GET()

        def do_POST(self):
            if not self.allowed() or self.headers.get('X-HandTrack') != '1':
                return self.respond({'message': 'Use the local HandTrack page'}, code=403)
            action = {'/api/start': bridge.start, '/api/stop': bridge.stop}.get(self.path)
            if action is None:
                return self.respond({'message': 'Unknown endpoint'}, code=404)
            try:
                self.respond(action())
            except Exception as exc:
                self.respond({'state': 'error', 'message': str(exc)}, code=500)

        def log_message(self, *_):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=4174)
    p.add_argument('--fyp', type=Path, default=Path.home() / 'Documents/FYP/Umetrack-with-YOLO-roi')
    p.add_argument('--calibration', type=Path, default=Path.home() / 'Documents/FYP/T265-dataset/t265_calibration.json')
    p.add_argument('--python', type=Path, default=Path('/opt/anaconda3/envs/umetrack/bin/python'))
    p.add_argument('--model-choice', choices=['pipeline1', 'pipeline2'], default=None)
    p.add_argument('--yolo-model', type=Path, default=None, help='Explicit YOLO OBB weights; overrides --model-choice')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.yolo_model = (args.yolo_model or
                       (args.fyp / 'models' / f'{args.model_choice}.pt' if args.model_choice else
                        Path.home() / 'Documents/Pipeline/pipeline2-3.pt')).resolve()
    if args.worker:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix='handtrack-live-') as folder:
        bridge = Bridge(args, Path(folder))
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(bridge))
        print(f'HandTrack: http://127.0.0.1:{args.port} — connect the T265, then click Connect camera.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            bridge.stop()
            server.server_close()


if __name__ == '__main__':
    main()
