"""Lifecycle regression checks without requiring a physical T265."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from live_server import Bridge, atomic


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bridge = Bridge(SimpleNamespace(python=Path('/python'), fyp=Path('/fyp'),
            calibration=Path('/calibration.json'), model_choice=None, yolo_model=Path('/models/pipeline2-3.pt')), self.root)
        self.addCleanup(self.bridge.stop)

    def test_start_is_idempotent_and_stop_releases_process(self):
        process = Mock()
        process.poll.return_value = None
        with patch('live_server.subprocess.Popen', return_value=process) as launch:
            self.assertEqual(self.bridge.start()['state'], 'starting')
            self.bridge.start()
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--yolo-model') + 1], '/models/pipeline2-3.pt')
            self.bridge.stop()
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
            self.assertEqual(self.bridge.status()['state'], 'idle')

    def test_fresh_stale_and_failed_capture(self):
        process = Mock()
        process.poll.return_value = None
        self.bridge.process = process
        atomic(self.root / 'status.json', json.dumps({'state':'live', 'updated':time.time()}).encode())
        self.assertEqual(self.bridge.status()['state'], 'live')
        atomic(self.root / 'status.json', json.dumps({'state':'live', 'updated':time.time()-20}).encode())
        self.assertEqual(self.bridge.status()['state'], 'error')
        process.poll.return_value = 1
        (self.root / 'worker.log').write_text('Traceback\nRuntimeError: No device connected\n')
        self.assertIn('No device connected', self.bridge.status()['message'])


if __name__ == '__main__':
    unittest.main()
