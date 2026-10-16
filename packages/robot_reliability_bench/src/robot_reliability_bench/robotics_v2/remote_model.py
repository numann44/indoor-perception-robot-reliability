"""Isolate optional learned inference from the minimal simulator environment."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import time
import uuid
from .ipc import predict_remote


class RemoteModelSession:
    """One fresh main-runtime inference process per context/episode."""
    def __init__(self, root, model_pack, output, *, backend='native', device='cpu', threads=1,
                 tracker=None, semantic_pack=None, tracker_lock=None, semantic_backend='native'):
        self.root = Path(root)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.socket = f'cache/v2/robotics/rpc-{uuid.uuid4().hex[:6]}.sock'
        self.command = [str(self.root/'environments/main/bin/python'), str(self.root/'tools/robotics_v2.py'),
            'serve', '--socket', self.socket, '--model-pack', str(model_pack), '--backend', backend,
            '--device', device, '--threads', str(threads), '--log', str(self.output/'ipc-events.jsonl')]
        if tracker:
            self.command += ['--pipeline-tracking', '--pipeline-tracker', tracker]
        if tracker_lock:
            self.command += ['--tracker-lock',str(tracker_lock)]
        if semantic_pack:
            self.command += ['--pipeline-semantic', str(semantic_pack),'--semantic-backend',semantic_backend]
        self.environment = os.environ.copy()
        self.environment['ROBOT_MODEL_THREADS'] = str(threads)
        self.environment['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        self.process = None
        self.log = None

    def __enter__(self):
        self.log = (self.output/'process.log').open('w')
        self.process = subprocess.Popen(self.command, cwd=self.root, env=self.environment,
                                       stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+120
        while not (self.root/self.socket).exists():
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(f'Isolated model initialization failed: {self.output}/process.log')
            if time.monotonic()>deadline:
                self.close()
                raise TimeoutError('Isolated model initialization exceeded120s')
            time.sleep(.02)
        return self

    def predict(self, frame):
        # macOS AF_UNIX path limit is108bytes; keep connection paths relative.
        socket_path=os.path.relpath(self.root/self.socket,Path.cwd())
        if len(os.fsencode(socket_path))>=104:
            raise ValueError('Unix socket path too long; run robotics CLI from its project directory')
        result = predict_remote(socket_path, frame, timeout=30.)
        if result.get('status') != 'ok':
            raise RuntimeError(f"Remote inference failed: {result.get('error')}")
        return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log is not None:
            self.log.close();self.log=None
        (self.root/self.socket).unlink(missing_ok=True)

    def __exit__(self, *_):
        self.close()
