"""Nonblocking desktop client for the training worker; no simulator shortcuts."""

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
from rl.training import atomic_json, read_json
from rl.training_config import TrainingConfig
from ui.live_view import interpolate_frames


class TrainingProcess:
    def __init__(self, root, runs_root=None):
        self.root = Path(root)
        self.runs_root = Path(runs_root) if runs_root else self.root / "runs/experiments"
        self.process = None
        self.directory = None
        self.config = None
        self.status = {}
        self.frame = None
        self.records = deque(maxlen=1000)
        self.control = {}
        self.output = None
        self.offset = 0
        self.last_poll = -float("inf")
        self.last_report = -float("inf")
        self.frames = deque(maxlen=12)
        self.frame_times = deque(maxlen=30)
        self.last_received = None

    @property
    def frame_hz(self):
        if len(self.frame_times) < 2:
            return 0.0
        return (len(self.frame_times) - 1) / max(0.001, self.frame_times[-1] - self.frame_times[0])

    @property
    def frame_age(self):
        return time.perf_counter() - self.last_received if self.last_received is not None else 0.0

    def display_frame(self):
        if self.status.get("state") not in ("running", "starting"):
            return self.frame
        return interpolate_frames(self.frames, time.time() - 0.075)

    @property
    def active(self):
        return self.process is not None and self.process.poll() is None

    @property
    def checkpoint(self):
        if not self.directory:
            return None
        path = Path(self.status.get("checkpoint", self.directory / "latest.pt"))
        return path if path.exists() else None

    def start(self, config):
        if self.active:
            raise ValueError("Já existe um treino ativo")
        if config.init_mode != "fresh" and not Path(config.checkpoint).is_file():
            raise ValueError("Arquivo de pesos não encontrado")
        if self.output:
            self.output.close()
        self.directory = (
            self.runs_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
        ).resolve()
        self.directory.mkdir(parents=True)
        config.save(self.directory / "config.json")
        self.config = config
        self.records.clear()
        self.frame = None
        self.frames.clear()
        self.frame_times.clear()
        self.last_received = None
        self.last_poll = self.last_report = -float("inf")
        self.status = dict(state="starting", target=config.effective_steps, session_transitions=0)
        self.offset = 0
        self.control = dict(paused=False, stop=False, speed=config.speed)
        atomic_json(self.directory / "control.json", self.control)
        self.output = (self.directory / "output.log").open("w", encoding="utf8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(self.root / "train.py"),
                "--config",
                str(self.directory / "config.json"),
                "--run-dir",
                str(self.directory),
            ],
            cwd=self.root,
            stdout=self.output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return self.directory

    def send(self, **changes):
        if self.active:
            self.control.update(changes)
            atomic_json(self.directory / "control.json", self.control)

    def poll(self, force=False):
        now = time.perf_counter()
        if self.directory is None or (not force and now - self.last_poll < 1 / 60):
            return
        self.last_poll = now
        frame = read_json(self.directory / "live.json", self.frame)
        if frame and (not self.frame or frame.get("timestamp") != self.frame.get("timestamp")):
            self.frame = frame
            self.frames.append(frame)
            self.frame_times.append(now)
            self.last_received = now
        if not force and now - self.last_report < 0.15:
            return
        self.last_report = now
        self.status = read_json(self.directory / "status.json", self.status)
        try:
            with (self.directory / "metrics.jsonl").open(encoding="utf8") as stream:
                stream.seek(self.offset)
                while line := stream.readline():
                    if not line.endswith("\n"):
                        break
                    self.records.append(json.loads(line))
                    self.offset = stream.tell()
        except (OSError, ValueError):
            pass
        if (
            self.process is not None
            and self.process.poll() is not None
            and self.status.get("state") not in ("completed", "stopped", "error")
        ):
            try:
                detail = (self.directory / "output.log").read_text(encoding="utf8")[-2000:]
            except OSError:
                detail = "Processo encerrado sem relatório"
            self.status = dict(self.status, state="error", error=detail)

    def close(self):
        if self.active:
            self.send(stop=True, paused=False)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        self.poll(True)
        if self.output:
            self.output.close()


def list_sessions(root, limit=30):
    root = Path(root)
    if not root.exists():
        return []
    rows = []
    for directory in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        try:
            config = TrainingConfig.load(directory / "config.json")
            status = read_json(directory / "status.json", {})
            checkpoint = Path(status.get("checkpoint", directory / "latest.pt"))
            rows.append(dict(path=directory, config=config, status=status, checkpoint=checkpoint))
        except (OSError, TypeError, ValueError):
            continue
        if len(rows) >= limit:
            break
    return rows
