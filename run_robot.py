"""Overhead camera → shared estimator → policy → BLE; dry-run by default."""

from __future__ import annotations
import argparse
import asyncio
from dataclasses import fields
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
import torch
from bridge.vespa_ble_bridge import VespaTeamBridge
from core.constants import PROFILES, RobotConfig, ArenaConfig
from rl.ppo_trainer import HockeyPPOTrainer
from vision.camera_tracker import ArenaHomography, HockeyVisionTracker
from vision.state import build_team_observation, tracking_ready


class LatestCamera:
    """Capture thread overwrites a one-frame slot; control never waits on read()."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera/video source")
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = -float("inf")
        self.running = True
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while self.running:
            ok, frame = self.cap.read()
            timestamp = time.monotonic()
            if not ok:
                break
            with self.lock:
                self.frame = frame
                self.timestamp = timestamp
        self.running = False

    def latest(self):
        with self.lock:
            return self.timestamp, self.frame

    def close(self):
        self.running = False
        self.cap.release()
        self.thread.join(timeout=1.0)


async def run(args):
    cfg = json.loads(Path(args.config).read_text())
    arena = ArenaConfig(**cfg["arena"]) if "arena" in cfg else PROFILES[cfg.get("profile", "micro-2024")]
    robot = RobotConfig(**cfg.get("robot", {}))
    h = ArenaHomography(cfg["floor_corners_px"], arena)
    rh = ArenaHomography(cfg["robot_corners_px"], arena) if "robot_corners_px" in cfg else h
    tracker = HockeyVisionTracker(
        h,
        rh,
        np.array(cfg["camera_matrix"]) if "camera_matrix" in cfg else None,
        np.array(cfg["dist_coeffs"]) if "dist_coeffs" in cfg else None,
    )
    team = int(cfg.get("team", 0))
    if team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    trainer = None
    if not args.track_only:
        trainer = HockeyPPOTrainer()
        metadata = trainer.load(args.policy)
        if not trainer.updates_count:
            raise ValueError("Policy has no recorded PPO updates")
        expected = metadata.get("configuration", {})
        if expected.get("arena") != arena.to_dict():
            raise ValueError("Policy arena differs from camera configuration")
        if expected.get("robot") != {f.name: getattr(robot, f.name) for f in fields(robot)}:
            raise ValueError("Policy robot parameters differ from hardware model")
    source = cfg.get("camera", 0)
    source = int(source) if str(source).isdigit() else source
    camera = LatestCamera(source)
    bridge = None
    previous = np.zeros((3, 2), np.float32)
    last_capture = -float("inf")
    last_seen_ready = False
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    next_tick = started
    print(
        "ARMED: HOCKEY-V2 hardware" if args.armed else "DRY RUN: camera and policy only; no BLE connection",
        flush=True,
    )
    try:
        if args.armed:
            if not cfg.get("protocol_verified", False):
                raise ValueError("Verify HOCKEY-V2 firmware and pin mapping in bench tests first")
            bridge = VespaTeamBridge(cfg["ble_addresses"])
            await bridge.connect_all()
        with log_path.open("w", encoding="utf8") as log:
            while not args.seconds or time.monotonic() - started < args.seconds:
                now = time.monotonic()
                capture, frame = camera.latest()
                # cap.read timestamp is host receipt, not sensor exposure. Subtract
                # measured capture-pipeline delay and budget it in the watchdog.
                capture -= float(cfg.get("capture_delay_seconds", 0.0))
                if frame is not None and capture > last_capture:
                    await asyncio.to_thread(tracker.detect_robots_and_puck, frame, capture)
                    last_capture = capture
                now = time.monotonic()
                state = tracker.estimator.snapshot(now)
                ready = tracking_ready(state, team, max_age=0.15)
                actions = np.zeros((3, 2), np.float32)
                if ready and trainer is not None:
                    obs = build_team_observation(state, arena, robot, team, previous)
                    actions, _, _ = trainer.select_action(obs, deterministic=True)
                # A transition into lost tracking sends stop immediately, bypassing
                # the regular radio rate limiter; firmware expires stale commands.
                if bridge:
                    if not ready and last_seen_ready:
                        await bridge.emergency_stop_all()
                    if ready:
                        oldest = min(
                            state.puck.last_seen, *(state.robots[team * 3 + i].last_seen for i in range(3))
                        )
                        if not await bridge.send_team_actions(actions, deadline=oldest + 0.15):
                            raise RuntimeError("Radio send failed or state expired; team stopped")
                    else:
                        await bridge.emergency_stop_all()
                previous = actions
                last_seen_ready = ready
                log.write(
                    json.dumps(
                        dict(
                            timestamp=now,
                            capture_timestamp=capture if np.isfinite(capture) else None,
                            tracking_ready=ready,
                            actions=actions.tolist(),
                            processing_ms=(time.monotonic() - now) * 1000,
                            robots={
                                str(i): dict(
                                    x=e.x,
                                    y=e.y,
                                    theta=e.theta,
                                    vx=e.vx,
                                    vy=e.vy,
                                    omega=e.omega,
                                    valid=e.valid(now),
                                    age_seconds=e.age(now),
                                )
                                for i, e in state.robots.items()
                            },
                            puck=dict(
                                x=state.puck.x,
                                y=state.puck.y,
                                vx=state.puck.vx,
                                vy=state.puck.vy,
                                valid=state.puck.valid(now),
                            ),
                        )
                    )
                    + "\n"
                )
                log.flush()
                if frame is not None and args.preview:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        "TRACKING" if ready else "STOP: TRACKING LOST",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 220, 0) if ready else (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Overhead robotics - Q to stop", preview)
                    if cv2.waitKey(1) & 255 == ord("q"):
                        break
                if frame is None and not camera.running:
                    raise RuntimeError("Camera failed")
                next_tick += 1 / 30
                if time.monotonic() - next_tick > 1 / 30:
                    next_tick = time.monotonic()
                await asyncio.sleep(max(0, next_tick - time.monotonic()))
    finally:
        if bridge:
            await bridge.disconnect_all()
        camera.close()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--policy")
    p.add_argument("--track-only", action="store_true", help="Record camera states without a policy or BLE")
    p.add_argument("--armed", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--log", default="runs/robot.jsonl")
    args = p.parse_args()
    if not args.track_only and not args.policy:
        p.error("Provide --policy or --track-only")
    if args.armed and args.track_only:
        p.error("--track-only cannot arm the motors")
    if args.seconds < 0:
        p.error("--seconds must be nonnegative")
    torch.set_num_threads(1)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
