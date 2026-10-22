"""Record actual frozen-policy episodes; no scripted actions or selected wins."""

import hashlib
import math
import os
from pathlib import Path
import numpy as np
import torch
from core.file_io import atomic_destination
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import atomic_json, frame_from_env, participation_record


def frozen_anchor(trainer, metadata):
    from copy import deepcopy

    anchor = deepcopy(trainer.net).eval()
    league = metadata.get("selfplay_state") or {}
    if league.get("anchor"):
        anchor.load_state_dict(league["anchor"])
    anchor.requires_grad_(False)
    return anchor


def record_episode(
    checkpoint, config, destination, *, seed=777, label="PPO", max_seconds=20, should_stop=lambda: False
):
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("GIF duration must be positive and finite")
    # SDL dummy renders the same arena as the UI without opening a window.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from PIL import Image
    from ui.arena_view import draw_arena

    pygame.font.init()
    font = pygame.font.SysFont("Segoe UI", 15)
    small = pygame.font.SysFont("Segoe UI", 13)
    surface = pygame.Surface((760, 430))
    trainer = HockeyPPOTrainer()
    metadata = trainer.load(checkpoint)
    trainer.net.eval()
    anchor = frozen_anchor(trainer, metadata) if config.opponent == "selfplay" else None
    env = config.make_env()
    obs, _ = env.reset(seed=seed)
    frames, times = [], []
    total, result, last_info = 0.0, "interrupted", None

    def capture(note):
        surface.fill((10, 16, 27))
        surface.blit(font.render(label, True, (236, 242, 250)), (15, 10))
        surface.blit(
            small.render(f"seed {seed} | {env.engine.episode_time:.1f}s | {note}", True, (180, 200, 220)),
            (15, 32),
        )
        draw_arena(
            surface,
            pygame.Rect(8, 58, 744, 344),
            env.arena,
            env.engine.robot_config,
            frame_from_env(env),
            small,
        )
        surface.blit(
            small.render(
                "PPO congelado | acoes deterministicas | episodio de seed fixa", True, (170, 189, 211)
            ),
            (15, 408),
        )
        frames.append(Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB")))
        times.append(env.engine.episode_time)

    capture("inicio")
    limit = min(config.episode_steps, max(1, round(max_seconds * 30)))
    for step in range(limit):
        if should_stop():
            break
        action = trainer.select_action(obs, deterministic=True)[0]
        red = np.zeros((config.opponent_robots, 2)) if config.opponent in ("none", "zero") else None
        if anchor is not None:
            with torch.no_grad():
                dist, _ = anchor.distribution(torch.from_numpy(env.get_team_obs(1)))
                red = dist.mean.tanh().numpy()
        obs, reward, term, trunc, last_info = env.step(action, red)
        total += reward
        if step % 3 == 2 or term or trunc:
            capture(last_info["result"])
        if term or trunc:
            result = last_info["result"]
            break
    else:
        result = "clip_limit" if limit < config.episode_steps else last_info["result"]
    if times[-1] != env.engine.episode_time:
        capture(result)
    durations = [max(10, round((b - a) * 100) * 10) for a, b in zip(times, times[1:])] + [800]
    destination = Path(destination)
    with atomic_destination(destination) as temporary:
        frames[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
    record = dict(
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        seed=seed,
        selected_example=False,
        deterministic=True,
        result=result,
        simulated_seconds=env.engine.episode_time,
        playback_ms=sum(durations),
        final_hold_ms=800,
        frames=len(frames),
        reward=total,
        config=config.to_dict(),
        max_recorded_seconds=max_seconds,
        opponent="initial_policy" if anchor is not None else config.opponent,
        participation=participation_record(last_info) if last_info is not None else None,
    )
    atomic_json(destination.with_suffix(".json"), record)
    return record
