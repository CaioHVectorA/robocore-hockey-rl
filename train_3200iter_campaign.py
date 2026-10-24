"""Run a second ultra-intensive 1,600 PPO updates campaign (reaching 3,200 cumulative updates) with warm start."""

import math
import shutil
import time
from dataclasses import replace
from pathlib import Path
import torch

from core.power import keep_system_awake
from rl.training import TrainingSession
from rl.training_config import TrainingConfig
from rl.recording import record_episode
from evaluate import benchmark

ARTIFACT_DIR = Path(r"C:\Users\caihe\.gemini\antigravity\brain\4d9de323-c3e6-49f7-b3ae-6f2299acb331")
RUN_DIR = Path("runs/training_3200iter_350rpm")
FINAL_CKPT = Path("checkpoints/solo-1v0-350rpm-3200iter.pt")
BASE_CKPT = Path("checkpoints/solo-1v0-350rpm-1600iter.pt")


def main():
    torch.set_num_threads(1)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CKPT.parent.mkdir(parents=True, exist_ok=True)

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"Base checkpoint {BASE_CKPT} not found")

    # 1,600 PPO updates * 4,096 transitions/update = 6,553,600 transitions
    total_steps = 1600 * 4096

    print("=== Starting Second 1,600 PPO Updates Campaign (Warm Start) ===")
    print(f"Total environment steps: {total_steps} (1,600 updates)")
    print(f"Base checkpoint: {BASE_CKPT}")
    print(f"Output directory: {RUN_DIR}")

    cfg = TrainingConfig(
        name="1v0 Solo 350 RPM 3200 Updates Campaign",
        profile="micro-practice",
        robots_per_team=1,
        opponent_robots=0,
        objective="score",
        scenario="solo_search",
        opponent="none",
        steps=total_steps,
        envs=8,
        rollout=512,
        episode_seconds=20,
        seed=3333,
        learning_rate=0.00005,
        gamma=0.999,
        gae_lambda=0.98,
        clip_eps=0.2,
        target_kl=0.015,
        entropy_coef=0.002,
        epochs=6,
        batch_size=512,
        domain_randomization=True,
        spawn_distance=75.0,
        random_heading=True,
        heading_spread=math.pi,
        setup_spread=math.pi,
        rehearsal_scenarios=("solo_finish", "solo_carry"),
        rehearsal_probability=0.25,
        stagnation_patience=0,
        init_mode="transfer",
        checkpoint=str(BASE_CKPT.resolve()),
        snapshot_every=20,
        eval_every=1000000,
    )

    t0 = time.time()
    with keep_system_awake(enabled=True):
        session = TrainingSession(cfg, RUN_DIR, save_path=RUN_DIR / "latest.pt", publish_live=False)
        session.run()

    elapsed = time.time() - t0
    print(f"=== Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min / {elapsed / 3600:.2f} h) ===")

    # Copy latest checkpoint to permanent destination
    shutil.copy2(RUN_DIR / "latest.pt", FINAL_CKPT)
    print(f"Saved master checkpoint to: {FINAL_CKPT}")

    # Benchmark on holdout seeds
    print("\n=== Benchmarking on 30 Holdout Episodes (solo_carry) ===")
    res_carry = benchmark(episodes=30, policy=str(FINAL_CKPT), scenario="solo_carry")
    ppo_carry = res_carry["controllers"]["ppo"]
    print(
        f"PPO carry wins: {ppo_carry['wins']}/30 ({ppo_carry['success_rate']*100:.1f}%), timeouts: {ppo_carry['timeouts']}"
    )

    print("\n=== Benchmarking on 30 Holdout Episodes (solo_search) ===")
    res_search = benchmark(episodes=30, policy=str(FINAL_CKPT), scenario="solo_search")
    ppo_search = res_search["controllers"]["ppo"]
    print(
        f"PPO search wins: {ppo_search['wins']}/30 ({ppo_search['success_rate']*100:.1f}%), timeouts: {ppo_search['timeouts']}"
    )

    # Generate Final Showcase GIFs
    print("\n=== Generating Final Showcase GIFs ===")
    gif_dir = RUN_DIR / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    gif_carry = gif_dir / "final-3200iter-carry-350rpm.gif"
    cfg_carry = replace(cfg, scenario="solo_carry", episode_seconds=15)
    meta_carry = record_episode(
        FINAL_CKPT, cfg_carry, gif_carry, label="1v0 Conducao e Gol Veloz (3200 Updates · 350 RPM)", seed=42
    )
    print(f"Carry GIF generated: {meta_carry['result']} in {meta_carry['simulated_seconds']:.2f}s")

    gif_search = gif_dir / "final-3200iter-search-350rpm.gif"
    cfg_search = replace(cfg, scenario="solo_search", episode_seconds=20)
    meta_search = record_episode(
        FINAL_CKPT, cfg_search, gif_search, label="1v0 Busca, Conducao e Gol (3200 Updates · 350 RPM)", seed=42
    )
    print(f"Search GIF generated: {meta_search['result']} in {meta_search['simulated_seconds']:.2f}s")

    # Copy GIFs to Artifact Directory
    if ARTIFACT_DIR.exists():
        shutil.copy2(gif_carry, ARTIFACT_DIR / "final-3200iter-carry-350rpm.gif")
        shutil.copy2(gif_search, ARTIFACT_DIR / "final-3200iter-search-350rpm.gif")
        print(f"Copied final GIFs to {ARTIFACT_DIR}")

    print("\n=== 3200 ITERATIONS CAMPAIGN COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
