from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import time
from types import SimpleNamespace
import pymunk
import pytest
import torch
from core.constants import PROFILES
from core.power import keep_system_awake
from rl.cli import main
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import TrainingSession, evaluate_opponent_suite, read_json
from rl.training_config import preset_config


@pytest.mark.parametrize("profile", list(PROFILES))
def test_duel_spawns_have_two_separated_robots_and_include_both_attack_roles(profile):
    cfg = replace(preset_config("duel_selfplay"), profile=profile)
    env = cfg.make_env()
    layouts = set()
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        layouts.add(info["initialization"])
        assert obs.shape == (1, 63) and env.get_team_obs(1).shape == (1, 63)
        assert len(env.engine.all_robots) == 2
        robot_shapes = [s for s in env.engine.space.shapes if isinstance(s, pymunk.Poly)]
        assert (
            min(s.point_query(env.engine.puck.body.position).distance for s in robot_shapes)
            >= env.arena.puck_radius - 1e-6
        )
        blue, red = env.engine.blue_robots[0], env.engine.red_robots[0]
        assert (blue.body.position - red.body.position).length > 20
        for shape in robot_shapes:
            vertices = [shape.body.local_to_world(p) for p in shape.get_vertices()]
            assert all(abs(p.x) < env.arena.half_length and abs(p.y) < env.arena.half_width for p in vertices)
    assert layouts == {"attack", "defend", "faceoff", "random"}


def test_selfplay_checkpoint_keeps_frozen_anchor_and_history_without_overwriting_poc(tmp_path):
    torch.set_num_threads(1)
    cfg = replace(
        preset_config("duel_selfplay"),
        steps=16,
        envs=2,
        rollout=4,
        epochs=1,
        snapshot_every=1,
        selfplay_pool_size=3,
        episode_seconds=1,
        eval_episodes=1,
    )
    poc = HockeyPPOTrainer()
    source = tmp_path / "poc.pt"
    solo = replace(preset_config("solo_goal"), profile=cfg.profile)
    poc.save(source, dict(configuration=solo.make_env().configuration(), training_config=solo.to_dict()))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    session = TrainingSession(
        replace(cfg, init_mode="transfer", checkpoint=str(source)), tmp_path / "first", publish_live=False
    )
    session.run()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    restored = HockeyPPOTrainer()
    meta = restored.load(tmp_path / "first/latest.pt")
    league = meta["selfplay_state"]
    assert len(league["pool"]) == 3
    for key, value in poc.net.state_dict().items():
        torch.testing.assert_close(league["anchor"][key], value)
    for opponent in session.selfplay_pool:
        assert not any(parameter.requires_grad for parameter in opponent.parameters())
    cfg = replace(session.config, init_mode="resume", checkpoint=str(tmp_path / "first/latest.pt"), steps=8)
    resumed = TrainingSession(cfg, tmp_path / "resumed", publish_live=False)
    trainer = resumed.run()
    assert trainer.environment_steps == 24
    for key, value in league["anchor"].items():
        torch.testing.assert_close(resumed.selfplay_anchor.state_dict()[key], value)
    assert read_json(tmp_path / "first/status.json")["evaluation"]["complete"]


def test_fixed_suite_uses_identical_reserved_seeds_layouts_and_does_not_modify_anchor():
    cfg = replace(preset_config("duel_selfplay"), episode_seconds=1, eval_episodes=2)
    trainer = HockeyPPOTrainer()
    anchor = deepcopy(trainer.net).eval()
    before = {key: value.clone() for key, value in anchor.state_dict().items()}
    results = evaluate_opponent_suite(trainer, cfg, anchor)
    assert results["complete"] and results["episodes"] == 4
    assert set(results["by_opponent"]) == {"heuristic", "initial_policy"}
    for result in results["by_opponent"].values():
        assert [row["seed"] for row in result["records"]] == [10000, 10001]
        assert [row["scenario"] for row in result["records"]] == ["faceoff", "random"]
    for key, value in before.items():
        torch.testing.assert_close(anchor.state_dict()[key], value)
    assert not evaluate_opponent_suite(trainer, cfg, anchor, should_stop=lambda: True)["complete"]


def test_wall_time_limit_exits_cooperatively_and_saves_checkpoint(tmp_path):
    cfg = replace(preset_config("duel_selfplay"), steps=100000, envs=1, evaluation_suite=False)
    # An already expired session deadline still saves the initialized/loaded model.
    session = TrainingSession(cfg, tmp_path, publish_live=False, deadline=time.time() - 1)
    session.run()
    status = read_json(tmp_path / "status.json")
    assert status["state"] == "stopped" and status["stop_reason"] == "time_limit"
    assert status["session_transitions"] == 0
    HockeyPPOTrainer().load(tmp_path / "latest.pt")


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-hours", "0"],
        ["--max-hours", "nan"],
        ["--until", "2026-01-01T07:00:00"],
        ["--until", "2020-01-01T07:00:00-03:00"],
    ],
)
def test_bad_deadline_fails_without_starting_training(tmp_path, flags):
    directory = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(["--preset", "duel_selfplay", "--run-dir", str(directory), *flags])
    assert not directory.exists()


def test_cli_until_timezone_is_saved_as_an_absolute_deadline(tmp_path, capsys):
    until = datetime.now(timezone(timedelta(hours=-3))) + timedelta(seconds=0.15)
    main(
        [
            "--preset",
            "duel_selfplay",
            "--run-dir",
            str(tmp_path),
            "--until",
            until.isoformat(),
            "--no-live",
            "--quiet",
        ]
    )
    assert capsys.readouterr().out == ""
    runtime = read_json(tmp_path / "runtime.json")
    assert runtime["deadline"] == pytest.approx(until.timestamp())
    assert read_json(tmp_path / "status.json")["stop_reason"] == "time_limit"


def test_keep_awake_releases_windows_execution_state_even_on_failure(monkeypatch):
    import ctypes
    import core.power as power

    calls = []

    def setter(flags):
        calls.append(flags)
        return 1

    monkeypatch.setattr(power, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=setter)),
        raising=False,
    )
    with pytest.raises(RuntimeError):
        with keep_system_awake(True):
            raise RuntimeError("worker failed")
    assert calls == [0x80000001, 0x80000000]
