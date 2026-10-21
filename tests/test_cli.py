"""Headless command contracts, actual worker control and checkpoint continuation."""

import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import pytest
from rl.cli import build_parser, main, resolve_config
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import read_json
from rl.training import TrainingSession
from rl.training_config import preset_config
from dataclasses import replace


def test_config_export_has_reward_ppo_and_layout_overrides_without_starting_training(tmp_path, capsys):
    target = tmp_path / "config.json"
    directory = tmp_path / "should-not-exist"
    main(
        [
            "--preset",
            "solo_goal",
            "--reward",
            "goal=25",
            "--reward",
            "approach=1,7",
            "--gamma",
            "0.99",
            "--epochs",
            "2",
            "--batch-size",
            "64",
            "--no-random-heading",
            "--no-domain-randomization",
            "--episode-seconds",
            "5",
            "--print-config",
            "--save-config",
            str(target),
            "--run-dir",
            str(directory),
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert json.loads(target.read_text(encoding="utf8")) == data
    assert data["robots_per_team"] == 1 and data["opponent_robots"] == 0
    assert data["reward"]["goal"] == 25 and data["reward"]["approach"] == 1.7
    assert data["gamma"] == 0.99 and data["epochs"] == 2 and data["batch_size"] == 64
    assert data["episode_seconds"] == 5
    assert not data["random_heading"] and not data["domain_randomization"]
    assert not directory.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--reward", "goal=-1"],
        ["--reward", "unknown=3"],
        ["--reward", "goal=nan"],
        ["--gamma", "1.2"],
        ["--envs", "0"],
        ["--resume", "missing.pt", "--init-from", "other.pt"],
        ["--episode-seconds", "5", "--episode-steps", "300"],
    ],
)
def test_invalid_cli_config_fails_before_creating_a_session(args, tmp_path):
    directory = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        main([*args, "--run-dir", str(directory)])
    assert exc.value.code != 0 and not directory.exists()


def test_zero_opponents_are_consistent_without_needing_a_preset():
    config = resolve_config(build_parser().parse_args(["--robots", "1", "--opponent-robots", "0"]))
    assert config.opponent == "none" and config.opponent_robots == 0


def test_headless_train_json_output_resume_eval_and_no_overwrite(tmp_path, capsys):
    source, resumed = tmp_path / "source", tmp_path / "resumed"
    main(
        [
            "--preset",
            "solo_goal",
            "--steps",
            "32",
            "--envs",
            "1",
            "--rollout",
            "8",
            "--epochs",
            "1",
            "--episode-seconds",
            "1",
            "--eval-episodes",
            "1",
            "--gamma",
            "0.99",
            "--reward",
            "goal=23",
            "--run-dir",
            str(source),
            "--output",
            "json",
            "--no-live",
        ]
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["event"] == "configuration" and records[-1]["event"] == "saved"
    assert any(row["event"] == "metrics" and row["updates"] == 4 for row in records)
    assert read_json(source / "status.json")["state"] == "completed"
    assert not (source / "live.json").exists()
    checkpoint = source / "latest.pt"
    main(["--resume", str(checkpoint), "--steps", "16", "--run-dir", str(resumed), "--quiet", "--no-live"])
    assert capsys.readouterr().out == ""
    config = read_json(resumed / "config.json")
    assert config["gamma"] == 0.99 and config["reward"]["goal"] == 23
    assert config["robots_per_team"] == 1 and config["opponent_robots"] == 0
    restored = HockeyPPOTrainer()
    restored.load(resumed / "latest.pt", resume=True)
    assert restored.environment_steps == 48 and restored.updates_count == 6
    main(["--eval-only", "--resume", str(resumed / "latest.pt"), "--output", "json"])
    result = json.loads(capsys.readouterr().out)
    assert result["episodes"] == 1 and result["opponent"] == "none"
    assert read_json(source / "status.json")["session_transitions"] == 32
    before = checkpoint.read_bytes()
    with pytest.raises(SystemExit):
        main(["--preset", "solo_goal", "--run-dir", str(source)])
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("launcher", ["train.py", "main.py"])
def test_real_worker_runs_without_pygame_and_accepts_terminal_pause_continue_stop(tmp_path, capsys, launcher):
    directory = tmp_path / "headless"
    root = Path(__file__).resolve().parents[1]
    argv = [
        "train.py",
        "--preset",
        "solo_goal",
        "--steps",
        "100000",
        "--envs",
        "1",
        "--rollout",
        "8",
        "--epochs",
        "1",
        "--episode-seconds",
        "1",
        "--eval-episodes",
        "1",
        "--speed",
        "1",
        "--no-live",
        "--quiet",
        "--run-dir",
        str(directory),
    ]
    script = (
        "import runpy,sys; sys.modules['pygame']=None; "
        f"sys.argv={([launcher, '--headless', *argv[1:]] if launcher == 'main.py' else argv)!r}; "
        f"runpy.run_path({launcher!r},run_name='__main__'); "
        "assert not any(name.startswith('ui.') for name in sys.modules)"
    )
    env = dict(os.environ, SDL_VIDEODRIVER="unavailable-headless-test", DISPLAY="")
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def until(predicate):
        deadline = time.perf_counter() + 20
        while time.perf_counter() < deadline:
            state = read_json(directory / "status.json", {})
            if predicate(state):
                return state
            if process.poll() is not None:
                pytest.fail(f"Headless worker exited: {process.communicate()}")
            time.sleep(0.025)
        pytest.fail(f"Worker timed out: {read_json(directory / 'status.json')}")

    try:
        until(lambda state: state.get("session_transitions", 0) >= 1)
        main(["--pause", str(directory), "--quiet"])
        paused = until(lambda state: state.get("state") == "paused")["session_transitions"]
        time.sleep(0.2)
        assert read_json(directory / "status.json")["session_transitions"] == paused
        main(["--status", str(directory), "--output", "json"])
        assert json.loads(capsys.readouterr().out)["state"] == "paused"
        main(["--continue", str(directory), "--quiet"])
        until(lambda state: state.get("session_transitions", 0) > paused)
        main(["--stop", str(directory), "--quiet"])
        until(lambda state: state.get("state") == "stopped")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert stdout == "" and stderr == ""
        assert (directory / "latest.pt").is_file() and not (directory / "live.json").exists()
        with pytest.raises(SystemExit):
            main(["--continue", str(directory)])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_interrupt_saves_partial_rollout_and_restores_signal_handler(tmp_path):
    previous = signal.getsignal(signal.SIGINT)
    interrupted = []

    def interrupt_during_collection(event, data):
        if event == "progress" and data["session_transitions"] > 0 and not interrupted:
            interrupted.append(True)
            signal.raise_signal(signal.SIGINT)

    config = replace(preset_config("solo_goal"), steps=1000, envs=1, epochs=1)
    TrainingSession(config, tmp_path, reporter=interrupt_during_collection, publish_live=False).run()
    status = read_json(tmp_path / "status.json")
    assert status["state"] == "stopped" and status["session_transitions"] < 1000
    restored = HockeyPPOTrainer()
    restored.load(tmp_path / "latest.pt", resume=True)
    assert restored.environment_steps == status["session_transitions"]
    assert restored.updates_count == 1
    assert signal.getsignal(signal.SIGINT) == previous
