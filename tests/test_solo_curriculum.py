from dataclasses import replace
import json
import numpy as np
import pymunk
import pytest
from core.arena_geometry import arena_outline, arena_wall_segments
from core.constants import PROFILES, SimulationConfig
from core.hockey_physics import HockeyPhysicsEngine
from rl.curriculum import SoloCurriculum, SOLO_STAGES
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import TrainingSession, curriculum_stage, evaluate_config, read_json
from rl.training_config import preset_config


@pytest.mark.parametrize("profile", list(PROFILES))
def test_shared_arena_boundary_has_real_open_goals_and_contains_both_recesses(profile):
    arena = PROFILES[profile]
    engine = HockeyPhysicsEngine(1, False, arena=arena, opponent_robots=0)
    vertices = arena_outline(arena)
    assert max(x for x, _ in vertices) == pytest.approx(arena.half_length + arena.goal_depth)
    assert min(x for x, _ in vertices) == pytest.approx(-arena.half_length - arena.goal_depth)
    assert max(y for _, y in vertices) == pytest.approx(arena.half_width)
    expected = set(arena_wall_segments(arena))
    colliders = {
        ((s.a.x, s.a.y), (s.b.x, s.b.y)) for s in engine.space.shapes if isinstance(s, pymunk.Segment)
    }
    assert colliders == expected
    for sign in (-1, 1):
        # A puck must fit through the actual opening without an invisible mouth wall.
        x = sign * arena.half_length
        walls = [s for s in engine.space.shapes if isinstance(s, pymunk.Segment)]
        assert min(s.point_query((x, 0)).distance for s in walls) > arena.puck_radius
        assert any(
            s.point_query((sign * (arena.half_length + arena.goal_depth), 0)).distance < 0 for s in walls
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_puck_physically_crosses_open_goal_and_back_wall_contains_it(sign):
    arena = PROFILES["micro-practice"]
    engine = HockeyPhysicsEngine(
        1,
        False,
        arena=arena,
        opponent_robots=0,
        simulation=SimulationConfig(puck_friction=0, radio_latency=0),
    )
    engine.place_puck(sign * (arena.half_length - 6), 0, sign * 70, 0)
    goals = []
    positions = []
    for _ in range(30):
        goals.append(engine.step([[0, 0]], np.zeros((0, 2)))["goal_scored"])
        positions.append(engine.puck.x * sign)
    assert goals.count(sign) == 1
    assert max(positions) <= arena.half_length + arena.goal_depth - arena.puck_radius + 0.2
    assert max(positions) > arena.half_length + arena.puck_radius


def test_curriculum_only_advances_on_enough_success_in_current_stage():
    curriculum = SoloCurriculum(preset_config("solo_curriculum"))
    records = [dict(stage="solo_carry", success=True)] * 100
    assert not curriculum.update(records) and curriculum.stage == "solo_finish"
    records = [dict(stage="solo_finish", success=True)] * 29
    assert not curriculum.update(records)
    records.append(dict(stage="solo_finish", success=False))
    assert curriculum.update(records) and curriculum.stage == "solo_carry"
    assert not curriculum.update(records) and curriculum.recent_count == 0
    records = [dict(stage="solo_carry", success=True)] * 21 + [dict(stage="solo_carry", success=False)] * 9
    assert curriculum.update(records) and curriculum.stage == "solo_search"
    assert not curriculum.update([dict(stage="solo_search", success=True)] * 100)
    restored = SoloCurriculum(preset_config("solo_curriculum"))
    restored.restore(curriculum.state())
    assert restored.stage == "solo_search"
    with pytest.raises(ValueError):
        restored.restore(dict(index=3))


def test_recent_window_does_not_promote_from_old_successes():
    config = replace(preset_config("solo_curriculum"), curriculum_window=30)
    curriculum = SoloCurriculum(config)
    records = [dict(stage="solo_finish", success=True)] * 70 + [dict(stage="solo_finish", success=False)] * 30
    assert not curriculum.update(records) and curriculum.success_rate == 0


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("stage", SOLO_STAGES)
def test_curriculum_spawns_are_seeded_collision_free_and_stay_solo(profile, stage):
    cfg = replace(preset_config("solo_curriculum"), profile=profile)
    env = cfg.make_env(stage)
    robot_sides = []
    for seed in range(12):
        first, _ = env.reset(seed=seed)
        expected = (env.engine.puck.x, env.engine.puck.y, env.engine.blue_robots[0].theta)
        second, _ = env.reset(seed=seed)
        np.testing.assert_array_equal(first, second)
        assert expected == (env.engine.puck.x, env.engine.puck.y, env.engine.blue_robots[0].theta)
        assert len(env.engine.all_robots) == 1 and len(env.engine.red_robots) == 0
        puck = env.engine.puck
        robot = env.engine.blue_robots[0]
        robot_sides.append(robot.x - puck.x)
        shapes = [s for s in env.engine.space.shapes if s.body is robot.body]
        assert min(s.point_query(puck.body.position).distance for s in shapes) >= env.arena.puck_radius - 1e-6
        assert abs(puck.x) + env.arena.puck_radius < env.arena.half_length
        assert abs(puck.y) + env.arena.puck_radius < env.arena.half_width
        if stage != "solo_search":
            assert robot.x < puck.x
    if stage == "solo_search":
        assert min(robot_sides) < 0 < max(robot_sides)  # both behind and ahead of the puck


@pytest.mark.parametrize("blue", [1, 3])
def test_signed_reward_components_reconcile_with_each_step_and_episode(blue):
    cfg = replace(preset_config("solo_goal"), robots_per_team=blue)
    env = cfg.make_env()
    env.reset(seed=5)
    total = 0
    for tick in range(12):
        _, reward, _, _, info = env.step(np.full((blue, 2), 0.4 + tick * 0.01))
        assert sum(info["reward_components"].values()) == pytest.approx(reward, abs=1e-6)
        assert info["reward_components"]["time_penalty"] == -cfg.reward.time_penalty
        assert info["reward_components"]["energy_penalty"] <= 0
        assert info["reward_components"]["reach_bonus"] == 0
        total += reward
    assert sum(info["episode_reward_components"].values()) == pytest.approx(total, abs=1e-5)
    env.reset()
    assert sum(env.episode_reward_components.values()) == 0


def test_curriculum_checkpoint_resume_preserves_stage_and_eval_uses_full_search(tmp_path):
    cfg = replace(
        preset_config("solo_curriculum"),
        steps=8,
        envs=1,
        rollout=8,
        epochs=1,
        episode_seconds=1,
        eval_episodes=1,
    )
    source = TrainingSession(cfg, tmp_path / "source")
    source.curriculum.index = 1
    trainer = HockeyPPOTrainer()
    trainer.environment_steps = 24
    path = tmp_path / "source.pt"
    trainer.save(path, source.metadata(cfg.make_env("solo_carry"), trainer))
    target = TrainingSession(replace(cfg, init_mode="resume", checkpoint=str(path)), tmp_path / "resume")
    target.run()
    state = read_json(tmp_path / "resume/status.json")
    assert state["curriculum"]["stage"] == "solo_carry"
    restored = HockeyPPOTrainer()
    meta = restored.load(tmp_path / "resume/latest.pt", resume=True)
    assert meta["curriculum"]["index"] == 1 and restored.environment_steps == 32
    result = evaluate_config(restored, cfg)
    assert result["scenario"] == "solo_search" and result["episodes"] == 1
    assert curriculum_stage(cfg, 1) == "solo_search"
    records = [json.loads(line) for line in (tmp_path / "resume/metrics.jsonl").read_text().splitlines()]
    assert records[-1]["evaluation"]["scenario"] == "solo_search"
