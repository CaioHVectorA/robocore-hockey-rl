"""Treino PPO headless: presets, recompensas, hiperparâmetros e controle pelo terminal."""

import argparse
from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4
import torch
from core.constants import PROFILES, RobotConfig
from core.power import keep_system_awake
from rl.console import ConsoleReporter
from rl.env import SCENARIOS
from rl.ppo_trainer import HockeyPPOTrainer
from rl.tasks import RewardConfig
from rl.training import TrainingSession, atomic_json, evaluate_config, evaluate_opponent_suite, read_json
from rl.training_config import TrainingConfig, PRESET_LABELS, preset_config, CURRICULA


def reward_argument(value):
    names = {field.name for field in fields(RewardConfig)}
    try:
        key, number = value.split("=", 1)
        if key not in names:
            raise ValueError()
        return key, float(number.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError("Use NOME=VALOR. Nomes: " + ", ".join(sorted(names))) from None


def deadline_argument(value):
    try:
        end = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if end.tzinfo is None:
            raise ValueError()
        return end.timestamp()
    except ValueError:
        raise argparse.ArgumentTypeError("Use data/hora ISO com fuso: 2026-10-22T07:00:00-03:00") from None


def resolve_deadline(args):
    if args.max_hours is not None:
        if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 168:
            raise ValueError("--max-hours deve ser maior que zero e no máximo 168")
        return time.time() + args.max_hours * 3600
    if args.until is not None and args.until <= time.time():
        raise ValueError("--until precisa indicar um horário futuro")
    return args.until


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemplos:
  python train.py --preset solo_goal --steps 100000 --no-live
  python train.py --preset solo_goal --reward goal=25 --reward approach=1.2
  python train.py --resume runs/experiments/SESSAO/latest.pt --steps 50000
  python train.py --preset solo_goal --print-config --save-config configs/solo.local.json
  python train.py --status runs/experiments/SESSAO
  python train.py --stop runs/experiments/SESSAO

Ctrl+C solicita encerramento com salvamento. Nenhuma janela e aberta.
""",
    )
    base = parser.add_argument_group("Exercício e configuração")
    base.add_argument("--headless", action="store_true", help="Modo sem janela (sempre ativo no train.py)")
    source = base.add_mutually_exclusive_group()
    source.add_argument("--config", help="JSON de experimento, inclusive salvo pela UI")
    source.add_argument("--preset", choices=PRESET_LABELS, help="Exercício inicial")
    base.add_argument("--list-presets", action="store_true", help="Lista exercícios e sai")
    base.add_argument(
        "--print-config", "--dry-run", action="store_true", help="Exibe o JSON final e sai, sem treinar"
    )
    base.add_argument("--save-config", help="Exporta o JSON final antes de executar")
    base.add_argument("--name", help="Nome da sessão")
    base.add_argument("--profile", choices=PROFILES)
    base.add_argument("--scenario", choices=(*SCENARIOS, *CURRICULA))
    base.add_argument("--opponent", choices=["selfplay", "mixed", "heuristic", "zero", "none"])
    base.add_argument("--robots", "--robots-per-team", dest="robots_per_team", type=int)
    base.add_argument("--opponent-robots", type=int)
    base.add_argument("--objective", choices=["reach", "score"])
    base.add_argument("--seed", type=int)
    base.add_argument("--spawn-distance", type=float, help="Distância inicial máxima no solo (cm)")
    base.add_argument("--random-heading", action=argparse.BooleanOptionalAction, default=None)
    base.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=None)
    base.add_argument("--evaluation-suite", action=argparse.BooleanOptionalAction, default=None)
    base.add_argument("--curriculum-eval-gate", action=argparse.BooleanOptionalAction, default=None)
    duration = base.add_mutually_exclusive_group()
    duration.add_argument("--episode-seconds", type=float)
    duration.add_argument("--episode-steps", type=int, help="Duração alternativa em passos de 30 Hz")
    base.add_argument("--opponent-difficulty", type=float)

    ppo = parser.add_argument_group("PPO, orçamento e recompensas")
    for key in (
        "steps",
        "envs",
        "rollout",
        "batch_size",
        "epochs",
        "snapshot_every",
        "eval_every",
        "eval_episodes",
        "curriculum_min_episodes",
        "curriculum_window",
        "selfplay_pool_size",
        "stagnation_patience",
    ):
        ppo.add_argument("--" + key.replace("_", "-"), type=int)
    for key in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_eps",
        "target_kl",
        "entropy_coef",
        "curriculum_success_threshold",
        "selfplay_anchor_probability",
    ):
        ppo.add_argument("--" + key.replace("_", "-"), type=float)
    ppo.add_argument(
        "--reward",
        type=reward_argument,
        action="append",
        default=[],
        metavar="NOME=VALOR",
        help="Repetível: " + ", ".join(field.name for field in fields(RewardConfig)),
    )
    ppo.add_argument("--speed", type=float, help="Ritmo: 0=máximo, 1=tempo real, 5=5x")
    weights = ppo.add_mutually_exclusive_group()
    weights.add_argument("--resume", help="Retoma pesos/otimizador e infere a configuração do checkpoint")
    weights.add_argument(
        "--init-from", help="Transfere pesos para o exercício selecionado, com novo otimizador"
    )
    ppo.add_argument("--eval-only", action="store_true", help="Avalia pesos congelados e sai")

    output = parser.add_argument_group("Saída e arquivos")
    output.add_argument("--run-dir", help="Nova pasta da sessão (não reutilize uma sessão existente)")
    output.add_argument("--save", help="Caminho alternativo para o checkpoint")
    output.add_argument("--log", help="Copia as métricas JSONL para este caminho ao encerrar")
    output.add_argument(
        "--no-live", action="store_true", help="Desativa live.json; mantém física, métricas e checkpoints"
    )
    output.add_argument(
        "--output",
        choices=["text", "json", "quiet"],
        default="text",
        help="Progresso legível, eventos JSONL ou silêncio",
    )
    output.add_argument("--quiet", dest="output", action="store_const", const="quiet")
    limits = output.add_mutually_exclusive_group()
    limits.add_argument("--max-hours", type=float, help="Encerra com salvamento após esta duração real")
    limits.add_argument(
        "--until", type=deadline_argument, help="Encerra no horário ISO indicado, incluindo o fuso"
    )
    output.add_argument(
        "--keep-awake", action="store_true", help="Inibe suspensão no Windows enquanto este processo treina"
    )
    controls = parser.add_argument_group("Consultar ou controlar uma sessão por outro terminal")
    action = controls.add_mutually_exclusive_group()
    for name in ("status", "pause", "continue", "stop"):
        action.add_argument(
            "--" + name,
            dest="control_" + name,
            metavar="PASTA",
            help={
                "status": "Mostra o estado salvo da sessão",
                "pause": "Solicita pausa",
                "continue": "Continua uma sessão pausada",
                "stop": "Solicita encerramento com salvamento",
            }[name],
        )
    return parser


def resolve_config(args):
    if args.config:
        config = TrainingConfig.load(args.config)
    elif args.preset:
        config = preset_config(args.preset)
    elif args.resume or (args.eval_only and args.init_from):
        meta = HockeyPPOTrainer().load(args.resume or args.init_from)
        if "training_config" not in meta:
            raise ValueError("Checkpoint sem configuração de treino; informe --config ou --preset")
        config = TrainingConfig.from_dict(meta["training_config"])
    else:
        config = TrainingConfig()
    updates = {
        field.name: getattr(args, field.name)
        for field in fields(TrainingConfig)
        if field.name != "reward" and getattr(args, field.name, None) is not None
    }
    if args.episode_steps is not None:
        updates["episode_seconds"] = args.episode_steps / 30
    if args.opponent_robots == 0 and args.opponent is None:
        updates["opponent"] = "none"
    if args.opponent == "none" and args.opponent_robots is None:
        updates["opponent_robots"] = 0
    if args.reward:
        updates["reward"] = replace(config.reward, **dict(args.reward))
    if args.resume or args.init_from:
        updates.update(
            init_mode="resume" if args.resume else "transfer",
            checkpoint=str(Path(args.resume or args.init_from).resolve()),
        )
    return replace(config, **updates)


def control_session(args):
    for action in ("status", "pause", "continue", "stop"):
        value = getattr(args, "control_" + action)
        if not value:
            continue
        directory = Path(value).resolve()
        status = read_json(directory / "status.json")
        if not isinstance(status, dict):
            raise ValueError(f"Estado da sessão não encontrado: {directory / 'status.json'}")
        if action != "status":
            if status.get("state") not in (
                "starting",
                "running",
                "updating",
                "saving",
                "evaluating",
                "paused",
            ):
                raise ValueError("Sessão encerrada. Use --resume CAMINHO/latest.pt para um novo treino")
            control = read_json(directory / "control.json", {})
            control.update(
                {"paused": True}
                if action == "pause"
                else {"paused": False}
                if action == "continue"
                else {"stop": True, "paused": False}
            )
            atomic_json(directory / "control.json", control)
            status = dict(requested=action, directory=str(directory), control=control)
        if args.output != "quiet":
            print(
                json.dumps(status, indent=None if args.output == "json" else 2, ensure_ascii=True), flush=True
            )
        return True
    return False


def main(argv=None):
    # Windows pipes otherwise default to the legacy code page, corrupting saved logs.
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if control_session(args):
            return
        if args.list_presets:
            if args.output == "json":
                print(json.dumps(PRESET_LABELS, ensure_ascii=True))
            elif args.output != "quiet":
                for name, label in PRESET_LABELS.items():
                    print(f"{name:14} {label}")
            return
        config = resolve_config(args)
        deadline = resolve_deadline(args)
        if args.save_config:
            config.save(args.save_config)
        if args.print_config:
            print(json.dumps(config.to_dict(), indent=2, ensure_ascii=True))
            return
        if args.eval_only:
            if not config.checkpoint:
                raise ValueError("--eval-only exige --resume, --init-from ou um JSON com checkpoint")
            trainer = HockeyPPOTrainer()
            meta = trainer.load(config.checkpoint)
            if meta.get("configuration", {}).get("arena") != PROFILES[config.profile].to_dict():
                raise ValueError("Checkpoint e arena selecionada são diferentes")
            if meta.get("configuration", {}).get("robot") != asdict(RobotConfig()):
                raise ValueError("Checkpoint e modelo físico do HockeyBot são diferentes")
            if config.evaluation_suite:
                anchor = deepcopy(trainer.net).eval()
                saved_league = meta.get("selfplay_state")
                if saved_league:
                    anchor.load_state_dict(saved_league["anchor"])
                result = evaluate_opponent_suite(trainer, config, anchor)
            else:
                result = evaluate_config(trainer, config)
            if args.output != "quiet":
                print(json.dumps(result, indent=None if args.output == "json" else 2))
            return
        directory = (
            Path(args.run_dir)
            if args.run_dir
            else Path("runs/experiments") / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
        ).resolve()
        # The desktop pre-populates config/control/output before starting its worker.
        if directory.exists() and any(
            (directory / name).exists()
            for name in (
                "status.json",
                "latest.pt",
                "metrics.jsonl",
                "episodes.jsonl",
            )
        ):
            raise ValueError("A pasta já contém uma sessão. Use uma nova --run-dir e --resume para continuar")
        save = Path(args.save).resolve() if args.save else None
        reporter = ConsoleReporter(args.output)
        reporter("configuration", dict(config=config.to_dict(), directory=str(directory)))
        atomic_json(
            directory / "runtime.json",
            dict(
                pid=os.getpid(), deadline=deadline, keep_awake=args.keep_awake, publish_live=not args.no_live
            ),
        )
        session = TrainingSession(
            config, directory, save, reporter=reporter, publish_live=not args.no_live, deadline=deadline
        )
        with keep_system_awake(args.keep_awake):
            session.run()
        if args.log:
            target = Path(args.log)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != (directory / "metrics.jsonl").resolve():
                shutil.copyfile(directory / "metrics.jsonl", target)
        reporter("saved", dict(directory=str(directory), checkpoint=str(session.save_path.resolve())))
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
