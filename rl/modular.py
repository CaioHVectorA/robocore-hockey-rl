"""Sequential PPO experiments: separate proof sets, immutable approvals and GIFs."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import signal
import time
import torch
from core.file_io import atomic_destination
from core.power import keep_system_awake
from rl.ppo_trainer import HockeyPPOTrainer
from rl.recording import frozen_anchor, record_episode
from rl.training import TrainingSession, atomic_json, evaluate_config, read_json, participation_summary
from rl.training_config import TrainingConfig


@contextmanager
def single_worker(directory):
    """OS lock survives stale files and is released even if a worker crashes."""
    path = Path(directory) / ".worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Já existe um worker ativo nesta sessão modular") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def module_config(plan, module):
    data = TrainingConfig.from_dict(plan["base"]).to_dict()
    overrides = dict(module.get("config", {}))
    data["reward"].update(overrides.pop("reward", {}))
    data.update(overrides, name=module["label"])
    return TrainingConfig.from_dict(data)


def load_plan(path):
    plan = read_json(path)
    if not isinstance(plan, dict) or not plan.get("modules"):
        raise ValueError("Plano modular ausente ou vazio")
    if not 0 < plan.get("threshold", 0) <= 1:
        raise ValueError("Meta de aprovação deve estar em (0, 1]")
    for field in (
        "evaluation_episodes",
        "confirmation_episodes",
        "regression_episodes",
        "chunk_steps",
        "max_stage_steps",
        "plateau_checks",
    ):
        if type(plan.get(field)) is not int or plan[field] <= 0:
            raise ValueError(f"Plano inválido: {field}")
    identifiers = []
    for module in plan["modules"]:
        key = module["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key) or key in identifiers:
            raise ValueError("Identificador de módulo inválido ou duplicado")
        identifiers.append(key)
        config = module_config(plan, module)
        if config.scenario in ("curriculum", "solo_curriculum"):
            raise ValueError("Cada módulo deve ter um cenário fixo")
        if config.opponent == "mixed":
            raise ValueError("Módulos usam adversários explícitos para avaliação e GIF")
    return plan


def passes(result, threshold, count):
    if not result.get("complete"):
        return False
    opponents = result.get("by_opponent") or {"single": result}
    return all(
        row["episodes"] >= count and row["active_success_rate"] >= threshold for row in opponents.values()
    )


def progress_quality(result):
    rows = list((result.get("by_opponent") or {"single": result}).values())
    return min(
        row["active_success_rate"]
        + 0.05 * row["touch_episode_rate"]
        - 0.01 * min(1.0, row["mean_distance_to_puck"] / 300)
        - 0.01 * row["mean_idle_fraction"]
        for row in rows
    )


def evaluate_module(checkpoint, config, count, seed, should_stop=lambda: False):
    trainer = HockeyPPOTrainer()
    metadata = trainer.load(checkpoint)
    trainer.net.eval()
    if config.opponent != "selfplay":
        return evaluate_config(trainer, config, episodes=count, seed=seed, should_stop=should_stop)
    anchor = frozen_anchor(trainer, metadata)
    results = {}
    for name, opponent in (("heuristic", None), ("initial_policy", anchor)):
        if should_stop():
            break
        results[name] = evaluate_config(
            trainer,
            config,
            episodes=count,
            seed=seed,
            should_stop=should_stop,
            opponent_net=opponent,
            opponent_name=name,
            scenarios=("faceoff", "random"),
        )
    records = [record for row in results.values() for record in row["records"]]
    return dict(
        episodes=len(records),
        complete=len(results) == 2 and all(r["complete"] for r in results.values()),
        by_opponent=results,
        records=records,
        **participation_summary(records),
    )


def copy_checkpoint(source, destination):
    with atomic_destination(destination) as temporary:
        temporary.write_bytes(Path(source).read_bytes())


class ModularChunk(TrainingSession):
    def __init__(self, owner, *args, **kwargs):
        self.owner = owner
        super().__init__(*args, **kwargs)

    def control(self):
        own_stop = super().control()
        if self.owner.stopping():
            self.stop_requested, self.stop_reason = True, self.owner.reason
        return self.stop_requested or own_stop

    def status(self, phase, **extra):
        super().status(phase, **extra)
        self.owner.status("training", child=read_json(self.directory / "status.json", {}))


class ModularRun:
    def __init__(self, plan, directory, initial, *, deadline=None, resume=False):
        self.plan, self.directory = plan, Path(directory).resolve()
        self.deadline = deadline
        self.started = time.time()
        self.interrupted = False
        self.reason = None
        self.last_phase = None
        self.cached_control = {}
        self.last_control = -math.inf
        self.directory.mkdir(parents=True, exist_ok=True)
        if resume:
            self.state = read_json(self.directory / "state.json")
            if not isinstance(self.state, dict):
                raise ValueError("Estado modular não encontrado")
        else:
            if (self.directory / "state.json").exists():
                raise ValueError("Sessão existente; use --resume")
            initial = Path(initial).resolve()
            HockeyPPOTrainer().load(initial)
            self.state = dict(
                index=0,
                checkpoint=str(initial),
                chunk=0,
                trained_steps=0,
                approved=[],
                best_quality=None,
                flat_checks=0,
                before_recorded=False,
                pending_chunk=None,
                initial_checkpoint=str(initial),
                initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
            )
            atomic_json(self.directory / "plan.json", plan)
            self.save()
        atomic_json(
            self.directory / "runtime.json",
            dict(pid=os.getpid(), deadline=deadline, started_at=datetime.now(timezone.utc).isoformat()),
        )

    def save(self):
        atomic_json(self.directory / "state.json", self.state)

    def status(self, phase, **extra):
        index = self.state["index"]
        module = self.plan["modules"][index] if index < len(self.plan["modules"]) else None
        data = dict(
            state=phase,
            current_index=index,
            current_module=module["id"] if module else None,
            label=module["label"] if module else "Todos os módulos aprovados",
            approved=[x["id"] for x in self.state["approved"]],
            checkpoint=self.state["checkpoint"],
            chunk=self.state["chunk"],
            stage_transitions=self.state["trained_steps"],
            elapsed_seconds=time.time() - self.started,
            updated_at=time.time(),
            deadline=self.deadline,
            stop_reason=self.reason,
            **extra,
        )
        atomic_json(self.directory / "status.json", data)
        if phase != self.last_phase or "evaluation" in extra:
            print(json.dumps(data, ensure_ascii=True), flush=True)
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.last_phase = phase

    def stopping(self):
        if time.monotonic() - self.last_control >= 0.1:
            self.cached_control = read_json(self.directory / "control.json", {})
            self.last_control = time.monotonic()
        control = self.cached_control
        if self.interrupted or control.get("stop"):
            self.reason = "requested"
            return True
        if self.deadline is not None and time.time() >= self.deadline:
            self.reason = "time_limit"
            return True
        while control.get("paused"):
            self.status("paused")
            time.sleep(0.2)
            control = read_json(self.directory / "control.json", {})
            self.cached_control = control
            if control.get("stop") or self.interrupted:
                self.reason = "requested"
                return True
            if self.deadline is not None and time.time() >= self.deadline:
                self.reason = "time_limit"
                return True
        return False

    def gallery(self):
        cards = read_json(self.directory / "gifs.json", [])
        text = [
            "# Treino PPO modular",
            "",
            "GIFs de seeds fixas, com ações reais da política. Consulte os JSONs ao lado de cada GIF.",
            "",
        ]
        blocks = []
        for card in cards:
            title, filename = card["label"], card["file"]
            text.extend(
                [
                    f"## {title}",
                    "",
                    f"![{title}]({filename})",
                    "",
                    f"Resultado: {card['result']} · seed {card['seed']}",
                    "",
                ]
            )
            blocks.append(
                f'<article><h2>{html.escape(title)}</h2><img loading="lazy" src="{html.escape(filename)}"><p>{html.escape(card["result"])} · seed {card["seed"]}</p><a href="{html.escape(str(Path(filename).with_suffix(".json")))}">Metadados</a></article>'
            )
        with atomic_destination(self.directory / "INDEX.md") as temporary:
            temporary.write_text("\n".join(text), encoding="utf-8")
        document = (
            '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>PPO modular · comportamento</title><style>body{background:#101927;color:#e9f1fa;font:16px system-ui;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}article{background:#1a2a40;border-radius:16px;padding:18px}img{width:100%;border-radius:10px}h2{font-size:18px}a{color:#67d8ca}</style><h1>PPO modular</h1><p>Episódios de seeds fixas. Aprovação exige validação e confirmação separadas.</p><main>'
            + "".join(blocks)
            + "</main></html>"
        )
        with atomic_destination(self.directory / "index.html") as temporary:
            temporary.write_text(document, encoding="utf-8")

    def record(self, checkpoint, config, filename, label):
        self.status("recording")
        metadata = record_episode(
            checkpoint, config, self.directory / filename, label=label, seed=777, should_stop=self.stopping
        )
        cards = read_json(self.directory / "gifs.json", [])
        cards.append(dict(label=label, file=filename, seed=777, result=metadata["result"]))
        atomic_json(self.directory / "gifs.json", cards)
        self.gallery()

    def recover_pending(self):
        pending = self.state.get("pending_chunk")
        if not pending:
            return
        checkpoint = Path(pending) / "latest.pt"
        if checkpoint.exists():
            metadata = HockeyPPOTrainer().load(checkpoint)
            self.state["trained_steps"] += int(metadata["session_transitions"])
            self.state["checkpoint"] = str(checkpoint)
        self.state["pending_chunk"] = None
        self.save()

    def run(self):
        torch.set_num_threads(1)
        self.recover_pending()
        self.status("starting")
        # Preserve current behaviour before any training, on two useful score tasks.
        if not (self.directory / "gifs.json").exists():
            base = TrainingConfig.from_dict(self.plan["base"])
            for scenario, title in (
                ("solo_carry", "Atual · condução 1v0"),
                ("solo_search", "Atual · busca e gol 1v0"),
            ):
                if self.stopping():
                    break
                cfg = replace(
                    base,
                    scenario=scenario,
                    objective="score",
                    opponent="none",
                    opponent_robots=0,
                    reach_speed_limit=0,
                    robots_per_team=1,
                    episode_seconds=15,
                )
                self.record(self.state["checkpoint"], cfg, f"gifs/000-current-{scenario}.gif", title)
        while self.state["index"] < len(self.plan["modules"]) and not self.stopping():
            index = self.state["index"]
            module = self.plan["modules"][index]
            cfg = module_config(self.plan, module)
            folder = self.directory / f"{index + 1:02d}-{module['id']}"
            folder.mkdir(exist_ok=True)
            cfg.save(folder / "module-config.json")
            if not self.state["before_recorded"]:
                self.record(
                    self.state["checkpoint"],
                    cfg,
                    f"gifs/{index + 1:02d}-{module['id']}-before.gif",
                    module["label"] + " · antes",
                )
                self.state["before_recorded"] = True
                self.save()
            self.status("evaluating")
            score = evaluate_module(
                self.state["checkpoint"],
                cfg,
                self.plan["evaluation_episodes"],
                70000 + index * 1000,
                self.stopping,
            )
            atomic_json(folder / f"evaluation-{self.state['chunk']:03d}.json", score)
            if not score.get("complete"):
                if self.stopping():
                    break
                raise RuntimeError("Avaliação incompleta; pesos não serão aprovados")
            quality = progress_quality(score)
            if self.state["best_quality"] is None or quality > self.state["best_quality"] + 0.005:
                self.state["best_quality"], self.state["flat_checks"] = quality, 0
                copy_checkpoint(self.state["checkpoint"], folder / "best.pt")
            else:
                self.state["flat_checks"] += 1
            self.status(
                "evaluating",
                evaluation={
                    key: value for key, value in score.items() if key not in ("records", "by_opponent")
                },
            )
            approved = passes(score, self.plan["threshold"], self.plan["evaluation_episodes"])
            confirmation, regressions = None, {}
            if approved and not self.stopping():
                confirmation = evaluate_module(
                    self.state["checkpoint"],
                    cfg,
                    self.plan["confirmation_episodes"],
                    90000 + index * 1000,
                    self.stopping,
                )
                atomic_json(folder / f"confirmation-{self.state['chunk']:03d}.json", confirmation)
                approved = passes(confirmation, self.plan["threshold"], self.plan["confirmation_episodes"])
            if approved:
                for previous in self.state["approved"]:
                    previous_module = self.plan["modules"][previous["index"]]
                    if not previous_module.get("retain", False) or self.stopping():
                        continue
                    result = evaluate_module(
                        self.state["checkpoint"],
                        module_config(self.plan, previous_module),
                        self.plan["regression_episodes"],
                        110000 + previous["index"] * 1000,
                        self.stopping,
                    )
                    regressions[previous["id"]] = result
                    approved &= passes(result, self.plan["threshold"], self.plan["regression_episodes"])
                atomic_json(folder / f"regressions-{self.state['chunk']:03d}.json", regressions)
            self.save()
            if self.stopping():
                break
            if approved:
                accepted = folder / "approved.pt"
                if accepted.exists():
                    raise ValueError("Checkpoint aprovado já existe; não será sobrescrito")
                copy_checkpoint(self.state["checkpoint"], accepted)
                atomic_json(
                    folder / "approval.json",
                    dict(
                        module=module,
                        threshold=self.plan["threshold"],
                        sha256=hashlib.sha256(accepted.read_bytes()).hexdigest(),
                        validation=score,
                        confirmation=confirmation,
                        regressions=regressions,
                    ),
                )
                self.record(
                    accepted,
                    cfg,
                    f"gifs/{index + 1:02d}-{module['id']}-approved.gif",
                    module["label"] + " · aprovado",
                )
                self.state["approved"].append(dict(id=module["id"], index=index, checkpoint=str(accepted)))
                self.state.update(
                    index=index + 1,
                    checkpoint=str(accepted),
                    chunk=0,
                    trained_steps=0,
                    best_quality=None,
                    flat_checks=0,
                    before_recorded=False,
                    pending_chunk=None,
                )
                self.save()
                continue
            if (
                self.state["trained_steps"] >= self.plan["max_stage_steps"]
                or self.state["flat_checks"] >= self.plan["plateau_checks"]
            ):
                self.reason = (
                    "stage_budget"
                    if self.state["trained_steps"] >= self.plan["max_stage_steps"]
                    else "plateau"
                )
                self.record(
                    self.state["checkpoint"],
                    cfg,
                    f"gifs/{index + 1:02d}-{module['id']}-final.gif",
                    module["label"] + " · sem aprovação",
                )
                self.status("needs_attention")
                return
            steps = min(self.plan["chunk_steps"], self.plan["max_stage_steps"] - self.state["trained_steps"])
            chunk_cfg = replace(
                cfg,
                steps=steps,
                seed=2000000 + index * 10000000 + cfg.seed + self.state["chunk"] * 100003,
                init_mode="resume" if self.state["trained_steps"] else "transfer",
                checkpoint=str(self.state["checkpoint"]),
            )
            self.state["chunk"] += 1
            chunk_folder = folder / f"train-{self.state['chunk']:03d}"
            if chunk_folder.exists():
                raise ValueError("Pasta de coleta já existe; use --resume para recuperar")
            self.state["pending_chunk"] = str(chunk_folder)
            self.save()
            chunk_folder.mkdir()
            atomic_json(
                chunk_folder / "runtime.json",
                dict(pid=os.getpid(), parent_run=str(self.directory), deadline=self.deadline),
            )
            session = ModularChunk(
                self,
                chunk_cfg,
                chunk_folder,
                reporter=lambda *args: None,
                publish_live=False,
                deadline=self.deadline,
            )
            session.run()
            self.state["trained_steps"] += session.finished
            self.state["checkpoint"] = str(session.save_path)
            self.state["pending_chunk"] = None
            self.save()
            copy_checkpoint(session.save_path, folder / "latest.pt")
            if not self.stopping():
                self.record(
                    session.save_path,
                    cfg,
                    f"gifs/{index + 1:02d}-{module['id']}-{self.state['chunk']:03d}.gif",
                    module["label"] + f" · coleta {self.state['chunk']}",
                )
            if session.stop_requested:
                self.reason = session.stop_reason or "requested"
                self.status("needs_attention" if session.stop_reason == "stagnation" else "stopped")
                return
        self.status("completed" if self.state["index"] == len(self.plan["modules"]) else "stopped")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/training/modular.json")
    parser.add_argument("--initial", default="checkpoints/foundation-candidate-20261022.pt")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Pasta modular existente; usa o plano salvo")
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--keep-awake", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    for name in ("status", "stop", "pause", "continue"):
        actions.add_argument("--" + name, dest="control_" + name)
    args = parser.parse_args(argv)
    for name in ("status", "stop", "pause", "continue"):
        directory = getattr(args, "control_" + name)
        if directory:
            root = Path(directory)
            data = read_json(root / "status.json")
            if not isinstance(data, dict):
                parser.error("Sessão modular não encontrada")
            if name != "status":
                if data["state"] in ("completed", "stopped", "needs_attention", "error"):
                    parser.error("Sessão encerrada; use --resume")
                controls = read_json(root / "control.json", {})
                controls.update(
                    {"stop": True, "paused": False} if name == "stop" else {"paused": name == "pause"}
                )
                atomic_json(root / "control.json", controls)
                data = dict(requested=name, control=controls)
            print(json.dumps(data, ensure_ascii=True))
            return
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 24:
        parser.error("--max-hours deve estar em (0, 24]")
    if args.resume and args.run_dir:
        parser.error("Use --resume ou --run-dir")
    root = Path(args.resume or args.run_dir or "runs/modular/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    plan = load_plan(root / "plan.json" if args.resume else args.plan)
    with single_worker(root):
        run = ModularRun(
            plan, root, args.initial, deadline=time.time() + args.max_hours * 3600, resume=bool(args.resume)
        )
        if args.resume:
            atomic_json(root / "control.json", dict(stop=False, paused=False))
        previous = {}

        def interrupt(*unused):
            run.interrupted = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        try:
            with keep_system_awake(args.keep_awake):
                run.run()
        except Exception as exc:
            run.reason = str(exc)
            run.status("error")
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
