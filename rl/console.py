"""Plain terminal and JSONL reporting without importing the desktop UI."""

import json
import time

PHASES = dict(
    starting="Iniciando",
    running="Coletando",
    paused="Pausado",
    updating="Atualizando PPO",
    evaluating="Avaliando",
    saving="Salvando",
    completed="Concluido",
    stopped="Encerrado e salvo",
    error="Erro",
)


class ConsoleReporter:
    def __init__(self, mode="text"):
        self.mode = mode
        self.phase = None
        self.last_progress = -float("inf")

    def __call__(self, event, data):
        if self.mode == "quiet":
            return
        if event == "progress":
            now = time.perf_counter()
            changed = data.get("state") != self.phase
            if not changed and now - self.last_progress < 1:
                return
            self.last_progress, self.phase = now, data.get("state")
        if self.mode == "json":
            print(json.dumps(dict(event=event, **data), ensure_ascii=True, allow_nan=False), flush=True)
        elif event == "configuration":
            cfg = data["config"]
            print(f"Hockey Lab | treino headless | {cfg['name']}", flush=True)
            print(
                f"Objetivo: {cfg['objective']} | robos: {cfg['robots_per_team']}v{cfg['opponent_robots']} | ambientes: {cfg['envs']}",
                flush=True,
            )
            print(f"Sessao: {data['directory']}\nCtrl+C solicita salvamento e encerramento.\n", flush=True)
        elif event == "progress":
            done, target = data["session_transitions"], data["target"]
            elapsed = data.get("wall_seconds", 0)
            rate = done / max(elapsed, 0.001)
            print(
                f"{PHASES.get(data['state'], data['state']):18} | {done:>9,}/{target:,} ({done / max(1, target):5.1%}) | {rate:,.0f} passos/s | {elapsed:.0f}s",
                flush=True,
            )
            curriculum = data.get("curriculum")
            if data.get("stop_reason") == "stagnation":
                print(
                    "  Parada por estagnacao: episodios sem sucesso ativo; checkpoint preservado.", flush=True
                )
            if curriculum:
                print(
                    f"  Curriculo: {curriculum['label']} | sucesso {curriculum['success_rate']:.0%} / meta {curriculum['threshold']:.0%} | episodios {curriculum['recent_count']} (minimo {curriculum['minimum']})",
                    flush=True,
                )
        elif event == "metrics":
            reward, success = data.get("mean_episode_reward"), data.get("success_rate")
            reward = "aguardando episodios" if reward is None else f"{reward:.3f}"
            success = "--" if success is None else f"{success:.1%}"
            print(
                f"  Update {data['updates']} | episodios {data['episodes']} | retorno {reward} | sucesso {success} | KL {data['approx_kl']:.4f}",
                flush=True,
            )
            parts = data.get("mean_reward_components")
            activity = data.get("participation")
            if activity:
                print(
                    f"  Participacao: toque em {activity['touch_episode_rate']:.0%} dos episodios | "
                    f"sucesso ativo {activity['active_success_rate']:.0%} | "
                    f"inativo {activity['mean_idle_fraction']:.0%} | "
                    f"distancia media {activity['mean_distance_to_puck']:.1f} cm",
                    flush=True,
                )
            if parts:
                print(
                    "  Retorno medio por componente: "
                    + " | ".join(f"{key}={value:+.3f}" for key, value in parts.items()),
                    flush=True,
                )
            if "evaluation" in data:
                score = data["evaluation"]
                print(
                    f"  Avaliacao reservada: {score['success_rate']:.1%} de sucesso em {score['episodes']} episodios",
                    flush=True,
                )
        elif event == "saved":
            print(f"\nPesos: {data['checkpoint']}\nSessao: {data['directory']}", flush=True)
        elif event == "error":
            # Errors are also emitted on stderr by the CLI, including in quiet mode.
            print(f"Falha: {data['error']}", flush=True)
