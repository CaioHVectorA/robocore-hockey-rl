"""Desktop experiment editor, live controls, metrics and saved-session browser."""

from dataclasses import replace
from pathlib import Path
import math
import pygame
from core.constants import PROFILES
from rl.training_config import TrainingConfig, PRESET_LABELS, preset_config
from ui.training_session import TrainingProcess, list_sessions

BG, PANEL, EDGE = (11, 18, 30), (21, 32, 48), (43, 59, 77)
TEXT, MUTED, ACCENT, GREEN = (232, 239, 245), (142, 164, 183), (255, 155, 72), (102, 213, 171)
STATES = dict(
    starting="Iniciando",
    running="Treinando",
    paused="Pausado",
    evaluating="Avaliação reservada",
    saving="Salvando",
    completed="Concluído",
    stopped="Encerrado e salvo",
    error="Falha no treino",
)


class TrainingPanel:
    def __init__(self, root, fonts, preview, evaluate, config=None):
        self.root, self.fonts = Path(root), fonts
        self.preview_callback, self.evaluate_callback = preview, evaluate
        self.config = config or TrainingConfig()
        self.worker = TrainingProcess(root)
        self.tab = "Treino"
        self.scroll = {tab: 0 for tab in ("Treino", "PPO", "Sessões", "Simular")}
        self.max_scroll = 0
        self.handlers = []
        self.fields = []
        self.focus = None
        self.edit = ""
        self.caret = 0
        self.select_all = False
        self.message = "Escolha um exercício e inicie o treino"
        self.error = False
        self.watch_live = False
        self.sessions = []
        self.selected_session = None
        self.last_scan = 0
        self.content = pygame.Rect(0, 0, 1, 1)
        self.y = 0
        self.screen = None

    @property
    def live(self):
        return self.watch_live and self.worker.frame is not None

    def notify(self, message, error=False):
        self.message, self.error = str(message), error

    def _apply(self, preview=False, **changes):
        try:
            self.config = replace(self.config, **changes)
            self.notify("Configuração pronta; aplicada ao iniciar a próxima sessão")
            if preview:
                self.preview()
        except (ValueError, TypeError) as exc:
            self.notify(exc, True)

    def set_preset(self, name):
        self.config = preset_config(name, self.config)
        self.focus = None
        self.preview()

    def preview(self):
        self.watch_live = False
        self.preview_callback(self.config)
        self.notify("Prévia do exercício com bot tático; ainda não é uma política treinada")

    def start(self):
        if not self.commit():
            return
        try:
            self.preview_callback(self.config)
            self.worker.start(self.config)
            self.watch_live = True
            self.notify("Treino ao vivo: ambiente 1, ações exploratórias do PPO")
        except (ValueError, OSError) as exc:
            self.notify(exc, True)

    def show_live(self):
        if self.worker.config:
            self.preview_callback(self.worker.config)
            self.watch_live = True
            self.notify("Ambiente real de coleta; exibição amostrada a até 10 quadros/s")

    def evaluate(self, path=None, config=None):
        path = path or self.worker.checkpoint
        config = config or self.worker.config or self.config
        if path is None or not Path(path).exists():
            self.notify("Ainda não há pesos salvos nesta sessão", True)
            return
        try:
            self.evaluate_callback(Path(path), config)
            self.watch_live = False
            self.notify("Avaliação visual com pesos congelados; o treino ativo continua separado")
        except (ValueError, RuntimeError, OSError) as exc:
            self.notify(exc, True)

    def toggle_pause(self):
        self.worker.send(paused=not self.worker.control.get("paused", False))

    def save_config(self):
        path = self.root / "runs/training-config.json"
        self.config.save(path)
        self.notify("Configuração salva em runs/training-config.json")

    def load_file(self, path):
        if self.worker.active:
            self.notify("Encerre o treino antes de trocar sua configuração", True)
            return
        try:
            path = Path(path).resolve()
            if path.suffix.lower() == ".json":
                self.config = TrainingConfig.load(path)
                self.preview()
            elif path.suffix.lower() == ".pt":
                self.config = replace(self.config, checkpoint=str(path), init_mode="transfer")
                self.notify("Pesos selecionados para iniciar um novo exercício por transferência")
        except (OSError, ValueError, TypeError) as exc:
            self.notify(exc, True)

    def commit(self):
        if self.focus is None:
            return True
        key, parser = self.focus
        try:
            value = parser(self.edit)
            if key.startswith("reward."):
                changes = dict(reward=replace(self.config.reward, **{key.split(".")[1]: value}))
            else:
                changes = {key: value}
            self.config = replace(self.config, **changes)
            self.focus = None
            self.error = False
            return True
        except (ValueError, TypeError) as exc:
            self.notify(f"Valor inválido: {exc}", True)
            return False

    def handle(self, event):
        if event.type == pygame.DROPFILE:
            self.load_file(event.file)
            return True
        if event.type == pygame.MOUSEWHEEL and self.content.collidepoint(pygame.mouse.get_pos()):
            self.scroll[self.tab] = max(0, min(self.max_scroll, self.scroll[self.tab] - event.y * 48))
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, callback in reversed(self.handlers):
                if rect.collidepoint(event.pos):
                    if self.commit():
                        callback()
                    return True
            for rect, key, parser in self.fields:
                if rect.collidepoint(event.pos):
                    if not self.commit():
                        return True
                    value = (
                        getattr(self.config.reward, key.split(".")[1])
                        if key.startswith("reward.")
                        else getattr(self.config, key)
                    )
                    self.focus = (key, parser)
                    self.edit = str(value)
                    self.caret = len(self.edit)
                    self.select_all = True
                    return True
            if self.focus is not None and not self.commit():
                return True
        if event.type == pygame.KEYDOWN and self.focus is not None:
            if event.key == pygame.K_ESCAPE:
                self.focus = None
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                self.commit()
            elif event.key == pygame.K_a and event.mod & pygame.KMOD_CTRL:
                self.select_all = True
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_HOME, pygame.K_END):
                self.caret = (
                    0
                    if event.key == pygame.K_HOME
                    else len(self.edit)
                    if event.key == pygame.K_END
                    else max(0, min(len(self.edit), self.caret + (-1 if event.key == pygame.K_LEFT else 1)))
                )
                self.select_all = False
            elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                if self.select_all:
                    self.edit, self.caret = "", 0
                elif event.key == pygame.K_BACKSPACE and self.caret:
                    self.edit = self.edit[: self.caret - 1] + self.edit[self.caret :]
                    self.caret -= 1
                elif event.key == pygame.K_DELETE:
                    self.edit = self.edit[: self.caret] + self.edit[self.caret + 1 :]
                self.select_all = False
            elif getattr(event, "unicode", "") and not event.mod & pygame.KMOD_CTRL:
                if self.select_all:
                    self.edit, self.caret = "", 0
                self.edit = self.edit[: self.caret] + event.unicode + self.edit[self.caret :]
                self.caret += len(event.unicode)
                self.select_all = False
            return True
        return False

    def text(self, value, x, y, size=14, color=TEXT, width=None):
        value = str(value)
        if width:
            while len(value) > 2 and self.fonts[size].size(value)[0] > width:
                value = value[:-2] + "…"
        self.screen.blit(self.fonts[size].render(value, True, color), (int(x), int(y)))

    def button(self, label, rect, callback, active=False, enabled=True):
        rect = pygame.Rect(rect)
        color = ACCENT if active else (37, 54, 73) if enabled else (26, 37, 51)
        pygame.draw.rect(self.screen, color, rect, border_radius=7)
        surf = self.fonts[12].render(label, True, BG if active else TEXT if enabled else MUTED)
        if surf.get_width() > rect.width - 10:
            label = label[: max(2, int(len(label) * (rect.width - 16) / surf.get_width()))] + "…"
            surf = self.fonts[12].render(label, True, BG if active else TEXT)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        visible = self.screen.get_clip().clip(rect)
        if enabled and visible.width and visible.height:
            self.handlers.append((visible, callback))

    def section(self, label):
        self.y += 12
        self.text(label.upper(), self.content.x, self.y, 12, ACCENT)
        self.y += 25

    def field(self, label, key, parser=float, x=None, width=None, advance=True):
        x = self.content.x if x is None else x
        width = self.content.width if width is None else width
        self.text(label, x, self.y, 12, MUTED, width)
        rect = pygame.Rect(x, self.y + 20, width, 31)
        focused = self.focus is not None and self.focus[0] == key
        pygame.draw.rect(self.screen, (12, 23, 37), rect, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT if focused else EDGE, rect, 1, border_radius=6)
        value = (
            getattr(self.config.reward, key.split(".")[1])
            if key.startswith("reward.")
            else getattr(self.config, key)
        )
        display = self.edit if focused else f"{value:g}" if isinstance(value, float) else str(value)
        if focused and self.select_all:
            pygame.draw.rect(self.screen, (53, 78, 98), rect.inflate(-8, -6), border_radius=3)
        self.text(display, x + 9, rect.y + 6, 14, TEXT if not self.worker.active else MUTED, width - 18)
        visible = self.screen.get_clip().clip(rect)
        if not self.worker.active and visible.height and visible.width:
            self.fields.append((visible, key, parser))
        if advance:
            self.y += 61

    def pair(self, first, second):
        width = (self.content.width - 12) // 2
        self.field(*first, width=width, advance=False)
        self.field(*second, x=self.content.x + width + 12, width=width)

    def choice(self, label, key, choices, labels=None, preview=False):
        self.text(label, self.content.x, self.y, 12, MUTED)
        current = getattr(self.config, key)

        def select(delta):
            value = choices[(choices.index(current) + delta) % len(choices)]
            changes = {key: value}
            if key == "opponent_robots":
                changes["opponent"] = (
                    "none"
                    if value == 0
                    else ("selfplay" if self.config.opponent == "none" else self.config.opponent)
                )
            self._apply(preview, **changes)

        self.button(
            "‹", (self.content.x, self.y + 20, 30, 30), lambda: select(-1), enabled=not self.worker.active
        )
        label = labels.get(current, str(current)) if labels else str(current)
        self.button(
            label,
            (self.content.x + 36, self.y + 20, self.content.width - 72, 30),
            lambda: select(1),
            enabled=not self.worker.active,
        )
        self.button(
            "›",
            (self.content.right - 30, self.y + 20, 30, 30),
            lambda: select(1),
            enabled=not self.worker.active,
        )
        self.y += 61

    def basic(self):
        self.section("Comece por um exercício")
        self.text("↓ Role para mais ajustes", self.content.right - 158, self.content.y + 12, 12, MUTED)
        width = (self.content.width - 10) // 2
        for i, (name, label) in enumerate(PRESET_LABELS.items()):
            preset = preset_config(name, self.config)
            chosen = all(
                getattr(preset, k) == getattr(self.config, k)
                for k in ("robots_per_team", "opponent_robots", "opponent", "objective", "scenario")
            )
            self.button(
                label,
                (self.content.x + (i % 2) * (width + 10), self.y + (i // 2) * 36, width, 30),
                lambda v=name: self.set_preset(v),
                active=chosen,
                enabled=not self.worker.active,
            )
        self.y += 116
        self.field("Nome da sessão", "name", str)
        self.pair(("Passos de ambiente", "steps", int), ("Ambientes paralelos", "envs", int))
        self.pair(("Duração do episódio (s)", "episode_seconds", float), ("Seed", "seed", int))
        self.choice("Arena", "profile", list(PROFILES), preview=True)
        self.choice("Robôs azuis que aprendem", "robots_per_team", [1, 2, 3], preview=True)
        self.choice(
            "Robôs vermelhos presentes",
            "opponent_robots",
            [0, 1, 2, 3],
            {0: "Nenhum — treino solo", 1: "1", 2: "2", 3: "3"},
            preview=True,
        )
        self.choice(
            "Objetivo",
            "objective",
            ["score", "reach"],
            {"score": "Buscar o disco e marcar gol", "reach": "Chegar até o disco"},
            preview=True,
        )
        self.choice(
            "Posições iniciais",
            "scenario",
            ["faceoff", "solo", "shooting", "defending", "random", "curriculum"],
            dict(
                faceoff="Face-off",
                solo="Disco e robô em posições variadas",
                shooting="Finalização perto do gol",
                defending="Disco vindo para defesa",
                random="Times em posições variadas",
                curriculum="Currículo progressivo",
            ),
            preview=True,
        )
        opponents = (
            ["none"] if self.config.opponent_robots == 0 else ["selfplay", "mixed", "heuristic", "zero"]
        )
        self.choice(
            "Controlador adversário",
            "opponent",
            opponents,
            dict(
                none="Sem adversário",
                selfplay="Self-play · snapshots da política",
                mixed="Misto · self-play e bots",
                heuristic="Bot tático",
                zero="Parado · obstáculo físico",
            ),
        )
        self.section("Dificuldade")
        self.field("Distância inicial máxima no exercício solo (cm)", "spawn_distance")
        self.choice(
            "Orientação inicial no solo",
            "random_heading",
            [True, False],
            {True: "Aleatória", False: "Apontando para o disco"},
            preview=True,
        )
        self.choice(
            "Variações físicas e ruído da câmera",
            "domain_randomization",
            [True, False],
            {True: "Ativados", False: "Desativados"},
            preview=True,
        )
        self.button(
            "Salvar configuração",
            (self.content.x, self.y, self.content.width, 31),
            self.save_config,
            enabled=not self.worker.active,
        )
        self.y += 42

    def advanced(self):
        if self.worker.records:
            self.section("Diagnóstico da atualização")
            record = self.worker.records[-1]
            values = [
                ("KL", f"{record.get('approx_kl', 0):.4f}"),
                ("Clipping", f"{record.get('clip_fraction', 0):.1%}"),
                ("Entropia", f"{record.get('entropy', 0):.3f}"),
                ("Variância explicada", f"{record.get('explained_variance', 0):.3f}"),
            ]
            for i, (label, value) in enumerate(values):
                self.text(
                    f"{label}: {value}",
                    self.content.x + i % 2 * self.content.width / 2,
                    self.y + i // 2 * 25,
                    12,
                    GREEN,
                )
            self.y += 59
            evaluation = next(
                (r["evaluation"] for r in reversed(self.worker.records) if "evaluation" in r), None
            )
            if evaluation:
                self.text(
                    f"Avaliação reservada: {evaluation['success_rate']:.0%} de sucesso · {evaluation['episodes']} episódios",
                    self.content.x,
                    self.y,
                    12,
                    MUTED,
                    self.content.width,
                )
                self.y += 28
        self.section("Política e retomada")
        self.field("Arquivo .pt · também pode arrastar para a janela", "checkpoint", str)
        self.button(
            "Avaliar arquivo neste exercício",
            (self.content.x, self.y, self.content.width, 30),
            lambda: self.evaluate(self.config.checkpoint, self.config),
            enabled=bool(self.config.checkpoint) and Path(self.config.checkpoint).is_file(),
        )
        self.y += 40
        self.choice(
            "Inicialização",
            "init_mode",
            ["fresh", "transfer", "resume"],
            dict(
                fresh="Do zero · novos pesos",
                transfer="Transferir pesos para este exercício",
                resume="Retomar pesos e otimizador",
            ),
        )
        self.section("PPO")
        self.pair(("Learning rate", "learning_rate", float), ("Entropia", "entropy_coef", float))
        self.pair(("Gamma", "gamma", float), ("Lambda GAE", "gae_lambda", float))
        self.pair(("Clipping", "clip_eps", float), ("KL alvo", "target_kl", float))
        self.pair(("Coleta por ambiente", "rollout", int), ("Minibatch", "batch_size", int))
        self.pair(("Épocas por atualização", "epochs", int), ("Novo snapshot a cada", "snapshot_every", int))
        self.pair(
            ("Avaliar a cada (updates)", "eval_every", int), ("Episódios de avaliação", "eval_episodes", int)
        )
        self.section("Recompensas")
        self.pair(("Gol", "reward.goal", float), ("Progresso do disco", "reward.progress", float))
        self.pair(
            ("Aproximação", "reward.approach", float), ("Bônus de chegada", "reward.reach_bonus", float)
        )
        self.pair(
            ("Custo por passo", "reward.time_penalty", float),
            ("Custo de esforço", "reward.energy_penalty", float),
        )
        self.field("Sucesso ao chegar: distância entre centros (cm)", "reward.reach_distance")
        self.field("Força do bot adversário (0 a 1)", "opponent_difficulty")

    def session_action(self, row, action):
        if action == "evaluate":
            self.evaluate(row["checkpoint"], row["config"])
        elif not self.worker.active:
            if action == "config":
                self.config = replace(row["config"], init_mode="fresh", checkpoint="")
            else:
                self.config = replace(
                    row["config"],
                    init_mode="resume" if action == "resume" else "transfer",
                    checkpoint=str(row["checkpoint"]),
                )
            self.tab = "Treino"
            self.preview()
            self.notify("Configuração restaurada; clique Iniciar treino")

    def history(self):
        now = pygame.time.get_ticks()
        if now - self.last_scan > 1500 or not self.sessions:
            self.sessions = list_sessions(self.worker.runs_root)
            self.last_scan = now
        self.section("Sessões salvas")
        if not self.sessions:
            self.text("Seu primeiro treino aparecerá aqui.", self.content.x, self.y, 14, MUTED)
            self.y += 30
        for row in self.sessions:
            cfg, status = row["config"], row["status"]
            rect = pygame.Rect(self.content.x, self.y, self.content.width, 145)
            pygame.draw.rect(self.screen, (15, 26, 41), rect, border_radius=9)
            self.text(cfg.name, rect.x + 10, rect.y + 9, 16, TEXT, rect.width - 20)
            self.text(
                f"{cfg.robots_per_team}v{cfg.opponent_robots} · {STATES.get(status.get('state'), 'Salvo')}",
                rect.x + 10,
                rect.y + 36,
                12,
                GREEN,
            )
            self.text(row["path"].name, rect.x + 10, rect.y + 57, 12, MUTED)
            self.text(
                f"{status.get('session_transitions', 0):,} passos · {status.get('updates', 0)} atualizações",
                rect.x + 10,
                rect.y + 76,
                12,
                MUTED,
            )
            labels = [
                ("Configurar", "config"),
                ("Avaliar", "evaluate"),
                ("Retomar", "resume"),
                ("Transferir", "transfer"),
            ]
            bw = (rect.width - 26) / 4
            for i, (label, action) in enumerate(labels):
                self.button(
                    label,
                    (rect.x + 7 + i * (bw + 4), rect.y + 104, bw, 29),
                    lambda r=row, a=action: self.session_action(r, a),
                    enabled=(action == "evaluate" or not self.worker.active)
                    and (action == "config" or row["checkpoint"].exists()),
                )
            self.y += 155
        self.y += 8
        self.button(
            "Carregar configuração salva",
            (self.content.x, self.y, self.content.width, 32),
            lambda: self.load_file(self.root / "runs/training-config.json"),
            enabled=not self.worker.active,
        )
        self.y += 44
        self.text(
            "Arraste .json ou .pt para carregar outros arquivos.",
            self.content.x,
            self.y,
            12,
            MUTED,
            self.content.width,
        )
        self.y += 30

    def draw(self, screen, rect):
        self.worker.poll()
        self.screen = screen
        self.handlers, self.fields = [], []
        pygame.draw.rect(screen, PANEL, rect, border_radius=16)
        x, width = rect.x + 16, rect.width - 32
        self.text("BANCADA DE TREINO", x, rect.y + 17, 14, ACCENT)
        phase = STATES.get(self.worker.status.get("state"), "Pronto para configurar")
        self.text(phase, x, rect.y + 43, 18, GREEN if self.worker.active else TEXT, width)
        tabs = ["Treino", "PPO", "Sessões", "Simular"]
        for i, tab in enumerate(tabs):
            self.button(
                tab,
                (x + i * (width / 4 + 1), rect.y + 77, width / 4 - 4, 30),
                lambda v=tab: self._tab(v),
                self.tab == tab,
            )
        bottom = rect.bottom - 204
        self.content = pygame.Rect(x, rect.y + 120, width - 8, max(70, bottom - rect.y - 135))
        old_clip = screen.get_clip()
        screen.set_clip(self.content)
        self.y = self.content.y - self.scroll[self.tab]
        if self.tab == "Treino":
            self.basic()
        elif self.tab == "PPO":
            self.advanced()
        elif self.tab == "Sessões":
            self.history()
        self.max_scroll = max(0, self.y + self.scroll[self.tab] - self.content.bottom)
        screen.set_clip(old_clip)
        if self.max_scroll:
            track = pygame.Rect(rect.right - 11, self.content.y, 3, self.content.height)
            pygame.draw.rect(screen, EDGE, track)
            height = max(20, round(track.height**2 / (track.height + self.max_scroll)))
            thumb = pygame.Rect(
                track.x,
                track.y + (track.height - height) * self.scroll[self.tab] / max(1, self.max_scroll),
                3,
                height,
            )
            pygame.draw.rect(screen, MUTED, thumb)
        pygame.draw.line(screen, EDGE, (x, bottom - 7), (rect.right - 16, bottom - 7))
        state = self.worker.status
        finished = max(
            state.get("session_transitions", 0), (self.worker.frame or {}).get("session_transitions", 0)
        )
        target = state.get("target", self.config.effective_steps)
        self.text(f"{finished:,} / {target:,} passos de ambiente", x, bottom + 3, 12, MUTED)
        bar = pygame.Rect(x, bottom + 25, width, 5)
        pygame.draw.rect(screen, EDGE, bar, border_radius=2)
        pygame.draw.rect(
            screen,
            ACCENT,
            (bar.x, bar.y, round(bar.width * min(1, finished / max(1, target))), 5),
            border_radius=2,
        )
        record = self.worker.records[-1] if self.worker.records else {}
        self.text(
            f"{record.get('updates', 0)} updates · {record.get('episodes', 0)} episódios · KL {record.get('approx_kl', 0):.4f}",
            x,
            bottom + 39,
            12,
            MUTED,
            width,
        )
        gap, half = 8, (width - 8) / 2
        if self.worker.active:
            self.button(
                "Continuar treino" if self.worker.control.get("paused") else "Pausar treino",
                (x, bottom + 63, half, 31),
                self.toggle_pause,
            )
            self.button(
                "Salvar e encerrar",
                (x + half + gap, bottom + 63, half, 31),
                lambda: self.worker.send(stop=True, paused=False),
            )
        else:
            self.button("Pré-visualizar exercício", (x, bottom + 63, half, 31), self.preview)
            self.button("Iniciar treino", (x + half + gap, bottom + 63, half, 31), self.start, True)
        self.button(
            "Ver treino ao vivo",
            (x, bottom + 103, half, 28),
            self.show_live,
            self.watch_live,
            self.worker.frame is not None,
        )
        self.button(
            "Avaliar últimos pesos",
            (x + half + gap, bottom + 103, half, 28),
            self.evaluate,
            enabled=self.worker.checkpoint is not None,
        )
        self.text("Ritmo da coleta", x, bottom + 143, 12, MUTED)
        for i, (speed, label) in enumerate(((1, "1×"), (5, "5×"), (0, "Máximo"))):
            self.button(
                label,
                (x + 121 + i * (width - 121) / 3, bottom + 137, (width - 121) / 3 - 4, 28),
                lambda v=speed: self._speed(v),
                (self.worker.control.get("speed", self.config.speed) == speed),
            )
        message = (
            self.worker.status.get("error", self.message)
            if self.worker.status.get("state") == "error"
            else self.message
        )
        self.text(
            str(message).splitlines()[0],
            x,
            bottom + 177,
            12,
            (255, 133, 121) if self.error or state.get("state") == "error" else MUTED,
            width,
        )
        return self.content

    def _tab(self, value):
        self.tab = value

    def _speed(self, value):
        if self.worker.active:
            self.worker.send(speed=value)
        else:
            self.config = replace(self.config, speed=value)

    def draw_metrics(self, screen, rect):
        self.screen = screen
        pygame.draw.rect(screen, PANEL, rect, border_radius=12)
        records = list(self.worker.records)
        width = (rect.width - 45) / 2
        for i, (key, title, color) in enumerate(
            (
                ("mean_episode_reward", "RETORNO NA COLETA · últimos 20 episódios", ACCENT),
                ("success_rate", "SUCESSO NA COLETA · últimos 20 episódios", GREEN),
            )
        ):
            x = rect.x + 16 + i * (width + 15)
            self.text(title, x, rect.y + 10, 12, MUTED)
            points = [
                (r["session_transitions"], r[key])
                for r in records
                if r.get(key) is not None and math.isfinite(r[key])
            ]
            if not points:
                self.text("Aguardando episódios completos", x, rect.y + 48, 12, MUTED, width)
                continue
            values = [p[1] for p in points]
            lo, hi = (0, 1) if key == "success_rate" else (min(values) - 0.5, max(values) + 0.5)
            plot = pygame.Rect(x, rect.y + 43, width - 60, rect.height - 58)
            pygame.draw.line(screen, EDGE, plot.bottomleft, plot.bottomright)
            start, end = points[0][0], points[-1][0]
            pixels = [
                (
                    plot.x + (step - start) / max(1, end - start) * plot.width,
                    plot.bottom - (value - lo) / max(1e-6, hi - lo) * plot.height,
                )
                for step, value in points
            ]
            if len(pixels) > 1:
                pygame.draw.lines(screen, color, False, pixels, 2)
            pygame.draw.circle(screen, color, pixels[-1], 3)
            self.text(
                f"{values[-1]:.0%}" if key == "success_rate" else f"{values[-1]:.2f}",
                x + width - 52,
                rect.y + 43,
                16,
                color,
            )
