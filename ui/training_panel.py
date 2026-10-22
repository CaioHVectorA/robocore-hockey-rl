"""Desktop experiment editor, live controls, metrics and saved-session browser."""

from dataclasses import replace
from pathlib import Path
import math
import pygame
from core.constants import PROFILES
from rl.training_config import TrainingConfig, PRESET_LABELS, preset_config
from ui.training_session import TrainingProcess, list_sessions
from ui.text_rendering import fit_text, render_text, wrap_text

BG, PANEL, EDGE = (10, 16, 27), (20, 30, 47), (43, 58, 79)
TEXT, MUTED, ACCENT, GREEN = (236, 242, 250), (151, 170, 193), (83, 218, 203), (126, 222, 164)
TABS = ("Treino", "Recompensas", "PPO", "Sessões", "Simular")
STATES = dict(
    starting="Iniciando",
    running="Treinando",
    updating="Atualizando a rede PPO",
    paused="Pausado",
    evaluating="Avaliação reservada",
    saving="Salvando",
    completed="Concluído",
    stopped="Encerrado e salvo",
    error="Falha no treino",
)
PHASE_HELP = dict(
    starting="Carregando os ambientes e os pesos iniciais…",
    running="Coletando ações exploratórias. Use 1× para observar o movimento.",
    updating="A coleta aguarda a atualização da rede. O campo fica parado nesta etapa.",
    evaluating="Testando pesos em episódios reservados. A coleta aguarda a avaliação.",
    saving="Gravando os pesos. A coleta continua após o salvamento.",
    paused="Treino pausado por você. Continuar preserva o estado do ambiente.",
    completed="Treino concluído. Avalie os pesos para observar a política aprendida.",
    stopped="Sessão encerrada e pesos salvos. Você já pode editar os parâmetros.",
    error="O processo encerrou com erro. Abra o diagnóstico para ver os detalhes.",
)

REWARD_ROWS = (
    ("goal", "Gol marcado / sofrido", "+ gol a favor; − gol contra.", "all"),
    ("progress", "Disco em direção ao gol", "Peso do progresso do disco até o gol da direita.", "score"),
    ("approach", "Aproximar do disco", "Peso da redução da distância robô–disco.", "all"),
    ("time_penalty", "Custo por passo", "Valor subtraído a cada passo de controle (30 Hz).", "all"),
    ("energy_penalty", "Custo dos motores", "Subtrai peso × média das ações ao quadrado.", "all"),
    ("reach_bonus", "Bônus de chegada", "Bônus ao atingir a distância de chegada.", "reach"),
    (
        "reach_distance",
        "Distância de chegada · cm",
        "Distância entre centros para concluir a chegada.",
        "reach",
    ),
    (
        "alignment",
        "Alinhar atrás do disco",
        "Potencial de proximidade, posição e orientação para o gol.",
        "score",
    ),
    ("first_touch", "Primeiro contato", "Bônus único por episódio; não repete por ficar encostado.", "all"),
    ("carry", "Condução com participação", "Novo progresso máximo após toque recente do azul.", "score"),
    (
        "idle_penalty",
        "Inatividade longe do disco",
        "Custo por passo após 3 s abaixo de 2 cm/s, a mais de 20 cm.",
        "all",
    ),
)


class TrainingPanel:
    def __init__(self, root, fonts, preview, evaluate, config=None):
        self.root, self.fonts = Path(root), fonts
        self.preview_callback, self.evaluate_callback = preview, evaluate
        self.config = config or TrainingConfig()
        self.worker = TrainingProcess(root)
        self.tab = "Treino"
        self.scroll = {tab: 0 for tab in TABS}
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
        self.details = False
        self.detail_scroll = 0
        self.detail_lines = 0
        self.detail_close = pygame.Rect(0, 0, 1, 1)
        self.scroll_track = pygame.Rect(0, 0, 0, 0)
        self.scroll_thumb = pygame.Rect(0, 0, 0, 0)
        self.scroll_drag = None

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
            self.notify("Coleta real: até 30 quadros/s; em 1×, as poses são suavizadas na exibição")

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
        self.control(paused=not self.worker.control.get("paused", False))

    def control(self, **changes):
        try:
            self.worker.send(**changes)
        except OSError as exc:
            self.notify(exc, True)

    def save_config(self):
        if not self.commit():
            return
        path = self.root / "runs/training-config.json"
        try:
            self.config.save(path)
            self.notify("Configuração salva em runs/training-config.json")
        except OSError as exc:
            self.notify(exc, True)

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
            value = parser(self.edit.replace(",", ".") if parser is float else self.edit)
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
        if self.details:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.details = False
            elif event.type == pygame.MOUSEBUTTONDOWN and self.detail_close.collidepoint(event.pos):
                self.details = False
            elif event.type == pygame.MOUSEWHEEL:
                self.detail_scroll = max(0, min(self.detail_lines, self.detail_scroll - event.y * 3))
            return event.type != pygame.QUIT
        if event.type == pygame.DROPFILE:
            self.load_file(event.file)
            return True
        if event.type == pygame.MOUSEBUTTONUP and self.scroll_drag is not None:
            self.scroll_drag = None
            return True
        if event.type == pygame.MOUSEMOTION and self.scroll_drag is not None:
            self.drag_scroll(event.pos[1])
            return True
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.scroll_track.collidepoint(event.pos)
        ):
            self.scroll_drag = (
                event.pos[1] - self.scroll_thumb.y
                if self.scroll_thumb.collidepoint(event.pos)
                else self.scroll_thumb.height / 2
            )
            self.drag_scroll(event.pos[1])
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
        if event.type == pygame.KEYDOWN and self.focus is None:
            if event.key in (pygame.K_PAGEDOWN, pygame.K_PAGEUP):
                delta = self.content.height * (1 if event.key == pygame.K_PAGEDOWN else -1)
                self.scroll[self.tab] = max(0, min(self.max_scroll, self.scroll[self.tab] + delta))
                return True
        if event.type == pygame.KEYDOWN and self.focus is not None:
            if event.key == pygame.K_ESCAPE:
                self.focus = None
            elif event.key == pygame.K_TAB:
                keys = [key for _, key, _ in self.fields]
                index = keys.index(self.focus[0]) if self.focus[0] in keys else -1
                if self.commit() and keys:
                    key = keys[(index + (-1 if event.mod & pygame.KMOD_SHIFT else 1)) % len(keys)]
                    parser = next(parser for _, name, parser in self.fields if name == key)
                    self.focus = (key, parser)
                    value = self.value(key)
                    self.edit, self.caret, self.select_all = str(value), len(str(value)), True
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
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
        clip = self.screen.get_clip()
        if y >= clip.bottom or y + self.fonts[size].get_linesize() <= clip.top:
            return
        value = fit_text(self.fonts[size], str(value), width)
        self.screen.blit(render_text(self.fonts[size], value, color), (int(x), int(y)))

    def button(self, label, rect, callback, active=False, enabled=True):
        rect = pygame.Rect(rect)
        visible = self.screen.get_clip().clip(rect)
        if not visible.width or not visible.height:
            return
        hover = enabled and rect.collidepoint(pygame.mouse.get_pos())
        color = ACCENT if active else (49, 70, 91) if hover else (33, 48, 68) if enabled else (25, 35, 50)
        pygame.draw.rect(self.screen, color, rect, border_radius=7)
        if not active:
            pygame.draw.rect(self.screen, EDGE, rect, 1, border_radius=7)
        label = fit_text(self.fonts[12], label, rect.width - 10)
        surf = render_text(self.fonts[12], label, BG if active else TEXT if enabled else MUTED)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        visible = self.screen.get_clip().clip(rect)
        if enabled and visible.width and visible.height:
            self.handlers.append((visible, callback))

    def section(self, label):
        self.y += 12
        self.text(label.upper(), self.content.x, self.y, 12, ACCENT)
        self.y += 25

    def paragraph(self, value, color=MUTED, width=None):
        width = width or self.content.width
        for line in wrap_text(self.fonts[12], str(value), width):
            self.text(line, self.content.x, self.y, 12, color)
            self.y += 19
        self.y += 5

    def value(self, key):
        return (
            getattr(self.config.reward, key.split(".")[1])
            if key.startswith("reward.")
            else getattr(self.config, key)
        )

    def field(self, label, key, parser=float, x=None, width=None, advance=True):
        x = self.content.x if x is None else x
        width = self.content.width if width is None else width
        self.text(label, x, self.y, 12, MUTED, width)
        rect = pygame.Rect(x, self.y + 20, width, 31)
        self.input(key, rect, parser)
        if advance:
            self.y += 61

    def input(self, key, rect, parser=float):
        if not self.screen.get_clip().colliderect(rect):
            return
        focused = self.focus is not None and self.focus[0] == key
        pygame.draw.rect(self.screen, (12, 23, 37), rect, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT if focused else EDGE, rect, 1, border_radius=6)
        value = self.value(key)
        display = self.edit if focused else f"{value:g}" if isinstance(value, float) else str(value)
        if focused and self.select_all:
            pygame.draw.rect(self.screen, (53, 78, 98), rect.inflate(-8, -6), border_radius=3)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(old_clip.clip(rect.inflate(-10, -2)))
        offset = max(0, self.fonts[14].size(display[: self.caret])[0] - rect.width + 24) if focused else 0
        self.text(display, rect.x + 9 - offset, rect.y + 6, 14, TEXT if not self.worker.active else MUTED)
        if focused and not self.select_all and pygame.time.get_ticks() % 1000 < 550:
            cx = rect.x + 9 - offset + self.fonts[14].size(display[: self.caret])[0]
            pygame.draw.line(self.screen, ACCENT, (cx, rect.y + 6), (cx, rect.bottom - 6))
        self.screen.set_clip(old_clip)
        visible = self.screen.get_clip().clip(rect)
        if not self.worker.active and visible.height and visible.width:
            self.fields.append((visible, key, parser))

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
        self.y += math.ceil(len(PRESET_LABELS) / 2) * 36 + 8
        self.section("Seu objetivo")
        self.paragraph(
            "Buscar o disco e marcar no gol da direita. Aproximar sozinho não conclui o episódio."
            if self.config.objective == "score"
            else "Chegar perto do disco, dentro da distância configurada. Não exige gol.",
            TEXT,
        )
        self.text(
            f"{self.config.robots_per_team} azul · {self.config.opponent_robots} vermelho · {self.config.episode_seconds:g}s por episódio",
            self.content.x,
            self.y,
            12,
            GREEN,
        )
        self.y += 27
        self.paragraph(
            "Recompensas e PPO têm abas próprias. Ajuste antes de iniciar; Enter aplica, Tab avança."
        )
        if self.config.scenario == "solo_curriculum":
            self.paragraph(
                "Currículo solo: finalizar perto do gol → conduzir → buscar de qualquer direção e marcar. Só avança após demonstrar sucesso; os pesos de recompensa permanecem iguais.",
                GREEN,
            )
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
            [
                "faceoff",
                "solo",
                "solo_finish",
                "solo_carry",
                "solo_search",
                "duel",
                "shooting",
                "defending",
                "random",
                "curriculum",
                "solo_curriculum",
            ],
            dict(
                faceoff="Face-off",
                solo="Disco e robô em posições variadas",
                shooting="Finalização perto do gol",
                defending="Disco vindo para defesa",
                random="Times em posições variadas",
                curriculum="Currículo progressivo",
                solo_curriculum="Solo · currículo por sucesso",
                solo_finish="Solo · finalizar perto do gol",
                solo_carry="Solo · conduzir até o gol",
                solo_search="Solo · buscar em todas as direções",
                duel="1v1 · saídas e situações variadas",
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
        if self.config.scenario == "solo_curriculum":
            self.field("Taxa de sucesso para avançar (0 a 1)", "curriculum_success_threshold")
            self.pair(
                ("Mínimo de episódios na etapa", "curriculum_min_episodes", int),
                ("Janela de episódios", "curriculum_window", int),
            )
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

    def rewards(self):
        self.section("Tabela de recompensas")
        self.paragraph(
            "Clique no valor para editar. Enter aplica; vírgula ou ponto são aceitos. Os custos são subtraídos automaticamente."
        )
        self.paragraph(
            "Os pesos ficam fixos na sessão. A recompensa calculada varia a cada passo; o acumulado aparece abaixo de cada componente durante a coleta."
        )
        self.paragraph(
            "Objetivo: buscar e marcar. O bônus e a distância de chegada não encerram este exercício."
            if self.config.objective == "score"
            else "Objetivo: ir ao disco. O peso de progresso até o gol não é usado neste exercício.",
            GREEN,
        )
        self.text("COMPONENTE / EFEITO", self.content.x + 12, self.y, 12, MUTED)
        self.text("PESO", self.content.right - 108, self.y, 12, MUTED)
        self.y += 25
        for index, (key, label, description, task) in enumerate(REWARD_ROWS):
            row = pygame.Rect(self.content.x, self.y, self.content.width, 86)
            pygame.draw.rect(
                self.screen, (25, 39, 58) if index % 2 == 0 else (17, 28, 44), row, border_radius=8
            )
            relevant = task == "all" or task == self.config.objective
            color = TEXT if relevant else MUTED
            self.text(label, row.x + 12, row.y + 10, 14, color, row.width - 144)
            self.text(description, row.x + 12, row.y + 39, 12, MUTED, row.width - 24)
            note = (
                "Ativo neste exercício"
                if relevant
                else "Só no objetivo: " + ("marcar gol" if task == "score" else "ir ao disco")
            )
            if key != "reach_distance" and self.live and self.worker.config == self.config:
                total = self.worker.frame.get("episode_reward_components", {}).get(key)
                if total is not None:
                    note += f" · neste episódio: {total:+.4f}"
            self.text(note, row.x + 12, row.y + 60, 12, GREEN if relevant else MUTED)
            self.input("reward." + key, pygame.Rect(row.right - 119, row.y + 7, 107, 31))
            self.y += 93
        self.section("Como interpretar")
        self.paragraph(
            "Progresso e aproximação usam diferenças de potencial com o gamma do PPO; estes pesos não são bônus fixos por contato. As penalidades são aplicadas por passo, e o esforço varia com a ação."
        )
        self.paragraph(
            f"Em {self.config.episode_seconds:g}s, o custo de tempo soma até {self.config.reward.time_penalty * self.config.episode_steps:g} por robô. A taxa de controle permanece em 30 Hz."
        )
        half = (self.content.width - 8) / 2
        self.button(
            "Restaurar pesos padrão",
            (self.content.x, self.y, half, 32),
            self.reset_rewards,
            enabled=not self.worker.active,
        )
        self.button(
            "Salvar configuração",
            (self.content.x + half + 8, self.y, half, 32),
            self.save_config,
            enabled=not self.worker.active,
        )
        self.y += 44

    def reset_rewards(self):
        preset = "solo_reach" if self.config.objective == "reach" else "solo_goal"
        if self.config.scenario == "solo_curriculum" and self.config.curriculum_eval_gate:
            preset = "solo_foundation"
        self._apply(reward=preset_config(preset).reward)

    def advanced(self):
        self.section("Hiperparâmetros PPO")
        self.paragraph(
            "Os valores abaixo são os usados ao iniciar a sessão. Durante o treino, ficam disponíveis para consulta."
        )
        self.pair(
            ("Learning rate · taxa de aprendizado", "learning_rate", float),
            ("Entropia · exploração", "entropy_coef", float),
        )
        self.pair(
            ("Gamma · desconto futuro", "gamma", float), ("Lambda · estimativa GAE", "gae_lambda", float)
        )
        self.pair(
            ("Clipping · limite PPO", "clip_eps", float), ("KL alvo · limite de mudança", "target_kl", float)
        )
        self.section("Coleta e otimização")
        self.pair(
            ("Passos por ambiente / atualização", "rollout", int),
            ("Amostras por minibatch", "batch_size", int),
        )
        self.pair(
            ("Épocas por atualização", "epochs", int), ("Snapshot a cada (updates)", "snapshot_every", int)
        )
        samples = self.config.rollout * self.config.envs * self.config.robots_per_team
        self.paragraph(
            f"Cada lote completo: {self.config.rollout} passos × {self.config.envs} ambientes × {self.config.robots_per_team} robôs = {samples:,} amostras. A coleta pausa enquanto a rede é atualizada."
        )
        self.section("Avaliação reservada")
        self.pair(
            ("Avaliar a cada (updates)", "eval_every", int), ("Episódios por avaliação", "eval_episodes", int)
        )
        self.paragraph(
            "Usa ações determinísticas e seeds reservadas. Durante essa etapa, o campo da coleta permanece parado; o estado aparece abaixo da arena."
        )
        self.field("Parar após episódios sem sucesso ativo (0 desativa)", "stagnation_patience", int)
        if self.config.scenario == "solo_curriculum":
            self.choice(
                "Exigir avaliação antes de avançar etapa",
                "curriculum_eval_gate",
                [True, False],
                {True: "Coleta e seeds reservadas", False: "Somente sucesso da coleta"},
            )
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
        self.field("Força do bot adversário (0 a 1)", "opponent_difficulty")
        if self.config.opponent_robots:
            self.section("Adversários e avaliação")
            self.pair(
                ("Versões no pool self-play", "selfplay_pool_size", int),
                ("Chance da política inicial (0 a 1)", "selfplay_anchor_probability", float),
            )
            self.choice(
                "Avaliar contra bot e política inicial",
                "evaluation_suite",
                [True, False],
                {True: "Suíte fixa · saídas e posições variadas", False: "Somente bot tático"},
            )

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
        tabs = TABS
        for i, tab in enumerate(tabs):
            self.button(
                tab,
                (x + i * width / len(tabs), rect.y + 77, width / len(tabs) - 4, 30),
                lambda v=tab: self._tab(v),
                self.tab == tab,
            )
        bottom = rect.bottom - 204
        self.text(
            "Em execução · encerre para editar"
            if self.worker.active
            else "Clique para editar · Enter aplica · role para mais",
            x,
            rect.y + 116,
            12,
            MUTED,
            width,
        )
        self.content = pygame.Rect(x, rect.y + 142, width - 8, max(70, bottom - rect.y - 155))
        old_clip = screen.get_clip()
        screen.set_clip(self.content)
        self.y = self.content.y - self.scroll[self.tab]
        if self.tab == "Treino":
            self.basic()
        elif self.tab == "PPO":
            self.advanced()
        elif self.tab == "Recompensas":
            self.rewards()
        elif self.tab == "Sessões":
            self.history()
        self.max_scroll = max(0, self.y + self.scroll[self.tab] - self.content.bottom)
        screen.set_clip(old_clip)
        self.scroll_track = pygame.Rect(0, 0, 0, 0)
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
            self.scroll_track = track.inflate(12, 0)
            self.scroll_thumb = thumb
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
                lambda: self.control(stop=True, paused=False),
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
        if self.error or state.get("state") == "error":
            self.button("Erro · abrir diagnóstico completo", (x, bottom + 171, width, 26), self.open_details)
        else:
            self.text(self.message, x, bottom + 177, 12, MUTED, width)
        return self.content

    def _tab(self, value):
        self.tab = value
        self.scroll_drag = None

    def drag_scroll(self, y):
        travel = self.scroll_track.height - self.scroll_thumb.height
        fraction = (y - self.scroll_drag - self.scroll_track.y) / max(1, travel)
        self.scroll[self.tab] = max(0, min(self.max_scroll, fraction * self.max_scroll))

    def _speed(self, value):
        if self.worker.active:
            self.control(speed=value)
        else:
            self.config = replace(self.config, speed=value)

    def open_details(self):
        self.details, self.detail_scroll = True, 0

    def draw_details(self, screen):
        if not self.details:
            return
        self.screen = screen
        shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 190))
        screen.blit(shade, (0, 0))
        rect = pygame.Rect(0, 0, min(1050, screen.get_width() - 80), min(650, screen.get_height() - 80))
        rect.center = screen.get_rect().center
        pygame.draw.rect(screen, PANEL, rect, border_radius=18)
        pygame.draw.rect(screen, EDGE, rect, 1, border_radius=18)
        self.text("Diagnóstico da sessão", rect.x + 24, rect.y + 22, 22)
        self.text(
            "Role para ler · os detalhes também ficam em output.log e status.json",
            rect.x + 24,
            rect.y + 58,
            12,
            MUTED,
        )
        self.detail_close = pygame.Rect(rect.right - 104, rect.y + 20, 80, 32)
        self.button("Fechar [Esc]", self.detail_close, lambda: None)
        state = self.worker.status
        body = "\n\n".join(
            str(part)
            for part in (
                state.get("error", self.message),
                state.get("detail", ""),
                f"Pasta da sessão: {self.worker.directory}" if self.worker.directory else "",
                f"Último checkpoint: {self.worker.checkpoint}"
                if self.worker.checkpoint
                else "Ainda não há checkpoint salvo.",
                "Você pode usar Sessões → Retomar para continuar a partir dos últimos pesos salvos.",
            )
            if part
        )
        viewport = pygame.Rect(rect.x + 24, rect.y + 93, rect.width - 48, rect.height - 117)
        lines = []
        for original in body.splitlines():
            line = ""
            for char in original:
                if self.fonts[14].size(line + char)[0] > viewport.width - 12:
                    lines.append(line)
                    line = ""
                line += char
            lines.append(line)
        visible = viewport.height // 22
        self.detail_lines = max(0, len(lines) - visible)
        self.detail_scroll = min(self.detail_scroll, self.detail_lines)
        screen.set_clip(viewport)
        for i, line in enumerate(lines[self.detail_scroll : self.detail_scroll + visible]):
            self.text(line, viewport.x, viewport.y + i * 22, 14, TEXT)
        screen.set_clip(None)

    def draw_health(self, screen, rect, frame, fps):
        self.screen = screen
        pygame.draw.rect(screen, PANEL, rect, border_radius=12)
        live = self.watch_live
        waiting = live and self.worker.status.get("state") != "running"
        stale = live and self.worker.active and not waiting and self.worker.frame_age > 1
        speed = (frame.get("robot_speeds") or [0])[0]
        cards = (
            ("JANELA", f"{fps:.0f} FPS", "desenho da interface"),
            (
                "QUADROS DO TREINO",
                "em espera" if waiting else f"{self.worker.frame_hz:.0f} Hz" if live else "prévia",
                "sem novos quadros" if stale else "até 30 Hz + suavização",
            ),
            (
                "PASSOS DE TREINO",
                f"{frame.get('transitions_per_second', 0):,.0f}/s" if live else "—",
                "média · todos os ambientes",
            ),
            ("ROBÔ AZUL 0", f"{speed:.1f} cm/s", f"disco a {frame.get('distance_to_puck', 0):.1f} cm"),
        )
        width = (rect.width - 24) / 4
        for i, (label, value, note) in enumerate(cards):
            x = rect.x + 12 + i * width
            if i:
                pygame.draw.line(screen, EDGE, (x - 6, rect.y + 12), (x - 6, rect.y + 65))
            self.text(label, x, rect.y + 9, 12, MUTED, width - 12)
            self.text(value, x, rect.y + 28, 18, ACCENT if i < 3 else GREEN, width - 12)
            self.text(note, x, rect.y + 52, 12, MUTED, width - 12)
        phase = self.worker.status.get("state")
        help_text = (
            PHASE_HELP.get(phase, "Prévia com bot tático. Inicie o treino para acompanhar o PPO.")
            if live
            else "Prévia / avaliação independente. Os controles de treino ficam no painel à direita."
        )
        if stale:
            help_text = "Aguardando novos quadros há mais de 1s. Confira o estado da sessão e o diagnóstico."
        curriculum = self.worker.status.get("curriculum")
        if self.worker.status.get("stop_reason") == "stagnation":
            help_text = "Treino interrompido por estagnação: episódios sem sucesso ativo. Pesos salvos; revise o exercício."
        if live and phase == "running" and curriculum:
            help_text = f"{curriculum['label']} · sucesso {curriculum['success_rate']:.0%} / meta {curriculum['threshold']:.0%} · {curriculum['recent_count']} episódios (mín. {curriculum['minimum']})"
        self.text(help_text, rect.x + 12, rect.y + 78, 12, MUTED, rect.width - 24)

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
