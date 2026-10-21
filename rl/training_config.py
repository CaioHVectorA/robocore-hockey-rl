"""Serializable experiments shared by the desktop interface and training CLI."""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import json
from core.constants import PROFILES
from rl.tasks import RewardConfig
from rl.env import SCENARIOS

CURRICULA = ("curriculum", "solo_curriculum")


@dataclass(frozen=True)
class TrainingConfig:
    name: str = "3v3 self-play"
    profile: str = "micro-2024"
    robots_per_team: int = 3
    opponent_robots: int = 3
    objective: str = "score"
    scenario: str = "faceoff"
    opponent: str = "selfplay"
    steps: int = 100000
    envs: int = 4
    rollout: int = 256
    episode_seconds: float = 30.0
    seed: int = 42
    learning_rate: float = 0.0003
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    target_kl: float = 0.02
    entropy_coef: float = 0.002
    epochs: int = 6
    batch_size: int = 256
    snapshot_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 3
    opponent_difficulty: float = 0.85
    domain_randomization: bool = True
    spawn_distance: float = 60.0
    random_heading: bool = True
    speed: float = 0.0  # 0 = unlimited; otherwise simulated seconds / wall second.
    init_mode: str = "fresh"
    checkpoint: str = ""
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum_success_threshold: float = 0.7
    curriculum_min_episodes: int = 30
    curriculum_window: int = 50
    selfplay_pool_size: int = 6
    selfplay_anchor_probability: float = 0.2
    evaluation_suite: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 120:
            raise ValueError("Nome da sessão: de 1 a 120 caracteres")
        if any(
            type(value) is not bool
            for value in (self.domain_randomization, self.random_heading, self.evaluation_suite)
        ):
            raise ValueError("Variações físicas e orientação aleatória devem ser booleanas")
        if self.profile not in PROFILES or self.objective not in ("score", "reach"):
            raise ValueError("Arena ou objetivo inválido")
        if self.scenario not in (*SCENARIOS, *CURRICULA):
            raise ValueError("Cenário inválido")
        if self.scenario == "duel" and (self.robots_per_team != 1 or self.opponent_robots != 1):
            raise ValueError("Cenário 1v1 exige exatamente um robô de cada time")
        if self.evaluation_suite and self.opponent_robots == 0:
            raise ValueError("Suíte de adversários exige robôs vermelhos")
        if self.scenario == "solo_curriculum" and (
            self.robots_per_team != 1
            or self.opponent_robots != 0
            or self.objective != "score"
            or self.opponent != "none"
        ):
            raise ValueError("Currículo solo exige um robô azul, nenhum adversário e objetivo de marcar gol")
        for key in (
            "steps",
            "envs",
            "rollout",
            "epochs",
            "batch_size",
            "snapshot_every",
            "eval_every",
            "eval_episodes",
            "curriculum_min_episodes",
            "curriculum_window",
            "selfplay_pool_size",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key}: informe um inteiro positivo")
        if type(self.robots_per_team) is not int or not 1 <= self.robots_per_team <= 3:
            raise ValueError("Escolha de 1 a 3 robôs azuis")
        if type(self.opponent_robots) is not int or not 0 <= self.opponent_robots <= 3:
            raise ValueError("Escolha de 0 a 3 robôs vermelhos")
        if self.opponent not in ("selfplay", "mixed", "heuristic", "zero", "none"):
            raise ValueError("Adversário inválido")
        if (self.opponent_robots == 0) != (self.opponent == "none"):
            raise ValueError("Sem adversário exige zero robôs vermelhos")
        if self.envs > 32 or self.rollout > 8192:
            raise ValueError("Máximo: 32 ambientes e coleta de 8192 passos")
        ranges = dict(
            episode_seconds=(1, 600),
            learning_rate=(1e-7, 0.01),
            gamma=(0.8, 1),
            gae_lambda=(0, 1),
            clip_eps=(0.01, 0.5),
            target_kl=(0.001, 0.5),
            entropy_coef=(0, 0.2),
            opponent_difficulty=(0, 1),
            spawn_distance=(15, 150),
            speed=(0, 1000),
            curriculum_success_threshold=(0.01, 1),
            selfplay_anchor_probability=(0, 1),
        )
        for key, (lo, hi) in ranges.items():
            value = getattr(self, key)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key}: intervalo permitido {lo:g}–{hi:g}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed deve ser um inteiro entre 0 e 4294967295")
        if self.init_mode not in ("fresh", "resume", "transfer"):
            raise ValueError("Modo inicial inválido")
        if self.init_mode != "fresh" and not self.checkpoint:
            raise ValueError("Selecione o arquivo de pesos para retomar ou transferir")
        if not isinstance(self.reward, RewardConfig):
            raise ValueError("Configuração de recompensas inválida")
        if not self.curriculum_min_episodes <= self.curriculum_window <= 100:
            raise ValueError("Currículo: episódios mínimos <= janela <= 100")
        if not 2 <= self.selfplay_pool_size <= 32:
            raise ValueError("Self-play: pool deve ter de 2 a 32 adversários")

    @property
    def episode_steps(self):
        return round(self.episode_seconds * 30)

    @property
    def effective_steps(self):
        return math.ceil(self.steps / self.envs) * self.envs

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if "reward" in data:
            data["reward"] = RewardConfig(**data["reward"])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf8")))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf8")

    def make_env(self, scenario=None):
        from rl.env import RoboCoreHockeyEnv

        return RoboCoreHockeyEnv(
            robots_per_team=self.robots_per_team,
            opponent_robots=self.opponent_robots,
            profile=self.profile,
            max_steps=self.episode_steps,
            scenario=scenario
            or (
                "shooting"
                if self.scenario == "curriculum"
                else "solo_finish"
                if self.scenario == "solo_curriculum"
                else self.scenario
            ),
            objective=self.objective,
            reward_config=self.reward,
            shaping_gamma=self.gamma,
            domain_randomization=self.domain_randomization,
            spawn_distance=self.spawn_distance,
            random_heading=self.random_heading,
            opponent_difficulty=self.opponent_difficulty,
        )


PRESET_LABELS = {
    "selfplay": "3v3 · self-play",
    "solo_goal": "Solo · buscar e marcar",
    "solo_curriculum": "Solo · progressivo até o gol",
    "solo_reach": "Solo · ir ao disco",
    "shooting": "Solo · finalização",
    "defending": "1v1 · defesa",
    "tactical": "3v3 · bot tático",
    "duel_selfplay": "1v1 · self-play",
}


def preset_config(name, base=None):
    base = base or TrainingConfig()
    common = dict(
        init_mode="fresh",
        checkpoint="",
        objective="score",
        reward=RewardConfig(),
        random_heading=True,
        evaluation_suite=False,
    )
    options = {
        "selfplay": dict(robots_per_team=3, opponent_robots=3, opponent="selfplay", scenario="faceoff"),
        "solo_goal": dict(robots_per_team=1, opponent_robots=0, opponent="none", scenario="solo"),
        "duel_selfplay": dict(
            robots_per_team=1,
            opponent_robots=1,
            opponent="selfplay",
            scenario="duel",
            profile="micro-practice",
            episode_seconds=60.0,
            envs=8,
            batch_size=512,
            learning_rate=0.0002,
            gamma=0.997,
            entropy_coef=0.005,
            snapshot_every=10,
            selfplay_pool_size=12,
            eval_every=100,
            eval_episodes=6,
            evaluation_suite=True,
            steps=1000000,
        ),
        "solo_curriculum": dict(
            robots_per_team=1,
            opponent_robots=0,
            opponent="none",
            scenario="solo_curriculum",
            profile="micro-practice",
            steps=1000000,
            envs=8,
            batch_size=512,
        ),
        "solo_reach": dict(
            robots_per_team=1,
            opponent_robots=0,
            opponent="none",
            scenario="solo",
            objective="reach",
            reward=RewardConfig(progress=0, approach=2),
        ),
        "shooting": dict(robots_per_team=1, opponent_robots=0, opponent="none", scenario="shooting"),
        "defending": dict(robots_per_team=1, opponent_robots=1, opponent="heuristic", scenario="defending"),
        "tactical": dict(robots_per_team=3, opponent_robots=3, opponent="heuristic", scenario="faceoff"),
    }
    if name not in options:
        raise ValueError("Preset desconhecido")
    common.update(options[name])
    return replace(base, name=PRESET_LABELS[name], **common)
