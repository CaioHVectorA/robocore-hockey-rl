"""Solo goal curriculum advances on observed success, never on elapsed budget."""

SOLO_STAGES = ("solo_finish", "solo_carry", "solo_search")
STAGE_LABELS = {
    "solo_finish": "1/3 · finalizar perto do gol",
    "solo_carry": "2/3 · conduzir até o gol",
    "solo_search": "3/3 · localizar, aproximar e marcar",
}


class SoloCurriculum:
    def __init__(self, config):
        self.index = 0
        self.threshold = config.curriculum_success_threshold
        self.minimum = config.curriculum_min_episodes
        self.window = config.curriculum_window
        self.recent_count = 0
        self.success_rate = 0.0
        self.eval_gate = config.curriculum_eval_gate
        self.eval_episodes = config.eval_episodes
        self.validation = None

    @property
    def stage(self):
        return SOLO_STAGES[self.index]

    def update(self, episodes):
        recent = [r for r in episodes if r["stage"] == self.stage][-self.window :]
        self.recent_count = len(recent)
        self.success_rate = sum(r.get("active_success", r["success"]) for r in recent) / max(
            1, self.recent_count
        )
        if self.ready and not self.eval_gate:
            return self._promote()
        return False

    @property
    def ready(self):
        return (
            self.index < len(SOLO_STAGES) - 1
            and self.recent_count >= self.minimum
            and self.success_rate >= self.threshold
        )

    def _promote(self):
        self.index += 1
        self.recent_count, self.success_rate = 0, 0.0
        self.validation = None
        return True

    def validate(self, stage, result):
        if stage != self.stage:
            return False
        self.validation = dict(stage=stage, **result)
        if (
            self.ready
            and result.get("complete", False)
            and result["episodes"] >= self.eval_episodes
            and result.get("active_success_rate", 0) >= self.threshold
        ):
            return self._promote()
        return False

    def state(self):
        return dict(
            index=self.index,
            stage=self.stage,
            label=STAGE_LABELS[self.stage],
            threshold=self.threshold,
            minimum=self.minimum,
            window=self.window,
            recent_count=self.recent_count,
            success_rate=self.success_rate,
            eval_gate=self.eval_gate,
            validation=self.validation,
        )

    def restore(self, data):
        index = data.get("index", 0)
        if type(index) is not int or not 0 <= index < len(SOLO_STAGES):
            raise ValueError("Invalid solo curriculum checkpoint stage")
        self.index = index


class ProgressGuard:
    """Stop unproductive episodes within each task; never roll back saved weights."""

    def __init__(self, patience):
        self.patience = patience
        self.counts = {}

    def record(self, episode):
        stage = episode["stage"]
        self.counts[stage] = (
            0 if episode.get("active_success", episode["success"]) else self.counts.get(stage, 0) + 1
        )

    def exhausted(self, stage):
        return self.patience > 0 and self.counts.get(stage, 0) >= self.patience

    def state(self):
        return dict(patience=self.patience, episodes_without_active_success=dict(self.counts))

    def restore(self, data):
        self.counts = dict(data.get("episodes_without_active_success", {}))
