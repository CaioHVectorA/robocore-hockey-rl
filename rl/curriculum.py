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

    @property
    def stage(self):
        return SOLO_STAGES[self.index]

    def update(self, episodes):
        recent = [r for r in episodes if r["stage"] == self.stage][-self.window :]
        self.recent_count = len(recent)
        self.success_rate = sum(r["success"] for r in recent) / max(1, self.recent_count)
        if (
            self.index < len(SOLO_STAGES) - 1
            and self.recent_count >= self.minimum
            and self.success_rate >= self.threshold
        ):
            self.index += 1
            self.recent_count, self.success_rate = 0, 0.0
            return True
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
        )

    def restore(self, data):
        index = data.get("index", 0)
        if type(index) is not int or not 0 <= index < len(SOLO_STAGES):
            raise ValueError("Invalid solo curriculum checkpoint stage")
        self.index = index
