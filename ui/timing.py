"""Frame pacing without SDL's coarse millisecond sleep on Windows."""

from collections import deque
import time


class FrameClock:
    def __init__(self):
        self.last = time.perf_counter()
        self.intervals = deque(maxlen=60)

    def tick(self, rate=60):
        now = time.perf_counter()
        delay = self.last + 1 / rate - now
        if delay > 0:
            time.sleep(delay)
            now = time.perf_counter()
        elapsed = now - self.last
        self.last = now
        self.intervals.append(elapsed)
        return elapsed * 1000

    def get_fps(self):
        return len(self.intervals) / sum(self.intervals) if self.intervals else 0.0
