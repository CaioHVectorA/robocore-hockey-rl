"""Short delayed interpolation of recorded poses; never advances training physics."""

import math


def interpolate_frames(frames, timestamp):
    if not frames:
        return None
    latest = frames[-1]
    if not latest.get("speed"):
        return latest
    for first, second in zip(frames, list(frames)[1:]):
        start, end = first["timestamp"], second["timestamp"]
        if not start <= timestamp < end:
            continue
        if (
            first.get("episode") != second.get("episode")
            or second.get("time", 0) < first.get("time", 0)
            or first.get("speed") != second.get("speed")
            or end - start > 0.25
            or len(first["robots"]) != len(second["robots"])
        ):
            return second
        fraction = (timestamp - start) / (end - start)

        def blend(a, b):
            return a + (b - a) * fraction

        robots = []
        for a, b in zip(first["robots"], second["robots"]):
            if a[3:] != b[3:]:
                return second
            angle = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
            robots.append([blend(a[0], b[0]), blend(a[1], b[1]), a[2] + angle * fraction, *b[3:]])
        return dict(
            second,
            robots=robots,
            puck=[blend(a, b) for a, b in zip(first["puck"], second["puck"])],
            time=blend(first["time"], second["time"]),
        )
    return latest
