"""Fit free sliding puck trials: CSV columns time,x,y in seconds and centimetres."""

import argparse
import json
import numpy as np


def fit_puck_trial(time, x, y):
    t = np.asarray(time, float)
    xy = np.column_stack((x, y)).astype(float)
    if len(t) < 8 or not np.isfinite(t).all() or not np.isfinite(xy).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Need at least eight finite samples with increasing timestamps")
    t = t - t[0]
    design = np.column_stack((np.ones_like(t), t, 0.5 * t * t))
    coefficients = np.linalg.lstsq(design, xy, rcond=None)[0]
    initial = coefficients[1]
    acceleration = coefficients[2]
    speed = np.linalg.norm(initial)
    if speed < 1:
        raise ValueError("Initial sliding speed too small")
    decel = -float(initial @ acceleration) / speed
    if decel <= 0:
        raise ValueError("Trial is not decelerating")
    prediction = design @ coefficients
    lateral = acceleration + decel * initial / speed
    return dict(
        friction_mu=decel / 981,
        initial_velocity_cm_s=initial.tolist(),
        acceleration_cm_s2=acceleration.tolist(),
        deceleration_cm_s2=decel,
        lateral_acceleration_cm_s2=float(np.linalg.norm(lateral)),
        rmse_cm=float(np.sqrt(np.mean(np.sum((prediction - xy) ** 2, axis=1)))),
        samples=len(t),
        duration_seconds=float(t[-1]),
        warning="Use only unobstructed sliding before rest; estimates do not automatically change configuration",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["puck"])
    p.add_argument("csv")
    p.add_argument("--output")
    args = p.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    result = fit_puck_trial(data["time"], data["x"], data["y"])
    result["source_file"] = args.csv
    text = json.dumps(result, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
