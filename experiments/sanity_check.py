# experiments/sanity_check.py
# Author: Avijit Roy
#
# Purpose:
# - Basic run integrity checks for attack CSV logs
# - Handles __HYPERPARAMS__ row
# - Checks per-image monotonicity of steps and elapsed_ms

import sys
import numpy as np
import pandas as pd


def main():
    path = sys.argv[1]
    df = pd.read_csv(path)

    # Drop the hyperparams/meta row if present
    df = df[df["image_id"] != "__HYPERPARAMS__"].copy()

    success = df[df["success"] == 1]
    grouped = success.groupby("image_id").size()

    print("Total rows:", len(df))
    print("Unique images:", df["image_id"].nunique())
    print("Images succeeded:", len(grouped))
    print("Max success per image:", int(grouped.max()) if len(grouped) else 0)

    failed = sorted(set(df["image_id"].unique()) - set(success["image_id"].unique()))
    print("Images failed:", len(failed))
    if failed:
        print("Example failed images:", failed[:10])

    # --- Monotonicity checks (per image) ---
    bad_steps = 0
    bad_time = 0
    bad_step_examples = []
    bad_time_examples = []

    for image_id, g in df.groupby("image_id"):
        g2 = g.sort_values("step")
        steps = g2["step"].to_numpy()
        times = g2["elapsed_ms"].to_numpy()

        # steps should be consecutive starting at 1
        if not (len(steps) > 0 and steps[0] == 1 and np.all(np.diff(steps) == 1)):
            bad_steps += 1
            if len(bad_step_examples) < 5:
                bad_step_examples.append(image_id)

        # elapsed_ms should be non-decreasing (allow equal due to rounding)
        if np.any(np.diff(times) < 0):
            bad_time += 1
            if len(bad_time_examples) < 5:
                bad_time_examples.append(image_id)

    print("Images with non-consecutive steps:", bad_steps)
    if bad_step_examples:
        print("Example step-bad images:", bad_step_examples)

    print("Images with decreasing elapsed_ms:", bad_time)
    if bad_time_examples:
        print("Example time-bad images:", bad_time_examples)


if __name__ == "__main__":
    main()