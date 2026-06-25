"""Sweep the "GOOD frames" anchor-residual threshold and plot its effect on gaze error.

The events_eval protocol (`--events_eval`) reports, per polynomial degree, the DoD error of
events anchored to "GOOD" frames — frames whose own polynomial residual is below a threshold
(default 5 deg). This script sweeps that threshold over several values and plots the resulting
degree-2 error against the threshold.

Because the threshold only affects the final masking step, all values are evaluated in a SINGLE
pipeline run (`--good_anchor_thresh 5 7 9 11 13`); the runner writes the per-(degree, threshold)
errors to a CSV, which we then read and plot.

Run from the repo root:
    source .venv/bin/activate
    python notebooks/threshold_sweep.py
"""
import os
import csv
import sys
import subprocess

import matplotlib.pyplot as plt

# --- experiment params (edit here) -------------------------------------------------
SUBJECT = 7
EYE = "left"
DATASET = "ev_eye"
MOTION = "saccadic"
EVAL_SPLIT = "blocks"
THRESHOLDS = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18]   # anchor-residual thresholds in degrees to sweep
DEGREE = 2                        # polynomial degree to plot
# -----------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO, "results",
                        f"threshold_sweep_s{SUBJECT}_{EYE}_{MOTION}.csv")
PNG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"threshold_sweep_s{SUBJECT}_{EYE}.png")


def run_pipeline_sweep():
    """Run main.py once, evaluating all thresholds; produces CSV_PATH."""
    cmd = [
        sys.executable, os.path.join("src", "main.py"),
        "--subject", str(SUBJECT), "--eye", EYE, "--dataset", DATASET,
        "--motion", MOTION, "--events_eval", "--eval_split", EVAL_SPLIT,
        "--good_anchor_thresh", *[str(t) for t in THRESHOLDS],
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def load_degree_rows(degree):
    """Read CSV_PATH, return (thresholds, g_means) sorted by threshold for one degree."""
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["degree"]) == degree:
                rows.append((float(r["threshold"]), float(r["g_mean"])))
    rows.sort()
    thresholds = [t for t, _ in rows]
    g_means = [g for _, g in rows]
    return thresholds, g_means


def main():
    run_pipeline_sweep()
    thresholds, g_means = load_degree_rows(DEGREE)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, g_means, marker="o")
    ax.set_xlabel("Threshold [deg]")
    ax.set_ylabel("DoD mean [deg]")
    ax.set_title(f"Subject {SUBJECT}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Saved plot -> {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
