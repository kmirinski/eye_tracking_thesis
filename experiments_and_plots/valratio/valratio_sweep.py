"""Sweep the calibration/validation split size (val_ratio) and plot its effect on gaze error.

In the events_eval `--eval_split blocks` protocol, `val_ratio` sets the fraction of time blocks
held out for evaluation; the rest calibrate the polynomial. This script sweeps `val_ratio` and
plots the resulting degree-2 frame-eval (25 Hz) DoD error against the split size.

Like the threshold sweep, all values are evaluated in a SINGLE pipeline run
(`--val_ratio_sweep 0.1 0.2 ...`): the event stream is extracted once, then each split is
re-fit/re-evaluated on the cached samples. The runner writes per-(degree, val_ratio) errors to a
CSV, which we read and plot.

Run from the repo root:
    source .venv/bin/activate
    python notebooks/valratio_sweep.py
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
VAL_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9]   # fraction of time blocks held out for eval
DEGREE = 2                                # polynomial degree to plot
# -----------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO, "results",
                        f"valratio_sweep_s{SUBJECT}_{EYE}_{MOTION}.csv")
PNG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"valratio_sweep_s{SUBJECT}_{EYE}.png")


def run_pipeline_sweep():
    """Run main.py once, evaluating all val_ratio values; produces CSV_PATH."""
    cmd = [
        sys.executable, os.path.join("src", "main.py"),
        "--subject", str(SUBJECT), "--eye", EYE, "--dataset", DATASET,
        "--motion", MOTION, "--events_eval", "--eval_split", EVAL_SPLIT,
        "--val_ratio_sweep", *[str(r) for r in VAL_RATIOS],
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def load_degree_rows(degree):
    """Read CSV_PATH, return (val_ratios, f_means) sorted by val_ratio for one degree."""
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["degree"]) == degree:
                rows.append((float(r["val_ratio"]), float(r["f_mean"])))
    rows.sort()
    val_ratios = [v for v, _ in rows]
    f_means = [g for _, g in rows]
    return val_ratios, f_means


def main():
    run_pipeline_sweep()
    val_ratios, f_means = load_degree_rows(DEGREE)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(val_ratios, f_means, marker="o")
    ax.set_xlabel("Evaluation set ratio")
    ax.set_ylabel("DoD mean [deg]")
    ax.set_title(f"Subject {SUBJECT}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Saved plot -> {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
