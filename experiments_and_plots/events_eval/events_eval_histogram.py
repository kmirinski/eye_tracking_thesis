"""
Run the events_eval regressor protocol across a few ev_eye subjects and plot a
grouped bar chart comparing two degree-2 diagnostics per subject:

  - "events anchored to GOOD frames (<5° anchor resid)" mean
  - "frames (25 Hz)" DoD mean

These numbers are only printed by run_regressor_events_eval (src/pipeline/runners.py),
not returned, so this script invokes src/main.py as a subprocess per subject and
parses the degree-2 block from stdout.

Usage (from repo root, with .venv active):
    python scripts/events_eval_histogram.py
"""

import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # save without a display; safe in headless runs
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subject 7 (already inspected) plus 4 random others, drawn from the verified-complete
# ev_eye subject list (src/config.py). Sorted so the x-axis is monotonic.
SUBJECTS = [5, 7, 22, 31, 36]
# SUBJECTS = [5, 7]
DEGREE = 2

BASE_ARGS = [
    "--eye", "left",
    "--dataset", "ev_eye",
    "--motion", "saccadic",
    "--events_eval",
    "--eval_split", "blocks",
]

OUT_PNG = os.path.join(REPO_ROOT, "scripts", "events_eval_deg2_histogram.png")

GOOD_RE = re.compile(r"events anchored to GOOD frames.*?mean=([\d.]+)")
FRAMES_RE = re.compile(r"frames\s+\(25 Hz\)\s+: DoD mean=([\d.]+)")


def run_subject(subject):
    """Run the pipeline for one subject and return its stdout (or None on failure)."""
    cmd = [sys.executable, "src/main.py", "--subject", str(subject)] + BASE_ARGS
    print(f"[subject {subject}] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[subject {subject}] WARNING: exited with code {proc.returncode}")
        # show the tail of stderr to help diagnose without dumping everything
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        if tail:
            print(tail)
        return None
    return proc.stdout


def parse_degree_block(stdout, degree):
    """Slice out the `--- Degree {degree} ---` section and parse the two means.

    The `frames (25 Hz)` line is printed for every degree, so we must restrict the
    search to this degree's block (up to the next `--- Degree` marker or end).
    Returns (good_mean, frames_mean) with NaN for any value not found.
    """
    start = stdout.find(f"--- Degree {degree} ---")
    if start == -1:
        return float("nan"), float("nan")
    rest = stdout[start + len(f"--- Degree {degree} ---"):]
    nxt = rest.find("--- Degree")
    block = rest if nxt == -1 else rest[:nxt]

    good_m = GOOD_RE.search(block)
    frames_m = FRAMES_RE.search(block)
    good = float(good_m.group(1)) if good_m else float("nan")
    frames = float(frames_m.group(1)) if frames_m else float("nan")
    if good_m is None:
        print(f"[degree {degree}] WARNING: GOOD-frames line not found "
              "(likely no events anchored to good frames)")
    if frames_m is None:
        print(f"[degree {degree}] WARNING: frames (25 Hz) line not found")
    return good, frames


def main():
    results = {}  # subject -> (good_mean, frames_mean)
    for s in sorted(SUBJECTS):
        stdout = run_subject(s)
        if stdout is None:
            results[s] = (float("nan"), float("nan"))
            continue
        results[s] = parse_degree_block(stdout, DEGREE)

    # Printed summary table.
    print("\n=== Degree-{} summary ===".format(DEGREE))
    print(f"{'subject':>8} {'GOOD mean':>12} {'frames(25Hz) mean':>20}")
    for s in sorted(results):
        g, f = results[s]
        print(f"{s:>8} {g:>12.2f} {f:>20.2f}")

    # Grouped bar chart.
    subjects = sorted(results)
    good_vals = np.array([results[s][0] for s in subjects], dtype=float)
    frame_vals = np.array([results[s][1] for s in subjects], dtype=float)

    x = np.arange(len(subjects))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(subjects)), 5))
    b1 = ax.bar(x - width / 2, good_vals, width,
                label="events anchored to GOOD frames (mean)", color="#4C72B0")
    b2 = ax.bar(x + width / 2, frame_vals, width,
                label="frames (25 Hz) DoD (mean)", color="#DD8452")

    ax.set_xlabel("Subject")
    ax.set_ylabel(f"Degree-{DEGREE} DoD mean (°)")
    ax.set_title(f"GOOD-frame events vs 25 Hz frames")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved chart to {OUT_PNG}")


if __name__ == "__main__":
    main()
