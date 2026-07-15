#!/usr/bin/env python3
"""Run the single-subject frame-only polynomial regressor for every ebveye subject
and summarise per-subject Distance Error / DoD, one table per polynomial degree.

Equivalent to running, for each subject S:

    python src/main.py --model regressor --subject S --motion saccadic --frame_only

This drives main.py as a subprocess (so each subject gets a clean run), captures
its stdout, and parses the per-degree blocks it prints:

    --- Degree D ---
    ...
    Distance Error: mean=X px  median=Y px
    DoD: mean=Z°  median=W°

then prints, for every polynomial degree, a summary table of the form:

    ============================================================
    Degree 4
    ============================================================
        4: Distance Error mean=72.91381px  median=61.66050px  |  DoD=3.88°
        5: ...

Usage:
    python experiments_and_plots/frame_only_regressor_sweep.py
    python experiments_and_plots/frame_only_regressor_sweep.py --subjects 6 7 15
"""
import argparse
import os
import re
import subprocess
import sys

# Repo root is the parent of experiments_and_plots/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(REPO_ROOT, 'src', 'main.py')

# Default subject list mirrors CROSS_SUBJECT_SUBJECTS['ebveye'] in src/config.py.
DEFAULT_SUBJECTS = [4, 5, 6, 7, 11, 12, 15, 18, 19, 21, 22]

DEGREE_RE = re.compile(r'^--- Degree (\d+) ---')
DIST_RE = re.compile(r'Distance Error:\s*mean=([\d.]+)px\s+median=([\d.]+)px')
DOD_RE = re.compile(r'DoD:\s*mean=([-\d.]+)°')


def run_subject(subject, motion, dataset):
    """Run main.py for one subject; return its stdout (also streamed live)."""
    cmd = [sys.executable, MAIN_PY,
           '--model', 'regressor',
           '--subject', str(subject),
           '--dataset', dataset,
           '--motion', motion,
           '--frame_only']
    print(f"\n>>> subject {subject}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"subject {subject} exited with code {proc.returncode}")
    return proc.stdout


def parse_degrees(stdout):
    """Parse stdout into {degree: {'mean':.., 'median':.., 'dod':..}}.

    The per-degree block prints the Distance Error line then the DoD line, so we
    track the current degree header and attach the next metrics we see to it.
    """
    results = {}
    cur = None
    for line in stdout.splitlines():
        m = DEGREE_RE.match(line.strip())
        if m:
            cur = int(m.group(1))
            results[cur] = {}
            continue
        if cur is None:
            continue
        m = DIST_RE.search(line)
        if m:
            results[cur]['mean'] = float(m.group(1))
            results[cur]['median'] = float(m.group(2))
            continue
        m = DOD_RE.search(line)
        if m:
            results[cur]['dod'] = float(m.group(1))
    # Keep only fully-populated degrees.
    return {d: v for d, v in results.items()
            if {'mean', 'median', 'dod'} <= v.keys()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', type=int, nargs='+', default=DEFAULT_SUBJECTS,
                    help='subjects to run (default: ebveye cross-subject list)')
    ap.add_argument('--motion', default='saccadic', choices=['saccadic', 'pursuit'])
    ap.add_argument('--dataset', default='ebveye', choices=['ebveye', 'ev_eye'])
    args = ap.parse_args()

    # per_subject[subject] = {degree: {'mean','median','dod'}}
    per_subject = {}
    for s in args.subjects:
        per_subject[s] = parse_degrees(run_subject(s, args.motion, args.dataset))

    # Collect the union of degrees seen, in ascending order.
    degrees = sorted({d for res in per_subject.values() for d in res})

    for deg in degrees:
        print('\n' + '=' * 60)
        print(f'Degree {deg}')
        print('=' * 60)
        for s in args.subjects:
            res = per_subject[s].get(deg)
            if res is None:
                print(f"  {s:>3}: (no result)")
                continue
            print(f"  {s:>3}: Distance Error mean={res['mean']:.5f}px  "
                  f"median={res['median']:.5f}px  |  DoD={res['dod']:.2f}°")


if __name__ == '__main__':
    main()
