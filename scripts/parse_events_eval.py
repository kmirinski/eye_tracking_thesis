#!/usr/bin/env python3
"""Parse the stdout of `main.py --events_eval` runs into a tidy CSV.

For each polynomial degree the events_eval run prints, among others:

    --- Degree 2 ---
      [diag deg2] events anchored to GOOD frames (<5° anchor resid): mean=3.57°  median=3.30°  (n=779/4958)
      frames     (25 Hz)    : DoD mean=8.03°  median=5.08°  (n=1303)

We keep only the two means the analysis cares about:
  * frames_dod_mean        <- "frames (25 Hz) : DoD mean="
  * events_good_anchor_mean <- "events anchored to GOOD frames (<...° anchor resid): mean="

One CSV is written per subject, with one row per (eval_split, degree). The
"events anchored to GOOD frames" line is only printed when at least one good
frame exists; if it is absent for a degree the value is recorded as empty.

Usage:
    python scripts/parse_events_eval.py \
        --subject 42 --eye left --motion saccadic \
        --out results/events_eval/s42.csv \
        --log blocks:logs/events_eval_s42_blocks.out \
        --log within:logs/events_eval_s42_within.out
"""
import argparse
import csv
import os
import re

DEGREE_RE = re.compile(r"^\s*---\s*Degree\s+(\d+)\s*---")
FRAMES_RE = re.compile(r"frames\s+\(25 Hz\)\s*:\s*DoD mean=([\d.]+|nan)")
GOOD_RE = re.compile(r"events anchored to GOOD frames.*?mean=([\d.]+|nan)")


def parse_log(path):
    """Return {degree: {'frames_dod_mean': str, 'events_good_anchor_mean': str}}."""
    results = {}
    current = None
    with open(path) as f:
        for line in f:
            m = DEGREE_RE.search(line)
            if m:
                current = int(m.group(1))
                results.setdefault(current, {})
                continue
            if current is None:
                continue
            m = GOOD_RE.search(line)
            if m:
                results[current]['events_good_anchor_mean'] = m.group(1)
                continue
            m = FRAMES_RE.search(line)
            if m:
                results[current]['frames_dod_mean'] = m.group(1)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--eye', default='left')
    ap.add_argument('--motion', default='saccadic')
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', action='append', default=[], metavar='SPLIT:PATH',
                    help='eval_split label and log path, e.g. blocks:logs/s42_blocks.out '
                         '(repeatable)')
    opt = ap.parse_args()

    fieldnames = ['subject', 'eye', 'motion', 'eval_split', 'degree',
                  'frames_dod_mean', 'events_good_anchor_mean']
    rows = []
    for spec in opt.log:
        split, _, path = spec.partition(':')
        if not os.path.exists(path):
            print(f"WARNING: log not found, skipping: {path}")
            continue
        parsed = parse_log(path)
        if not parsed:
            print(f"WARNING: no degree blocks parsed from {path}")
        for degree in sorted(parsed):
            vals = parsed[degree]
            rows.append({
                'subject': opt.subject, 'eye': opt.eye, 'motion': opt.motion,
                'eval_split': split, 'degree': degree,
                'frames_dod_mean': vals.get('frames_dod_mean', ''),
                'events_good_anchor_mean': vals.get('events_good_anchor_mean', ''),
            })

    os.makedirs(os.path.dirname(os.path.abspath(opt.out)), exist_ok=True)
    with open(opt.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {opt.out}")


if __name__ == '__main__':
    main()
