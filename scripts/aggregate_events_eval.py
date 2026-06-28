#!/usr/bin/env python3
"""Merge the per-subject events_eval CSVs into one summary CSV.

Run after the `run_events_eval.sh` SLURM array finishes. Globs
results/events_eval/s*.csv and concatenates them, sorted by
(subject, eval_split, degree), into results/events_eval_summary.csv.
"""
import argparse
import csv
import glob
import os

FIELDNAMES = ['subject', 'eye', 'motion', 'eval_split', 'degree',
              'frames_dod_mean', 'events_good_anchor_mean']


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in_dir', default=os.path.join(repo_root, 'results', 'events_eval'))
    ap.add_argument('--out', default=os.path.join(repo_root, 'results', 'events_eval_summary.csv'))
    opt = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(opt.in_dir, 's*.csv')))
    if not paths:
        print(f"No per-subject CSVs found in {opt.in_dir}")
        return

    rows = []
    for path in paths:
        with open(path) as f:
            rows.extend(csv.DictReader(f))

    def sort_key(r):
        return (int(r['subject']), r['eval_split'], int(r['degree']))

    rows.sort(key=sort_key)

    with open(opt.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(paths)} files, {len(rows)} rows -> {opt.out}")


if __name__ == '__main__':
    main()
