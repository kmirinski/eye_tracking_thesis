"""
Merge per-fold cross-subject result CSVs into a single summary.

Each SLURM job writes its own file to results/folds/ (see src/results_io.py).
This script concatenates them into results/summary.csv and prints a compact
pivot: mean +/- std of mean_error and dod_mean across subjects, grouped by
(model, motion, combined, fine_tune, phase), plus the baseline->finetuned gain
within the fine-tuned jobs.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results_dir results
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mean_std(values):
    n = len(values)
    if n == 0:
        return float('nan'), float('nan')
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def load_rows(folds_dir):
    rows = []
    files = sorted(glob.glob(os.path.join(folds_dir, '*.csv')))
    for path in files:
        with open(path, newline='') as f:
            rows.extend(csv.DictReader(f))
    return rows, files


def write_summary(rows, summary_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_pivot(rows):
    # For the regressor a fold has several degrees; keep the best (lowest mean_error)
    # per (fold-config, phase) so each subject contributes once per group.
    best = {}
    for r in rows:
        key = (r['model'], r['dataset'], r['motion'], r['eye'], r['combined'],
               r['fine_tune'], r['phase'], r['val_subject'])
        me = float(r['mean_error'])
        if key not in best or me < float(best[key]['mean_error']):
            best[key] = r

    groups = defaultdict(list)
    for r in best.values():
        gkey = (r['model'], r['motion'], r['combined'], r['fine_tune'], r['phase'])
        groups[gkey].append(r)

    header = f"{'model':<10} {'motion':<9} {'comb':<5} {'ft':<3} {'phase':<10} " \
             f"{'n':>3} {'mean_error':>18} {'dod_mean':>16}"
    print(header)
    print('-' * len(header))
    for gkey in sorted(groups):
        model, motion, combined, ft, phase = gkey
        members = groups[gkey]
        me_mean, me_std = _mean_std([float(r['mean_error']) for r in members])
        dod_mean, dod_std = _mean_std([float(r['dod_mean']) for r in members])
        comb = combined if combined != '' else '-'
        print(f"{model:<10} {motion:<9} {comb:<5} {ft:<3} {phase:<10} "
              f"{len(members):>3} {me_mean:>9.4f} +/- {me_std:<6.4f} "
              f"{dod_mean:>7.3f} +/- {dod_std:<6.3f}")

    # baseline -> finetuned gain within the fine-tuned (ft=1) jobs
    print("\nFine-tune gain (baseline -> finetuned, mean_error across subjects):")
    deltas = {}
    for (model, motion, combined, ft, phase), members in groups.items():
        if ft != '1':
            continue
        me_mean, _ = _mean_std([float(r['mean_error']) for r in members])
        deltas[(model, motion, combined, phase)] = me_mean
    seen = sorted({(m, mo, c) for (m, mo, c, p) in deltas})
    for model, motion, combined in seen:
        base = deltas.get((model, motion, combined, 'baseline'))
        ft = deltas.get((model, motion, combined, 'finetuned'))
        if base is None or ft is None:
            continue
        comb = combined if combined != '' else '-'
        print(f"  {model:<10} {motion:<9} comb={comb:<5} "
              f"{base:.4f} -> {ft:.4f}  (delta {base - ft:+.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', default=os.path.join(REPO_ROOT, 'results'),
                        help='Directory containing folds/ and where summary.csv is written')
    opt = parser.parse_args()

    folds_dir = os.path.join(opt.results_dir, 'folds')
    rows, files = load_rows(folds_dir)
    if not rows:
        print(f"No per-fold CSVs found in {folds_dir}")
        return

    summary_path = os.path.join(opt.results_dir, 'summary.csv')
    write_summary(rows, summary_path)
    print(f"Merged {len(files)} files ({len(rows)} rows) -> {summary_path}\n")
    print_pivot(rows)


if __name__ == '__main__':
    main()
