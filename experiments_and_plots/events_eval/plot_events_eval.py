"""Visualize the events_eval regressor summaries (whole FoV and 40x20 FoV).

Reads the two aggregated CSVs produced by scripts/aggregate_events_eval.py and, for
EACH FoV setting, writes 5 per-subject bar plots into its own folder:

    experiments_and_plots/events_eval/whole_fov/
    experiments_and_plots/events_eval/fov_40x20/
        best_per_subject.png          best degree/subject over blocks+within
        best_per_subject_blocks.png   best degree/subject, blocks split only
        best_per_subject_within.png   best degree/subject, within split only
        degree2.png                   degree 2, two panels (blocks, within)
        degree3.png                   degree 3, two panels (blocks, within)

Every panel shows two metrics per subject:
    * Frame gaze (25 Hz)        -> frames_dod_mean        (paper model-based ref 9.72 deg)
    * Events @ good frames (<5) -> events_good_anchor_mean (paper EV-Eye ref   4.71 deg)
"best" means the row with the lowest frame DoD for that subject within the selected rows.

The across-subject mean of BOTH metrics for every panel is printed to stdout.

Run from the repo root:
    python experiments_and_plots/events_eval/plot_events_eval.py
"""
import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Metric columns, display labels, colors, and the paper reference line for each.
FRAME = dict(col='frames_dod_mean', label='Frame gaze (25 Hz)',
             color='steelblue', ref=9.72, ref_label='EVBEYE model-based (9.72°, whole FoV)')
EVENT = dict(col='events_good_anchor_mean', label='Events @ good frames (<5°)',
             color='darkorange', ref=4.71, ref_label='EV-Eye method (4.71°, whole FoV)')

# Whole-FoV vs central-40x20 frame-DoD comparison (one metric, two FoV settings).
WHOLE_FOV = dict(label='Whole FoV', color='steelblue',
                 ref=9.72, ref_label='EVBEYE model-based (9.72°, whole FoV)')
FOV40 = dict(label='Central 40×20° FoV', color='darkorange')


def _f(v):
    """Parse a CSV cell to float; empty / 'nan' -> NaN."""
    v = (v or '').strip()
    if v == '' or v.lower() == 'nan':
        return float('nan')
    return float(v)


def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def best_per_subject(rows):
    """{subject: (frame, events)} keeping, per subject, the row with min frame DoD."""
    out = {}
    for r in rows:
        frame = _f(r['frames_dod_mean'])
        if math.isnan(frame):
            continue
        s = int(r['subject'])
        if s not in out or frame < out[s][0]:
            out[s] = (frame, _f(r['events_good_anchor_mean']))
    return out


def at_degree_split(rows, degree, split):
    """{subject: (frame, events)} for one (degree, eval_split)."""
    out = {}
    for r in rows:
        if int(r['degree']) == degree and r['eval_split'] == split:
            out[int(r['subject'])] = (_f(r['frames_dod_mean']),
                                      _f(r['events_good_anchor_mean']))
    return out


def _nanmean(vals):
    arr = np.array([v for v in vals if not math.isnan(v)], dtype=float)
    return (float(arr.mean()) if arr.size else float('nan')), arr.size


def draw_panel(ax, data, title, tag):
    """data: {subject: (frame, events)}. Draws grouped bars + ref/mean lines, prints means."""
    subjects = sorted(data)
    x = np.arange(len(subjects))
    width = 0.4
    frame_vals = [data[s][0] for s in subjects]
    event_vals = [data[s][1] for s in subjects]

    f_mean, f_n = _nanmean(frame_vals)
    e_mean, e_n = _nanmean(event_vals)
    print(f"  {tag} | {title}")
    print(f"      {FRAME['label']:<28} mean = {f_mean:6.2f}°  (n={f_n})")
    print(f"      {EVENT['label']:<28} mean = {e_mean:6.2f}°  (n={e_n})")

    ax.bar(x - width / 2, frame_vals, width, color=FRAME['color'],
           label=f"{FRAME['label']} (mean {f_mean:.2f}°)")
    ax.bar(x + width / 2, event_vals, width, color=EVENT['color'],
           label=f"{EVENT['label']} (mean {e_mean:.2f}°)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects], fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('Subject', fontsize=17)
    ax.set_ylabel('DoD [deg]', fontsize=17)
    ax.grid(True, axis='y', alpha=0.3)
    # Legend outside the axes (above), so it never overlaps the bars.
    ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2)


def draw_figure(panels, fig_title, out_path, tag):
    """panels: list of (title, data). One row per panel. Titles are used only for the
    stdout means summary; they are not drawn on the figure."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.2 * n), squeeze=False)
    for ax, (title, data) in zip(axes[:, 0], panels):
        draw_panel(ax, data, title, tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> saved {out_path}\n")


def _frame(data):
    """{subject: (frame, events)} -> {subject: frame} (frame DoD only)."""
    return {s: v[0] for s, v in data.items()}


def draw_compare_panel(ax, whole, fov, title, tag):
    """Per-subject frame DoD: whole FoV (blue) vs central 40x20 (orange). Both dicts are
    {subject: frame_dod}. Mirrors draw_panel's styling; prints both means."""
    subjects = sorted(set(whole) | set(fov))
    x = np.arange(len(subjects))
    width = 0.4
    w_vals = [whole.get(s, float('nan')) for s in subjects]
    f_vals = [fov.get(s, float('nan')) for s in subjects]

    w_mean, w_n = _nanmean(w_vals)
    f_mean, f_n = _nanmean(f_vals)
    print(f"  {tag} | {title}")
    print(f"      {WHOLE_FOV['label']:<22} frame DoD mean = {w_mean:6.2f}°  (n={w_n})")
    print(f"      {FOV40['label']:<22} frame DoD mean = {f_mean:6.2f}°  (n={f_n})")

    ax.bar(x - width / 2, w_vals, width, color=WHOLE_FOV['color'],
           label=WHOLE_FOV['label'])
    ax.bar(x + width / 2, f_vals, width, color=FOV40['color'],
           label=FOV40['label'])

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects], fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('Subject', fontsize=17)
    ax.set_ylabel('DoD [deg]', fontsize=17)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2)


def draw_compare_figure(panels, out_path, tag):
    """panels: list of (title, whole_data, fov_data). One row per panel."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.2 * n), squeeze=False)
    for ax, (title, whole, fov) in zip(axes[:, 0], panels):
        draw_compare_panel(ax, whole, fov, title, tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> saved {out_path}\n")


def build_whole_vs_fov(whole_csv, fov_csv, out_dir):
    if not (os.path.exists(whole_csv) and os.path.exists(fov_csv)):
        print(f"SKIP whole-vs-fov: need both CSVs ({whole_csv}, {fov_csv})")
        return
    os.makedirs(out_dir, exist_ok=True)
    W, F = load_rows(whole_csv), load_rows(fov_csv)
    Wb = [r for r in W if r['eval_split'] == 'blocks']
    Ww = [r for r in W if r['eval_split'] == 'within']
    Fb = [r for r in F if r['eval_split'] == 'blocks']
    Fw = [r for r in F if r['eval_split'] == 'within']
    tag = 'Whole-vs-40x20'

    print(f"\n================  Whole FoV vs central 40x20 (frame DoD)  ================")

    draw_compare_figure(
        [('Best degree per subject (blocks + within)',
          _frame(best_per_subject(W)), _frame(best_per_subject(F)))],
        os.path.join(out_dir, 'best_per_subject.png'), tag)
    draw_compare_figure(
        [('Best degree per subject — blocks split',
          _frame(best_per_subject(Wb)), _frame(best_per_subject(Fb)))],
        os.path.join(out_dir, 'best_per_subject_blocks.png'), tag)
    draw_compare_figure(
        [('Best degree per subject — within split',
          _frame(best_per_subject(Ww)), _frame(best_per_subject(Fw)))],
        os.path.join(out_dir, 'best_per_subject_within.png'), tag)
    draw_compare_figure(
        [('Degree 2 — blocks split',
          _frame(at_degree_split(W, 2, 'blocks')), _frame(at_degree_split(F, 2, 'blocks'))),
         ('Degree 2 — within split',
          _frame(at_degree_split(W, 2, 'within')), _frame(at_degree_split(F, 2, 'within')))],
        os.path.join(out_dir, 'degree2.png'), tag)
    draw_compare_figure(
        [('Degree 3 — blocks split',
          _frame(at_degree_split(W, 3, 'blocks')), _frame(at_degree_split(F, 3, 'blocks'))),
         ('Degree 3 — within split',
          _frame(at_degree_split(W, 3, 'within')), _frame(at_degree_split(F, 3, 'within')))],
        os.path.join(out_dir, 'degree3.png'), tag)


def build_for_csv(csv_path, out_dir, label):
    if not os.path.exists(csv_path):
        print(f"SKIP {label}: CSV not found at {csv_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    rows = load_rows(csv_path)
    blocks = [r for r in rows if r['eval_split'] == 'blocks']
    within = [r for r in rows if r['eval_split'] == 'within']

    print(f"\n================  {label}  ({len(rows)} rows)  ================")

    draw_figure([('Best degree per subject (blocks + within)', best_per_subject(rows))],
                f'{label} — best per subject',
                os.path.join(out_dir, 'best_per_subject.png'), label)
    draw_figure([('Best degree per subject — blocks split', best_per_subject(blocks))],
                f'{label} — best per subject (blocks)',
                os.path.join(out_dir, 'best_per_subject_blocks.png'), label)
    draw_figure([('Best degree per subject — within split', best_per_subject(within))],
                f'{label} — best per subject (within)',
                os.path.join(out_dir, 'best_per_subject_within.png'), label)
    draw_figure([('Degree 2 — blocks split', at_degree_split(rows, 2, 'blocks')),
                 ('Degree 2 — within split', at_degree_split(rows, 2, 'within'))],
                f'{label} — degree 2',
                os.path.join(out_dir, 'degree2.png'), label)
    draw_figure([('Degree 3 — blocks split', at_degree_split(rows, 3, 'blocks')),
                 ('Degree 3 — within split', at_degree_split(rows, 3, 'within'))],
                f'{label} — degree 3',
                os.path.join(out_dir, 'degree3.png'), label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--whole_csv', default=os.path.join(REPO, 'results', 'events_eval_summary.csv'))
    ap.add_argument('--fov_csv', default=os.path.join(REPO, 'results',
                                                      'events_eval_fov40x20_summary.csv'))
    ap.add_argument('--out_base', default=PLOT_DIR)
    opt = ap.parse_args()

    build_for_csv(opt.whole_csv, os.path.join(opt.out_base, 'whole_fov'), 'Whole FoV')
    build_for_csv(opt.fov_csv, os.path.join(opt.out_base, 'fov_40x20'), 'FoV 40x20')
    build_whole_vs_fov(opt.whole_csv, opt.fov_csv,
                       os.path.join(opt.out_base, 'whole_vs_fov40x20'))


if __name__ == '__main__':
    main()
