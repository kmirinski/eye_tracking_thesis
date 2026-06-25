"""
Per-fold result persistence for cross-subject evaluation.

Each SLURM job (one fold of one config) writes its own CSV file to
``results/folds/`` via :func:`save_fold`. Unique filenames (see
:func:`fold_filename`) make parallel writes race-free — no locking needed.
``scripts/aggregate_results.py`` later merges all per-fold files into
``results/summary.csv``.

Rows are stored in long format: one row per (fold x phase), and for the
regressor also per polynomial degree. The ``phase`` column is ``baseline`` or
``finetuned``; the ``fine_tune`` column (0/1) distinguishes a no-FT job
(baseline evaluated on the full validation subject) from an FT job (baseline +
finetuned evaluated on the held-out eval split).
"""

import csv
import os

from pipeline.runners import errors_to_degrees

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FOLDS_DIR = os.path.join(RESULTS_DIR, 'folds')

# Fixed column order for every per-fold CSV, so files concatenate cleanly.
CONFIG_FIELDS = [
    'model', 'dataset', 'motion', 'eye', 'val_subject',
    'fine_tune', 'relabel', 'combined', 'degree', 'phase', 'n_eval',
]
METRIC_FIELDS = [
    'mse', 'rmse', 'mean_error', 'std_error', 'max_error', 'median_error',
    'mean_error_v', 'mean_error_h', 'median_error_v', 'median_error_h',
    'h_deg', 'v_deg', 'dod_mean', 'dod_median',
]
FIELDNAMES = CONFIG_FIELDS + METRIC_FIELDS


def metrics_row(metrics, gaze_config, normalized):
    """Flatten a metrics dict into the metric columns, adding per-axis degrees.

    ``metrics`` is the dict returned by ``estimator.evaluate`` augmented with
    ``dod_mean``/``dod_median``. Per-axis degrees are derived from the pixel
    errors via :func:`errors_to_degrees` (reused from the runners module) so the
    CSV is directly readable in degrees.
    """
    v_deg, h_deg = errors_to_degrees(
        metrics['mean_error_v'], metrics['mean_error_h'], gaze_config, normalized)
    row = {k: float(metrics[k]) for k in METRIC_FIELDS if k in metrics}
    row['h_deg'] = float(h_deg)
    row['v_deg'] = float(v_deg)
    return row


def fold_filename(model, dataset, motion, eye, val_subject,
                  fine_tune, combined=False, relabel=False):
    """Build a unique per-fold filename (mirrors the data_cache naming style).

    The ``ft0``/``ft1`` tag keeps the no-FT and FT jobs for the same fold from
    colliding; ``comb`` and ``rel`` tags disambiguate the remaining dimensions.
    """
    ft = 'ft1' if fine_tune else 'ft0'
    comb = 'comb1' if combined else 'comb0'
    rel = 'rel1' if relabel else 'rel0'
    return f'{model}_{dataset}_{motion}_{eye}_sub{val_subject}_{ft}_{comb}_{rel}.csv'


def save_accumulated(rows, filename):
    """Write all rows of a full leave-one-out run to ``results/<filename>``.

    Unlike :func:`save_fold` (one file per fold under ``folds/``), this writes a
    single CSV at the results root holding every fold of one LOO run, so the
    sweep is self-contained in one file without re-running the aggregator.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})
    print(f"Saved accumulated LOO results -> {path} ({len(rows)} rows)")
    return path


def save_fold(rows, filename):
    """Write ``rows`` (list of dicts) to ``results/folds/<filename>``.

    Missing metric columns (e.g. ``degree`` for the LSTM) are written blank.
    """
    os.makedirs(FOLDS_DIR, exist_ok=True)
    path = os.path.join(FOLDS_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})
    print(f"  Saved fold result -> {path} ({len(rows)} rows)")
    return path
