"""Per-subject DoD comparison of the three EVB-Eye gaze configurations.

Grouped bar chart (one group per subject, three bars per group):
    * Single-subject regressor (per-subject calibration, degree 6)
    * Cross-subject LSTM       (zero-shot, no calibration)
    * Cross-subject regressor  (fine-tuned on a calibration fraction)

Per-subject DoD values are taken directly from the run summaries (see thesis
results section). Styling mirrors experiments_and_plots/events_eval.

Run from the repo root:
    python experiments_and_plots/model_comparison/plot_model_comparison.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

SUBJECTS = [4, 5, 6, 7, 11, 12, 15, 18, 19, 21, 22]

# Per-subject DoD (degrees) for each configuration.
DOD_SINGLE_REG = {4: 2.29, 5: 3.43, 6: 2.10, 7: 1.73, 11: 2.06, 12: 2.33,
                  15: 1.10, 18: 1.76, 19: 1.71, 21: 2.06, 22: 1.78}
DOD_CROSS_LSTM = {4: 3.88, 5: 4.30, 6: 2.14, 7: 1.92, 11: 2.13, 12: 2.01,
                  15: 1.49, 18: 2.64, 19: 2.21, 21: 2.53, 22: 2.22}
DOD_CROSS_REG = {4: 3.46, 5: 5.54, 6: 2.39, 7: 2.52, 11: 2.74, 12: 2.51,
                 15: 2.08, 18: 2.28, 19: 2.28, 21: 2.71, 22: 2.32}

# Per-subject distance error (mean, pixels) for each configuration.
DE_SINGLE_REG = {4: 42.68, 5: 66.88, 6: 40.40, 7: 33.45, 11: 39.91, 12: 46.59,
                 15: 21.72, 18: 34.29, 19: 32.91, 21: 41.43, 22: 36.62}
DE_CROSS_LSTM = {4: 72.91, 5: 81.59, 6: 42.66, 7: 37.93, 11: 41.86, 12: 40.00,
                 15: 28.75, 18: 50.39, 19: 43.78, 21: 49.98, 22: 43.83}
DE_CROSS_REG = {4: 65.40, 5: 106.68, 6: 47.08, 7: 49.46, 11: 52.95, 12: 50.41,
                15: 40.64, 18: 44.70, 19: 44.22, 21: 52.80, 22: 47.29}

# Per-configuration display label and color (drawn left-to-right within each group).
CONFIGS = [
    ('Single-subject regressor (per-subject calib.)', 'seagreen'),
    ('Cross-subject LSTM (zero-shot)', 'steelblue'),
    ('Cross-subject regressor (calibrated)', 'darkorange'),
]


def draw(series, ylabel, unit, out_path):
    """series: list of (data dict, label, color). One grouped bar chart, saved to out_path."""
    x = np.arange(len(SUBJECTS))
    n = len(series)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (data, label, color) in enumerate(series):
        vals = [data[s] for s in SUBJECTS]
        mean = float(np.mean(vals))
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=color, label=label)
        print(f'{label:<45} mean = {mean:7.2f}{unit}  std = {np.std(vals):.2f}{unit}')

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SUBJECTS], fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('Subject', fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'-> saved {out_path}\n')


def main():
    dod_series = [(d, lbl, c) for (lbl, c), d in
                  zip(CONFIGS, [DOD_SINGLE_REG, DOD_CROSS_LSTM, DOD_CROSS_REG])]
    de_series = [(d, lbl, c) for (lbl, c), d in
                 zip(CONFIGS, [DE_SINGLE_REG, DE_CROSS_LSTM, DE_CROSS_REG])]

    print('=== DoD ===')
    draw(dod_series, 'DoD [deg]', '°',
         os.path.join(PLOT_DIR, 'model_comparison_per_subject.png'))
    print('=== Distance error ===')
    draw(de_series, 'Distance error [px]', ' px',
         os.path.join(PLOT_DIR, 'model_comparison_per_subject_de.png'))


if __name__ == '__main__':
    main()
