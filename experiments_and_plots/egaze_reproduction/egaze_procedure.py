#!/usr/bin/env python3
"""Render the E-Gaze pupil-extraction procedure as a step-by-step figure (for the report).

This reproduces the eye-tracking block of:

    N. Li, M. Chang, A. Raychowdhury, "E-Gaze: Gaze Estimation With Event Camera",
    IEEE TPAMI 46(7), 2024.  (papers/E-Gaze_Gaze_Estimation_With_Event_Camera.pdf)

It runs the reproduction code that lives in `src/processing/` (the unused KDE alternative
pipeline) on a single 2000-event set and dumps one PNG per stage:

    a_event_set.png        2000 consecutive events scattered (ON = green, OFF = red),
                           cf. E-Gaze Fig. 1(b) / Fig. 3 — the raw input to the procedure.
    b_polarity_images.png  the 3 count images generated from one set: positive-only,
                           negative-only, and combined (Sec. IV-A "Accumulating Events").
    c_filtering_masks.png  image-processing + morphology segmentation (Sec. IV / Fig. 4):
                           noise mask, eyelid & glint mask, eyelash mask, and the
                           pupil & iris events left after removing all three.
    d_kde_center.png       non-parametric pupil-center finding — the donut-kernel KDE
                           density (event image convolved with an annulus) and its argmax.
    e_pupil_ellipse.png    the segmented pupil events (circular ROI around the KDE center)
                           and the fitted pupil ellipse — the feature passed to the RNN.
    panels_grid.png        all of the above assembled into one overview figure.

Run from the repo root (venv active):
    python experiments_and_plots/egaze_reproduction/egaze_procedure.py
    python experiments_and_plots/egaze_reproduction/egaze_procedure.py --subject 22 --eye 0
    python experiments_and_plots/egaze_reproduction/egaze_procedure.py --subject 22 --set_index 1486
"""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'src'))

from config import DATASET_PATHS                                               # noqa: E402
from data.loaders import EyeDataset                                           # noqa: E402
from processing.preprocessing import accumulate_events, event_to_image         # noqa: E402
from processing.filtering import (generate_noise_mask, generate_eyelid_glint_mask,  # noqa: E402
                                  generate_eyelash_mask, generate_pupil_iris_mask,
                                  apply_mask)
from scipy.signal import fftconvolve                                            # noqa: E402

H, W = 260, 346          # DAVIS event-camera resolution (rows, cols)
ON_COLOR = '#00b04f'     # ON / positive events (polarity 1) — green
OFF_COLOR = '#d6263a'    # OFF / negative events (polarity 0) — red

# Donut (annulus) kernel geometry, in pixels, matched to the pupil-boundary ring. The
# pupil events form a small ring; convolving the event image with a donut kernel of that
# radius gives a density that peaks at the ring centre = the pupil centre.
DONUT_R_IN = 5
DONUT_R_OUT = 16
SEG_RADIUS = 20          # radius of the circular ROI used to segment the pupil events


def _donut_kernel(r_in, r_out):
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    return ((d >= r_in) & (d <= r_out)).astype(np.float32)


def donut_kde_center(img, r_in=DONUT_R_IN, r_out=DONUT_R_OUT):
    """Donut-kernel pupil-centre finder (E-Gaze Sec. IV "KDE with donut kernel").

    A donut (annulus) kernel is the natural kernel for locating the centre of a ring of
    events: KDE with that kernel = convolving the event image with the annulus, whose
    response peaks where a pupil-radius ring of events is centred. This replaces the
    difference-of-Gaussians in src/processing/pupil_finding.locate_pupil_center_kde, which
    peaks on dense ring *regions* (e.g. the iris/eyelash arc) rather than the ring centre,
    and additionally had a grid-ordering/reshape bug (density reshaped (H,W) instead of
    (W,H).T) that scrambled it into diagonal bands.
    """
    if not (img > 0).any():
        raise ValueError("No events found in the image")
    density = fftconvolve((img > 0).astype(np.float32), _donut_kernel(r_in, r_out), mode='same')
    density = np.maximum(density, 0.0)
    cy, cx = np.unravel_index(np.argmax(density), density.shape)
    return int(cx), int(cy), density


# ---------------------------------------------------------------------------------------
# Procedure: this mirrors processing.filtering.generate_and_apply_masks +
# processing.pupil_finding, but keeps every intermediate so we can plot each stage.
# ---------------------------------------------------------------------------------------
def compute_masks(event_set):
    """Cheap part of the procedure: the three count images + segmentation masks.

    event_set: (n_events, 4) = [polarity, row, col, timestamp]. No KDE here so this can
    be run over many candidate sets to find the active ones before the expensive step.
    """
    # --- Sec. IV-A: three images from one event set (positive / negative / combined) ---
    neg_set = event_set[event_set[:, 0] == 0]
    pos_set = event_set[event_set[:, 0] == 1]
    img_neg = event_to_image(neg_set)
    img_pos = event_to_image(pos_set)
    img = event_to_image(event_set)

    # --- Sec. IV: eye-part segmentation by image processing + morphology -> Fig. 4 ---
    noise_mask = generate_noise_mask(img_neg, img_pos)
    eyelid_glint_mask = generate_eyelid_glint_mask(img_neg, img_pos, noise_mask)
    eyelash_mask = generate_eyelash_mask(img, eyelid_glint_mask)
    pupil_iris_mask = generate_pupil_iris_mask(noise_mask, eyelid_glint_mask, eyelash_mask)
    pupil_iris = apply_mask(img, pupil_iris_mask, keep_masked=True)

    return dict(event_set=event_set, neg_set=neg_set, pos_set=pos_set,
                img_neg=img_neg, img_pos=img_pos, img=img,
                noise_mask=noise_mask, eyelid_glint_mask=eyelid_glint_mask,
                eyelash_mask=eyelash_mask, pupil_iris_mask=pupil_iris_mask,
                pupil_iris=pupil_iris, n_pupil_iris_events=int((pupil_iris > 0).sum()))


def finish_procedure(res):
    """Expensive part: donut-kernel pupil centre, circular segmentation, ellipse fit."""
    # --- pupil-center finding: donut-kernel KDE over the pupil/iris events ---
    center_x, center_y, density = donut_kde_center(res['pupil_iris'])

    # --- pupil segmentation (circular ROI) + ellipse fit (the feature for the RNN) ---
    yy, xx = np.indices(res['pupil_iris'].shape)
    pupil_mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2 <= SEG_RADIUS ** 2)
    pupil_events = (res['pupil_iris'] > 0) & pupil_mask
    rows, cols = np.nonzero(pupil_events)
    ellipse = None
    if len(rows) >= 5:
        pts = np.column_stack([cols, rows]).astype(np.float32).reshape(-1, 1, 2)
        try:
            ellipse = cv2.fitEllipse(pts)
        except cv2.error:
            ellipse = None

    res.update(center=(center_x, center_y), density=density, pupil_mask=pupil_mask,
               ellipse=ellipse, n_pupil_events=int(len(rows)))
    return res


def run_procedure(event_set):
    """Full E-Gaze eye-tracking block on one 2000-event set (masks + KDE + ellipse)."""
    return finish_procedure(compute_masks(event_set))


def score_set(res, margin=45):
    """How 'pupil-like' is the extraction? Used to auto-pick a clean set for the figure.

    Favours a compact, well-populated pupil whose centre sits away from the FoV border
    (a centre hugging the edge usually means a cut-off eye or edge noise, not a pupil).
    """
    cx, cy = res['center']
    if cx < margin or cx > W - margin or cy < margin or cy > H - margin:
        return 0.0
    if res['ellipse'] is None:
        return 0.0
    (_, _), (ew, eh), _ = res['ellipse']
    axis = max(ew, eh)
    if axis < 8 or axis > 120:          # implausible pupil size
        return 0.0
    # many segmented events packed into a small ellipse => clean, compact pupil
    return res['n_pupil_events'] / axis


# ---------------------------------------------------------------------------------------
# Per-stage plots
# ---------------------------------------------------------------------------------------
def _eye_axes(ax, title):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)        # row 0 at the top (image orientation)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_polarity(ax, event_set, s=4, alpha=0.6):
    pol, row, col = event_set[:, 0], event_set[:, 1], event_set[:, 2]
    neg, pos = pol == 0, pol == 1
    ax.scatter(col[neg], row[neg], c=OFF_COLOR, s=s, alpha=alpha, edgecolors='none',
               label='OFF (negative)')
    ax.scatter(col[pos], row[pos], c=ON_COLOR, s=s, alpha=alpha, edgecolors='none',
               label='ON (positive)')


def show_count_image(ax, img, cmap, title):
    vmax = max(1, np.percentile(img[img > 0], 99)) if (img > 0).any() else 1
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    _eye_axes(ax, title)


def show_mask(ax, mask, title, color):
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask.astype(bool)] = color
    ax.imshow(np.ones((*mask.shape, 3)), interpolation='nearest')  # white bg
    ax.imshow(rgba, interpolation='nearest')
    _eye_axes(ax, title)


def plot_a_event_set(res, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.4), dpi=160)
    scatter_polarity(ax, res['event_set'])
    _eye_axes(ax, f"(a) {len(res['event_set'])} accumulated events")
    ax.legend(loc='lower right', framealpha=0.9, markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_b_polarity_images(res, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=160)
    show_count_image(axes[0], res['img_pos'], 'Greens', 'Positive (ON) image')
    show_count_image(axes[1], res['img_neg'], 'Reds', 'Negative (OFF) image')
    show_count_image(axes[2], res['img'], 'gray_r', 'Combined image')
    fig.suptitle('(b) Three count images generated from one event set', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_c_filtering_masks(res, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=160)
    show_mask(axes[0], res['noise_mask'], 'Noise mask', (0.4, 0.4, 0.4, 1))
    show_mask(axes[1], res['eyelid_glint_mask'], 'Eyelid & glint mask', (0.1, 0.7, 0.2, 1))
    show_mask(axes[2], res['eyelash_mask'], 'Eyelash mask', (0.2, 0.4, 0.9, 1))
    show_count_image(axes[3], res['pupil_iris'], 'magma', 'Pupil & iris events (kept)')
    fig.suptitle('(c) Eye-part segmentation: image processing + morphology (cf. E-Gaze Fig. 4)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_d_kde_center(res, out_path):
    cx, cy = res['center']
    fig, ax = plt.subplots(figsize=(7, 5.4), dpi=160)
    im = ax.imshow(res['density'], cmap='inferno', interpolation='nearest')
    ax.scatter([cx], [cy], marker='+', s=240, c='cyan', linewidths=2.5,
               label=f'pupil center ({cx}, {cy})')
    _eye_axes(ax, '(d) Donut-kernel KDE density & pupil center')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='donut density')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


def _draw_ellipse(ax, ellipse, color='#00e0ff', lw=2.2):
    (ecx, ecy), (ew, eh), ang = ellipse
    t = np.linspace(0, 2 * np.pi, 200)
    a, b, th = ew / 2.0, eh / 2.0, np.deg2rad(ang)
    ex, ey = a * np.cos(t), b * np.sin(t)
    ax.plot(ecx + ex * np.cos(th) - ey * np.sin(th),
            ecy + ex * np.sin(th) + ey * np.cos(th), color=color, lw=lw,
            label='fitted pupil ellipse')


def plot_e_pupil_ellipse(res, out_path):
    cx, cy = res['center']
    fig, ax = plt.subplots(figsize=(7, 5.4), dpi=160)
    # all pupil/iris events in grey, the segmented pupil events emphasised
    rows, cols = np.nonzero(res['pupil_iris'] > 0)
    ax.scatter(cols, rows, c='0.75', s=4, alpha=0.6, edgecolors='none', label='pupil & iris events')
    pupil_events = (res['pupil_iris'] > 0) & res['pupil_mask'].astype(bool)
    prows, pcols = np.nonzero(pupil_events)
    ax.scatter(pcols, prows, c='#1f77b4', s=6, alpha=0.85, edgecolors='none',
               label='segmented pupil events')
    ax.add_patch(Circle((cx, cy), SEG_RADIUS, fill=False, ls='--',
                        ec='0.3', lw=1.2))
    ax.scatter([cx], [cy], marker='+', s=200, c='black', linewidths=2)
    if res['ellipse'] is not None:
        _draw_ellipse(ax, res['ellipse'])
    _eye_axes(ax, '(e) Pupil segmentation & fitted ellipse')
    ax.legend(loc='lower right', framealpha=0.9, markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_grid(res, out_path):
    """All stages in one 3x3 overview figure for the report."""
    fig = plt.figure(figsize=(15, 13), dpi=150)
    gs = fig.add_gridspec(3, 4, hspace=0.28, wspace=0.18)

    ax = fig.add_subplot(gs[0, 0:2]); scatter_polarity(ax, res['event_set'])
    _eye_axes(ax, f"(a) {len(res['event_set'])} accumulated events")
    ax.legend(loc='lower right', markerscale=2, fontsize=8)

    show_count_image(fig.add_subplot(gs[0, 2]), res['img_pos'], 'Greens', '(b) Positive image')
    show_count_image(fig.add_subplot(gs[0, 3]), res['img_neg'], 'Reds', '(b) Negative image')

    show_mask(fig.add_subplot(gs[1, 0]), res['noise_mask'], '(c) Noise mask', (0.4, 0.4, 0.4, 1))
    show_mask(fig.add_subplot(gs[1, 1]), res['eyelid_glint_mask'], '(c) Eyelid & glint', (0.1, 0.7, 0.2, 1))
    show_mask(fig.add_subplot(gs[1, 2]), res['eyelash_mask'], '(c) Eyelash mask', (0.2, 0.4, 0.9, 1))
    show_count_image(fig.add_subplot(gs[1, 3]), res['pupil_iris'], 'magma', '(c) Pupil & iris events')

    cx, cy = res['center']
    axd = fig.add_subplot(gs[2, 0:2])
    axd.imshow(res['density'], cmap='inferno', interpolation='nearest')
    axd.scatter([cx], [cy], marker='+', s=220, c='cyan', linewidths=2.5)
    _eye_axes(axd, '(d) Donut-KDE density & pupil center')

    axe = fig.add_subplot(gs[2, 2:4])
    rows, cols = np.nonzero(res['pupil_iris'] > 0)
    axe.scatter(cols, rows, c='0.75', s=4, alpha=0.6, edgecolors='none')
    pupil_events = (res['pupil_iris'] > 0) & res['pupil_mask'].astype(bool)
    prows, pcols = np.nonzero(pupil_events)
    axe.scatter(pcols, prows, c='#1f77b4', s=6, alpha=0.85, edgecolors='none')
    axe.add_patch(Circle((cx, cy), SEG_RADIUS, fill=False, ls='--', ec='0.3', lw=1.2))
    if res['ellipse'] is not None:
        _draw_ellipse(axe, res['ellipse'])
    _eye_axes(axe, '(e) Pupil segmentation & fitted ellipse')

    fig.suptitle('E-Gaze pupil-feature extraction from a single 2000-event set', fontsize=15, y=0.995)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', type=int, default=1)
    ap.add_argument('--eye', type=int, default=0, choices=[0, 1], help='0 = left, 1 = right')
    ap.add_argument('--n_events', type=int, default=2000, help='events per accumulated set')
    ap.add_argument('--set_index', type=int, default=None,
                    help='which event set to render; default = auto-pick the cleanest')
    ap.add_argument('--search_start', type=int, default=None,
                    help='first set index of the auto-pick scan block (default: 10%% in)')
    ap.add_argument('--search_count', type=int, default=600,
                    help='number of consecutive sets scanned (pass 1) during auto-pick')
    ap.add_argument('--top_k', type=int, default=30,
                    help='how many of the most-active sets get the full KDE pipeline (pass 2)')
    ap.add_argument('--out_dir', default=PLOT_DIR)
    opt = ap.parse_args()

    ds = EyeDataset(DATASET_PATHS['ebveye'], opt.subject, mode='np')
    events = ds.load_events_sorted(opt.eye)               # (N, 4) [pol, row, col, ts]
    event_sets = accumulate_events(events, n_events=opt.n_events)
    print(f"Subject {opt.subject}, eye {opt.eye}: {len(events)} events "
          f"-> {len(event_sets)} sets of {opt.n_events}")

    if opt.set_index is not None:
        res = run_procedure(event_sets[opt.set_index])
        print(f"Using set {opt.set_index} (score {score_set(res):.2f})")
    else:
        # Two-pass auto-pick. Pass 1 (cheap, no KDE): scan a consecutive block and rank
        # sets by how many pupil/iris events survive the masking — active sets where the
        # eye is clearly visible. Pass 2: run the full KDE pipeline on the top candidates
        # only, and keep the one with the cleanest, best-centred pupil.
        n = len(event_sets)
        lo = opt.search_start if opt.search_start is not None else n // 10
        lo = max(0, min(lo, n - 1))
        hi = min(lo + opt.search_count, n)
        pass1 = []
        for i in range(lo, hi):
            m = compute_masks(event_sets[i])
            if m['n_pupil_iris_events'] > 0:
                pass1.append((m['n_pupil_iris_events'], i, m))
        pass1.sort(key=lambda t: -t[0])
        print(f"Pass 1: {len(pass1)} active sets in [{lo}, {hi}); "
              f"running KDE on top {min(opt.top_k, len(pass1))}")

        best, best_res, best_idx = -1.0, None, None
        fb, fb_res, fb_idx = -1, None, None
        for _, i, m in pass1[:opt.top_k]:
            try:
                r = finish_procedure(m)
            except ValueError:
                continue
            if r['n_pupil_events'] > fb:
                fb, fb_res, fb_idx = r['n_pupil_events'], r, i
            s = score_set(r)
            if s > best:
                best, best_res, best_idx = s, r, i
        if best_res is None:               # nothing passed the quality gate; use fallback
            best_res, best_idx, best = fb_res, fb_idx, float(fb)
        if best_res is None:
            raise RuntimeError("No usable event set found; pass --set_index explicitly.")
        res = best_res
        print(f"Auto-picked set {best_idx} (score {best:.2f}); "
              f"override with --set_index {best_idx}")

    os.makedirs(opt.out_dir, exist_ok=True)
    plot_a_event_set(res, os.path.join(opt.out_dir, 'a_event_set.png'))
    plot_b_polarity_images(res, os.path.join(opt.out_dir, 'b_polarity_images.png'))
    plot_c_filtering_masks(res, os.path.join(opt.out_dir, 'c_filtering_masks.png'))
    plot_d_kde_center(res, os.path.join(opt.out_dir, 'd_kde_center.png'))
    plot_e_pupil_ellipse(res, os.path.join(opt.out_dir, 'e_pupil_ellipse.png'))
    plot_grid(res, os.path.join(opt.out_dir, 'panels_grid.png'))


if __name__ == '__main__':
    main()
