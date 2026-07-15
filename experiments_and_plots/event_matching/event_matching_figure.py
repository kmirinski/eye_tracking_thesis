"""Generate a Figure-4-style illustration of the event-based pupil tracking.

Reproduces, on real data, the three panels of EV-Eye's Figure 4:
  (a) Candidate subset    -- near-eye frame with the events overlaid, the two
                             concentric solid circles (radii lambda1*gamma_bar and
                             lambda2*gamma_bar) that define the candidate annulus.
  (b) Before matching     -- the discrete template boundary pixels together with 
                             only the candidate events (colored by polarity).
  (c) After matching      -- the same candidate events after applying the estimated
                             translation T, now aligned with the boundary.

Run from the repo root:
    source .venv/bin/activate
    python notebooks/event_matching_figure.py
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- make src/ importable (same layout main.py relies on) --------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import (get_frame_detection_config, TemplateTrackingConfig)
from data.loaders import EyeDataset, EvEyeDataset
from pipeline.pipeline import (pupil_extraction_stage, noise_flagging_stage,
                               build_valid_mask)
from tracking import (sample_ellipse_boundary, select_candidate_events,
                      points_to_edge_matching)

# --- experiment params (edit here) -------------------------------------------------
SUBJECT = 7
EYE = "left"
DATASET = "ev_eye"          # "ev_eye" or "ebveye"
MOTION = "saccadic"
DATA_DIR = os.path.join(REPO_ROOT, "eye_data",
                        "ev_eye/raw_data" if DATASET == "ev_eye" else "ebveye")

TEMPLATE_FRAME = 186
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "event_matching_figure.png")
GAZE_AXES_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "gaze_axes_figure.png")
# ----------------------------------------------------------------------------------


def make_gaze_axes_figure(gray, center, gamma_bar, vis_cols, vis_rows,
                          vis_pos_mask, vis_neg_mask, out_path):
    """Near-eye frame + events with a 3D coordinate triad and a gaze-direction arrow
    anchored at the pupil center (no lambda circles)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(gray, cmap="gray")
    ax.scatter(vis_cols[vis_neg_mask], vis_rows[vis_neg_mask], s=14, c="red",
               edgecolors="none", zorder=2)
    ax.scatter(vis_cols[vis_pos_mask], vis_rows[vis_pos_mask], s=14, c="green",
               edgecolors="none", zorder=2)

    cx, cy = float(center[0]), float(center[1])
    L = 2.8 * gamma_bar          # axis length
    # 2D screen directions of the projected 3D axes (image y points down), chosen to
    # match the reference: green = up, blue = right, red = down-left, magenta = gaze.
    arrows = [
        ((0.00, -1.00), L,        "#22dd22"),   # Y axis  (up)    green
        ((1.00,  0.14), L,        "#2a6dff"),   # X axis  (right) blue
        ((-0.55, 0.84), L,        "#ff2a2a"),   # Z axis  (depth) red
        ((-0.72, -0.62), 0.82 * L, "#ff2ad4"),  # gaze direction  magenta
    ]
    for (ux, uy), length, color in arrows:
        ax.annotate("", xy=(cx + ux * length, cy + uy * length), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=3.5,
                                    mutation_scale=26, shrinkA=0, shrinkB=0),
                    zorder=10)

    pad_x, pad_y = 6.0 * gamma_bar, 5.0 * gamma_bar
    ax.set_xlim(cx - pad_x, cx + pad_x)
    ax.set_ylim(cy + pad_y, cy - pad_y)
    ax.axis("off")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

def load_pipeline_state():
    """Load one subject/eye and run the early pipeline stages."""
    frame_config = get_frame_detection_config(SUBJECT, EYE, DATASET)

    if DATASET == "ev_eye":
        ds = EvEyeDataset(DATA_DIR, SUBJECT, motion=MOTION, mode="np")
        eye_key = EYE
        ds.collect_data(eye=eye_key)
    else:
        ds = EyeDataset(DATA_DIR, SUBJECT, mode="stack")
        eye_key = 0 if EYE == "left" else 1
        ds.collect_data(eye=eye_key, motion=MOTION)

    pupil_centers, ellipses, screen_coords = pupil_extraction_stage(ds, frame_config)
    blink_mask = noise_flagging_stage(pupil_centers)

    from config import get_gaze_config
    gaze_config = get_gaze_config(SUBJECT)
    valid_mask = build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        skip_label_changes=False,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=getattr(ds, "alignment_gaps", None),
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    events_np = ds.load_events_sorted(eye_key)
    return ds, ellipses, valid_mask, events_np


def pick_template_idx(valid_chron, ellipses_chron):
    """Indices (chronological) of frames with a valid detection."""
    valid_idx = [i for i in range(len(valid_chron))
                 if valid_chron[i] and ellipses_chron[i] is not None]
    if not valid_idx:
        raise RuntimeError("No valid frame with an ellipse found.")
    if TEMPLATE_FRAME is not None:
        cand = [i for i in valid_idx if i >= TEMPLATE_FRAME]
        return cand[0] if cand else valid_idx[-1]
    return valid_idx[len(valid_idx) // 2]


def main():
    ds, ellipses, valid_mask, events_np = load_pipeline_state()

    frame_list_chron = ds.frame_list[::-1]
    ellipses_chron = ellipses[::-1]
    valid_chron = valid_mask[::-1]
    frame_ts = np.array([f.timestamp for f in frame_list_chron], dtype=np.int64)

    cfg = TemplateTrackingConfig()

    t_idx = pick_template_idx(valid_chron, ellipses_chron)
    template_ellipse = ellipses_chron[t_idx]
    t_ts = frame_ts[t_idx]

    boundary_Q = sample_ellipse_boundary(template_ellipse, cfg.num_boundary)
    center = np.array(template_ellipse[0], dtype=np.float64)
    gamma_bar = float(np.mean(np.linalg.norm(boundary_Q - center, axis=1)))
    
    # Create the discrete pixel boundary (stair-step integer pixels)
    boundary_Q_discrete = np.unique(np.round(boundary_Q).astype(int), axis=0)

    start = int(np.searchsorted(events_np[:, 3], t_ts, side="right"))
    win_events = []          
    candidates = []          # Store (col, row, polarity)
    last_ts = t_ts
    
    for i in range(start, len(events_np)):
        polarity, row, col, ts = events_np[i]
        win_events.append((col, row, polarity))
        dist = np.hypot(col - center[0], row - center[1])
        if cfg.lambda1 * gamma_bar < dist < cfg.lambda2 * gamma_bar:
            candidates.append((col, row, polarity))
        last_ts = ts
        if len(candidates) >= 50:
            break

    candidates = np.array(candidates, dtype=np.float64)
    if len(candidates) < 5:
        raise RuntimeError(f"Only {len(candidates)} candidates found; try another frame.")

    # Only pass (col, row) coordinates for matching
    T, residual = points_to_edge_matching(candidates[:, :2], boundary_Q,
                                           max_iter=cfg.max_icp_iter,
                                           convergence=cfg.convergence)
    print(f"Template frame idx={t_idx}, ts={t_ts}, gamma_bar={gamma_bar:.1f}px, "
          f"|candidates|={len(candidates)}, T={T}, residual={residual:.2f}px")

    # Fetch exactly 3500 events for dense visualization in Panel A only
    vis_count = min(2500, len(events_np) - start)
    vis_events_raw = events_np[start : start + vis_count]
    vis_pols = vis_events_raw[:, 0]
    vis_rows = vis_events_raw[:, 1]
    vis_cols = vis_events_raw[:, 2]
    
    vis_pos_mask = vis_pols > 0
    vis_neg_mask = ~vis_pos_mask

    # -- figure aesthetics setup --------------------------------------------------
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    img = cv2.imread(frame_list_chron[t_idx].img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Separate figure: frame + events with the gaze coordinate triad (no lambda circles).
    make_gaze_axes_figure(gray, center, gamma_bar, vis_cols, vis_rows,
                          vis_pos_mask, vis_neg_mask, GAZE_AXES_OUT_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # (a) candidate subset over the near-eye frame
    ax = axes[0]
    ax.imshow(gray, cmap="gray")
    
    # Plotting 3500 events (dense background)
    ax.scatter(vis_cols[vis_neg_mask], vis_rows[vis_neg_mask], s=16, c="red", edgecolors="none", zorder=2)
    ax.scatter(vis_cols[vis_pos_mask], vis_rows[vis_pos_mask], s=16, c="green", edgecolors="none", zorder=2)
    
    r1, r2 = cfg.lambda1 * gamma_bar, cfg.lambda2 * gamma_bar
    ax.add_patch(plt.Circle(center, r1, fill=False, color="white", ls="-", lw=2.5, zorder=5))
    ax.add_patch(plt.Circle(center, r2, fill=False, color="yellow", ls="-", lw=2.5, zorder=5))
    
    # Lambda 1 Annotation (Magenta)
    theta1 = np.radians(260) 
    dx1, dy1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    ax.arrow(center[0], center[1], dx1, dy1, 
             facecolor='white', edgecolor='white', width=0.2, head_width=3.5, head_length=3.5, 
             length_includes_head=True, zorder=10)
    ax.text(center[0] + dx1*1.1, center[1] + dy1*1.1 - 6, r'$\lambda_1\bar{\gamma}$', 
            color="white", fontsize=30, ha='center', va='bottom')

    # Lambda 2 Annotation (Yellow)
    theta2 = np.radians(20)
    dx2, dy2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
    ax.arrow(center[0], center[1], dx2, dy2, 
             facecolor='yellow', edgecolor='yellow', width=0.2, head_width=3.5, head_length=3.5, 
             length_includes_head=True, zorder=10)
    ax.text(center[0] + dx2 + 4, center[1] + dy2, r'$\lambda_2\bar{\gamma}$', 
            color="yellow", fontsize=30, ha='left', va='center')

    # Zoom out
    pad = 4.5 * gamma_bar
    ax.set_xlim(center[0] - pad, center[0] + pad)
    ax.set_ylim(center[1] + pad, center[1] - pad)
    ax.axis("off")
    # Aligned and lowered caption
    ax.text(0.5, -0.35, "(a) Candidate subset", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    # Separation logic for candidata polarity (to be used in panels B and C)
    cand_pos = candidates[:, 2] > 0
    cand_neg = ~cand_pos

    # (b) before matching: Discrete pixels (UNDER) + ONLY candidates (OVER)
    ax = axes[1]
    ax.scatter(boundary_Q_discrete[:, 0], boundary_Q_discrete[:, 1], s=20, c="dimgray", marker="o", edgecolors="none", zorder=2)
    ax.scatter(candidates[cand_neg, 0], candidates[cand_neg, 1], s=16, c="red", marker="o", edgecolors="none", zorder=5)
    ax.scatter(candidates[cand_pos, 0], candidates[cand_pos, 1], s=16, c="green", marker="o", edgecolors="none", zorder=5)
    # Aligned and lowered caption
    ax.text(0.5, -0.35, "(b) Before matching", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    # (c) after matching: Discrete pixels (UNDER) + ONLY shifted candidates (OVER)
    ax = axes[2]
    shifted = candidates[:, :2] + T
    ax.scatter(boundary_Q_discrete[:, 0], boundary_Q_discrete[:, 1], s=20, c="dimgray", marker="o", edgecolors="none", zorder=2)
    ax.scatter(shifted[cand_neg, 0], shifted[cand_neg, 1], s=16, c="red", marker="o", edgecolors="none", zorder=5)
    ax.scatter(shifted[cand_pos, 0], shifted[cand_pos, 1], s=16, c="green", marker="o", edgecolors="none", zorder=5)
    # Aligned and lowered caption
    ax.text(0.5, -0.35, "(c) After matching", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    # Dynamic limits for Panels B & C centered around the actual pupil
    cx, cy = int(np.round(center[0])), int(np.round(center[1]))
    span = 25
    
    for ax_idx in [1, 2]:
        ax = axes[ax_idx]
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy + span, cy - span)  # Inverted Y
        ax.set_aspect("equal")
        ax.set_facecolor('white') # Ensure background is white
        
        ticks_x = [cx - span, cx, cx + span]
        ticks_y = [cy - span, cy, cy + span]
        
        ax.set_xticks(ticks_x)
        ax.set_yticks(ticks_y)
        ax.set_xticklabels(ticks_x, fontsize=14)
        ax.set_yticklabels(ticks_y, fontsize=14)
        
        # Apply padding specifically to prevent corner tick overlaps
        ax.tick_params(direction='in', length=5, width=1.5, top=True, right=True)
        ax.tick_params(axis='x', pad=12) # Pushes X ticks down, clearing Y ticks
        ax.tick_params(axis='y', pad=4)  # standard spacing for Y ticks
        
        ax.set_xlabel(r'$\mathbf{x}$', fontsize=16, labelpad=8)
        ax.set_ylabel(r'$\mathbf{y}$', fontsize=16, labelpad=8)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Added bottom spacing to make room for lowered captions
    plt.subplots_adjust(wspace=0.35, bottom=0.30)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUT_PATH}")

if __name__ == "__main__":
    main()