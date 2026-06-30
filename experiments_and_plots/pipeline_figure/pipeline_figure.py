#!/usr/bin/env python3
"""Render the building-block plots for a pipeline-overview figure (cf. EV-Eye Fig. 3).

This is a standalone visualisation script (it does NOT touch the pipeline). It runs the
normal single-subject stages — frame pupil extraction + template-based event tracking —
on a short window of three consecutive frames and produces three transparent PNGs that
you can assemble into a pipeline diagram in external software:

    1_frames_events.png    3 frames as image planes along the time (T) axis with the raw
                           DVS events (red = ON, blue = OFF) scattered in between.
    2_events_only.png      The same event cloud on its own (the "template-based tracking"
                           input), no frames.
    3_frames_centers.png   The 3 frames plus the extracted pupil centers threaded between
                           them — frame ellipse centers (large) + event-tracked centers
                           (small), connected as the trajectory C = {c_1, ..., c_L}.

Run from the repo root:
    python experiments_and_plots/pipeline_figure/pipeline_figure.py
    python experiments_and_plots/pipeline_figure/pipeline_figure.py --subject 7 --start_frame 1200
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'src'))

from data.loaders import EyeDataset, EvEyeDataset                            # noqa: E402
from config import (DATASET_PATHS, get_frame_detection_config,               # noqa: E402
                    get_gaze_config, TemplateTrackingConfig)
from pipeline.pipeline import (pupil_extraction_stage, noise_flagging_stage,  # noqa: E402
                               build_valid_mask, template_tracking_stage)

ON_COLOR = '#00b04f'    # ON / positive events (polarity 1) — green
OFF_COLOR = '#d6263a'   # OFF / negative events (polarity 0) — red


# ---------------------------------------------------------------------------------------
# Data loading (mirrors the single-subject pipeline path used elsewhere)
# ---------------------------------------------------------------------------------------
def load_subject(subject, eye, dataset, motion):
    data_dir = DATASET_PATHS[dataset]
    frame_config = get_frame_detection_config(subject, eye, dataset)
    gaze_config = get_gaze_config(subject, dataset)

    if dataset == 'ev_eye':
        ds = EvEyeDataset(data_dir, subject, motion=motion, mode='np')
        eye_key = eye
        ds.collect_data(eye=eye_key)
    else:
        ds = EyeDataset(data_dir, subject, mode='stack')
        eye_key = 0 if eye == 'left' else 1
        ds.collect_data(eye=eye_key, motion=motion)

    pupil_centers, ellipses, screen_coords = pupil_extraction_stage(ds, frame_config)
    blink_mask = noise_flagging_stage(pupil_centers)
    skip_label_changes = (dataset == 'ebveye') and (motion == 'saccadic')
    valid_mask = build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=getattr(ds, 'alignment_gaps', None),
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )
    events_np = ds.load_events_sorted(eye_key)
    event_samples = template_tracking_stage(
        events_np, ds.frame_list, ellipses, screen_coords, valid_mask,
        TemplateTrackingConfig())
    return ds, pupil_centers, ellipses, valid_mask, events_np, event_samples


# ---------------------------------------------------------------------------------------
# Window selection: pick 3 consecutive valid frames (chronological) with the largest
# pupil displacement (i.e. a saccade) so the event cloud is rich.
# ---------------------------------------------------------------------------------------
def select_window(ds, pupil_centers, ellipses, valid_mask, start_frame=None):
    frames_chron = ds.frame_list[::-1]
    pc_chron = pupil_centers[::-1]
    ell_chron = ellipses[::-1]
    valid_chron = valid_mask[::-1]
    n = len(frames_chron)

    def ok(i):
        return (0 <= i < n and ell_chron[i] is not None and valid_chron[i]
                and not np.all(pc_chron[i] == -1))

    if start_frame is not None:
        # use the first triple of consecutive good frames at/after start_frame
        i = start_frame
        while i + 2 < n and not (ok(i) and ok(i + 1) and ok(i + 2)):
            i += 1
        best_i = i
    else:
        best_i, best_score = None, -1.0
        for i in range(n - 2):
            if not (ok(i) and ok(i + 1) and ok(i + 2)):
                continue
            d = (np.linalg.norm(pc_chron[i + 1] - pc_chron[i]) +
                 np.linalg.norm(pc_chron[i + 2] - pc_chron[i + 1]))
            if d > best_score:
                best_score, best_i = d, i
    if best_i is None:
        raise RuntimeError("No triple of consecutive valid frames with ellipses found.")

    idxs = [best_i, best_i + 1, best_i + 2]
    frames = [frames_chron[i] for i in idxs]
    centers = [pc_chron[i] for i in idxs]              # (col, row)
    frame_ellipses = [ell_chron[i] for i in idxs]      # ((cx,cy),(w,h),angle)
    ts = [int(frames_chron[i].timestamp) for i in idxs]
    print(f"Selected chron frames {idxs} | ts {ts} | "
          f"displacement {np.linalg.norm(centers[2]-centers[0]):.1f}px")
    return frames, centers, frame_ellipses, ts


def crop_to_content(path):
    """Crop a transparent PNG to the bounding box of its non-transparent pixels."""
    img = Image.open(path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    bbox = img.split()[-1].getbbox()   # bbox of non-zero alpha
    if bbox:
        img.crop(bbox).save(path)


def load_gray(path, downsample=2):
    img = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    if downsample > 1:
        img = img[::downsample, ::downsample]
    return img


# ---------------------------------------------------------------------------------------
# 3D drawing helpers. Coordinate mapping: plot-X = time (frame plane position),
# plot-Y = image column, plot-Z = (H - image row) so the image is upright.
# ---------------------------------------------------------------------------------------
def add_frame_plane(ax, img, x_pos, W, H, border=True):
    nr, nc = img.shape
    cols = np.linspace(0, W, nc)
    rows = np.linspace(0, H, nr)
    YY, ROWS = np.meshgrid(cols, rows)
    ZZ = H - ROWS
    XX = np.full_like(YY, x_pos)
    rgba = plt.cm.gray(img)
    ax.plot_surface(XX, YY, ZZ, facecolors=rgba, rstride=1, cstride=1,
                    shade=False, antialiased=False, linewidth=0, zorder=1)
    if border:
        bx = [x_pos] * 5
        by = [0, W, W, 0, 0]
        bz = [0, 0, H, H, 0]
        ax.plot(bx, by, bz, color='0.35', lw=1.2, zorder=2)


def ellipse_boundary(ellipse, n=160):
    """Return (col, row) boundary points of a cv2-style ellipse ((cx,cy),(w,h),angle_deg)."""
    (cx, cy), (w, h), ang = ellipse
    a, b = w / 2.0, h / 2.0
    th = np.deg2rad(ang)
    t = np.linspace(0, 2 * np.pi, n)
    ex, ey = a * np.cos(t), b * np.sin(t)
    col = cx + ex * np.cos(th) - ey * np.sin(th)
    row = cy + ex * np.sin(th) + ey * np.cos(th)
    return col, row


def draw_ellipse_in_plane(ax, ellipse, x_pos, H, color, lw=2.0, zorder=7):
    col, row = ellipse_boundary(ellipse)
    ax.plot(np.full_like(col, x_pos), col, H - row, color=color, lw=lw, zorder=zorder)


def scatter_events(ax, ev, t_positions, frame_ts, W, H, max_pts=6000, size=2.0):
    """ev: (N,4) [pol,row,col,ts]; map ts -> plot-X by interpolation over the frame ts."""
    if len(ev) == 0:
        return
    if len(ev) > max_pts:
        sel = np.random.default_rng(0).choice(len(ev), max_pts, replace=False)
        ev = ev[sel]
    pol, row, col, ts = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3].astype(np.float64)
    x = np.interp(ts, frame_ts, t_positions)
    z = H - row
    colors = np.where(pol == 1, ON_COLOR, OFF_COLOR)
    ax.scatter(x, col, z, c=colors, s=size, depthshade=False, alpha=0.75,
               edgecolors='none', zorder=3)


def style_3d(ax, x_positions, W, H, depth_visual=380, ylim=None, zlim=None,
             elev=16, azim=-66, box_aspect=None):
    # box_aspect sets the *visual* proportions of the three axes regardless of data
    # range; the time axis has a tiny data span (0..2) so give it an explicit visual
    # length so the three frame planes separate cleanly along T. Passing box_aspect
    # directly lets a caller stretch an axis (spread the points) independently of lims.
    span = max(x_positions) - min(x_positions)
    ylim = ylim or (0, W)
    zlim = zlim or (0, H)
    if box_aspect is None:
        box_aspect = (depth_visual, ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_box_aspect(box_aspect)
    ax.set_xlim(min(x_positions) - 0.08 * span, max(x_positions) + 0.08 * span)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def get_event_window(events_np, t0, t2):
    lo = int(np.searchsorted(events_np[:, 3], t0, side='left'))
    hi = int(np.searchsorted(events_np[:, 3], t2, side='right'))
    return events_np[lo:hi]


# ---------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------
def plot_single_frame(frame, ellipse, out_path, with_pupil, W, H, downsample):
    """Render a single frame as a tilted 3D image plane (same look as the frames in the
    3-subsequent-frames plot), optionally with the fitted pupil ellipse overlaid."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    add_frame_plane(ax, load_gray(frame.img, downsample), 0.0, W, H)
    if with_pupil and ellipse is not None:
        draw_ellipse_in_plane(ax, ellipse, 0.0, H, color='#00e0ff', lw=2.0)
        ax.scatter([0.0], [ellipse[0][0]], [H - ellipse[0][1]], c='#00e0ff', s=18,
                   depthshade=False, zorder=8)
    # Same projection geometry as the multi-frame plot so the plane is skewed identically.
    ax.set_box_aspect((380, W, H))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.view_init(elev=16, azim=-66)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    crop_to_content(out_path)
    print(f"  -> {out_path}")


def plot_frames_events(frames, ts, ev_window, W, H, out_path, downsample):
    x_positions = np.linspace(0, 2.0, 3)
    frame_ts = np.array(ts, dtype=np.float64)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    for fr, xp in zip(frames, x_positions):
        add_frame_plane(ax, load_gray(fr.img, downsample), xp, W, H)
    scatter_events(ax, ev_window, x_positions, frame_ts, W, H, size=1.3)
    style_3d(ax, x_positions, W, H)
    fig.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    crop_to_content(out_path)
    print(f"  -> {out_path}  ({len(ev_window)} events in window)")


def plot_events_only(ts, ev_window, W, H, out_path):
    x_positions = np.linspace(0, 2.0, 3)
    frame_ts = np.array(ts, dtype=np.float64)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter_events(ax, ev_window, x_positions, frame_ts, W, H, max_pts=9000, size=2.6)

    # Zoom into the actual event cloud extent (with a small margin) so it fills the
    # frame, and stretch the time axis so the cloud spreads out along T.
    col = ev_window[:, 2]
    z = H - ev_window[:, 1]
    my = 0.08 * (col.max() - col.min() + 1)
    mz = 0.08 * (z.max() - z.min() + 1)
    ylim = (col.min() - my, col.max() + my)
    zlim = (z.min() - mz, z.max() + mz)
    # Previous 3D look (elev=16, azim=-66). Limits stay tight to the cloud; the box
    # aspect stretches the time (x) and image-y (z) axes so the events spread out more
    # in both visible directions while the column (y) axis stays compressed.
    span_y = ylim[1] - ylim[0]
    span_z = zlim[1] - zlim[0]
    style_3d(ax, x_positions, W, H, ylim=ylim, zlim=zlim, elev=16, azim=-66,
             box_aspect=(2.2 * span_y, span_y, 2.6 * span_z))
    fig.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    crop_to_content(out_path)
    print(f"  -> {out_path}")


def plot_frames_centers(frames, frame_centers, frame_ellipses, ts, event_samples,
                        W, H, out_path, downsample):
    x_positions = np.linspace(0, 2.0, 3)
    frame_ts = np.array(ts, dtype=np.float64)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    for fr, xp in zip(frames, x_positions):
        add_frame_plane(ax, load_gray(fr.img, downsample), xp, W, H)

    # Fitted pupil ellipse drawn on each frame plane.
    for ell, xp in zip(frame_ellipses, x_positions):
        if ell is not None:
            draw_ellipse_in_plane(ax, ell, xp, H, color='#00e0ff', lw=1.1, zorder=7)

    # Event-tracked ellipses within the window (floating between the frames). There are
    # many, so only draw every 3rd to keep it readable; the frame ellipses always stay.
    ev = [s for s in event_samples if ts[0] <= s['timestamp'] <= ts[2]]
    for s in ev[::3]:
        xp = float(np.interp(s['timestamp'], frame_ts, x_positions))
        draw_ellipse_in_plane(ax, s['ellipse'], xp, H, color='#ff8c00', lw=0.7, zorder=6)

    # Center markers C = {c_1, ..., c_L}: frame centers + event centers (no connecting line).
    pts = [(t, c[0], c[1], 'frame') for c, t in zip(frame_centers, ts)]
    pts += [(s['timestamp'], s['ellipse'][0][0], s['ellipse'][0][1], 'event') for s in ev]
    pts.sort(key=lambda p: p[0])

    xs = np.interp([p[0] for p in pts], frame_ts, x_positions)
    ys = np.array([p[1] for p in pts])          # col
    zs = H - np.array([p[2] for p in pts])      # row -> upright

    ev_mask = np.array([p[3] == 'event' for p in pts])
    if ev_mask.any():
        ax.scatter(xs[ev_mask], ys[ev_mask], zs[ev_mask], c='#ff8c00', s=34,
                   depthshade=False, edgecolors='black', linewidths=0.5, zorder=9)
    fr_mask = ~ev_mask
    ax.scatter(xs[fr_mask], ys[fr_mask], zs[fr_mask], c='#00b894', s=80,
               depthshade=False, edgecolors='black', linewidths=0.8, zorder=10)

    style_3d(ax, x_positions, W, H)
    fig.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    crop_to_content(out_path)
    print(f"  -> {out_path}  ({int(ev_mask.sum())} event centers in window)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', type=int, default=7)
    ap.add_argument('--eye', default='left', choices=['left', 'right'])
    ap.add_argument('--dataset', default='ev_eye', choices=list(DATASET_PATHS.keys()))
    ap.add_argument('--motion', default='saccadic', choices=['saccadic', 'pursuit'])
    ap.add_argument('--start_frame', type=int, default=None,
                    help='chron frame index to start the 3-frame window; default = '
                         'largest-saccade triple')
    ap.add_argument('--downsample', type=int, default=2,
                    help='image downsample factor for the frame planes')
    ap.add_argument('--out_dir', default=PLOT_DIR)
    ap.add_argument('--from_cache', action='store_true',
                    help='reuse the cached window data instead of reloading + tracking')
    opt = ap.parse_args()

    W, H = 346, 260   # Davis frame resolution
    cache_path = os.path.join(opt.out_dir,
                              f'_cache_s{opt.subject}_{opt.eye}_{opt.motion}.npz')

    have_cache = opt.from_cache and os.path.exists(cache_path)
    if have_cache:
        cz = np.load(cache_path, allow_pickle=True)
        have_cache = 'frame_ellipses' in cz   # old caches lack the fitted ellipses
        if not have_cache:
            print("Cache is missing 'frame_ellipses'; rebuilding from scratch.")

    if have_cache:
        print(f"Loading cached window from {cache_path}")
        frame_paths = list(cz['frame_paths'])
        centers = list(cz['centers'])
        frame_ellipses = list(cz['frame_ellipses'])
        ts = list(cz['ts'])
        ev_window = cz['ev_window']
        event_samples = list(cz['event_samples'])
    else:
        ds, pupil_centers, ellipses, valid_mask, events_np, event_samples = load_subject(
            opt.subject, opt.eye, opt.dataset, opt.motion)
        frames_sel, centers, frame_ellipses, ts = select_window(
            ds, pupil_centers, ellipses, valid_mask, start_frame=opt.start_frame)
        frame_paths = [fr.img for fr in frames_sel]
        ev_window = get_event_window(events_np, ts[0], ts[2])
        np.savez(cache_path, frame_paths=np.array(frame_paths, dtype=object),
                 centers=np.array(centers),
                 frame_ellipses=np.array(frame_ellipses, dtype=object),
                 ts=np.array(ts), ev_window=ev_window,
                 event_samples=np.array(event_samples, dtype=object))
        print(f"Cached window to {cache_path}")

    Frame = __import__('collections').namedtuple('Frame', 'img')
    frames = [Frame(p) for p in frame_paths]

    os.makedirs(opt.out_dir, exist_ok=True)
    tag = f"s{opt.subject}_{opt.eye}_{opt.motion}"
    plot_single_frame(frames[0], frame_ellipses[0],
                      os.path.join(opt.out_dir, f'{tag}_0a_first_frame.png'),
                      with_pupil=False, W=W, H=H, downsample=opt.downsample)
    plot_single_frame(frames[0], frame_ellipses[0],
                      os.path.join(opt.out_dir, f'{tag}_0b_first_frame_pupil.png'),
                      with_pupil=True, W=W, H=H, downsample=opt.downsample)
    plot_frames_events(frames, ts, ev_window, W, H,
                       os.path.join(opt.out_dir, f'{tag}_1_frames_events.png'), opt.downsample)
    plot_events_only(ts, ev_window, W, H,
                     os.path.join(opt.out_dir, f'{tag}_2_events_only.png'))
    plot_frames_centers(frames, centers, frame_ellipses, ts, event_samples, W, H,
                        os.path.join(opt.out_dir, f'{tag}_3_frames_centers.png'), opt.downsample)


if __name__ == '__main__':
    main()
