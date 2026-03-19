import cv2
import numpy as np
import matplotlib.pyplot as plt



def plot_event_set(event_set: np.ndarray, ax, title, img_width=346, img_height=260):
    """
    Plot a single event set.
    
    Args:
        event_set: Event set (n_events, 4) with [polarity, row, col, timestamp]
        ax: Matplotlib axis
        title: Plot title
        img_width: Width of the image frame
        img_height: Height of the image frame
    """
    pols = event_set[:, 0]
    rows = event_set[:, 1]
    cols = event_set[:, 2]

    neg_mask = pols == 0
    pos_mask = pols == 1
    
    if neg_mask.any():
        ax.scatter(cols[neg_mask], rows[neg_mask], c='red', s=12, label='Polarity 0', alpha=0.5)
    if pos_mask.any():
        ax.scatter(cols[pos_mask], rows[pos_mask], c='green', s=12, label='Polarity 1', alpha=0.5)
    
    ax.set_xlabel('Column (x)')
    ax.set_xlim(0, img_width)
    ax.set_ylabel('Row (y)')
    ax.set_ylim(0, img_height)
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

def plot_event_image(img: np.ndarray,ax , title, img_width=346, img_height=260):

    rows, cols = np.nonzero(img)

    counts = img[rows, cols]
    rows_rep = np.repeat(rows, counts)
    cols_rep = np.repeat(cols, counts)

    ax.scatter(cols_rep, rows_rep, c='black', s=12, alpha=0.5)

    ax.set_xlabel('Column (x)')
    ax.set_xlim(0, img_width)
    ax.set_ylabel('Row (y)')
    ax.set_ylim(0, img_height)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

def plot_event_image_standalone(img: np.ndarray, title, img_width=346, img_height=260):
    """
    Plot event image in a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    rows, cols = np.nonzero(img)
    counts = img[rows, cols]
    rows_rep = np.repeat(rows, counts)
    cols_rep = np.repeat(cols, counts)
    
    ax.scatter(cols_rep, rows_rep, c='black', s=12, alpha=0.5)
    ax.set_xlabel('Column (x)')
    ax.set_xlim(0, img_width)
    ax.set_ylabel('Row (y)')
    ax.set_ylim(0, img_height)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_pupil_centers_over_time_all(pupil_centers, screen_coords, valid_mask=None):
    pc = pupil_centers[::-1]
    sc = screen_coords[::-1]
    mask = valid_mask[::-1] if valid_mask is not None else np.ones(len(pc), dtype=bool)

    t = np.arange(len(pc))

    _, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True, dpi=150)

    # --- Pupil X ---
    axes[0].plot(t, pc[:, 0], lw=0.8, color='steelblue', label='pupil x')
    axes[0].set_ylabel('Pupil X (px)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # --- Pupil Y ---
    axes[1].plot(t, pc[:, 1], lw=0.8, color='tomato', label='pupil y')
    axes[1].set_ylabel('Pupil Y (px)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # --- Screen target ---
    axes[2].plot(t, sc[:, 0], lw=0.8, color='green', label='screen row')
    axes[2].plot(t, sc[:, 1], lw=0.8, color='orange', label='screen col')
    axes[2].set_ylabel('Screen coord (px)')
    axes[2].set_xlabel('Frame index (chronological)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    invalid = ~mask
    if invalid.any():
        starts = np.where(np.diff(np.concatenate([[False], invalid, [False]])))[0]
        for s, e in zip(starts[::2], starts[1::2]):
            for ax in axes:
                ax.axvspan(s, e, color='gray', alpha=0.25, linewidth=0)

    plt.suptitle('Pupil centers over time — all frames (grey = filtered out)', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_pupil_centers_over_time(pupil_centers, screen_coords, valid_mask=None):
    pc = pupil_centers[::-1]
    sc = screen_coords[::-1]
    mask = valid_mask[::-1] if valid_mask is not None else np.ones(len(pc), dtype=bool)

    # Keep only valid frames; use original indices on x-axis so gaps are visible
    t = np.where(mask)[0]
    pc = pc[mask]
    sc = sc[mask]

    _, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True, dpi=150)

    # --- Pupil X ---
    axes[0].plot(t, pc[:, 0], lw=0.8, color='steelblue', label='pupil x')
    axes[0].set_ylabel('Pupil X (px)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # --- Pupil Y ---
    axes[1].plot(t, pc[:, 1], lw=0.8, color='tomato', label='pupil y')
    axes[1].set_ylabel('Pupil Y (px)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # --- Screen target ---
    axes[2].plot(t, sc[:, 0], lw=0.8, color='green', label='screen row')
    axes[2].plot(t, sc[:, 1], lw=0.8, color='orange', label='screen col')
    axes[2].set_ylabel('Screen coord (px)')
    axes[2].set_xlabel('Frame index (chronological)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Pupil centers over time (valid frames only)', fontsize=11)
    plt.tight_layout()
    plt.show()


def browse_ellipse_frames(frame_list, ellipses, screen_coords, saccade_mask=None, start=0):
    """
    Interactive frame browser. Navigate with left/right arrow keys.
    Press Q or close the window to quit.
    """
    n = len(frame_list)
    # Build chronological index list (same order as the video)
    chron_indices = list(range(n - 1, 0, -1))  # list_idx values in video order

    state = {'pos': start, 'input': ''}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.subplots_adjust(bottom=0.08)

    def render(pos):
        list_idx = chron_indices[pos]
        frame_idx = n - 1 - list_idx  # chronological frame number shown in video
        frame = frame_list[list_idx]
        ellipse = ellipses[list_idx]
        sc = screen_coords[list_idx]

        img = cv2.imread(frame.img)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.clear()
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', transform=ax.transAxes)

        if ellipse is not None:
            from matplotlib.patches import Ellipse as MEllipse
            cx, cy = ellipse[0]
            w, h = ellipse[1]
            angle = ellipse[2]
            patch = MEllipse((cx, cy), w, h, angle=angle,
                             edgecolor='lime', facecolor='none', linewidth=1.5)
            ax.add_patch(patch)
            ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=1.5)
            status = f'ellipse center ({cx:.1f}, {cy:.1f})'
        else:
            status = 'NO DETECTION'

        discard_tag = '  |  SACCADE (discarded)' if (saccade_mask is not None and saccade_mask[list_idx]) else ''
        jump_hint = f'  jump: {state["input"]}_' if state['input'] else '  type a number + Enter to jump'
        ax.set_title(f'frame {frame_idx}  |  screen ({int(sc[0])}, {int(sc[1])})  |  {status}{discard_tag}\n'
                     f'[{pos + 1}/{len(chron_indices)}]  ←/→ navigate  Q quit{jump_hint}')
        ax.axis('off')
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'right':
            state['input'] = ''
            state['pos'] = min(state['pos'] + 1, len(chron_indices) - 1)
            render(state['pos'])
        elif event.key == 'left':
            state['input'] = ''
            state['pos'] = max(state['pos'] - 1, 0)
            render(state['pos'])
        elif event.key in ('q', 'Q'):
            plt.close(fig)
        elif event.key in '0123456789':
            state['input'] += event.key
            render(state['pos'])
        elif event.key == 'backspace':
            state['input'] = state['input'][:-1]
            render(state['pos'])
        elif event.key == 'enter':
            if state['input']:
                target = int(state['input'])
                state['input'] = ''
                # target is a frame index (as shown in the title); find its pos
                target_list_idx = n - 1 - target
                if 0 <= target_list_idx < n and target_list_idx in chron_indices:
                    state['pos'] = chron_indices.index(target_list_idx)
                else:
                    state['pos'] = max(0, min(target, len(chron_indices) - 1))
                render(state['pos'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    render(state['pos'])
    plt.show()


def write_ellipse_video(frame_list, ellipses, screen_coords, output_path='ellipse_detection.mp4', fps=30):
    """
    Write a video of all frames with the fitted ellipse and center drawn 
    on each frame. Frames with no detection are labelled.
    """
    n = len(frame_list)

    # Get frame size
    sample = cv2.imread(frame_list[1].img)
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Iterate the list backwards
    for list_idx in range(n - 1, 0, -1):
        idx = n - 1 - list_idx
        frame = frame_list[list_idx]
        ellipse = ellipses[list_idx]
        sc = screen_coords[list_idx]

        img = cv2.imread(frame.img)
        if img is None:
            continue

        if ellipse is not None:
            cv2.ellipse(img, ellipse, (0, 220, 0), 1)
            cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        else:
            cv2.putText(img, 'NO DETECTION', (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(img, f'frame {idx}', (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(img, f'screen ({int(sc[0])}, {int(sc[1])})', (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(img)

    writer.release()
    print(f'Video written to {output_path}')


def plot_gaze_predictions(screen_pred, screen_gt, title='Gaze prediction vs ground truth'):
    """
    Scatter of GT vs predicted gaze points on the screen plane with error vectors,
    plus an error distribution panel.

    Args:
        screen_pred: (N, 2) predicted screen coords (row, col)
        screen_gt:   (N, 2) ground-truth screen coords (row, col)
        title:       figure suptitle
    """
    screen_pred = np.asarray(screen_pred)
    screen_gt   = np.asarray(screen_gt)

    euclidean = np.sqrt(np.sum((screen_pred - screen_gt) ** 2, axis=1))
    sort_idx = np.argsort(euclidean)  # draw worst-error vectors on top

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # --- Left: screen-plane scatter ---
    ax = axes[0]
    # error vectors
    for i in sort_idx:
        ax.plot(
            [screen_gt[i, 1], screen_pred[i, 1]],
            [screen_gt[i, 0], screen_pred[i, 0]],
            color='gray', alpha=0.3, linewidth=0.6, zorder=1
        )
    sc = ax.scatter(screen_pred[:, 1], screen_pred[:, 0],
                    c=euclidean, cmap='plasma', s=14, zorder=3,
                    label='predicted', vmin=0)
    ax.scatter(screen_gt[:, 1], screen_gt[:, 0],
               c='steelblue', s=10, marker='x', zorder=2, label='ground truth')
    fig.colorbar(sc, ax=ax, label='Euclidean error (px)')
    ax.set_xlabel('Screen col (px)')
    ax.set_ylabel('Screen row (px)')
    ax.invert_yaxis()
    ax.set_title('Screen plane  —  GT (×) vs predicted (●)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right: error distribution ---
    ax2 = axes[1]
    ax2.hist(euclidean, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
    ax2.axvline(np.mean(euclidean),   color='red',    linestyle='--', label=f'mean  {np.mean(euclidean):.1f}')
    ax2.axvline(np.median(euclidean), color='orange', linestyle='--', label=f'median {np.median(euclidean):.1f}')
    ax2.set_xlabel('Euclidean error (px)')
    ax2.set_ylabel('Count')
    ax2.set_title('Error distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_pupil_diffs(pupil_centers, screen_coords):
    """
    Plot frame-to-frame pupil displacement (L2 norm) over time in chronological order.
    Red vertical lines mark label changes. Blink frames (pupil == -1) are skipped when
    computing diffs — displacement is always measured against the last valid pupil position.
    """
    pc = pupil_centers[::-1]
    sc = screen_coords[::-1]
    n = len(pc)

    # Blink-aware diffs: compare each valid frame to the last valid position
    diffs = np.zeros(n)
    last_valid = None
    for i in range(n):
        if np.all(pc[i] == -1):
            continue  # blink — skip, don't update last_valid
        if last_valid is not None:
            diffs[i] = np.linalg.norm(pc[i] - last_valid)
        last_valid = pc[i].copy()

    # Find label change indices (ignore blink frames when comparing)
    change_indices = []
    prev_label = None
    for i in range(n):
        if np.all(pc[i] == -1):
            continue
        if prev_label is not None and not np.array_equal(sc[i], prev_label):
            change_indices.append(i)
        prev_label = sc[i].copy()

    fig, ax = plt.subplots(figsize=(20, 4), dpi=150)
    ax.plot(np.arange(n), diffs, lw=0.8, color='steelblue', label='pupil displacement (px)')
    for ci in change_indices:
        ax.axvline(ci, color='red', linestyle='--', linewidth=0.6, alpha=0.7)
    # Dummy line for legend
    ax.axvline(-1, color='red', linestyle='--', linewidth=0.6, alpha=0.7, label='label change')
    ax.set_xlabel('Frame index (chronological)')
    ax.set_ylabel('Displacement (px)')
    ax.set_title('Frame-to-frame pupil displacement — spikes should appear ~10-15 frames after each label change')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_relabeling_diagnostic(pupil_chron, screen_chron_original, phase_labels, blink_mask_chron):
    """
    Two-panel diagnostic for relabeling phase assignment.

    Top: displacement curve for the saccade section with Phase A/B/C shading and blink markers.
    Bottom: stacked bar chart of Phase A/B/C frame counts per label transition.

    All inputs in chronological order.
    """
    from pipeline.pipeline import _find_sections

    PHASE_COLORS = {'A': '#aec6e8', 'B': '#f4a9a8', 'C': '#b7e4c7'}

    sections = _find_sections(screen_chron_original)
    if not sections:
        print("No saccade section found — nothing to plot.")
        return
    sac_start, sac_end = sections[0]

    # Blink-aware displacement (same logic as plot_pupil_diffs, restricted to saccade section)
    n = len(pupil_chron)
    diffs = np.zeros(n)
    last_valid = None
    for i in range(n):
        if np.all(pupil_chron[i] == -1):
            continue
        if last_valid is not None:
            diffs[i] = np.linalg.norm(pupil_chron[i] - last_valid)
        last_valid = pupil_chron[i].copy()

    # Label change indices (from original screen_coords, within saccade section)
    change_indices = []
    prev = screen_chron_original[sac_start].copy()
    for i in range(sac_start + 1, sac_end):
        if not np.array_equal(screen_chron_original[i], prev):
            change_indices.append(i)
            prev = screen_chron_original[i].copy()

    # Per-transition phase counts for bar chart
    # Boundaries: [sac_start, change_indices[0], ..., change_indices[-1], sac_end]
    boundaries = [sac_start] + change_indices + [sac_end]
    n_transitions = len(boundaries) - 1
    counts = {'A': np.zeros(n_transitions, dtype=int),
              'B': np.zeros(n_transitions, dtype=int),
              'C': np.zeros(n_transitions, dtype=int)}
    for t in range(n_transitions):
        seg = phase_labels[boundaries[t]:boundaries[t + 1]]
        for p in ('A', 'B', 'C'):
            counts[p][t] = np.sum(seg == p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 8), dpi=200,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # --- Top panel: displacement + shading ---
    xs = np.arange(sac_start, sac_end)
    ax1.plot(xs, diffs[sac_start:sac_end], lw=0.8, color='steelblue',
             label='pupil displacement (px)', zorder=3)

    # Phase background shading
    i = sac_start
    while i < sac_end:
        p = phase_labels[i]
        if p in PHASE_COLORS:
            j = i + 1
            while j < sac_end and phase_labels[j] == p:
                j += 1
            ax1.axvspan(i, j, color=PHASE_COLORS[p], alpha=0.5, zorder=1)
            i = j
        else:
            i += 1

    # Label change markers
    for ci in change_indices:
        ax1.axvline(ci, color='black', linestyle='--', linewidth=0.7, alpha=0.8, zorder=4)

    # Blink markers
    blink_xs = np.where(blink_mask_chron[sac_start:sac_end])[0] + sac_start
    if len(blink_xs):
        ax1.plot(blink_xs, np.zeros(len(blink_xs)), 'v', color='orange',
                 markersize=5, label='blink', zorder=5)

    # Legend patches
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=PHASE_COLORS['A'], alpha=0.7, label='Phase A (relabeled)'),
        mpatches.Patch(color=PHASE_COLORS['B'], alpha=0.7, label='Phase B (saccade discard)'),
        mpatches.Patch(color=PHASE_COLORS['C'], alpha=0.7, label='Phase C (new label kept)'),
        plt.Line2D([0], [0], color='steelblue', lw=0.8, label='displacement (px)'),
        plt.Line2D([0], [0], color='black', linestyle='--', lw=0.7, label='label change'),
        plt.Line2D([0], [0], marker='v', color='orange', lw=0, markersize=5, label='blink'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=7)
    ax1.set_xlabel('Frame index (chronological)')
    ax1.set_ylabel('Displacement (px)')
    ax1.set_title('Relabeling phase diagnostic — saccade section')
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: stacked bar chart ---
    x = np.arange(n_transitions)
    ax2.bar(x, counts['A'], color=PHASE_COLORS['A'], label='Phase A')
    ax2.bar(x, counts['B'], bottom=counts['A'], color=PHASE_COLORS['B'], label='Phase B')
    ax2.bar(x, counts['C'], bottom=counts['A'] + counts['B'], color=PHASE_COLORS['C'], label='Phase C')
    ax2.set_xlabel('Transition index')
    ax2.set_ylabel('Frame count')
    ax2.set_title('Phase A/B/C frames per label transition')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def browse_pupil_extraction(frame_list, config, screen_coords, start=0):
    """
    Interactive browser for per-frame pupil extraction stages.
    Shows the 6-panel detection figure for each frame.
    Navigate with left/right arrows, type digits + Enter to jump, Q to quit.
    """
    import matplotlib.patches as mpatches
    from processing.frame_detection import _run_detection

    n = len(frame_list)
    chron_indices = list(range(n - 1, 0, -1))

    state = {'pos': start, 'input': ''}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    plt.subplots_adjust(bottom=0.08)

    def render(pos):
        list_idx = chron_indices[pos]
        frame_idx = n - 1 - list_idx
        frame = frame_list[list_idx]
        sc = screen_coords[list_idx]

        img, binary, opened, contour_img, selected_points, ellipse = _run_detection(frame, config)

        for ax in axes.flat:
            ax.clear()
            ax.axis('off')

        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')

        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Binarized (Hθ)')

        axes[0, 2].imshow(opened, cmap='gray')
        axes[0, 2].set_title('After Opening (◦ Sσ)')

        axes[1, 0].imshow(contour_img, cmap='gray')
        axes[1, 0].set_title('Contours')

        axes[1, 1].imshow(img, cmap='gray')
        if len(selected_points) > 0:
            axes[1, 1].scatter(selected_points[:, 0], selected_points[:, 1],
                               c='red', s=1, alpha=0.5)
        axes[1, 1].set_title(f'Selected Contour ({len(selected_points)} points)')

        img_with_ellipse = img.copy()
        if ellipse is not None:
            cv2.ellipse(img_with_ellipse, ellipse, 255, 1)
        axes[1, 2].imshow(img_with_ellipse, cmap='gray')
        if ellipse is not None:
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            axes[1, 2].set_title(f'Fitted Ellipse  Center: {center}')
        else:
            axes[1, 2].set_title('No Ellipse Fitted')

        jump_hint = f'  jump: {state["input"]}_' if state['input'] else '  type number + Enter to jump'
        status = f'ellipse {(int(ellipse[0][0]), int(ellipse[0][1]))}' if ellipse is not None else 'NO DETECTION'
        fig.suptitle(
            f'frame {frame_idx}  |  screen ({int(sc[0])}, {int(sc[1])})  |  {status}\n'
            f'[{pos + 1}/{len(chron_indices)}]  ←/→ navigate  Q quit{jump_hint}',
            fontsize=10
        )
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'right':
            state['input'] = ''
            state['pos'] = min(state['pos'] + 1, len(chron_indices) - 1)
            render(state['pos'])
        elif event.key == 'left':
            state['input'] = ''
            state['pos'] = max(state['pos'] - 1, 0)
            render(state['pos'])
        elif event.key in ('q', 'Q'):
            plt.close(fig)
        elif event.key in '0123456789':
            state['input'] += event.key
            render(state['pos'])
        elif event.key == 'backspace':
            state['input'] = state['input'][:-1]
            render(state['pos'])
        elif event.key == 'enter':
            if state['input']:
                target = int(state['input'])
                state['input'] = ''
                target_list_idx = n - 1 - target
                if 0 <= target_list_idx < n and target_list_idx in chron_indices:
                    state['pos'] = chron_indices.index(target_list_idx)
                else:
                    state['pos'] = max(0, min(target, len(chron_indices) - 1))
                render(state['pos'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    render(state['pos'])
    plt.show()


def plot_axes(rows: int, cols: int, images: list[tuple], centers: list[tuple]):
    _, axes = plt.subplots(rows, cols, figsize=(20, 10), dpi=200)
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]

    for idx, (img, title) in enumerate(images):
        ax = axes_flat[idx]
        plot_event_image(img, ax, title)

        if centers is not None:
            n = len(centers)
            center_idx = idx if idx < n else idx - n
            
            if center_idx < len(centers) and centers[center_idx] is not None:
                cx, cy = centers[center_idx]
                ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)


    