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


def browse_ellipse_frames(frame_list, ellipses, screen_coords, start=0):
    """
    Interactive frame browser. Navigate with left/right arrow keys.
    Press Q or close the window to quit.
    """
    n = len(frame_list)
    # Build chronological index list (same order as the video)
    chron_indices = list(range(n - 1, 0, -1))  # list_idx values in video order

    state = {'pos': start, 'input': ''}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
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

        jump_hint = f'  jump: {state["input"]}_' if state['input'] else '  type a number + Enter to jump'
        ax.set_title(f'frame {frame_idx}  |  screen ({int(sc[0])}, {int(sc[1])})  |  {status}\n'
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


    