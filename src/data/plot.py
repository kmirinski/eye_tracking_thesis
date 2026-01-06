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

def plot_axes(rows: int, cols: int, images: list[tuple]):
    _, axes = plt.subplots(rows, cols, figsize=(20, 10), dpi=200)
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for idx, (img, title) in enumerate(images):
        plot_event_image(img, axes[idx], title)


    