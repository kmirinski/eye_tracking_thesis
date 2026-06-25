"""Compose the 6 pupil-extraction panels into a 2x3 grid with (a)-(f) captions."""
import os
import cv2
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(HERE, 'noise_example')
OUT_PATH  = os.path.join(PANEL_DIR, 'panels_grid.png')

# (filename, caption) in reading order: top row a,b,c then bottom row d,e,f.
panels = [
    ('a_grayscale.png',        '(a)'),
    ('b_threshold.png',        '(b)'),
    ('c_opened.png',           '(c)'),
    ('d_contours.png',         '(d)'),
    ('e_all_ellipses.png',     '(e)'),
    ('f_selected_ellipse.png', '(f)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=200)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                    wspace=0.05, hspace=0.28)

for ax, (fname, caption) in zip(axes.flat, panels):
    img = cv2.imread(os.path.join(PANEL_DIR, fname))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # black outline around each image (keep frame, hide ticks)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.text(0.5, -0.07, caption, transform=ax.transAxes,
            ha='center', va='top', fontsize=26)

fig.savefig(OUT_PATH, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
print(f"Saved grid -> {OUT_PATH}")
