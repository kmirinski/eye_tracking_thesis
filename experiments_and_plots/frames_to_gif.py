"""Build a GIF from 50 consecutive ebveye frames of a single subject.

The GIF contains only the raw frame images played one after another — no
axes, indices, borders, or any other overlay.

Run from the repo root:
    python experiments_and_plots/frames_to_gif.py
"""

import os

from PIL import Image

# --- Settings ---------------------------------------------------------------
SUBJECT = 15
EYE = 0  # 0 = left, 1 = right
START_INDEX = 1000  # first frame index to include (inclusive)
END_INDEX = 1120  # last frame index to include (inclusive)
FRAME_DURATION_MS = 100  # per-frame display time in the GIF

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(
    REPO_ROOT, "eye_data", "ebveye", f"user{SUBJECT}", str(EYE), "frames"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"subject{SUBJECT}_frames.gif"
)


def frame_index(filename):
    """Parse the leading integer index from `{idx}_{row}_{col}_{type}_{ts}.png`."""
    return int(filename.split("_", 1)[0])


def main():
    filenames = [f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")]
    filenames.sort(key=frame_index)

    selected = [
        f for f in filenames if START_INDEX <= frame_index(f) <= END_INDEX
    ]
    if not selected:
        raise RuntimeError(
            f"No frames with index in [{START_INDEX}, {END_INDEX}] "
            f"found in {FRAMES_DIR}."
        )

    images = [Image.open(os.path.join(FRAMES_DIR, f)).convert("P") for f in selected]

    images[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
    )
    print(f"Wrote {len(images)} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
