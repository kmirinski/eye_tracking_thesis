# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
cd /home/kmirinski/uni/msc/thesis/eye_tracking_thesis
source .venv/bin/activate
python src/main.py --subject 22 --eye left --data_dir eye_data
```

All arguments have defaults (`--subject 22`, `--data_dir <cwd>/eye_data`). Scripts must be run from the repo root so that relative imports inside `src/` resolve correctly.

## Architecture

The system is an event-based eye tracking pipeline for DVS (Dynamic Vision Sensor) cameras, mapping pupil position to screen gaze coordinates via polynomial regression.

**Data flow:**
1. `data/loaders.py` — loads per-subject frame images and raw `.aerdat` binary event streams into `Frame` and `Event` namedtuples; `EyeDataset` holds both and merges them chronologically.
2. `processing/frame_detection.py` — detects the pupil in APS frames: grayscale → threshold → morphological opening → contour fitting → ellipse selection filtered by aspect ratio, axis length, area, and center position. All thresholds live in `FrameDetectionConfig`.
3. `pipeline.py` — orchestrates the full run: loads data, extracts pupil centers for all frames, builds a `valid_mask` (drops failed detections, zero screen coords, and post-saccade frames), then splits and trains gaze estimators.
4. `gaze_estimator.py` — fits two independent `sklearn` `LinearRegression` models on `PolynomialFeatures`-expanded pupil coordinates, one for screen X and one for screen Y. Degrees 3–10 are swept.
5. `tracking.py` — event-based tracker using `lsq-ellipse`; accumulates N events, fits an ellipse, falls back to ROI-filtered events when a previous ellipse is known.
6. `processing/filtering.py` + `processing/pupil_finding.py` — alternative event-image pipeline: noise mask → eyelid/glint mask → eyelash mask → KDE-based pupil centre localisation. Not used in the current main pipeline.

**Config:** All algorithm parameters are dataclasses in `src/config.py` (`FrameDetectionConfig`, `TrackingConfig`, `KDEConfig`, `GazeConfig`). Edit there rather than in individual processing files.

**Visualisation helpers** (`data/visualization.py`):
- `write_ellipse_video` — renders annotated MP4 (`ellipse_detection.mp4`) at the repo root.
- `browse_ellipse_frames` — interactive matplotlib browser; ←/→ to step, type digits + Enter to jump to a frame index, Q to quit.
- `extract_pupil(..., visualize=True)` — shows a 6-panel matplotlib figure for a single frame (useful for debugging individual detections).

## Data layout

```
eye_data/
  user{1..27}/
    0/          # left eye
      frames/   # APS frame images, filename encodes metadata
      events.aerdat
    1/          # right eye
```

Frame filenames encode `(index, row, col, stimulus_type, timestamp)` parsed by `get_path_info()` in `loaders.py`. The frame list is stored in reverse chronological order; most processing iterates with negative indices or explicit reversal.

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

Key non-standard packages: `lsq-ellipse` (ellipse fitting), `opencv-python`, `scikit-learn`, `scipy`.

## Tests

No pytest suite. Tests are in `notebooks/testing.ipynb` (run with `jupyter notebook`).
