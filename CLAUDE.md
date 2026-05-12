# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
cd /home/kmirinski/uni/msc/thesis/eye_tracking_thesis
source .venv/bin/activate
python src/main.py --subject 22 --eye left --data_dir eye_data
```

All arguments have defaults (`--subject 22`, `--data_dir <cwd>/eye_data`). Scripts must be run from the repo root so that relative imports inside `src/` resolve correctly.

**Key flags:**
- `--model regressor|lstm` — polynomial regression (default) or LSTM gaze estimator
- `--relabel` — relabel pre-saccade frames to the previous target label; discards active saccade frames via `relabeling_stage`
- `--fov WIDTH_DEG HEIGHT_DEG` — restrict training to a centered FoV window (e.g. `--fov 40 20`)
- `--cross_subject` — leave-one-out cross-subject evaluation; add `--val_subject N` to run a single fold, `--fine_tune` to fine-tune on the held-out subject

**Debug/visualisation flags:** `--video`, `--f_browse`, `--pe_browse`, `--pe_plots`, `--ge_plots`, `--relabel_diag`, `--diff_plot`, `--event_diag`, `--loss_plot`

## Architecture

The system is an event-based eye tracking pipeline for DVS (Dynamic Vision Sensor) cameras, mapping pupil position to screen gaze coordinates.

**Data flow (single-subject):**
1. `data/loaders.py` — loads per-subject APS frame images and raw `.aerdat` binary event streams into `Frame` and `Event` namedtuples; `EyeDataset.collect_data()` populates `frame_list` and event data.
2. `processing/frame_detection.py` — detects the pupil in APS frames: grayscale → threshold → morphological opening → contour fitting → ellipse selection filtered by aspect ratio, area, and center position. All thresholds live in `FrameDetectionConfig`.
3. `pipeline/pipeline.py` — orchestrates the full run: `pupil_extraction_stage` → `noise_flagging_stage` → optional `relabeling_stage` → `build_valid_mask` → model training. Also contains `event_extraction_stage` (ROI-based batch ellipse fitting from events) and `merge_frame_event_samples`.
4. `pipeline/runners.py` — model-specific training: `run_regressor` (sweeps polynomial degrees), `run_lstm` (frame-only LSTM), `run_lstm_combined` (frame + event ellipses merged chronologically).
5. `models/polynomial.py` — `GazeEstimator`: two independent `sklearn` `LinearRegression` models on `PolynomialFeatures`-expanded pupil coordinates, one per screen axis.
6. `models/lstm.py` — `LSTMGazeEstimator`: Keras LSTM → dense head. `ellipse_to_21d` encodes each OpenCV ellipse as a 21D outer-product vector; `build_lstm_sequences` builds sliding windows from valid frames.
7. `tracking.py` — `fit_ellipse` wrapper around `lsq-ellipse`; used by `event_extraction_stage` to fit ellipses to ROI-filtered event batches.
8. `cross_subject_lstm.py` / `cross_subject_regressor.py` — leave-one-out evaluation over `SUBJECTS = [4,5,6,7,11,12,15,18,19,21,22]`. Preprocessed sequences cached to `data_cache/` as `.npz` files; delete to force re-preprocessing.

**Unused alternative pipeline:** `processing/filtering.py` + `processing/pupil_finding.py` — KDE-based pupil localisation from event images. Not called from `main.py`.

**Config:** All algorithm parameters are dataclasses in `src/config.py`: `FrameDetectionConfig`, `TrackingConfig`, `KDEConfig`, `GazeConfig`, `LSTMConfig`. Per-subject overrides are dicts (`SUBJECT_FRAME_DETECTION_OVERRIDES`, `SUBJECT_GAZE_OVERRIDES`); `get_frame_detection_config(subject, eye)` and `get_gaze_config(subject)` apply them.

**Array ordering convention:** `EyeDataset.frame_list` and derived arrays (`pupil_centers`, `ellipses`, `screen_coords`, `valid_mask`) are stored in **reverse chronological order** (newest first). Processing functions flip with `[::-1]` to get chronological order internally and flip back before returning. This is a pervasive invariant — breaking it silently corrupts temporal logic.

**Visualisation helpers** (`data/visualization.py`):
- `write_ellipse_video` — renders annotated MP4 (`ellipse_detection.mp4`) at the repo root.
- `browse_ellipse_frames` — interactive matplotlib browser; ←/→ to step, digits + Enter to jump, Q to quit.
- `browse_pupil_extraction` — per-frame 6-panel debug view of detection stages.

## Data layout

```
eye_data/
  user{1..27}/
    0/          # left eye
      frames/   # APS frame images, filename encodes metadata
      events.aerdat
    1/          # right eye
data_cache/     # cached cross-subject .npz sequences (auto-created)
```

Frame filenames encode `(index, row, col, stimulus_type, timestamp)` parsed by `get_path_info()` in `loaders.py`.

## Dependencies

```bash
pip install -r requirements.txt
```

Key non-standard packages: `lsq-ellipse` (ellipse fitting), `opencv-python`, `scikit-learn`, `scipy`, `tensorflow`.

## Tests

No pytest suite. Tests are in `notebooks/testing.ipynb` (run with `jupyter notebook`).
