# E-Gaze pupil-extraction procedure (reproduction figure)

Step-by-step visualisation of the **E-Gaze** event-only eye-tracking block, for the
thesis report. Reproduces the pupil-feature extraction of:

> N. Li, M. Chang, A. Raychowdhury,
> *"E-Gaze: Gaze Estimation With Event Camera"*, IEEE TPAMI 46(7), 2024.
> (`papers/E-Gaze_Gaze_Estimation_With_Event_Camera.pdf`)

This was an early approach explored in the thesis: accumulate a fixed number of events,
then recover the pupil from the resulting event image with classical image processing +
non-parametric statistics (instead of the frame-based ellipse detector adopted later).
The underlying code lives in `src/processing/` (the *unused alternative pipeline*):

- `processing/preprocessing.py` — `accumulate_events` (2000-event sets), `event_to_image`
- `processing/filtering.py` — noise / eyelid-glint / eyelash / pupil-iris masks (morphology)
- `processing/pupil_finding.py` — pupil-centre finding + circular segmentation

## Run

```bash
source .venv/bin/activate
python experiments_and_plots/egaze_reproduction/egaze_procedure.py --subject 22 --eye 0
python experiments_and_plots/egaze_reproduction/egaze_procedure.py --subject 22 --set_index 1486
```

Without `--set_index` the script auto-picks a clean, well-centred pupil: pass 1 scans a
block of consecutive event sets cheaply (masking only) to find the active ones, pass 2
runs the full pupil-finding pipeline on the top candidates and keeps the best. It prints
the chosen index so you can reproduce it deterministically with `--set_index`. The figures
checked in here are subject 22, eye 0, set **1486**.

## Outputs

| File | Stage |
|------|-------|
| `a_event_set.png`       | the raw 2000 accumulated events (ON = green, OFF = red), cf. E-Gaze Fig. 1(b)/3 |
| `b_polarity_images.png` | the three count images from one set: positive, negative, combined (Sec. IV-A) |
| `c_filtering_masks.png` | eye-part segmentation: noise / eyelid-glint / eyelash masks → pupil & iris events (Fig. 4) |
| `d_kde_center.png`      | donut-kernel KDE density and the detected pupil centre |
| `e_pupil_ellipse.png`   | circular pupil segmentation + the fitted pupil ellipse (the feature fed to the RNN) |
| `panels_grid.png`       | all stages assembled into one overview figure |

## Pupil-centre finding (note vs. the source code)

E-Gaze locates the pupil centre with *"KDE with a donut kernel"*. The pupil-boundary
events form a small ring, and a donut (annulus) kernel matched to that radius gives a
density that **peaks at the ring centre** = the pupil centre. This script implements that
directly: it convolves the pupil/iris event image with an annulus kernel
(`DONUT_R_IN`/`DONUT_R_OUT`, in px) and takes the `argmax`.

The original `src/processing/pupil_finding.py:locate_pupil_center_kde` instead used a
*difference of two Gaussian KDEs*. That has two problems for this purpose: (1) it peaks on
dense ring *regions* (the iris/eyelash arc) rather than at the ring's centre, so the
segmentation circle lands off the pupil; and (2) it builds the FFTKDE grid sorted
`(x outer, y inner)` but reshapes the returned flat density straight into `(H, W)` — since
the density is in `(x, y)` order this transposes/scrambles it (diagonal banding) and biases
the `argmax`; the correct reshape is `(W, H).T`. The donut-kernel version here avoids both.
