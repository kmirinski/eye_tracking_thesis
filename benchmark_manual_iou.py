#!/usr/bin/env python3
"""
IoU / F1 / PE benchmark of our pupil-extraction method against the EV-Eye
*manual* ground-truth annotations (VGG Image Annotator ellipses stored in the
per-session ``user_{N}.csv`` files).

This is the apples-to-apples comparison with the EV-Eye paper, which reports for
frame-based pupil segmentation (averaged over subjects, vs. the same manual GT):

    DL-based (U-Net)    : IoU 0.9187 | F1 0.9560 | PE 0.64 px
    Model-based (EBVEYE): IoU 0.8360 | F1 0.9075 | PE 1.30 px

Run from the repo root:
    python benchmark_manual_iou.py
"""
import os, sys, csv, json, glob, math, collections
import numpy as np
import cv2

sys.path.append("src")
from data.loaders import Frame
from processing.frame_detection import extract_pupil
from config import get_frame_detection_config, EV_EYE_FRAME_DETECTION_OVERRIDES

# Subjects we actually fine-tuned (non-empty override entry in config.py).
TUNED_SUBJECTS = {s for s, ov in EV_EYE_FRAME_DETECTION_OVERRIDES.items() if ov}

DATA_ROOT = "eye_data/ev_eye/raw_data/data_davis"
MASK_SHAPE = (260, 346)  # (H, W) of the DAVIS346 sensor


def gt_ellipse_to_mask(d, shape=MASK_SHAPE):
    """Rasterize a VGG ellipse dict {cx,cy,rx,ry,theta(rad)} into a filled mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(d["cx"])), int(round(d["cy"]))),
        (int(round(d["rx"])), int(round(d["ry"]))),  # rx,ry are semi-axes
        math.degrees(d.get("theta", 0.0)),
        0, 360, color=1, thickness=-1,
    )
    return mask


def pred_ellipse_to_mask(ellipse, shape=MASK_SHAPE):
    """Rasterize our cv2 ellipse ((cx,cy),(w,h),angle_deg) into a filled mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if ellipse is None:
        return mask
    (cx, cy), (w, h), angle = ellipse
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (int(round(w / 2)), int(round(h / 2))),
                angle, 0, 360, color=1, thickness=-1)
    return mask


def ellipse_center(ellipse):
    if ellipse is None:
        return None
    (cx, cy), _, _ = ellipse
    return np.array([cx, cy])


def metrics(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 0.0
    f1 = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0
    return iou, f1


def parse_csv_subject_eye(path):
    """user{N}/{left|right}/... -> (int N, 'left'/'right')."""
    parts = path.replace("\\", "/").split("/")
    subj = next(int(p[4:]) for p in parts if p.startswith("user") and p[4:].isdigit())
    eye = "left" if "left" in parts else "right" if "right" in parts else "left"
    return subj, eye


def main():
    csvs = sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True))
    print(f"Found {len(csvs)} annotation CSV files under {DATA_ROOT}\n")

    # per (subject, eye) accumulators
    records = collections.defaultdict(list)   # key -> list of (iou, f1)
    pe_records = collections.defaultdict(list) # key -> list of center pixel errors
    config_cache = {}
    n_anno = n_scored = n_no_detect = n_missing_png = n_empty = 0

    for csv_path in csvs:
        subj, eye = parse_csv_subject_eye(csv_path)
        if eye != "left":          # research is left-eye only
            continue
        if subj not in TUNED_SUBJECTS:   # only subjects we fine-tuned in config.py
            continue
        base_dir = os.path.dirname(csv_path)
        # index every png under this CSV's directory by filename
        png_index = {os.path.basename(p): p
                     for p in glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)}

        ckey = (subj, eye)
        if ckey not in config_cache:
            config_cache[ckey] = get_frame_detection_config(subj, eye, dataset="ev_eye")
        config = config_cache[ckey]

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    shape = json.loads(row["region_shape_attributes"])
                except Exception:
                    continue
                if shape.get("name") != "ellipse":
                    continue
                n_anno += 1
                fname = row["filename"]
                png_path = png_index.get(fname)
                if png_path is None:
                    n_missing_png += 1
                    continue

                gt = gt_ellipse_to_mask(shape)
                if not gt.any():
                    n_empty += 1
                    continue

                _, ellipse = extract_pupil(Frame(0, 0, png_path, 0), config)
                if ellipse is None:
                    n_no_detect += 1
                pred = pred_ellipse_to_mask(ellipse)

                iou, f1 = metrics(pred, gt)
                records[ckey].append((iou, f1))
                n_scored += 1

                c = ellipse_center(ellipse)
                if c is not None:
                    pe_records[ckey].append(
                        np.hypot(c[0] - shape["cx"], c[1] - shape["cy"]))

    # ---- report -----------------------------------------------------------
    print("=" * 72)
    print("Manual-GT pupil-segmentation benchmark (our method)")
    print("=" * 72)
    print(f"Annotated ellipses found : {n_anno}")
    print(f"Frames scored            : {n_scored}")
    print(f"Skipped (no raw png)     : {n_missing_png}")
    print(f"Skipped (empty GT)       : {n_empty}")
    print(f"Detection failures       : {n_no_detect} "
          f"({n_no_detect / max(n_scored,1):.2%})\n")

    print(f"{'subject/eye':>14} | {'n':>5} | {'IoU':>6} | {'F1':>6} | {'PE px':>6}")
    print("-" * 50)
    subj_iou, subj_f1, subj_pe = [], [], []
    for key in sorted(records):
        arr = np.array(records[key])
        miou, mf1 = arr[:, 0].mean(), arr[:, 1].mean()
        mpe = np.mean(pe_records[key]) if pe_records[key] else float("nan")
        subj_iou.append(miou); subj_f1.append(mf1); subj_pe.append(mpe)
        print(f"{f'{key[0]} {key[1]}':>14} | {len(arr):>5} | "
              f"{miou:>6.4f} | {mf1:>6.4f} | {mpe:>6.2f}")

    all_arr = np.array([r for v in records.values() for r in v])
    print("-" * 50)
    print(f"{'MICRO (all)':>14} | {len(all_arr):>5} | "
          f"{all_arr[:,0].mean():>6.4f} | {all_arr[:,1].mean():>6.4f} | "
          f"{np.mean([p for v in pe_records.values() for p in v]):>6.2f}")
    print(f"{'MACRO (subj)':>14} | {len(subj_iou):>5} | "
          f"{np.mean(subj_iou):>6.4f} | {np.mean(subj_f1):>6.4f} | "
          f"{np.nanmean(subj_pe):>6.2f}")

    print("\nPaper reference (macro over subjects, same manual GT):")
    print(f"{'EV-Eye U-Net':>14} |       | 0.9187 | 0.9560 |   0.64")
    print(f"{'EV-Eye model':>14} |       | 0.8360 | 0.9075 |   1.30")


if __name__ == "__main__":
    main()
