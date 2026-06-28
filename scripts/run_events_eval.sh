#!/bin/bash
#SBATCH --job-name=events_eval_reg
#SBATCH --output=logs/events_eval_%a.out
#SBATCH --time=00:30:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-19

# Regressor events_eval sweep over every tuned ev_eye subject.
# One array task per subject; each runs BOTH eval-split protocols
# (--eval_split blocks and --eval_split within) and parses the two means
# of interest (frames 25 Hz DoD mean + events-anchored-to-GOOD-frames mean)
# for every polynomial degree into a per-subject CSV.
#
# Subjects = keys of EV_EYE_FRAME_DETECTION_OVERRIDES in src/config.py.
# CPU-only (sklearn polynomial regression); no GPU needed.
#
# Submit:   sbatch scripts/run_events_eval.sh
# Then merge per-subject CSVs into one summary on the login node:
#           python scripts/aggregate_events_eval.py

SUBJECTS=(4 5 6 7 8 22 25 29 30 31 32 33 34 35 36 40 41 42 43 44)
EYE=left
MOTION=saccadic
SPLITS=(blocks within)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/eye_tracking_thesis/.venv/bin/activate
cd ~/eye_tracking_thesis
mkdir -p logs results/events_eval

echo "=== Subject $SUBJECT | eye=$EYE | motion=$MOTION ==="

LOG_ARGS=()
for SPLIT in "${SPLITS[@]}"; do
    LOG="logs/events_eval_s${SUBJECT}_${SPLIT}.out"
    echo "--- Running subject $SUBJECT | eval_split=$SPLIT ---"
    python src/main.py \
        --subject "$SUBJECT" \
        --dataset ev_eye \
        --eye "$EYE" \
        --motion "$MOTION" \
        --events_eval \
        --eval_split "$SPLIT" \
        2>&1 | tee "$LOG"
    LOG_ARGS+=(--log "${SPLIT}:${LOG}")
done

# Parse this subject's two run logs into one CSV (race-free: unique filename).
python scripts/parse_events_eval.py \
    --subject "$SUBJECT" \
    --eye "$EYE" \
    --motion "$MOTION" \
    --out "results/events_eval/s${SUBJECT}.csv" \
    "${LOG_ARGS[@]}"
