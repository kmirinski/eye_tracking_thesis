#!/bin/bash
#SBATCH --job-name=events_eval_reg
#SBATCH --output=logs/events_eval_%a.out
#SBATCH --time=00:30:00
#SBATCH --account=tdsei17279
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-19

# NOTE: account tdsei17279 has GPU budget only (no CPU partition), so this
# CPU-bound regressor runs on a GPU node. To spend less of the shared SBU
# budget you can switch --partition to gpu_mig (a fractional A100 slice);
# drop --cpus-per-task to ~9 if you do, since MIG slices have fewer cores.

# Regressor events_eval sweep over every tuned ev_eye subject.
# One array task per subject; each runs BOTH eval-split protocols
# (--eval_split blocks and --eval_split within) for TWO FoV settings:
#   * whole field of view (no --fov)      -> results/events_eval/
#   * central 40x20 deg window (--fov 40 20) -> results/events_eval_fov40x20/
# Each setting is parsed into its OWN per-subject CSV (two separate result
# trees, aggregated into two separate summaries — see bottom).
#
# Subjects = keys of EV_EYE_FRAME_DETECTION_OVERRIDES in src/config.py.
# CPU-only (sklearn polynomial regression); no GPU needed.
#
# Submit:   sbatch scripts/run_events_eval.sh
# Then merge each tree's per-subject CSVs on the login node:
#   python scripts/aggregate_events_eval.py \
#       --in_dir results/events_eval          --out results/events_eval_summary.csv
#   python scripts/aggregate_events_eval.py \
#       --in_dir results/events_eval_fov40x20 --out results/events_eval_fov40x20_summary.csv

SUBJECTS=(4 5 6 7 8 22 25 29 30 31 32 33 34 35 36 40 41 42 43 44)
EYE=left
MOTION=saccadic
SPLITS=(blocks within)

# Parallel arrays: a label (used for dirs/log names) and the matching --fov args.
FOV_LABELS=(full fov40x20)
FOV_ARGS=("" "--fov 40 20")
FOV_DIRS=(results/events_eval results/events_eval_fov40x20)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/eye_tracking_thesis/.venv/bin/activate
cd ~/eye_tracking_thesis
mkdir -p logs "${FOV_DIRS[@]}"

echo "=== Subject $SUBJECT | eye=$EYE | motion=$MOTION ==="

for i in "${!FOV_LABELS[@]}"; do
    FOV_LABEL=${FOV_LABELS[$i]}
    FOV_ARG=${FOV_ARGS[$i]}
    FOV_DIR=${FOV_DIRS[$i]}

    LOG_ARGS=()
    for SPLIT in "${SPLITS[@]}"; do
        LOG="logs/events_eval_s${SUBJECT}_${FOV_LABEL}_${SPLIT}.out"
        echo "--- Running subject $SUBJECT | fov=$FOV_LABEL | eval_split=$SPLIT ---"
        # $FOV_ARG is intentionally unquoted so "--fov 40 20" splits into 3 args
        # and "" expands to nothing.
        python src/main.py \
            --subject "$SUBJECT" \
            --dataset ev_eye \
            --eye "$EYE" \
            --motion "$MOTION" \
            --events_eval \
            --eval_split "$SPLIT" \
            $FOV_ARG \
            2>&1 | tee "$LOG"
        LOG_ARGS+=(--log "${SPLIT}:${LOG}")
    done

    # One CSV per (subject, fov setting); race-free unique filename per array task.
    python scripts/parse_events_eval.py \
        --subject "$SUBJECT" \
        --eye "$EYE" \
        --motion "$MOTION" \
        --out "${FOV_DIR}/s${SUBJECT}.csv" \
        "${LOG_ARGS[@]}"
done
