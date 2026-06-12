#!/bin/bash
#SBATCH --job-name=eye_lstm_ev
#SBATCH --output=logs/fold_%a.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --array=0-127

# 16 subjects x 2 motions x 2 fine_tune x 2 combined = 128 jobs
# Index decomposition (LSB first):
#   bit 0: combined (0=frame only, 1=frame+events)
#   bit 1: fine_tune (0=no, 1=yes)
#   bit 2: motion (0=saccadic, 1=pursuit)
#   bits 3-6: subject index (0-15)

SUBJECTS=(4 5 6 7 8 22 25 29 30 31 32 33 34 35 36 44)
MOTIONS=(saccadic pursuit)

IDX=$SLURM_ARRAY_TASK_ID
COMBINED_IDX=$((IDX % 2));   IDX=$((IDX / 2))
FINE_TUNE_IDX=$((IDX % 2));  IDX=$((IDX / 2))
MOTION_IDX=$((IDX % 2));     IDX=$((IDX / 2))
SUBJECT_IDX=$IDX

VAL_SUBJECT=${SUBJECTS[$SUBJECT_IDX]}
MOTION=${MOTIONS[$MOTION_IDX]}

EXTRA_ARGS=""
[ $FINE_TUNE_IDX -eq 1 ] && EXTRA_ARGS="$EXTRA_ARGS --fine_tune"
[ $COMBINED_IDX -eq 1 ]  && EXTRA_ARGS="$EXTRA_ARGS --lstm_events"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
# Check available CUDA versions with: module spider CUDA
module load CUDA/12.1.1

source ~/eye_tracking_thesis/.venv/bin/activate
cd ~/eye_tracking_thesis
mkdir -p logs

echo "=== Subject $VAL_SUBJECT | Motion $MOTION | fine_tune=$FINE_TUNE_IDX | combined=$COMBINED_IDX ==="

# Entry point is main.py: cross_subject_lstm has no __main__ block of its own.
# Each job writes its own results/folds/*.csv (race-free, unique filename).
python src/main.py \
    --cross_subject \
    --model lstm \
    --val_subject $VAL_SUBJECT \
    --dataset ev_eye \
    --eye left \
    --motion $MOTION \
    $EXTRA_ARGS

# After the whole array finishes, merge the per-fold CSVs into results/summary.csv:
#   python scripts/aggregate_results.py
# (run on the login node, or submit with:
#   sbatch --dependency=afterok:<arrayjobid> --wrap "python scripts/aggregate_results.py")
