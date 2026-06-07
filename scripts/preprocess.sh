#!/bin/bash
#SBATCH --job-name=eye_preprocess
#SBATCH --output=logs/preprocess_%a.out
#SBATCH --time=02:00:00
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3

# Warms all subject caches for every (motion x combined) combination.
# Running one fold per combo is enough — it loads all subjects as a side effect.
# Submit this first; only submit run_fold.sh after all 4 jobs complete.

MOTIONS=(saccadic pursuit)
MOTION_IDX=$((SLURM_ARRAY_TASK_ID / 2))
COMBINED_IDX=$((SLURM_ARRAY_TASK_ID % 2))

MOTION=${MOTIONS[$MOTION_IDX]}
EXTRA_ARGS=""
[ $COMBINED_IDX -eq 1 ] && EXTRA_ARGS="--lstm_events"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/eye_tracking_thesis/.venv/bin/activate
cd ~/eye_tracking_thesis
mkdir -p logs

echo "=== Preprocessing: motion=$MOTION combined=$COMBINED_IDX ==="

# Entry point is main.py. Running one fold warms every subject's cache as a side
# effect; the trained model here is discarded (the GPU array does the real runs).
python src/main.py \
    --cross_subject \
    --model lstm \
    --val_subject 4 \
    --dataset ev_eye \
    --eye left \
    --motion $MOTION \
    $EXTRA_ARGS
