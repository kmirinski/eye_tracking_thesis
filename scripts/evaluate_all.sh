#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

RESULTS="results.txt"
> "$RESULTS"

for subject in $(seq 4 27); do
    echo "=== SUBJECT $subject ===" | tee -a "$RESULTS"
    python src/main.py --subject $subject --eye left --model regressor 2>&1 | tee -a "$RESULTS"
done

echo ""
echo "=== SUMMARY ==="
grep -E "Subject [0-9]+|Validation Mean Error" "$RESULTS"
