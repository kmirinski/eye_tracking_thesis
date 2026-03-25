#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

read -rp "Enter subject numbers: " input
subjects=($input)

declare -A results

for subject in "${subjects[@]}"; do
    echo ""
    echo "======================================================"
    echo "  Subject $subject"
    echo "======================================================"
    output=$(python src/tune.py --subject "$subject" --eye left --relabel --fov 40 20)
    echo "$output"
    best=$(echo "$output" | grep "^\s*${subject}:" | tail -1)
    results[$subject]="$best"
done

echo ""
echo "======================================================"
echo "  Summary — paste into SUBJECT_FRAME_DETECTION_OVERRIDES"
echo "======================================================"
for subject in "${subjects[@]}"; do
    echo "    ${results[$subject]}"
done
