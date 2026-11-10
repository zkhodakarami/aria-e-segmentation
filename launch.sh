#!/usr/bin/env bash
# Detachable launcher for the ARIA-E nnU-Net pipeline.
#   ./launch.sh              start (tmux session: ariae)
#   tmux attach -t ariae     watch
#   Ctrl-b d                 detach (training keeps running)
#   tail -f logs/train_*.log follow without tmux
set -euo pipefail
cd "$(dirname "$0")"
SESSION=ariae
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/nnunet_ariae}"
PY="$ENV_DIR/bin/python"
# nnUNetV2Runner shells out to the `nnUNetv2_train` console script, so the env's
# bin MUST be on PATH or every fold dies instantly with "not found".
export PATH="$ENV_DIR/bin:$PATH"
mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running -> tmux attach -t $SESSION"; exit 1
fi
tmux new-session -d -s "$SESSION" \
  "stdbuf -oL -eL $PY scripts/run_pipeline.py ${*:-} 2>&1 | tee $LOG"
echo "started tmux session '$SESSION'"
echo "log: $(pwd)/$LOG"
echo "attach: tmux attach -t $SESSION   |   detach: Ctrl-b d"
