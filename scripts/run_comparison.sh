#!/usr/bin/env bash
#
# End-to-end FitNets comparison on a single GPU box.
#
#   1. Train ONE teacher (Maxout-CNN), or reuse an existing checkpoint.
#   2. Train student A with the ORIGINAL FitNets recipe (hint MSE + KD).
#   3. Train student B with the NEW projected-logit recipe.
#
# Both students share the same teacher, backbone, data pipeline and KD stage, so
# the only difference is the hint stage -> a clean A/B comparison.
#
# Disconnect-safe usage (recommended):
#   nohup bash scripts/run_comparison.sh > runs/compare.out 2>&1 &
#   # ...then you can safely close the terminal; follow progress with:
#   tail -f runs/compare_cifar100/logs/*.log
#
# Or run it inside tmux/screen. Every stage also tees its own log file.
#
# Everything is configurable via environment variables (defaults in []):
#   DATASET[cifar100] DEVICE[cuda] SEED[1337] BATCH_SIZE[128] NUM_WORKERS[4]
#   WHITEN[1] AMP[1] DOWNLOAD[1]
#   TEACHER_EPOCHS[288] STAGE0_EPOCHS[20] HINT_EPOCHS[40] KD_EPOCHS[288]
#   TEACHER_CKPT[checkpoints/<dataset>_teacher.pt] RUN_ROOT[runs/compare_<dataset>]

set -euo pipefail

DATASET="${DATASET:-cifar100}"
DATA_ROOT="${DATA_ROOT:-./data}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1337}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WHITEN="${WHITEN:-1}"        # 1 = GCN+ZCA (match original FitNets), 0 = mean/std
AMP="${AMP:-1}"             # 1 = mixed precision (CUDA only)
DOWNLOAD="${DOWNLOAD:-1}"   # 1 = download the dataset if missing

TEACHER_EPOCHS="${TEACHER_EPOCHS:-288}"
STAGE0_EPOCHS="${STAGE0_EPOCHS:-20}"  # projected variant: teacher-projection training
HINT_EPOCHS="${HINT_EPOCHS:-40}"      # stage-1 hint epochs (both methods)
KD_EPOCHS="${KD_EPOCHS:-288}"         # final KD epochs (both methods)

RUN_ROOT="${RUN_ROOT:-runs/compare_${DATASET}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/${DATASET}_teacher.pt}"
PY="${PYTHON:-python}"

mkdir -p "$LOG_DIR" "$(dirname "$TEACHER_CKPT")"

# Assemble shared optional flags.
COMMON_FLAGS=(--dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE"
              --seed "$SEED" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
[ "$WHITEN" = "1" ] && COMMON_FLAGS+=(--whiten)
[ "$AMP" = "1" ] && COMMON_FLAGS+=(--amp)
[ "$DOWNLOAD" = "1" ] && COMMON_FLAGS+=(--download)

echo "=================================================================="
echo " FitNets comparison run"
echo "   dataset=$DATASET device=$DEVICE whiten=$WHITEN amp=$AMP seed=$SEED"
echo "   teacher_ckpt=$TEACHER_CKPT  run_root=$RUN_ROOT"
echo "=================================================================="

# ----------------------------------------------------------------------------
# Step 1: teacher
# ----------------------------------------------------------------------------
if [ -f "$TEACHER_CKPT" ]; then
  echo "[teacher] $TEACHER_CKPT already exists -> skipping teacher training."
else
  echo "[teacher] training Maxout-CNN teacher for $TEACHER_EPOCHS epochs..."
  "$PY" train_teacher_torch.py \
    "${COMMON_FLAGS[@]}" \
    --output "$TEACHER_CKPT" \
    --epochs "$TEACHER_EPOCHS" \
    2>&1 | tee "$LOG_DIR/teacher.log"
fi

# ----------------------------------------------------------------------------
# Step 2: student A -- original FitNets baseline (hint MSE + KD)
# ----------------------------------------------------------------------------
echo "[student A] original FitNets baseline (hint MSE + KD)..."
"$PY" train_fitnets_baseline_torch.py \
  "${COMMON_FLAGS[@]}" \
  --teacher-ckpt "$TEACHER_CKPT" \
  --output-dir "$RUN_ROOT/fitnets_baseline" \
  --hint-epochs "$HINT_EPOCHS" \
  --kd-epochs "$KD_EPOCHS" \
  2>&1 | tee "$LOG_DIR/fitnets_baseline.log"

# ----------------------------------------------------------------------------
# Step 3: student B -- new projected-logit recipe
# ----------------------------------------------------------------------------
echo "[student B] new projected-logit recipe..."
"$PY" train_projected_logits_torch.py \
  "${COMMON_FLAGS[@]}" \
  --teacher-ckpt "$TEACHER_CKPT" \
  --output-dir "$RUN_ROOT/projected_logits" \
  --stage0-epochs "$STAGE0_EPOCHS" \
  --stage1-epochs "$HINT_EPOCHS" \
  --stage2-epochs "$KD_EPOCHS" \
  2>&1 | tee "$LOG_DIR/projected_logits.log"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo "=================================================================="
echo " DONE. Best eval accuracies:"
echo "   teacher        : $(grep -h 'best teacher checkpoint' "$LOG_DIR/teacher.log" 2>/dev/null | tail -1 || echo 'n/a (reused checkpoint)')"
echo "   FitNets baseline: $(grep -h 'best student eval_acc' "$LOG_DIR/fitnets_baseline.log" | tail -1 || echo 'n/a')"
echo "   projected logits: $(grep -h 'best student eval_acc' "$LOG_DIR/projected_logits.log" | tail -1 || echo 'n/a')"
echo " Checkpoints under: $RUN_ROOT"
echo "=================================================================="
