#!/usr/bin/env bash
# Four isolated CO2 regularizer arms on CPU (sequential, cheap → expensive):
#   none, kl_only, prior_fp, chem_tether
# 1,000 episodes, batch 64. Does not edit train.yaml.
#
#   bash run_tether_ablation_cpu.sh
#   MGW_NO_PENALTIES=1 MGW_ABLATION_OUT=outputs/ablation_cpu_nopenalty \
#     bash run_tether_ablation_cpu.sh
#
# Requires weights/guacamol_nc.pt. The chem_tether arm also needs
# weights/chem_tether_bank.npz (build_chemical_tether_bank.py) and
#   uv sync --extra tether
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

N_EPISODES=1000
BATCH_SIZE=64
BANK="$ROOT/weights/chem_tether_bank.npz"
OUT_ROOT="${MGW_ABLATION_OUT:-$ROOT/outputs/ablation_cpu}"
LOG="$OUT_ROOT/ablation.log"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

export CUDA_VISIBLE_DEVICES=""
export MGW_OBJECTIVE=co2
export MGW_INCLUDE_PFC=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== tether ablation (CPU) $(date -Iseconds) ==="
echo "episodes=$N_EPISODES batch=$BATCH_SIZE PFC=off NO_PENALTIES=${MGW_NO_PENALTIES:-0} out=$OUT_ROOT"

if [[ ! -f "$ROOT/weights/guacamol_nc.pt" ]]; then
  echo "Missing weights/guacamol_nc.pt" >&2
  exit 1
fi
if [[ ! -f "$BANK" ]]; then
  echo "Missing $BANK" >&2
  echo "Run: uv sync --extra tether && PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py" >&2
  exit 1
fi

write_arm_yaml() {
  local arm="$1"
  uv run python - "$ROOT" "$OUT_ROOT" "$arm" "$N_EPISODES" "$BATCH_SIZE" "$BANK" <<'PY'
import sys
from pathlib import Path

import yaml

root, out_root, arm, n_ep, batch, bank = sys.argv[1:7]
n_ep, batch = int(n_ep), int(batch)
src = Path(root) / "train.yaml"
cfg = yaml.safe_load(src.read_text())
arm_dir = Path(out_root) / arm
arm_dir.mkdir(parents=True, exist_ok=True)

cfg["DEVICE"] = "cpu"
cfg["OUTPUTS"]["DIR"] = str(arm_dir)
cfg["GENERATE"]["CKPT_PATH"] = str(Path(root) / "weights" / "guacamol_nc.pt")
cfg["GENERATE"]["N_CPU"] = 1
rl = cfg["RL_TRAIN"]
rl["N_EPISODES"] = n_ep
rl["BATCH_SIZE"] = batch
rl["EVAL_BATCH_SIZE"] = batch
rl["SNAPSHOT_BATCH_SIZE"] = min(int(rl.get("SNAPSHOT_BATCH_SIZE", 64)), batch)
rl["PRIOR_FP_CACHE"] = str(Path(root) / "weights" / "prior_fp_bank.smi")
rl["CHEM_TETHER_BANK"] = str(Path(bank))
rl["CHEM_TETHER_LOG"] = False

rl["KL_COEF"] = 0.0
rl["KL_COEF_START"] = 0.0
rl["KL_COEF_END"] = 0.0
rl["PRIOR_FP_FLOOR"] = 0.0
rl["CHEM_TETHER_COEF"] = 0.0
rl["CHEM_TETHER_COEF_START"] = 0.0
rl["CHEM_TETHER_COEF_END"] = 0.0

if arm == "kl_only":
    rl["KL_COEF"] = 0.75
    rl["KL_COEF_START"] = 0.75
    rl["KL_COEF_END"] = 0.40
elif arm == "prior_fp":
    rl["PRIOR_FP_FLOOR"] = 0.3
elif arm == "chem_tether":
    rl["CHEM_TETHER_COEF"] = 0.75
    rl["CHEM_TETHER_COEF_START"] = 0.75
    rl["CHEM_TETHER_COEF_END"] = 0.40
elif arm != "none":
    raise SystemExit(f"unknown arm {arm}")

out = arm_dir / "config.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

run_arm() {
  local arm="$1"
  local extra=()
  if [[ "$arm" == "chem_tether" ]]; then
    extra=(--extra tether)
  fi
  write_arm_yaml "$arm"
  local yaml="$OUT_ROOT/$arm/config.yaml"
  echo
  echo "── arm $arm  $(date -Iseconds)  config=$yaml"
  uv run "${extra[@]}" python train.py --config "$yaml" \
    | tee "$OUT_ROOT/$arm/train.log"
  echo "── arm $arm done  $(date -Iseconds)"
}

# Cheap arms first so a chem_tether crash still leaves the other three.
for arm in none kl_only prior_fp chem_tether; do
  run_arm "$arm"
done

echo
echo "=== all four arms finished $(date -Iseconds) ==="
echo "Results under $OUT_ROOT/{none,kl_only,prior_fp,chem_tether}/"
echo "Compare rl_metrics.csv in each timestamped run dir."
