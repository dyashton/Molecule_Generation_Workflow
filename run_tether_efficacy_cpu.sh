#!/usr/bin/env bash
# Matched multi-seed CPU evaluation for chemical-tether efficacy.
#
# Defaults are intentionally smaller than the production run only in the
# number of seeds; override all controls with environment variables:
#
#   MGW_EVAL_SEEDS="42 43 44 45 46" \
#   MGW_EVAL_ARMS="none kl_only prior_fp chem_tether current current_plus_tether" \
#   MGW_EVAL_OUT=outputs/tether_efficacy \
#   bash run_tether_efficacy_cpu.sh
#
# Set MGW_NO_PENALTIES=1 to evaluate the no-penalty objective consistently.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

N_EPISODES="${MGW_EVAL_EPISODES:-1000}"
BATCH_SIZE="${MGW_EVAL_BATCH_SIZE:-64}"
OUT_ROOT="${MGW_EVAL_OUT:-$ROOT/outputs/tether_efficacy}"
SEEDS="${MGW_EVAL_SEEDS:-42 43 44}"
ARMS="${MGW_EVAL_ARMS:-none kl_only prior_fp chem_tether current current_plus_tether}"
BANK="$ROOT/weights/chem_tether_bank.npz"

if [[ ! -f "$ROOT/weights/guacamol_nc.pt" ]]; then
  echo "Missing weights/guacamol_nc.pt" >&2
  exit 1
fi
if [[ ! -f "$BANK" ]]; then
  echo "Missing $BANK" >&2
  echo "Run: uv sync --extra tether && PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export MGW_OBJECTIVE=co2
export MGW_INCLUDE_PFC=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

write_config() {
  local seed="$1"
  local arm="$2"
  local output_dir="$3"
  uv run python - "$ROOT" "$seed" "$arm" "$output_dir" "$N_EPISODES" "$BATCH_SIZE" "$BANK" <<'PY'
import sys
from pathlib import Path

import yaml

root, seed, arm, output_dir, n_ep, batch, bank = sys.argv[1:8]
src = Path(root) / "train.yaml"
cfg = yaml.safe_load(src.read_text())
cfg["DEVICE"] = "cpu"
cfg["RANDOM_SEED"] = int(seed)
cfg["OUTPUTS"]["DIR"] = str(Path(output_dir).resolve())
cfg["GENERATE"]["CKPT_PATH"] = str((Path(root) / "weights" / "guacamol_nc.pt").resolve())
cfg["GENERATE"]["N_CPU"] = 1

rl = cfg["RL_TRAIN"]
rl["N_EPISODES"] = int(n_ep)
rl["BATCH_SIZE"] = int(batch)
rl["EVAL_BATCH_SIZE"] = int(batch)
rl["SNAPSHOT_BATCH_SIZE"] = min(int(rl.get("SNAPSHOT_BATCH_SIZE", 64)), int(batch))
rl["PRIOR_FP_CACHE"] = str((Path(root) / "weights" / "prior_fp_bank.smi").resolve())
rl["CHEM_TETHER_BANK"] = str(Path(bank).resolve())
rl["CHEM_TETHER_LOG"] = False

# Disable every optional regularizer before selecting exactly one comparison arm.
rl["KL_COEF"] = 0.0
rl["KL_COEF_START"] = 0.0
rl["KL_COEF_END"] = 0.0
rl["PRIOR_FP_FLOOR"] = 0.0
rl["CHEM_TETHER_COEF"] = 0.0
rl["CHEM_TETHER_COEF_START"] = 0.0
rl["CHEM_TETHER_COEF_END"] = 0.0

if arm in {"kl_only", "current", "current_plus_tether"}:
    rl["KL_COEF"] = 0.75
    rl["KL_COEF_START"] = 0.75
    rl["KL_COEF_END"] = 0.40
if arm in {"prior_fp", "current", "current_plus_tether"}:
    rl["PRIOR_FP_FLOOR"] = 0.3
if arm in {"chem_tether", "current_plus_tether"}:
    rl["CHEM_TETHER_COEF"] = 0.75
    rl["CHEM_TETHER_COEF_START"] = 0.75
    rl["CHEM_TETHER_COEF_END"] = 0.40
if arm not in {
    "none", "kl_only", "prior_fp", "chem_tether", "current",
    "current_plus_tether",
}:
    raise SystemExit(f"unknown arm: {arm}")

# Matched checkpoints are more informative than arm-specific early stopping.
rl["EARLY_STOP_PATIENCE"] = max(int(n_ep) + 1, 1001)
rl["SAVE_MOLECULE_SNAPSHOTS"] = True
rl["SNAPSHOT_EVERY"] = 10
rl["EVAL_EVERY"] = 50

path = Path(output_dir) / "config.yaml"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(path)
PY
}

for seed in $SEEDS; do
  for arm in $ARMS; do
    run_root="$OUT_ROOT/seed_${seed}/${arm}"
    config="$(write_config "$seed" "$arm" "$run_root")"
    echo "=== seed=$seed arm=$arm config=$config ==="
    if [[ "$arm" == "chem_tether" || "$arm" == "current_plus_tether" ]]; then
      uv run --extra tether python train.py --config "$config" \
        | tee "$run_root/train.log"
    else
      uv run python train.py --config "$config" \
        | tee "$run_root/train.log"
    fi
  done
done

echo "=== matched tether-efficacy runs complete: $OUT_ROOT ==="
