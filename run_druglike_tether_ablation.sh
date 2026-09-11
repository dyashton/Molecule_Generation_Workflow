#!/usr/bin/env bash
# Mode I tether ablation on the druglike multi-objective.
# Arms: none, kl_only, chem_tether, current (KL + tether). No prior-FP.
# Does not edit train.yaml.
#
#   bash run_druglike_tether_ablation.sh
#   MGW_ABLATION_EPISODES=500 MGW_ABLATION_DEVICE=cuda:0 \
#     MGW_ABLATION_OUT=outputs/druglike_tether_ablation bash run_druglike_tether_ablation.sh
#
# Requires weights/guacamol_nc.pt, weights/chem_tether_bank.npz (for tether arms),
# and: uv sync --extra tether
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

N_EPISODES="${MGW_ABLATION_EPISODES:-1000}"
BATCH_SIZE="${MGW_ABLATION_BATCH_SIZE:-128}"
DEVICE="${MGW_ABLATION_DEVICE:-cuda:0}"
OUT_ROOT="${MGW_ABLATION_OUT:-$ROOT/outputs/druglike_tether_ablation}"
BANK="$ROOT/weights/chem_tether_bank.npz"
LOG="$OUT_ROOT/ablation.log"
TETHER_COEF="${MGW_ABLATION_TETHER_COEF:-0.75}"
ARMS="${MGW_ABLATION_ARMS:-none kl_only chem_tether current}"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

export MGW_OBJECTIVE=druglike
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== druglike tether ablation $(date -Iseconds) ==="
echo "episodes=$N_EPISODES batch=$BATCH_SIZE device=$DEVICE out=$OUT_ROOT"
echo "arms=$ARMS tether_coef=$TETHER_COEF (fixed when on) prior_fp=off"

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
  uv run python - "$ROOT" "$OUT_ROOT" "$arm" "$N_EPISODES" "$BATCH_SIZE" "$BANK" "$DEVICE" "$TETHER_COEF" <<'PY'
import sys
from pathlib import Path

import yaml

root, out_root, arm, n_ep, batch, bank, device, tether_coef = sys.argv[1:9]
n_ep, batch = int(n_ep), int(batch)
tether_coef = float(tether_coef)
src = Path(root) / "train.yaml"
cfg = yaml.safe_load(src.read_text())
arm_dir = Path(out_root) / arm
arm_dir.mkdir(parents=True, exist_ok=True)

cfg["DEVICE"] = device
cfg["RUN_NAME"] = f"druglike_{arm}"
cfg["OUTPUTS"]["DIR"] = str(arm_dir.resolve())
cfg["GENERATE"]["CKPT_PATH"] = str((Path(root) / "weights" / "guacamol_nc.pt").resolve())
if str(device).startswith("cpu"):
    cfg["GENERATE"]["N_CPU"] = 1

rl = cfg["RL_TRAIN"]
rl["N_EPISODES"] = n_ep
rl["BATCH_SIZE"] = batch
rl["EVAL_BATCH_SIZE"] = batch
rl["SNAPSHOT_BATCH_SIZE"] = min(int(rl.get("SNAPSHOT_BATCH_SIZE", 64)), batch)
rl["PRIOR_FP_CACHE"] = str((Path(root) / "weights" / "prior_fp_bank.smi").resolve())
rl["CHEM_TETHER_BANK"] = str(Path(bank).resolve())
rl["CHEM_TETHER_LOG"] = True
rl["CHEM_TETHER_MODE"] = "regularize"
rl["PRIOR_FP_FLOOR"] = 0.0

rl["KL_COEF"] = 0.0
rl["KL_COEF_START"] = 0.0
rl["KL_COEF_END"] = 0.0
rl["CHEM_TETHER_COEF"] = 0.0
rl["CHEM_TETHER_COEF_START"] = 0.0
rl["CHEM_TETHER_COEF_END"] = 0.0

if arm in {"kl_only", "current"}:
    rl["KL_COEF"] = 0.75
    rl["KL_COEF_START"] = 0.75
    rl["KL_COEF_END"] = 0.40
if arm in {"chem_tether", "current"}:
    rl["CHEM_TETHER_COEF"] = tether_coef
    rl["CHEM_TETHER_COEF_START"] = tether_coef
    rl["CHEM_TETHER_COEF_END"] = tether_coef
if arm not in {"none", "kl_only", "chem_tether", "current"}:
    raise SystemExit(f"unknown arm {arm}")

rl["EARLY_STOP_PATIENCE"] = max(n_ep + 1, 1001)
rl["SAVE_MOLECULE_SNAPSHOTS"] = True
rl["SNAPSHOT_EVERY"] = 10
rl["EVAL_EVERY"] = 50

out = arm_dir / "config.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
}

run_arm() {
  local arm="$1"
  local yaml
  yaml="$(write_arm_yaml "$arm")"
  echo
  echo "── arm $arm  $(date -Iseconds)  config=$yaml"
  if [[ "$arm" == "chem_tether" || "$arm" == "current" ]]; then
    uv run --extra tether python train.py --config "$yaml" \
      | tee "$OUT_ROOT/$arm/train.log"
  else
    uv run python train.py --config "$yaml" \
      | tee "$OUT_ROOT/$arm/train.log"
  fi
  echo "── arm $arm done  $(date -Iseconds)"
}

for arm in $ARMS; do
  run_arm "$arm"
done

echo
echo "=== druglike tether ablation finished $(date -Iseconds) ==="
echo "Results under $OUT_ROOT/{none,kl_only,chem_tether,current}/"
