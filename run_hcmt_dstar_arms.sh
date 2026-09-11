#!/usr/bin/env bash
# Mode-II explore arms with static novelty radii D* on calibrated d
# (d'_c = d_c / s_c, s_c = bank-median leave-one-out k-NN; typical d≈1).
# Arms: 0.80 / 1.00 / 1.30 / 2.00 around the on-manifold baseline.
# Shared: explore mode, PRIOR_FP_FLOOR=0, fixed tether coef 4.0, fixed KL 0.50.
# Does not edit train.yaml. Writes a new out dir so λ=0.75 runs stay intact.
#
#   bash run_hcmt_dstar_arms.sh
#   MGW_DSTAR_EPISODES=500 MGW_DSTAR_DEVICE=cuda:0 \
#     MGW_DSTAR_OUT=outputs/druglike_dstar_arms_nn_c4 bash run_hcmt_dstar_arms.sh
#
# After runs, plot shells (d′) then chemistry (t-SNE):
#   PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_shells.py \
#     --root outputs/druglike_dstar_arms_nn_c4
#   PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_tsne.py \
#     --root outputs/druglike_dstar_arms_nn_c4
#
# Requires weights/guacamol_nc.pt, weights/chem_tether_bank.npz, and
#   uv sync --extra tether
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

N_EPISODES="${MGW_DSTAR_EPISODES:-1000}"
BATCH_SIZE="${MGW_DSTAR_BATCH_SIZE:-128}"
DEVICE="${MGW_DSTAR_DEVICE:-cuda:0}"
OUT_ROOT="${MGW_DSTAR_OUT:-$ROOT/outputs/druglike_dstar_arms_nn_c4}"
BANK="$ROOT/weights/chem_tether_bank.npz"
LOG="$OUT_ROOT/dstar_arms.log"
# Fixed λ_tether (no anneal) so arms differ only by D*.
TETHER_COEF="${MGW_DSTAR_TETHER_COEF:-4.0}"
# Moderate fixed KL: protect validity while allowing shells to move.
KL_COEF="${MGW_DSTAR_KL_COEF:-0.50}"

# space-separated "arm_name:d_star" (NN-units; 1 = typical bank neighbor)
ARMS="${MGW_DSTAR_ARMS:-dstar_080:0.80 dstar_100:1.00 dstar_130:1.30 dstar_200:2.00}"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

export MGW_OBJECTIVE="${MGW_OBJECTIVE:-druglike}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== HCMT D* explore arms $(date -Iseconds) ==="
echo "objective=$MGW_OBJECTIVE episodes=$N_EPISODES batch=$BATCH_SIZE device=$DEVICE out=$OUT_ROOT"
echo "arms=$ARMS PRIOR_FP_FLOOR=0 MODE=explore SCHEDULE=static"
echo "tether_coef=$TETHER_COEF (fixed) kl_coef=$KL_COEF (fixed) d=calibrated_mean (NN-units)"

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
  local d_star="$2"
  uv run python - "$ROOT" "$OUT_ROOT" "$arm" "$d_star" "$N_EPISODES" "$BATCH_SIZE" "$BANK" "$DEVICE" "$TETHER_COEF" "$KL_COEF" <<'PY'
import sys
from pathlib import Path

import yaml

(
    root,
    out_root,
    arm,
    d_star,
    n_ep,
    batch,
    bank,
    device,
    tether_coef,
    kl_coef,
) = sys.argv[1:11]
n_ep, batch = int(n_ep), int(batch)
d_star = float(d_star)
tether_coef = float(tether_coef)
kl_coef = float(kl_coef)
if arm == "no_tether":
    tether_coef = 0.0
src = Path(root) / "train.yaml"
cfg = yaml.safe_load(src.read_text())
arm_dir = Path(out_root) / arm
arm_dir.mkdir(parents=True, exist_ok=True)

cfg["DEVICE"] = device
cfg["RUN_NAME"] = f"hcmt_{arm}"
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

# Explore shell hygiene: no prior-FP floor fighting (d − D*)².
rl["PRIOR_FP_FLOOR"] = 0.0

# Fixed KL (weaker than train.yaml default) so D* can move the policy.
rl["KL_COEF"] = kl_coef
rl["KL_COEF_START"] = kl_coef
rl["KL_COEF_END"] = kl_coef

# Mode II: static novelty radius on normalized d (arms differ only here).
rl["CHEM_TETHER_MODE"] = "explore"
rl["CHEM_TETHER_D_STAR_SCHEDULE"] = "static"
rl["CHEM_TETHER_D_STAR"] = d_star
# Fixed λ_tether (START=END) so D* comparison is not confounded by anneal.
rl["CHEM_TETHER_COEF"] = tether_coef
rl["CHEM_TETHER_COEF_START"] = tether_coef
rl["CHEM_TETHER_COEF_END"] = tether_coef

# Matched length: disable early stop.
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
  local d_star="$2"
  local yaml
  yaml="$(write_arm_yaml "$arm" "$d_star")"
  echo
  echo "── arm $arm  D*=$d_star  $(date -Iseconds)  config=$yaml"
  uv run --extra tether python train.py --config "$yaml" \
    | tee "$OUT_ROOT/$arm/train.log"
  echo "── arm $arm done  $(date -Iseconds)"
}

for spec in $ARMS; do
  arm="${spec%%:*}"
  d_star="${spec##*:}"
  run_arm "$arm" "$d_star"
done

echo
echo "=== all D* arms finished $(date -Iseconds) ==="
echo "Results under $OUT_ROOT/{dstar_080,dstar_100,dstar_130,dstar_200}/"
echo "Plot: PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_tsne.py --root $OUT_ROOT"
