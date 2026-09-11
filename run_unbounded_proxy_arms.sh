#!/usr/bin/env bash
# Unbounded single-proxy reward-hacking sweep.
# For each MGW_PROXY, default arms: D* ∈ {1.00, 1.30, 2.00} (no D*=0.50 for now).
# Soft MW penalty: expm1((MW-400)/50) subtracted from proxy (see MGW_MW_PENALTY_*).
#
#   bash run_unbounded_proxy_arms.sh
#   MGW_PROXY_OUT=outputs/unbounded_proxy_hacking_mw400 MGW_PROXY_LIST=mollogp \
#     bash run_unbounded_proxy_arms.sh
#   # include no_tether / D*=0.50:
#   MGW_DSTAR_ARMS="no_tether:1.00 dstar_050:0.50 dstar_100:1.00 dstar_130:1.30 dstar_200:2.00" ...
#
# Layout: outputs/unbounded_proxy_hacking/<proxy>/<arm>/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

N_EPISODES="${MGW_DSTAR_EPISODES:-1000}"
BATCH_SIZE="${MGW_DSTAR_BATCH_SIZE:-128}"
DEVICE="${MGW_DSTAR_DEVICE:-cuda:0}"
OUT_ROOT="${MGW_PROXY_OUT:-$ROOT/outputs/unbounded_proxy_hacking}"
BANK="$ROOT/weights/chem_tether_bank.npz"
LOG="$OUT_ROOT/proxy_arms.log"
TETHER_COEF="${MGW_DSTAR_TETHER_COEF:-25.0}"
KL_COEF="${MGW_DSTAR_KL_COEF:-1.0}"

# space-separated proxies
PROXIES="${MGW_PROXY_LIST:-mollogp molwt heavyatomcount bertzct numrotatablebonds tpsa mollogp_molwt}"
# per proxy: D* shells (D*=0.50 dropped for now — inward shell + MolLogP fought too hard)
ARMS="${MGW_DSTAR_ARMS:-dstar_100:1.00 dstar_130:1.30 dstar_200:2.00}"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

export MGW_OBJECTIVE=unbounded_proxy
export MGW_MW_PENALTY_START="${MGW_MW_PENALTY_START:-400}"
export MGW_MW_PENALTY_TAU="${MGW_MW_PENALTY_TAU:-50}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== unbounded proxy hacking arms $(date -Iseconds) ==="
echo "proxies=$PROXIES arms=$ARMS episodes=$N_EPISODES batch=$BATCH_SIZE device=$DEVICE"
echo "out=$OUT_ROOT tether_coef=$TETHER_COEF kl_coef=$KL_COEF"
echo "mw_penalty start=$MGW_MW_PENALTY_START tau=$MGW_MW_PENALTY_TAU (soft expm1)"

if [[ ! -f "$ROOT/weights/guacamol_nc.pt" ]]; then
  echo "Missing weights/guacamol_nc.pt" >&2
  exit 1
fi
if [[ ! -f "$BANK" ]]; then
  echo "Missing $BANK" >&2
  exit 1
fi

write_arm_yaml() {
  local proxy="$1"
  local arm="$2"
  local d_star="$3"
  local proxy_out="$OUT_ROOT/$proxy"
  uv run python - "$ROOT" "$proxy_out" "$arm" "$d_star" "$N_EPISODES" "$BATCH_SIZE" "$BANK" "$DEVICE" "$TETHER_COEF" "$KL_COEF" "$proxy" <<'PY'
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
    proxy,
) = sys.argv[1:12]
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
cfg["RUN_NAME"] = f"proxy_{proxy}_{arm}"
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
rl["PRIOR_FP_FLOOR"] = 0.0
rl["KL_COEF"] = kl_coef
rl["KL_COEF_START"] = kl_coef
rl["KL_COEF_END"] = kl_coef
rl["CHEM_TETHER_MODE"] = "explore"
rl["CHEM_TETHER_D_STAR_SCHEDULE"] = "static"
rl["CHEM_TETHER_D_STAR"] = d_star
rl["CHEM_TETHER_COEF"] = tether_coef
rl["CHEM_TETHER_COEF_START"] = tether_coef
rl["CHEM_TETHER_COEF_END"] = tether_coef
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
  local proxy="$1"
  local arm="$2"
  local d_star="$3"
  local yaml
  export MGW_PROXY="$proxy"
  yaml="$(write_arm_yaml "$proxy" "$arm" "$d_star")"
  echo
  echo "── proxy=$proxy arm=$arm D*=$d_star  $(date -Iseconds)  config=$yaml"
  uv run --extra tether python train.py --config "$yaml" \
    | tee "$OUT_ROOT/$proxy/$arm/train.log"
  echo "── proxy=$proxy arm=$arm done  $(date -Iseconds)"
}

for proxy in $PROXIES; do
  mkdir -p "$OUT_ROOT/$proxy"
  for spec in $ARMS; do
    arm="${spec%%:*}"
    d_star="${spec##*:}"
    run_arm "$proxy" "$arm" "$d_star"
  done
done

echo
echo "=== all unbounded proxy arms finished $(date -Iseconds) ==="
echo "Results under $OUT_ROOT/<proxy>/{dstar_100,dstar_130,dstar_200}/"
