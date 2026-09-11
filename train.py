"""
REINFORCE training with pluggable rewards (`reward.compute_reward_batch`).

Mitigations vs peak→crash→plateau:
  1. EMA monitor; fixed eval for logging; best.pt + early-stop on train EMA
  2. KL(π || π_pretrained) anchor (KL_COEF)
  3. Entropy + temperature schedules; larger default batch
  4. CO2: unique-canonical learn set, elite replay, soft-bounded z-rewards,
     Morgan-FP soft floor vs cached prior SMILES (PRIOR_FP_FLOOR)

Policy gradients use advantages; plots/checkpoints use EMA (or eval EMA).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from torch.cuda.amp import GradScaler

from objectives.co2_rl_helpers import (
    EliteReplayBuffer,
    baseline_relative_advantages,
    batch_health_metrics,
    canonical_smiles_or_raw,
    decline_streak_should_restore,
    early_stop_should_stop,
    explore_schedule_progress,
    inception_replay_k,
    linear_schedule,
    should_skip_thin_batch,
    top_quantile_indices,
    unique_canonical_learn_indices,
    update_ema,
)
from objectives.chemical_tether import ChemicalTether, SHELL_HIT_BAND, tether_cost, tether_loss
from objectives.prior_fp_tether import load_smiles_cache, save_smiles_cache
from reward import (
    RewardBatch,
    compute_reward_batch,
    configure_co2_scorer,
    ensure_co2_scorer,
)
from transpharmer.dataset import pad_token, stoi, tokenize, untokenize
from transpharmer.model import GPT, sample
from transpharmer.SMILEValidifier import valid_smiles
from transpharmer.utils.io import load_config
from transpharmer.utils.mol import get_props_from_mols
from transpharmer.utils.optim import build_optimizer
from transpharmer.utils.seed import set_seed


def _co2_objective() -> bool:
    return os.environ.get("MGW_OBJECTIVE", "nlrp").strip().lower() == "co2"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _run_name_from_env_or_config(config_run_name: str) -> str:
    """CO2/druglike get env-based names; else YAML RUN_NAME."""
    obj = os.environ.get("MGW_OBJECTIVE", "nlrp").strip().lower()
    if obj == "druglike":
        return "druglike"
    if obj == "unbounded_proxy":
        proxy = os.environ.get("MGW_PROXY", "mollogp").strip().lower() or "mollogp"
        return f"proxy_{proxy}"
    if obj != "co2":
        return config_run_name
    pfc = "full" if _env_flag("MGW_INCLUDE_PFC", default=True) else "nopf"
    nopen = "_nopen" if _env_flag("MGW_NO_PENALTIES", default=False) else ""
    if _env_flag("MGW_CO2_SIGMOID", default=False):
        # Sigmoid uses RANGES as soft limits; hard ceil clamp does not apply.
        return f"co2_{pfc}_sig{nopen}"
    ceil = "z_ceil" if _env_flag("MGW_OBJECTIVE_CEILINGS", default=False) else "z_uncapped"
    return f"co2_{pfc}_{ceil}{nopen}"


def _new_run_dir(out_root: Path, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_root / f"{ts}_{run_name}"


def _fmt_metric(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return f"{float(value):.4f}"


def _fmt_num(value: float | None, width: int = 6, prec: int = 4) -> str:
    """Compact float for episode logs; non-finite → '—'."""
    if value is None:
        return f"{'—':>{width}}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{'—':>{width}}"
    if math.isnan(v) or math.isinf(v):
        return f"{'—':>{width}}"
    return f"{v:{width}.{prec}f}"


def _fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "—"
    return f"{100.0 * float(num) / float(den):.0f}%"


def _nanmean(x: np.ndarray) -> float:
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float("nan")
    return float(np.nanmean(x))


def _print_episode_summary(
    ep: int, n_episodes: int, metrics: dict, *, batch_size: int
) -> None:
    """Human-readable per-episode training snapshot."""
    skipped = float(metrics.get("skipped_update", 0.0)) >= 0.5
    status = "SKIP (too few gated)" if skipped else "updated"

    n_valid = int(metrics.get("n_valid", 0))
    n_gated = int(metrics.get("n_gated", 0))
    n_learn = int(metrics.get("n_learn") or 0)
    n_replay = int(metrics.get("n_replay") or 0)
    n_mw_zero = int(metrics.get("n_mw_zero") or 0)

    reward = metrics.get("gated_mean_reward", metrics.get("average_policy_reward"))
    p90 = metrics.get("gated_p90")
    ema = metrics.get("average_raw_objective")
    phys_ema = metrics.get("average_physchem_objective")
    base = metrics.get("baseline")
    loss = metrics.get("policy_loss")
    kl = metrics.get("kl")
    kl_coef = metrics.get("kl_coef")
    temp = metrics.get("temperature")
    hcoef = metrics.get("entropy_coef")
    top_q = metrics.get("top_quantile")
    mean_mw = metrics.get("mean_mw_scale")

    print(f"\n── episode {ep}/{n_episodes}  {status} ──", flush=True)
    print(
        f"  molecules   valid {n_valid}/{batch_size} ({_fmt_pct(n_valid, batch_size)})  "
        f"gated {n_gated}/{n_valid} ({_fmt_pct(n_gated, n_valid)})  "
        f"unique_gated {int(metrics.get('n_unique_gated') or 0)} "
        f"({_fmt_pct(int(metrics.get('n_unique_gated') or 0), batch_size)})  "
        f"policy_batch {n_learn}  elite_replay {n_replay}"
        + (f"  boost" if metrics.get("inception_boost") else ""),
        flush=True,
    )
    if n_gated > 0:
        mw_detail = (
            f"mean_scale {_fmt_num(mean_mw)}  "
            f"MW_out_of_band {n_mw_zero}/{n_gated} ({_fmt_pct(n_mw_zero, n_gated)})"
        )
    else:
        mw_detail = f"mean_scale {_fmt_num(mean_mw)}  MW_out_of_band —"
    print(f"  mw_scale    {mw_detail}", flush=True)
    print(
        f"  reward      gated_mean {_fmt_num(reward)}  "
        f"p90 {_fmt_num(p90)}  "
        f"EMA_full {_fmt_num(ema)}  "
        f"EMA_physchem {_fmt_num(phys_ema)}  "
        f"baseline {_fmt_num(base)}",
        flush=True,
    )
    print(
        f"  optim       loss {_fmt_num(loss)}  "
        f"KL {_fmt_num(kl)} (coef {_fmt_num(kl_coef)})  "
        f"temp {_fmt_num(temp, prec=3)}  "
        f"entropy_coef {_fmt_num(hcoef)}  "
        f"top_fraction {_fmt_num(top_q, prec=3)}",
        flush=True,
    )
    if metrics.get("valid_frac_ema") is not None or metrics.get("unique_gated_frac_ema") is not None:
        print(
            f"  health      valid {_fmt_num(metrics.get('valid_frac'), prec=3)} "
            f"(ema {_fmt_num(metrics.get('valid_frac_ema'), prec=3)})  "
            f"unique_gated {_fmt_num(metrics.get('unique_gated_frac'), prec=3)} "
            f"(ema {_fmt_num(metrics.get('unique_gated_frac_ema'), prec=3)})",
            flush=True,
        )
    print(
        f"  regularize  KL {_fmt_num(kl)} (coef {_fmt_num(kl_coef)})  "
        f"prior_fp {_fmt_num(metrics.get('prior_fp_mean_factor'))} "
        f"(floor {_fmt_num(metrics.get('prior_fp_floor'))})  "
        f"tether {_fmt_num(metrics.get('tether'))} "
        f"(coef {_fmt_num(metrics.get('tether_coef'))})  "
        f"d {_fmt_num(metrics.get('tether_d'))} "
        f"d* {_fmt_num(metrics.get('tether_d_star'))} "
        f"|d-d*| {_fmt_num(metrics.get('tether_abs_dev'))} "
        f"hit {_fmt_num(metrics.get('tether_shell_hit'))}  "
        f"d'_m {_fmt_num(metrics.get('tether_dp_morgan'))} "
        f"d'_2d {_fmt_num(metrics.get('tether_dp_2d'))} "
        f"d'_3d {_fmt_num(metrics.get('tether_dp_3d'))}  "
        f"raw_m {_fmt_num(metrics.get('tether_d_morgan'))} "
        f"raw_2d {_fmt_num(metrics.get('tether_d_2d'))} "
        f"raw_3d {_fmt_num(metrics.get('tether_d_3d'))}",
        flush=True,
    )


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve(config_dir: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


def _plot_rewards(
    all_rewards: list[float],
    out_path: Path,
    title: str = "Mean raw objective (valid SMILES)",
    ylabel: str = "Mean raw objective",
    physchem_rewards: list[float] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(all_rewards) + 1),
        all_rewards,
        label="EMA gated mean (full)",
        color="blue",
    )
    if physchem_rewards is not None and len(physchem_rewards) > 0:
        n = min(len(all_rewards), len(physchem_rewards))
        plt.plot(
            range(1, n + 1),
            physchem_rewards[:n],
            label="EMA gated physchem",
            color="orange",
        )
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def _plot_eval_curve(
    episodes: list[int],
    eval_raw: list[float],
    eval_ema: list[float],
    out_path: Path,
    *,
    best_episode: int | None = None,
) -> None:
    """Sparse fixed-eval monitor (the metric that selects best.pt when EVAL_EVERY>0)."""
    if not episodes:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, eval_raw, label="fixed eval raw", color="green", marker="o", ms=3)
    plt.plot(episodes, eval_ema, label="fixed eval EMA", color="purple", lw=2)
    if best_episode is not None and best_episode > 0:
        plt.axvline(best_episode, color="gray", ls="--", lw=1, label=f"best @ ep {best_episode}")
    plt.xlabel("Episode")
    plt.ylabel("Gated mean reward")
    plt.title("Fixed eval (best-checkpoint monitor)")
    plt.grid()
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def _write_eval_curve(
    path: Path,
    episodes: list[int],
    eval_raw: list[float],
    eval_ema: list[float],
) -> None:
    with path.open("w") as f:
        f.write("episode,eval_raw,eval_ema\n")
        for ep, raw, ema in zip(episodes, eval_raw, eval_ema):
            f.write(f"{ep},{raw},{ema}\n")


def _write_molecule_snapshot(
    path: Path,
    episode: int,
    smiles_list: list[str],
    batch: RewardBatch | None,
    valid_mask: list[bool],
    extra_channels: dict[str, np.ndarray] | None = None,
) -> None:
    """Write scored SMILES snapshot; invalids get NaN score columns."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "episode",
        "smiles",
        "valid",
        "policy",
        "raw_total",
        "gate_ok",
        "raw_physchem",
        "quality",
        "mw_scale",
        "raw_ba",
        "raw_div",
        "tether_d_morgan",
        "tether_d_2d",
        "tether_d_3d",
        "tether_dp_morgan",
        "tether_dp_2d",
        "tether_dp_3d",
    ]
    nan = float("nan")
    extra = extra_channels or {}

    def _extra(name: str, i: int) -> float:
        arr = extra.get(name)
        if arr is None or i >= len(arr):
            return nan
        v = arr[i]
        try:
            return float(v)
        except (TypeError, ValueError):
            return nan

    valid_i = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for smi, is_valid in zip(smiles_list, valid_mask):
            if is_valid and batch is not None:
                row = [
                    episode,
                    smi,
                    True,
                    batch.policy[valid_i],
                    batch.raw_total[valid_i],
                    batch.gate_ok[valid_i],
                    batch.raw_physchem[valid_i],
                    batch.quality[valid_i],
                    batch.mw_scale[valid_i],
                    batch.raw_ba[valid_i],
                    batch.raw_div[valid_i],
                    _extra("morgan", valid_i),
                    _extra("2d", valid_i),
                    _extra("3d", valid_i),
                    _extra("dp_morgan", valid_i),
                    _extra("dp_2d", valid_i),
                    _extra("dp_3d", valid_i),
                ]
                valid_i += 1
            else:
                row = [
                    episode,
                    smi,
                    False,
                    nan,
                    nan,
                    False,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                ]
            w.writerow(row)


def _write_best_pointer(best_pt: Path, src: Path) -> None:
    """Copy checkpoint to best.pt (symlink if possible)."""
    best_pt.parent.mkdir(parents=True, exist_ok=True)
    try:
        if best_pt.exists() or best_pt.is_symlink():
            best_pt.unlink()
        os.symlink(src.resolve(), best_pt)
    except OSError:
        shutil.copy2(src, best_pt)


def _cleanup_old_periodic(
    checkpoint_dir: Path, run_name: str, keep_last: int
) -> None:
    if keep_last <= 0:
        return
    pat = re.compile(rf"^periodic_{re.escape(run_name)}_ep(\d+)\.pt$")
    files: list[tuple[int, Path]] = []
    for p in checkpoint_dir.glob(f"periodic_{run_name}_ep*.pt"):
        m = pat.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    while len(files) > keep_last:
        _, path = files.pop(0)
        path.unlink(missing_ok=True)


def _cleanup_old_best(checkpoint_dir: Path, run_name: str, keep: Path) -> None:
    """Delete prior best_*.pt files except the current keep path."""
    keep_res = keep.resolve()
    for p in checkpoint_dir.glob(f"best_{run_name}_ep*.pt"):
        try:
            if p.resolve() != keep_res:
                p.unlink(missing_ok=True)
        except OSError:
            pass


class RLPipeline:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str | None = None,
        *,
        resume_dir: str | Path | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.config = load_config(str(self.config_path))

        out_root = _resolve(self.config_dir, str(self.config.OUTPUTS.DIR))
        self.run_name = _run_name_from_env_or_config(
            str(getattr(self.config, "RUN_NAME", "default"))
        )

        self.start_episode = 0  # last completed episode; training continues at start_episode+1
        resume_ckpt: dict | None = None
        resume_path: Path | None = None

        if resume_dir is not None:
            rd = Path(resume_dir)
            if not rd.is_absolute():
                rd = _resolve(self.config_dir, rd)
            if rd.is_file() and rd.suffix == ".pt":
                resume_path = rd.resolve()
                self.run_dir = resume_path.parent.parent
            elif rd.is_dir():
                self.run_dir = rd.resolve()
                cand = self.run_dir / "checkpoints" / "best.pt"
                if not cand.exists():
                    raise FileNotFoundError(f"No best.pt under {cand}")
                resume_path = cand.resolve()
            else:
                raise FileNotFoundError(f"Resume path not found: {rd}")
            self.checkpoint_dir = self.run_dir / "checkpoints"
            self.graph_dir = self.run_dir / "graphs"
            self.ckpt_path = resume_path
            print(f"Resuming run directory: {self.run_dir}", flush=True)
            print(f"Resume checkpoint: {self.ckpt_path}", flush=True)
        else:
            ckpt = ckpt_path or str(self.config.GENERATE.CKPT_PATH)
            self.ckpt_path = _resolve(self.config_dir, ckpt)
            self.run_dir = _new_run_dir(out_root, self.run_name)
            self.checkpoint_dir = self.run_dir / "checkpoints"
            self.graph_dir = self.run_dir / "graphs"
            print(f"Run directory: {self.run_dir}", flush=True)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.model = GPT(self.config)
        self.device = self.config.DEVICE
        self.reward_baseline = 0.0
        self.baseline_momentum = float(self.config.RL_TRAIN.BASELINE_MOMENTUM)
        self.entropy_coef = float(self.config.RL_TRAIN.ENTROPY_COEF)
        self.grad_clip = float(self.config.RL_TRAIN.GRAD_NORM_CLIP)
        self.save_every = int(self.config.RL_TRAIN.SAVE_EVERY)
        self.keep_last_n_periodic = int(self.config.RL_TRAIN.KEEP_LAST_N_PERIODIC)
        rl = self.config.RL_TRAIN
        self.min_gated = int(getattr(rl, "MIN_GATED", 8))
        top_q_default = float(getattr(rl, "TOP_QUANTILE", 1.0))
        self.top_quantile_start = float(
            getattr(rl, "TOP_QUANTILE_START", top_q_default)
        )
        self.top_quantile_end = float(
            getattr(rl, "TOP_QUANTILE_END", top_q_default)
        )
        self.top_quantile = self.top_quantile_start
        self.monitor_ema_momentum = float(getattr(rl, "MONITOR_EMA_MOMENTUM", 0.9))
        self.decline_patience = int(getattr(rl, "DECLINE_PATIENCE", 200))
        self.decline_margin = float(getattr(rl, "DECLINE_MARGIN", 0.08))
        self.unique_gated_frac_floor = float(
            getattr(rl, "UNIQUE_GATED_FRAC_FLOOR", 0.10)
        )
        self.early_stop_patience = int(getattr(rl, "EARLY_STOP_PATIENCE", 500))
        self.eval_every = int(getattr(rl, "EVAL_EVERY", 0))
        self.eval_batch_size = int(
            getattr(rl, "EVAL_BATCH_SIZE", self.config.RL_TRAIN.BATCH_SIZE)
        )
        self.save_molecule_snapshots = bool(
            getattr(rl, "SAVE_MOLECULE_SNAPSHOTS", True)
        )
        self.snapshot_every = int(getattr(rl, "SNAPSHOT_EVERY", 5))
        self.snapshot_batch_size = int(getattr(rl, "SNAPSHOT_BATCH_SIZE", 64))
        self.molecule_snapshot_dir = self.run_dir / "molecule_snapshots"
        if self.save_molecule_snapshots:
            self.molecule_snapshot_dir.mkdir(parents=True, exist_ok=True)
        kl_default = float(getattr(rl, "KL_COEF", 0.0))
        self.kl_coef_start = float(getattr(rl, "KL_COEF_START", kl_default))
        self.kl_coef_end = float(getattr(rl, "KL_COEF_END", kl_default))
        self.kl_coef = self.kl_coef_start
        self.explore_episodes = int(getattr(rl, "EXPLORE_EPISODES", 0))
        entropy_default = float(self.config.RL_TRAIN.ENTROPY_COEF)
        self.entropy_coef_start = float(
            getattr(rl, "ENTROPY_COEF_START", entropy_default)
        )
        self.entropy_coef_end = float(
            getattr(rl, "ENTROPY_COEF_END", entropy_default)
        )
        temp_default = float(self.config.GENERATE.TEMPERATURE)
        self.temperature_start = float(
            getattr(rl, "TEMPERATURE_START", temp_default)
        )
        self.temperature_end = float(
            getattr(rl, "TEMPERATURE_END", temp_default)
        )
        self.decline_streak = 0
        self.episodes_since_best = 0
        self.monitor_ema = float("nan")
        self.physchem_ema = float("nan")
        self.valid_frac_ema = float("nan")
        self.unique_gated_frac_ema = float("nan")
        self.eval_ema = float("nan")
        self.best_train_ema = float("-inf")
        self.reward_std_ema = float("nan")
        self.scheduled_temperature = self.temperature_start
        self.current_temperature = self.temperature_start
        self.entropy_coef = self.entropy_coef_start
        self.co2_mode = _co2_objective()
        self.prior_fp_floor = float(getattr(rl, "PRIOR_FP_FLOOR", 0.0))
        self.prior_fp_bank_size = int(getattr(rl, "PRIOR_FP_BANK_SIZE", 2048))
        self.prior_fp_cache = _resolve(
            self.config_dir,
            str(getattr(rl, "PRIOR_FP_CACHE", "./weights/prior_fp_bank.smi")),
        )
        tether_default = float(getattr(rl, "CHEM_TETHER_COEF", 0.0))
        self.chem_tether_coef_start = float(
            getattr(rl, "CHEM_TETHER_COEF_START", tether_default)
        )
        self.chem_tether_coef_end = float(
            getattr(rl, "CHEM_TETHER_COEF_END", tether_default)
        )
        self.chem_tether_coef = self.chem_tether_coef_start
        self.chem_tether_lambda_morgan = float(
            getattr(rl, "CHEM_TETHER_LAMBDA_MORGAN", 1.0)
        )
        self.chem_tether_lambda_2d = float(getattr(rl, "CHEM_TETHER_LAMBDA_2D", 1.0))
        self.chem_tether_lambda_3d = float(getattr(rl, "CHEM_TETHER_LAMBDA_3D", 1.0))
        self.chem_tether_k = int(getattr(rl, "CHEM_TETHER_K", 1))
        self.chem_tether_bank = _resolve(
            self.config_dir,
            str(getattr(rl, "CHEM_TETHER_BANK", "./weights/chem_tether_bank.npz")),
        )
        self.chem_tether_2d_model = str(
            getattr(rl, "CHEM_TETHER_2D_MODEL", "DeepChem/ChemBERTa-77M-MLM")
        )
        mode_raw = str(getattr(rl, "CHEM_TETHER_MODE", "regularize")).strip().lower()
        if mode_raw not in ("regularize", "explore"):
            raise ValueError(
                f"CHEM_TETHER_MODE must be regularize|explore, got {mode_raw!r}"
            )
        self.chem_tether_mode = mode_raw
        sched_raw = str(
            getattr(rl, "CHEM_TETHER_D_STAR_SCHEDULE", "static")
        ).strip().lower()
        if sched_raw not in ("static", "adaptive"):
            raise ValueError(
                f"CHEM_TETHER_D_STAR_SCHEDULE must be static|adaptive, got {sched_raw!r}"
            )
        self.chem_tether_d_star_schedule = sched_raw
        self.chem_tether_d_star_fixed = float(getattr(rl, "CHEM_TETHER_D_STAR", 1.0))
        self.chem_tether_d_star_start = float(
            getattr(rl, "CHEM_TETHER_D_STAR_START", 0.8)
        )
        self.chem_tether_d_star_end = float(getattr(rl, "CHEM_TETHER_D_STAR_END", 1.5))
        self.chem_tether_d_star = float("nan")
        log_raw = getattr(rl, "CHEM_TETHER_LOG", False)
        if isinstance(log_raw, str):
            self.chem_tether_log = log_raw.strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
        else:
            self.chem_tether_log = bool(log_raw)
        # Lagging baseline for tether REINFORCE (avoids zero advantages when the
        # whole batch drifts off-shell together).
        self.chem_tether_cost_ema = float("nan")
        self.chem_tether_cost_ema_decay = float(
            getattr(rl, "CHEM_TETHER_COST_EMA", 0.9)
        )
        self.chem_tether: ChemicalTether | None = None
        if self.co2_mode:
            configure_co2_scorer(
                scaffold_bucket=int(getattr(rl, "SCAFFOLD_BUCKET", 0)),
                scaffold_penalty=float(getattr(rl, "SCAFFOLD_PENALTY", 0.0)),
                use_scaffold_filter=bool(int(getattr(rl, "SCAFFOLD_BUCKET", 0)) > 0),
                n_workers=int(getattr(rl, "REWARD_WORKERS", 0)),
                prior_fp_floor=self.prior_fp_floor,
            )
            # Fork pool before CUDA — spawn IPC erases most of the speedup.
            scorer = ensure_co2_scorer()
            self.include_pfc = bool(scorer.include_pfc)
            self.use_objective_ceilings = bool(scorer.use_objective_ceilings)
            self.use_sigmoid_reward = bool(scorer.use_sigmoid_reward)
        else:
            self.include_pfc = True
            self.use_objective_ceilings = False
            self.use_sigmoid_reward = False
        self.inception_capacity = int(getattr(rl, "INCEPTION_CAPACITY", 128))
        self.inception_batch = int(getattr(rl, "INCEPTION_BATCH", 0))
        self.inception_batch_boost = int(
            getattr(rl, "INCEPTION_BATCH_BOOST", max(self.inception_batch, 32))
        )
        self.inception_boost_patience = int(getattr(rl, "INCEPTION_BOOST_PATIENCE", 25))
        self.inception_max_frac = float(getattr(rl, "INCEPTION_MAX_FRAC", 0.5))
        self.elite_buffer = EliteReplayBuffer(capacity=max(self.inception_capacity, 1))
        self._elite_rng = np.random.default_rng(int(self.config.RANDOM_SEED) + 7)
        explore_s = (
            str(self.explore_episodes)
            if self.explore_episodes > 0
            else "full_run"
        )
        print(
            "RL mitigations: "
            f"ema={self.monitor_ema_momentum}, early_stop={self.early_stop_patience}, "
            f"eval_every={self.eval_every}, explore={explore_s}, "
            f"kl={self.kl_coef_start}→{self.kl_coef_end}, "
            f"chem_tether={self.chem_tether_coef_start}→{self.chem_tether_coef_end}, "
            f"mode={self.chem_tether_mode}, "
            f"entropy={self.entropy_coef_start}→{self.entropy_coef_end}, "
            f"temp={self.temperature_start}→{self.temperature_end}, "
            f"top_q={self.top_quantile_start}→{self.top_quantile_end}",
            flush=True,
        )
        if self.co2_mode:
            pfc_bit = "physchem+PFC" if self.include_pfc else "physchem (no PFC)"
            score_bit = "sigmoid(RANGES)" if self.use_sigmoid_reward else "z-score"
            print(
                f"CO2 RL mode: soft-bounded {score_bit} {pfc_bit} + "
                f"prior-FP floor + elite replay; "
                f"min_gated={self.min_gated}, "
                f"scaffold_bucket={getattr(rl, 'SCAFFOLD_BUCKET', 0)}, "
                f"scaffold_penalty={getattr(rl, 'SCAFFOLD_PENALTY', 0.0)}, "
                f"prior_fp_floor={self.prior_fp_floor}, "
                f"prior_fp_bank={self.prior_fp_bank_size}, "
                f"reward_workers={getattr(rl, 'REWARD_WORKERS', 0)}, "
                f"inception_capacity={self.inception_capacity}, "
                f"inception_batch={self.inception_batch}, "
                f"inception_boost={self.inception_batch_boost}@"
                f"patience{self.inception_boost_patience} "
                f"max_frac={self.inception_max_frac}, "
                f"unique_gated_floor={self.unique_gated_frac_floor}, "
                f"decline_patience={self.decline_patience}, "
                f"decline_margin={self.decline_margin}, "
                f"include_pfc={self.include_pfc}, "
                f"objective_ceilings={self.use_objective_ceilings}, "
                f"sigmoid_reward={self.use_sigmoid_reward}, "
                f"lr={getattr(rl.LR, 'LEARNING_RATE', '?')}",
                flush=True,
            )

        if self.config.DEVICE == "cpu":
            checkpoint = torch.load(
                str(self.ckpt_path), map_location="cpu", weights_only=False
            )
        else:
            checkpoint = torch.load(str(self.ckpt_path), map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            resume_ckpt = checkpoint
            print("Loaded model state dict from training checkpoint.", flush=True)
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded model state dict from model checkpoint.", flush=True)

        self.model = self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config.RL_TRAIN)
        self.ref_model = self._build_ref_model()
        if self.co2_mode and self.prior_fp_floor > 0:
            self._ensure_prior_fp_bank()
        self._maybe_load_chem_tether()

        if self.device == "cuda" or str(self.device).startswith("cuda:"):
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.best_reward = float("-inf")
        self.curve_path = self.run_dir / "reward_curve.csv"
        self.metrics_path = self.run_dir / "rl_metrics.csv"
        self.eval_curve_path = self.run_dir / "eval_curve.csv"
        self.raw_curve: list[float] = []
        self.physchem_curve: list[float] = []
        self.eval_episodes: list[int] = []
        self.eval_raw_curve: list[float] = []
        self.eval_ema_curve: list[float] = []
        self.best_eval_episode = 0
        self.best_eval_ema = float("-inf")

        if resume_ckpt is not None and resume_dir is not None:
            if "optimizer_state_dict" in resume_ckpt:
                try:
                    self.optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                    print("Restored optimizer state.", flush=True)
                except Exception as e:
                    print(f"Optimizer restore warning: {e}", flush=True)
            self.start_episode = int(resume_ckpt.get("episode", 0))
            if "best_reward" in resume_ckpt:
                self.best_reward = float(resume_ckpt["best_reward"])
            if "reward_baseline" in resume_ckpt:
                self.reward_baseline = float(resume_ckpt["reward_baseline"])
            if "monitor_ema" in resume_ckpt:
                self.monitor_ema = float(resume_ckpt["monitor_ema"])
            if "physchem_ema" in resume_ckpt:
                self.physchem_ema = float(resume_ckpt["physchem_ema"])
            if "reward_std_ema" in resume_ckpt:
                self.reward_std_ema = float(resume_ckpt["reward_std_ema"])
            if "eval_ema" in resume_ckpt:
                self.eval_ema = float(resume_ckpt["eval_ema"])
            if "best_train_ema" in resume_ckpt:
                self.best_train_ema = float(resume_ckpt["best_train_ema"])
            if "best_eval_episode" in resume_ckpt:
                self.best_eval_episode = int(resume_ckpt["best_eval_episode"])
            if "best_eval_ema" in resume_ckpt:
                self.best_eval_ema = float(resume_ckpt["best_eval_ema"])
            if "episodes_since_best" in resume_ckpt:
                self.episodes_since_best = int(resume_ckpt["episodes_since_best"])
            if "elite_buffer" in resume_ckpt and resume_ckpt["elite_buffer"]:
                self.elite_buffer.load(
                    [(str(s), float(q)) for s, q in resume_ckpt["elite_buffer"]]
                )
                print(
                    f"Restored elite replay buffer: n={len(self.elite_buffer)}",
                    flush=True,
                )
            if self.curve_path.exists():
                import csv

                with self.curve_path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            self.raw_curve.append(float(row["reward"]))
                        except (KeyError, ValueError):
                            self.raw_curve.append(float("nan"))
                        try:
                            self.physchem_curve.append(float(row["physchem"]))
                        except (KeyError, ValueError):
                            self.physchem_curve.append(float("nan"))
                # Pad/truncate to start_episode length if needed
                if len(self.raw_curve) > self.start_episode:
                    self.raw_curve = self.raw_curve[: self.start_episode]
                if len(self.physchem_curve) > self.start_episode:
                    self.physchem_curve = self.physchem_curve[: self.start_episode]
                while len(self.raw_curve) < self.start_episode:
                    self.raw_curve.append(float("nan"))
                while len(self.physchem_curve) < self.start_episode:
                    self.physchem_curve.append(float("nan"))
            else:
                self.raw_curve = [float("nan")] * self.start_episode
                self.physchem_curve = [float("nan")] * self.start_episode
            if self.eval_curve_path.exists():
                import csv

                with self.eval_curve_path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            ep_i = int(float(row["episode"]))
                        except (KeyError, ValueError):
                            continue
                        if ep_i > self.start_episode:
                            continue
                        self.eval_episodes.append(ep_i)
                        try:
                            self.eval_raw_curve.append(float(row["eval_raw"]))
                        except (KeyError, ValueError):
                            self.eval_raw_curve.append(float("nan"))
                        try:
                            self.eval_ema_curve.append(float(row["eval_ema"]))
                        except (KeyError, ValueError):
                            self.eval_ema_curve.append(float("nan"))
            print(
                f"Resume state: last_episode={self.start_episode} "
                f"best_reward={self.best_reward:.4f} baseline={self.reward_baseline:.4f}",
                flush=True,
            )

        set_seed(self.config.RANDOM_SEED)

    def _build_ref_model(self) -> GPT | None:
        """Frozen pretrained policy for KL anchoring (always from GENERATE.CKPT_PATH)."""
        if max(self.kl_coef_start, self.kl_coef_end) <= 0:
            return None
        path = _resolve(self.config_dir, str(self.config.GENERATE.CKPT_PATH))
        ref = GPT(self.config).to(self.device)
        if self.config.DEVICE == "cpu":
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        else:
            checkpoint = torch.load(
                str(path), map_location=self.device, weights_only=False
            )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            ref.load_state_dict(checkpoint["model_state_dict"])
        else:
            ref.load_state_dict(checkpoint)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(
            f"KL ref policy loaded from {path} "
            f"(coef={self.kl_coef_start}→{self.kl_coef_end})",
            flush=True,
        )
        return ref

    def _frozen_prior_policy(self) -> GPT:
        """Frozen GuacaMol prior for bank sampling (ref_model if present)."""
        if self.ref_model is not None:
            return self.ref_model
        return self._build_ref_model_always()

    def _build_ref_model_always(self) -> GPT:
        """Load GENERATE.CKPT_PATH as a frozen policy (even when KL_COEF=0)."""
        path = _resolve(self.config_dir, str(self.config.GENERATE.CKPT_PATH))
        ref = GPT(self.config).to(self.device)
        if self.config.DEVICE == "cpu":
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        else:
            checkpoint = torch.load(
                str(path), map_location=self.device, weights_only=False
            )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            ref.load_state_dict(checkpoint["model_state_dict"])
        else:
            ref.load_state_dict(checkpoint)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    def _ensure_prior_fp_bank(self) -> None:
        """Load prior SMILES cache; top up by sampling the frozen prior if short."""
        scorer = ensure_co2_scorer()
        target = max(int(self.prior_fp_bank_size), 1)
        cache = Path(self.prior_fp_cache)
        smiles = load_smiles_cache(cache)
        print(
            f"Prior FP cache {cache}: loaded {len(smiles)} SMILES "
            f"(target={target})",
            flush=True,
        )
        if len(smiles) < target:
            need = target - len(smiles)
            print(
                f"Prior FP bank short by {need}; sampling frozen prior…",
                flush=True,
            )
            prior = self._frozen_prior_policy()
            seen = set(smiles)
            # GENERATE.TEMPERATURE for bank (stable, not RL explore schedule).
            bank_temp = float(self.config.GENERATE.TEMPERATURE)
            batch = int(self.config.RL_TRAIN.BATCH_SIZE)
            guard = 0
            max_guard = max(50, (need // max(batch, 1)) * 20 + 20)
            from rdkit import Chem

            while len(smiles) < target and guard < max_guard:
                guard += 1
                raw, _, _ = self.generate_molecules(
                    batch_size=batch,
                    temperature=bank_temp,
                    policy=prior,
                )
                for smi in raw:
                    if not valid_smiles(smi):
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    can = Chem.MolToSmiles(mol, canonical=True)
                    if can in seen:
                        continue
                    seen.add(can)
                    smiles.append(can)
                    if len(smiles) >= target:
                        break
            if len(smiles) < target:
                print(
                    f"Prior FP bank: only got {len(smiles)}/{target} after sampling "
                    f"({guard} batches); proceeding with what we have.",
                    flush=True,
                )
            save_smiles_cache(cache, smiles[:target])
            print(f"Prior FP cache written: {cache} n={min(len(smiles), target)}", flush=True)
        scorer.set_prior_fp_bank(smiles[:target])

    def _maybe_load_chem_tether(self) -> None:
        need = (
            max(self.chem_tether_coef_start, self.chem_tether_coef_end) > 0.0
            or self.chem_tether_log
        )
        if not need:
            self.chem_tether = None
            return
        bank = Path(self.chem_tether_bank)
        if not bank.is_file():
            raise FileNotFoundError(
                f"Chemical tether bank missing: {bank}. "
                "Run: PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py"
            )
        self.chem_tether = ChemicalTether.from_npz(
            bank,
            lambda_morgan=self.chem_tether_lambda_morgan,
            lambda_2d=self.chem_tether_lambda_2d,
            lambda_3d=self.chem_tether_lambda_3d,
            k=self.chem_tether_k,
            model_2d=self.chem_tether_2d_model,
            device=str(self.device),
        )
        print(
            f"Chemical tether: bank={bank} n={len(self.chem_tether.smiles)} "
            f"λ_m={self.chem_tether_lambda_morgan} λ_2d={self.chem_tether_lambda_2d} "
            f"λ_3d={self.chem_tether_lambda_3d} k={self.chem_tether_k} "
            f"s_m={self.chem_tether.scales['morgan']:.4f} "
            f"s_2d={self.chem_tether.scales['2d']:.4f} "
            f"s_3d={self.chem_tether.scales['3d']:.4f} "
            f"coef={self.chem_tether_coef_start}→{self.chem_tether_coef_end} "
            f"mode={self.chem_tether_mode} "
            f"d*_sched={self.chem_tether_d_star_schedule} "
            f"d*={self.chem_tether_d_star_fixed} "
            f"d*_range={self.chem_tether_d_star_start}→{self.chem_tether_d_star_end} "
            f"log={self.chem_tether_log}",
            flush=True,
        )

    def _mean_prior_fp_factor(self, smiles: list[str]) -> float:
        if self.prior_fp_floor <= 0.0 or not self.co2_mode:
            return 1.0
        if not smiles:
            return float("nan")
        scorer = ensure_co2_scorer()
        tether = scorer.prior_fp_tether
        if not tether.enabled:
            return 1.0
        return float(np.mean([tether.factor(s) for s in smiles]))

    def _regularizer_metrics(self, **overrides) -> dict:
        out = {
            "kl": 0.0,
            "kl_coef": self.kl_coef,
            "prior_fp_mean_factor": 1.0 if self.prior_fp_floor <= 0.0 else float("nan"),
            "prior_fp_floor": self.prior_fp_floor,
            "tether": float("nan"),
            "tether_coef": self.chem_tether_coef,
            "tether_cost_ema": (
                float(self.chem_tether_cost_ema)
                if math.isfinite(self.chem_tether_cost_ema)
                else float("nan")
            ),
            "tether_d": float("nan"),
            "tether_d_star": self.chem_tether_d_star,
            "tether_abs_dev": float("nan"),
            "tether_shell_hit": float("nan"),
            "tether_d_morgan": float("nan"),
            "tether_d_2d": float("nan"),
            "tether_d_3d": float("nan"),
            "tether_dp_morgan": float("nan"),
            "tether_dp_2d": float("nan"),
            "tether_dp_3d": float("nan"),
        }
        out.update(overrides)
        return out

    def _chem_tether_term(
        self,
        sequence_log_probs: torch.Tensor,
        smiles: list[str],
    ) -> tuple[torch.Tensor, dict]:
        """REINFORCE tether increment when coef>0; else zeros / empty overrides."""
        zero = sequence_log_probs.new_zeros(())
        if (
            self.chem_tether is None
            or self.chem_tether_coef <= 0.0
            or not smiles
        ):
            return zero, {}
        d, channels = self.chem_tether.distances(
            smiles, device=sequence_log_probs.device
        )
        cost = tether_cost(
            d,
            mode=self.chem_tether_mode,
            d_star=float(self.chem_tether_d_star)
            if self.chem_tether_mode == "explore"
            else 0.0,
        )
        cost_np = cost.detach().float().cpu().numpy()
        batch_mean = _nanmean(cost_np)
        if not math.isfinite(self.chem_tether_cost_ema):
            # Seed EMA; first-step advantages ~0 (same as batch-mean centering).
            self.chem_tether_cost_ema = (
                float(batch_mean) if math.isfinite(batch_mean) else 0.0
            )
        baseline = float(self.chem_tether_cost_ema)
        if math.isfinite(batch_mean):
            decay = float(self.chem_tether_cost_ema_decay)
            self.chem_tether_cost_ema = (
                decay * self.chem_tether_cost_ema + (1.0 - decay) * float(batch_mean)
            )
        tloss = tether_loss(sequence_log_probs, cost, baseline=baseline)
        d_np = d.detach().float().cpu().numpy()
        primed = self.chem_tether.calibrate_channels(channels)
        metrics = {
            "tether": float(tloss.item()),
            "tether_cost_ema": baseline,
            "tether_d": _nanmean(d_np),
            "tether_d_star": float(self.chem_tether_d_star),
            "tether_d_morgan": _nanmean(channels["morgan"]),
            "tether_d_2d": _nanmean(channels["2d"]),
            "tether_d_3d": _nanmean(channels["3d"]),
            "tether_dp_morgan": _nanmean(primed["morgan"]),
            "tether_dp_2d": _nanmean(primed["2d"]),
            "tether_dp_3d": _nanmean(primed["3d"]),
        }
        if self.chem_tether_mode == "explore" and math.isfinite(
            float(self.chem_tether_d_star)
        ):
            d_star = float(self.chem_tether_d_star)
            abs_dev = np.abs(d_np - d_star)
            metrics["tether_abs_dev"] = _nanmean(abs_dev)
            finite = np.isfinite(abs_dev)
            metrics["tether_shell_hit"] = (
                float(np.mean(abs_dev[finite] <= SHELL_HIT_BAND))
                if finite.any()
                else float("nan")
            )
        return self.chem_tether_coef * tloss, metrics

    def _channel_dist_for_smiles(self, smiles: list[str]) -> dict[str, np.ndarray]:
        n = len(smiles)
        nan = np.full((n,), np.nan, dtype=np.float64)
        empty = {"morgan": nan.copy(), "2d": nan.copy(), "3d": nan.copy()}
        if self.chem_tether is None or n == 0:
            return empty
        _d, channels = self.chem_tether.distances(smiles)
        return channels

    def _schedule_progress(self, episode: int, n_episodes: int) -> float:
        if self.explore_episodes > 0:
            return explore_schedule_progress(episode, self.explore_episodes)
        # Fallback: linear over full run when EXPLORE_EPISODES unset/0
        denom = max(int(n_episodes) - 1, 1)
        return float(episode - 1) / float(denom)

    def _apply_schedules(self, episode: int, n_episodes: int) -> None:
        progress = self._schedule_progress(episode, n_episodes)
        self.entropy_coef = linear_schedule(
            self.entropy_coef_start, self.entropy_coef_end, progress
        )
        self.scheduled_temperature = linear_schedule(
            self.temperature_start, self.temperature_end, progress
        )
        self.current_temperature = self.scheduled_temperature
        self.kl_coef = linear_schedule(
            self.kl_coef_start, self.kl_coef_end, progress
        )
        self.chem_tether_coef = linear_schedule(
            self.chem_tether_coef_start, self.chem_tether_coef_end, progress
        )
        if self.chem_tether_mode == "explore":
            if self.chem_tether_d_star_schedule == "static":
                self.chem_tether_d_star = self.chem_tether_d_star_fixed
            else:
                self.chem_tether_d_star = linear_schedule(
                    self.chem_tether_d_star_start,
                    self.chem_tether_d_star_end,
                    progress,
                )
        else:
            self.chem_tether_d_star = float("nan")
        self.top_quantile = linear_schedule(
            self.top_quantile_start, self.top_quantile_end, progress
        )

    def _co2_monitor_payload(
        self,
        *,
        n_valid: int,
        valid_smiles_list: list[str],
        gated_local_indices: np.ndarray,
    ) -> dict[str, float | int]:
        batch_size = int(self.config.RL_TRAIN.BATCH_SIZE)
        health = batch_health_metrics(
            n_valid=n_valid,
            valid_smiles_list=valid_smiles_list,
            gated_local_indices=gated_local_indices,
            batch_size=batch_size,
        )
        health["valid_frac_ema"] = float(self.valid_frac_ema)
        health["unique_gated_frac_ema"] = float(self.unique_gated_frac_ema)
        health["scheduled_temperature"] = float(self.scheduled_temperature)
        return health

    def _update_co2_health_ema(self, metrics: dict) -> None:
        vf = metrics.get("valid_frac")
        if vf is not None and math.isfinite(float(vf)):
            self.valid_frac_ema = update_ema(
                self.valid_frac_ema, float(vf), self.monitor_ema_momentum
            )
        uf = metrics.get("unique_gated_frac")
        if uf is not None and math.isfinite(float(uf)):
            self.unique_gated_frac_ema = update_ema(
                self.unique_gated_frac_ema, float(uf), self.monitor_ema_momentum
            )
        metrics["valid_frac_ema"] = float(self.valid_frac_ema)
        metrics["unique_gated_frac_ema"] = float(self.unique_gated_frac_ema)

    def generate_molecules(
        self,
        batch_size: int = 32,
        temperature: float | None = None,
        *,
        policy: GPT | None = None,
    ):
        model = self.model if policy is None else policy
        model.eval()
        temp = (
            float(self.current_temperature)
            if temperature is None
            else float(temperature)
        )
        if isinstance(self.config.GENERATE.CONTEXT, str):
            context = torch.tensor(
                tokenize(self.config.GENERATE.CONTEXT), dtype=torch.long
            )
            context = context.repeat(batch_size, 1)
        else:
            context = torch.tensor(
                tokenize(self.config.GENERATE.CONTEXT[0]), dtype=torch.long
            )
            context = context.repeat(batch_size, 1)

        context = context.to(self.device)

        if self.config.MODEL.NUM_PROPS > 0:
            import rdkit.Chem as Chem

            mols = [Chem.MolFromSmiles(t) for t in self.config.GENERATE.TEMPLATES]
            props = get_props_from_mols(mols, self.config.MODEL.NUM_PROPS)
            all_props = torch.repeat_interleave(
                torch.tensor(props, dtype=torch.int),
                batch_size // len(self.config.GENERATE.TEMPLATES) + 1,
                dim=0,
            )
            all_props = all_props[:batch_size].to(self.device)
            with torch.no_grad():
                gen_tokens = sample(
                    model=model,
                    x=context,
                    steps=self.config.MODEL.MAX_LEN,
                    temperature=temp,
                    sample_=True,
                    top_k=None,
                    prop=all_props,
                )
        else:
            with torch.no_grad():
                gen_tokens = sample(
                    model=model,
                    x=context,
                    steps=self.config.MODEL.MAX_LEN,
                    temperature=temp,
                    sample_=True,
                    top_k=None,
                    prop=None,
                )

        gen_tokens = gen_tokens.cpu().numpy().tolist()
        smiles_list = [untokenize(tokens) for tokens in gen_tokens]
        print(f"Generated {len(smiles_list)} molecules (temp={temp:.3f})", flush=True)
        return smiles_list, gen_tokens, context

    def rl_step(self):
        smiles_list, gen_tokens, _context = self.generate_molecules(
            batch_size=self.config.RL_TRAIN.BATCH_SIZE
        )

        # Keep eval() so teacher-forced log-probs match sampling (dropout off).
        valid_indices = [i for i, smi in enumerate(smiles_list) if valid_smiles(smi)]
        n_valid = len(valid_indices)

        empty = {
            "policy_loss": 0.0,
            "average_policy_reward": 0.0,
            "average_raw_objective": float("nan"),
            "average_physchem_objective": float("nan"),
            "mean_raw_ba_gated": float("nan"),
            "gated_valid_frac": 0.0,
            "gated_mean_reward": float("nan"),
            "gated_p90": float("nan"),
            "n_valid": 0,
            "n_gated": 0,
            "n_learn": 0,
            "mean_mw_scale": float("nan"),
            "n_mw_zero": 0,
            "skipped_update": 1.0,
            "max_policy_reward": 0.0,
            "min_policy_reward": 0.0,
            "baseline": self.reward_baseline,
            "kl": 0.0,
            "kl_coef": self.kl_coef,
            "entropy_coef": self.entropy_coef,
            "temperature": self.current_temperature,
            "scheduled_temperature": self.scheduled_temperature,
            "top_quantile": self.top_quantile,
            "valid_frac": 0.0,
            "unique_gated_frac": 0.0,
            "n_unique_gated": 0,
            "valid_frac_ema": float(self.valid_frac_ema),
            "unique_gated_frac_ema": float(self.unique_gated_frac_ema),
        }
        empty.update(self._regularizer_metrics())

        if n_valid == 0:
            if self.co2_mode:
                self._update_co2_health_ema(empty)
            return empty

        batch = compute_reward_batch(smiles_list)
        policy_rewards = np.array(batch.policy, dtype=np.float64)
        raw_totals = np.array(batch.raw_total, dtype=np.float64)
        raw_bas = np.array(batch.raw_ba, dtype=np.float64)
        raw_physchem = np.array(batch.raw_physchem, dtype=np.float64)
        qualities = np.array(batch.quality, dtype=np.float64)
        gate_ok = np.array(batch.gate_ok, dtype=bool)
        mw_scales = np.array(batch.mw_scale, dtype=np.float64)

        policy_rewards = policy_rewards[valid_indices]
        raw_totals = raw_totals[valid_indices]
        raw_bas = raw_bas[valid_indices]
        raw_physchem = raw_physchem[valid_indices]
        qualities = qualities[valid_indices]
        gate_ok = gate_ok[valid_indices]
        mw_scales = mw_scales[valid_indices]
        valid_smiles_list = [smiles_list[i] for i in valid_indices]

        if self.co2_mode:
            return self._rl_step_co2(
                policy_rewards,
                raw_totals,
                raw_bas,
                raw_physchem,
                qualities,
                gate_ok,
                mw_scales,
                gen_tokens,
                valid_indices,
                valid_smiles_list,
                n_valid=n_valid,
            )

        avg_policy_reward = float(np.mean(policy_rewards))
        avg_raw_objective = float(np.mean(raw_totals))
        self.monitor_ema = update_ema(
            self.monitor_ema, avg_raw_objective, self.monitor_ema_momentum
        )

        gated_mask = gate_ok
        if np.any(gated_mask):
            mean_raw_ba_gated = float(np.mean(raw_bas[gated_mask]))
            gated_valid_frac = float(np.mean(gated_mask))
        else:
            mean_raw_ba_gated = float("nan")
            gated_valid_frac = 0.0

        gen_tokens_tensor = torch.tensor(gen_tokens, dtype=torch.long, device=self.device)
        gen_tokens_tensor = gen_tokens_tensor[valid_indices]

        log_probs, selected_log_probs, tok_mask = self._teacher_force_log_probs(
            gen_tokens_tensor
        )

        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * avg_policy_reward
        )
        adjusted_rewards = torch.as_tensor(
            policy_rewards - self.reward_baseline, dtype=torch.float32, device=self.device
        )

        entropy = self._masked_token_entropy(log_probs, tok_mask)
        kl = self._kl_to_ref(log_probs, gen_tokens_tensor, tok_mask)

        sequence_log_probs = (selected_log_probs * tok_mask).sum(dim=1)
        policy_loss = (
            -(sequence_log_probs * adjusted_rewards).mean()
            - entropy * self.entropy_coef
            + self.kl_coef * kl
        )
        tether_term, tether_m = self._chem_tether_term(
            sequence_log_probs, valid_smiles_list
        )
        policy_loss = policy_loss + tether_term

        self._optimizer_step(policy_loss)

        out = {
            "policy_loss": policy_loss.item(),
            "average_policy_reward": avg_policy_reward,
            "average_raw_objective": float(self.monitor_ema),
            "average_physchem_objective": float("nan"),
            "mean_raw_ba_gated": mean_raw_ba_gated,
            "gated_valid_frac": gated_valid_frac,
            "gated_mean_reward": float("nan"),
            "gated_p90": float("nan"),
            "n_valid": n_valid,
            "n_gated": int(np.sum(gate_ok)),
            "n_learn": int(len(valid_indices)),
            "skipped_update": 0.0,
            "max_policy_reward": float(np.max(policy_rewards)),
            "min_policy_reward": float(np.min(policy_rewards)),
            "baseline": self.reward_baseline,
            "entropy_coef": self.entropy_coef,
            "temperature": self.current_temperature,
            "top_quantile": self.top_quantile,
        }
        out.update(
            self._regularizer_metrics(
                kl=float(kl.item()),
                prior_fp_mean_factor=1.0,
                **tether_m,
            )
        )
        return out

    def _rl_step_co2(
        self,
        policy_rewards: np.ndarray,
        raw_totals: np.ndarray,
        raw_bas: np.ndarray,
        raw_physchem: np.ndarray,
        qualities: np.ndarray,
        gate_ok: np.ndarray,
        mw_scales: np.ndarray,
        gen_tokens: list,
        valid_indices: list[int],
        valid_smiles_list: list[str],
        *,
        n_valid: int,
    ) -> dict:
        gated_mask = gate_ok & np.isfinite(policy_rewards)
        n_gated = int(np.sum(gated_mask))
        gated_valid_frac = float(np.mean(gate_ok)) if gate_ok.size else 0.0
        gated_local = np.flatnonzero(gated_mask)
        monitor = self._co2_monitor_payload(
            n_valid=n_valid,
            valid_smiles_list=valid_smiles_list,
            gated_local_indices=gated_local,
        )

        if n_gated > 0 and np.any(np.isfinite(mw_scales[gated_mask])):
            scales_g = mw_scales[gated_mask]
            mean_mw_scale = float(np.nanmean(scales_g))
            n_mw_zero = int(np.sum(np.isfinite(scales_g) & (scales_g <= 0.0)))
        else:
            mean_mw_scale = float("nan")
            n_mw_zero = 0

        # Store quality elites (post prior-FP floor) for replay.
        if n_gated > 0:
            from rdkit import Chem

            for i in np.flatnonzero(gate_ok & np.isfinite(qualities)):
                mol = Chem.MolFromSmiles(valid_smiles_list[int(i)])
                if mol is None:
                    continue
                can = Chem.MolToSmiles(mol)
                self.elite_buffer.offer(can, float(qualities[int(i)]))

        if n_gated > 0:
            gated_rewards = policy_rewards[gated_mask]
            gated_mean = float(np.mean(gated_rewards))
            gated_p90 = float(np.percentile(gated_rewards, 90))
            mean_raw_ba_gated = (
                float(np.nanmean(raw_bas[gated_mask]))
                if np.any(np.isfinite(raw_bas[gated_mask]))
                else float("nan")
            )
            physchem_gated = raw_physchem[gated_mask]
            gated_physchem_mean = (
                float(np.nanmean(physchem_gated))
                if np.any(np.isfinite(physchem_gated))
                else float("nan")
            )
        else:
            gated_rewards = np.array([], dtype=np.float64)
            gated_mean = float("nan")
            gated_p90 = float("nan")
            mean_raw_ba_gated = float("nan")
            gated_physchem_mean = float("nan")

        if should_skip_thin_batch(n_gated, self.min_gated):
            skip = {
                "policy_loss": 0.0,
                "average_policy_reward": gated_mean if n_gated else 0.0,
                "average_raw_objective": float("nan"),
                "average_physchem_objective": float("nan"),
                "mean_raw_ba_gated": mean_raw_ba_gated,
                "gated_valid_frac": gated_valid_frac,
                "gated_mean_reward": gated_mean,
                "gated_p90": gated_p90,
                "n_valid": n_valid,
                "n_gated": n_gated,
                "n_learn": 0,
                "n_replay": 0,
                "inception_boost": 0.0,
                "mean_mw_scale": mean_mw_scale,
                "n_mw_zero": n_mw_zero,
                "skipped_update": 1.0,
                "max_policy_reward": float(np.nanmax(policy_rewards)) if n_gated else 0.0,
                "min_policy_reward": float(np.nanmin(policy_rewards)) if n_gated else 0.0,
                "baseline": self.reward_baseline,
                "entropy_coef": self.entropy_coef,
                "temperature": self.current_temperature,
                "scheduled_temperature": self.scheduled_temperature,
                "top_quantile": self.top_quantile,
            }
            skip.update(monitor)
            gated_smiles = [
                valid_smiles_list[int(i)] for i in np.flatnonzero(gated_mask)
            ]
            skip.update(
                self._regularizer_metrics(
                    prior_fp_mean_factor=self._mean_prior_fp_factor(gated_smiles)
                    if n_gated
                    else float("nan"),
                )
            )
            self._update_co2_health_ema(skip)
            return skip

        gated_local_idx = np.flatnonzero(gated_mask)
        # One gradient per canonical SMILES (highest reward wins), then top quantile.
        unique_idx = unique_canonical_learn_indices(
            valid_smiles_list, gated_local_idx, policy_rewards
        )
        if unique_idx.size == 0:
            skip = {
                "policy_loss": 0.0,
                "average_policy_reward": gated_mean if n_gated else 0.0,
                "average_raw_objective": float("nan"),
                "average_physchem_objective": float("nan"),
                "mean_raw_ba_gated": mean_raw_ba_gated,
                "gated_valid_frac": gated_valid_frac,
                "gated_mean_reward": gated_mean,
                "gated_p90": gated_p90,
                "n_valid": n_valid,
                "n_gated": n_gated,
                "n_learn": 0,
                "n_replay": 0,
                "inception_boost": 0.0,
                "mean_mw_scale": mean_mw_scale,
                "n_mw_zero": n_mw_zero,
                "skipped_update": 1.0,
                "max_policy_reward": float(np.nanmax(policy_rewards)) if n_gated else 0.0,
                "min_policy_reward": float(np.nanmin(policy_rewards)) if n_gated else 0.0,
                "baseline": self.reward_baseline,
                "entropy_coef": self.entropy_coef,
                "temperature": self.current_temperature,
                "scheduled_temperature": self.scheduled_temperature,
                "top_quantile": self.top_quantile,
            }
            skip.update(monitor)
            gated_smiles = [
                valid_smiles_list[int(i)] for i in np.flatnonzero(gated_mask)
            ]
            skip.update(
                self._regularizer_metrics(
                    prior_fp_mean_factor=self._mean_prior_fp_factor(gated_smiles)
                    if n_gated
                    else float("nan"),
                )
            )
            self._update_co2_health_ema(skip)
            return skip

        unique_rewards = policy_rewards[unique_idx]
        top_local = top_quantile_indices(unique_rewards, self.top_quantile)
        learn_idx = unique_idx[top_local]
        learn_rewards = policy_rewards[learn_idx]
        learn_smiles = [valid_smiles_list[int(i)] for i in learn_idx]

        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * gated_mean
        )
        batch_std = float(np.std(gated_rewards)) if n_gated > 1 else 1.0
        self.reward_std_ema = update_ema(
            self.reward_std_ema, batch_std, self.baseline_momentum
        )
        self.monitor_ema = update_ema(
            self.monitor_ema, gated_mean, self.monitor_ema_momentum
        )
        if math.isfinite(gated_physchem_mean):
            self.physchem_ema = update_ema(
                self.physchem_ema, gated_physchem_mean, self.monitor_ema_momentum
            )

        gen_tokens_tensor = torch.tensor(gen_tokens, dtype=torch.long, device=self.device)
        learn_tokens = gen_tokens_tensor[valid_indices][learn_idx]

        # Inception: pause when uniqueness is below floor; never replay SMILES
        # already in the on-policy learn set.
        n_replay = 0
        unique_frac = float(monitor.get("unique_gated_frac", 1.0))
        uniqueness_ok = (
            math.isfinite(unique_frac) and unique_frac >= self.unique_gated_frac_floor
        )
        boost_active = (
            uniqueness_ok
            and self.inception_boost_patience > 0
            and self.decline_streak >= self.inception_boost_patience
            and math.isfinite(self.best_train_ema)
        )
        replay_k = 0
        if uniqueness_ok:
            replay_k = inception_replay_k(
                self.inception_batch,
                self.inception_batch_boost,
                boost_active=boost_active,
                n_on_policy=int(learn_tokens.size(0)),
                max_frac=self.inception_max_frac,
            )
        if replay_k > 0 and len(self.elite_buffer) > 0:
            elites = self.elite_buffer.sample(
                replay_k, self._elite_rng, weighted=boost_active
            )
            if elites:
                max_len = int(self.config.MODEL.MAX_LEN)
                tgt_len = int(learn_tokens.size(1))
                learn_canons = {canonical_smiles_or_raw(s) for s in learn_smiles}
                elite_rows = []
                elite_qs = []
                elite_smiles = []
                for smi, q in elites:
                    can = canonical_smiles_or_raw(smi)
                    if can in learn_canons:
                        continue
                    tok = tokenize(smi, max_len=max(max_len, tgt_len))
                    if len(tok) < tgt_len:
                        tok = tok + [stoi[pad_token]] * (tgt_len - len(tok))
                    elite_rows.append(tok[:tgt_len])
                    elite_qs.append(float(q))
                    elite_smiles.append(smi)
                    learn_canons.add(can)
                if elite_rows:
                    elite_tensor = torch.tensor(
                        elite_rows, dtype=torch.long, device=self.device
                    )
                    learn_tokens = torch.cat([learn_tokens, elite_tensor], dim=0)
                    learn_rewards = np.concatenate(
                        [learn_rewards, np.asarray(elite_qs, dtype=np.float64)]
                    )
                    learn_smiles.extend(elite_smiles)
                    n_replay = len(elite_qs)

        advantages = baseline_relative_advantages(
            learn_rewards, self.reward_baseline, self.reward_std_ema
        )

        log_probs, selected_log_probs, tok_mask = self._teacher_force_log_probs(
            learn_tokens
        )

        adjusted_rewards = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        entropy = self._masked_token_entropy(log_probs, tok_mask)
        kl = self._kl_to_ref(log_probs, learn_tokens, tok_mask)
        sequence_log_probs = (selected_log_probs * tok_mask).sum(dim=1)
        policy_loss = (
            -(sequence_log_probs * adjusted_rewards).mean()
            - entropy * self.entropy_coef
            + self.kl_coef * kl
        )
        tether_term, tether_m = self._chem_tether_term(
            sequence_log_probs, learn_smiles
        )
        policy_loss = policy_loss + tether_term

        self._optimizer_step(policy_loss)

        gated_smiles = [valid_smiles_list[int(i)] for i in np.flatnonzero(gated_mask)]
        out = {
            "policy_loss": policy_loss.item(),
            "average_policy_reward": gated_mean,
            # Checkpoint monitor: smoothed gated mean (not noisy top-quartile).
            "average_raw_objective": float(self.monitor_ema),
            "average_physchem_objective": float(self.physchem_ema),
            "mean_raw_ba_gated": mean_raw_ba_gated,
            "gated_valid_frac": gated_valid_frac,
            "gated_mean_reward": gated_mean,
            "gated_p90": gated_p90,
            "n_valid": n_valid,
            "n_gated": n_gated,
            "n_learn": int(len(learn_idx)),
            "n_replay": n_replay,
            "inception_boost": 1.0 if boost_active else 0.0,
            "mean_mw_scale": mean_mw_scale,
            "n_mw_zero": n_mw_zero,
            "skipped_update": 0.0,
            "max_policy_reward": float(np.max(learn_rewards)),
            "min_policy_reward": float(np.min(learn_rewards)),
            "baseline": self.reward_baseline,
            "entropy_coef": self.entropy_coef,
            "temperature": self.current_temperature,
            "scheduled_temperature": self.scheduled_temperature,
            "top_quantile": self.top_quantile,
        }
        out.update(monitor)
        out.update(
            self._regularizer_metrics(
                kl=float(kl.item()),
                prior_fp_mean_factor=self._mean_prior_fp_factor(gated_smiles),
                **tether_m,
            )
        )
        self._update_co2_health_ema(out)
        return out

    def _teacher_force_log_probs(
        self, gen_tokens_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eval-mode teacher force; returns log_probs, selected, pad credit mask."""
        self.model.eval()
        logits, _ = self.model(gen_tokens_tensor[:, :-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        target_tokens = gen_tokens_tensor[:, 1 : log_probs.size(1) + 1]
        selected_log_probs = log_probs.gather(
            2, target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        tok_mask = self._pad_credit_mask(target_tokens)
        return log_probs, selected_log_probs, tok_mask

    def _pad_credit_mask(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Float mask [B, T]: 1 before first pad; 0 at pad and after."""
        pad_id = int(stoi[pad_token])
        is_pad = target_tokens == pad_id
        seen_pad = is_pad.cumsum(dim=1) > 0
        return (~seen_pad).to(dtype=torch.float32)

    def _masked_token_entropy(
        self, log_probs: torch.Tensor, tok_mask: torch.Tensor
    ) -> torch.Tensor:
        token_h = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        denom = tok_mask.sum().clamp(min=1.0)
        return (token_h * tok_mask).sum() / denom

    def _kl_to_ref(
        self,
        log_probs: torch.Tensor,
        gen_tokens_tensor: torch.Tensor,
        tok_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean token KL(π_θ || π_ref) over credited positions. Zero if KL disabled."""
        if self.ref_model is None or self.kl_coef <= 0:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            ref_logits, _ = self.ref_model(gen_tokens_tensor[:, :-1])
            log_ref = torch.log_softmax(ref_logits, dim=-1)
        t = min(log_probs.size(1), log_ref.size(1))
        lp = log_probs[:, :t]
        lr = log_ref[:, :t]
        p = lp.exp()
        kl_tok = (p * (lp - lr)).sum(dim=-1)
        if tok_mask is None:
            return kl_tok.mean()
        mask = tok_mask[:, :t]
        denom = mask.sum().clamp(min=1.0)
        return (kl_tok * mask).sum() / denom

    def _optimizer_step(self, policy_loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(policy_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

    def _run_fixed_eval(self) -> float:
        """
        Fixed-seed eval batch for stable checkpointing.
        Uses end-of-schedule temperature; restores RNG after.
        """
        cpu_state = torch.get_rng_state()
        np_state = np.random.get_state()
        cuda_states = None
        if torch.cuda.is_available():
            try:
                cuda_states = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_states = None

        set_seed(int(self.config.RANDOM_SEED) + 10_000)
        smiles_list, _, _ = self.generate_molecules(
            batch_size=self.eval_batch_size,
            temperature=self.temperature_end,
        )
        valid = [s for s in smiles_list if valid_smiles(s)]
        score = float("nan")
        if valid:
            batch = compute_reward_batch(valid)
            rewards = np.array(batch.raw_total, dtype=np.float64)
            if self.co2_mode:
                gate = np.array(batch.gate_ok, dtype=bool)
                gated = rewards[gate & np.isfinite(rewards)]
                if gated.size:
                    score = float(np.mean(gated))
            else:
                finite = rewards[np.isfinite(rewards)]
                if finite.size:
                    score = float(np.mean(finite))
            if self.chem_tether is not None:
                ch = self._channel_dist_for_smiles(valid)
                primed = self.chem_tether.calibrate_channels(ch)
                print(
                    f"Fixed eval tether d'_m={_fmt_num(_nanmean(primed['morgan']))} "
                    f"d'_2d={_fmt_num(_nanmean(primed['2d']))} "
                    f"d'_3d={_fmt_num(_nanmean(primed['3d']))} "
                    f"(raw m/2d/3d {_fmt_num(_nanmean(ch['morgan']))} "
                    f"{_fmt_num(_nanmean(ch['2d']))} "
                    f"{_fmt_num(_nanmean(ch['3d']))})",
                    flush=True,
                )

        torch.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass
        return score

    def _run_molecule_snapshot(self, episode: int) -> None:
        """
        Fixed-seed scored SMILES dump for metric-over-time figures.
        Uses end-of-schedule temperature; restores RNG after. Logging-only.
        """
        cpu_state = torch.get_rng_state()
        np_state = np.random.get_state()
        cuda_states = None
        if torch.cuda.is_available():
            try:
                cuda_states = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_states = None

        set_seed(int(self.config.RANDOM_SEED) + 10_000)
        smiles_list, _, _ = self.generate_molecules(
            batch_size=self.snapshot_batch_size,
            temperature=self.temperature_end,
        )
        valid_mask = [valid_smiles(s) for s in smiles_list]
        valid = [s for s, ok in zip(smiles_list, valid_mask) if ok]
        batch = compute_reward_batch(valid) if valid else None
        extra = None
        if self.chem_tether is not None and valid:
            extra = self._channel_dist_for_smiles(valid)
            primed = self.chem_tether.calibrate_channels(extra)
            extra = {
                **extra,
                "dp_morgan": primed["morgan"],
                "dp_2d": primed["2d"],
                "dp_3d": primed["3d"],
            }
        out = self.molecule_snapshot_dir / f"ep{episode}.csv"
        _write_molecule_snapshot(
            out, episode, smiles_list, batch, valid_mask, extra_channels=extra
        )
        n_valid = sum(valid_mask)
        print(
            f"Molecule snapshot ep{episode}: {n_valid}/{len(smiles_list)} valid -> {out}",
            flush=True,
        )

        torch.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

    def save_checkpoint(self, episode: int, kind: str) -> Path:
        """kind: 'best' | 'periodic'. Atomic write; raises OSError on disk failure."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        prefix = f"{kind}_{self.run_name}_ep{episode}"
        checkpoint_path = self.checkpoint_dir / f"{prefix}.pt"
        tmp_path = self.checkpoint_dir / f".{prefix}.pt.tmp"
        payload = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "best_reward": self.best_reward,
            "reward_baseline": self.reward_baseline,
            "monitor_ema": self.monitor_ema,
            "physchem_ema": self.physchem_ema,
            "reward_std_ema": self.reward_std_ema,
            "eval_ema": self.eval_ema,
            "best_train_ema": self.best_train_ema,
            "best_eval_episode": self.best_eval_episode,
            "best_eval_ema": self.best_eval_ema,
            "episodes_since_best": self.episodes_since_best,
            "elite_buffer": self.elite_buffer.to_list()
            if getattr(self, "elite_buffer", None) is not None
            else [],
        }
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, checkpoint_path)
        except (OSError, RuntimeError) as e:
            tmp_path.unlink(missing_ok=True)
            # Free a periodic if disk-full, then retry once.
            _cleanup_old_periodic(
                self.checkpoint_dir, self.run_name, max(1, self.keep_last_n_periodic)
            )
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, checkpoint_path)
            except (OSError, RuntimeError):
                tmp_path.unlink(missing_ok=True)
                print(
                    f"ERROR: failed to save checkpoint ({e}). "
                    "Disk may be full — free space and resume from best.pt.",
                    flush=True,
                )
                raise
        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def rl_train(self) -> None:
        print("Starting RL training (REINFORCE)...", flush=True)
        raw_curve: list[float] = list(getattr(self, "raw_curve", []))
        physchem_curve: list[float] = list(getattr(self, "physchem_curve", []))
        n_episodes = int(self.config.RL_TRAIN.N_EPISODES)
        start_ep = int(getattr(self, "start_episode", 0))
        if start_ep >= n_episodes:
            print(
                f"Nothing to do: start_episode={start_ep} >= N_EPISODES={n_episodes}",
                flush=True,
            )
            return
        best_pt = self.checkpoint_dir / "best.pt"
        plot_title = (
            "EMA gated mean reward"
            if self.co2_mode
            else "EMA mean raw objective (valid SMILES)"
        )
        plot_ylabel = "EMA gated mean" if self.co2_mode else "EMA mean raw objective"
        print(f"Continuing from episode {start_ep + 1} → {n_episodes}", flush=True)
        eval_episodes = list(getattr(self, "eval_episodes", []))
        eval_raw_curve = list(getattr(self, "eval_raw_curve", []))
        eval_ema_curve = list(getattr(self, "eval_ema_curve", []))

        for episode in range(start_ep, n_episodes):
            ep = episode + 1
            self._apply_schedules(ep, n_episodes)

            metrics = self.rl_step()
            train_monitor = metrics["average_raw_objective"]
            physchem_monitor = metrics.get(
                "average_physchem_objective", float("nan")
            )
            raw_curve.append(train_monitor)
            physchem_curve.append(
                float(physchem_monitor)
                if physchem_monitor is not None
                else float("nan")
            )
            self._write_reward_curve(raw_curve, physchem_curve)
            _append_rl_metrics(self.metrics_path, ep, metrics)

            _print_episode_summary(
                ep,
                n_episodes,
                metrics,
                batch_size=int(self.config.RL_TRAIN.BATCH_SIZE),
            )

            if (
                self.save_molecule_snapshots
                and self.snapshot_every > 0
                and ep % self.snapshot_every == 0
            ):
                self._run_molecule_snapshot(ep)

            # Fixed eval: logging only (eval.png). best.pt / early-stop use train EMA.
            if self.eval_every > 0 and ep % self.eval_every == 0:
                eval_score = self._run_fixed_eval()
                if math.isfinite(eval_score):
                    self.eval_ema = update_ema(
                        self.eval_ema, eval_score, self.monitor_ema_momentum
                    )
                eval_episodes.append(ep)
                eval_raw_curve.append(
                    float(eval_score) if math.isfinite(eval_score) else float("nan")
                )
                eval_ema_curve.append(
                    float(self.eval_ema)
                    if math.isfinite(self.eval_ema)
                    else float("nan")
                )
                self.eval_episodes = eval_episodes
                self.eval_raw_curve = eval_raw_curve
                self.eval_ema_curve = eval_ema_curve
                if math.isfinite(self.eval_ema) and self.eval_ema > self.best_eval_ema:
                    self.best_eval_ema = float(self.eval_ema)
                    self.best_eval_episode = ep
                _write_eval_curve(
                    self.eval_curve_path,
                    eval_episodes,
                    eval_raw_curve,
                    eval_ema_curve,
                )
                _plot_eval_curve(
                    eval_episodes,
                    eval_raw_curve,
                    eval_ema_curve,
                    self.graph_dir / "eval.png",
                    best_episode=self.best_eval_episode or None,
                )
                print(
                    f"Fixed eval: raw={_fmt_metric(eval_score)} "
                    f"ema={_fmt_metric(self.eval_ema)}",
                    flush=True,
                )

            plot_physchem = physchem_curve if self.co2_mode else None
            unique_frac = float(metrics.get("unique_gated_frac", float("nan")))
            uniqueness_ok = (not self.co2_mode) or (
                math.isfinite(unique_frac)
                and unique_frac >= self.unique_gated_frac_floor
            )
            if (
                math.isfinite(train_monitor)
                and train_monitor > self.best_train_ema
                and uniqueness_ok
            ):
                self.best_train_ema = float(train_monitor)
                self.decline_streak = 0

            # best.pt + early-stop: train EMA peak (matches final.png).
            # CO2: only promote best while unique gated frac is above floor.
            if (
                math.isfinite(train_monitor)
                and train_monitor > self.best_reward
                and uniqueness_ok
            ):
                self.best_reward = float(train_monitor)
                path = self.save_checkpoint(ep, "best")
                _write_best_pointer(best_pt, path)
                _cleanup_old_best(self.checkpoint_dir, self.run_name, path)
                _plot_rewards(
                    raw_curve,
                    self.graph_dir / "best.png",
                    title=plot_title + " (best train EMA)",
                    ylabel=plot_ylabel,
                    physchem_rewards=plot_physchem,
                )
                print(
                    f"New best train EMA: {self.best_reward:.4f} @ ep {ep}",
                    flush=True,
                )
                self.episodes_since_best = 0
            else:
                if (
                    self.co2_mode
                    and math.isfinite(train_monitor)
                    and train_monitor > self.best_reward
                    and not uniqueness_ok
                ):
                    print(
                        f"Best skip ep {ep}: EMA {_fmt_metric(train_monitor)} "
                        f"> peak {_fmt_metric(self.best_reward)} but "
                        f"unique_gated_frac={unique_frac:.3f} "
                        f"< floor {self.unique_gated_frac_floor}",
                        flush=True,
                    )
                self.episodes_since_best += 1
                if early_stop_should_stop(
                    self.episodes_since_best, self.early_stop_patience
                ):
                    print(
                        f"Early stop at episode {ep}: no train-EMA best improvement "
                        f"for {self.episodes_since_best} episodes "
                        f"(patience={self.early_stop_patience}).",
                        flush=True,
                    )
                    final_graph = self.graph_dir / "final.png"
                    _plot_rewards(
                        raw_curve,
                        final_graph,
                        title=plot_title,
                        ylabel=plot_ylabel,
                        physchem_rewards=plot_physchem,
                    )
                    if eval_episodes:
                        _plot_eval_curve(
                            eval_episodes,
                            eval_raw_curve,
                            eval_ema_curve,
                            self.graph_dir / "eval.png",
                            best_episode=self.best_eval_episode or None,
                        )
                    self.save_checkpoint(ep, "periodic")
                    break

            # Track decline for inception boost only (no mid-run weight restore).
            if self.co2_mode and math.isfinite(self.best_train_ema):
                _, self.decline_streak = decline_streak_should_restore(
                    train_monitor if math.isfinite(train_monitor) else float("nan"),
                    self.best_train_ema,
                    self.decline_streak,
                    margin=self.decline_margin,
                    patience=self.decline_patience,
                    relative=True,
                )

            final_graph = self.graph_dir / "final.png"
            _plot_rewards(
                raw_curve,
                final_graph,
                title=plot_title,
                ylabel=plot_ylabel,
                physchem_rewards=plot_physchem,
            )

            is_last = ep == n_episodes
            if is_last or (self.save_every > 0 and ep % self.save_every == 0):
                p = self.save_checkpoint(ep, "periodic")
                _cleanup_old_periodic(
                    self.checkpoint_dir, self.run_name, self.keep_last_n_periodic
                )
                print(f"Periodic checkpoint at episode {ep}: {p}", flush=True)

        print("RL training completed.", flush=True)
        print(f"Raw objective curve length: {len(raw_curve)}", flush=True)

    def _write_reward_curve(
        self, raw_curve: list[float], physchem_curve: list[float] | None = None
    ) -> None:
        path = getattr(self, "curve_path", self.run_dir / "reward_curve.csv")
        phys = physchem_curve or []
        with path.open("w") as f:
            f.write("episode,reward,physchem\n")
            for i, r in enumerate(raw_curve, start=1):
                p = phys[i - 1] if i - 1 < len(phys) else float("nan")
                f.write(f"{i},{r},{p}\n")


_RL_METRIC_KEYS = (
    "policy_loss",
    "gated_mean_reward",
    "average_raw_objective",
    "average_physchem_objective",
    "gated_valid_frac",
    "valid_frac",
    "valid_frac_ema",
    "unique_gated_frac",
    "unique_gated_frac_ema",
    "n_unique_gated",
    "n_valid",
    "n_gated",
    "n_learn",
    "skipped_update",
    "kl",
    "kl_coef",
    "prior_fp_mean_factor",
    "prior_fp_floor",
    "tether",
    "tether_coef",
    "tether_d",
    "tether_d_star",
    "tether_abs_dev",
    "tether_shell_hit",
    "tether_d_morgan",
    "tether_d_2d",
    "tether_d_3d",
    "tether_dp_morgan",
    "tether_dp_2d",
    "tether_dp_3d",
    "entropy_coef",
    "scheduled_temperature",
    "temperature",
    "top_quantile",
)


def _append_rl_metrics(path: Path, episode: int, metrics: dict) -> None:
    """Append one episode of regularizer/reward columns (same schema every run)."""
    new_file = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        if new_file:
            f.write("episode," + ",".join(_RL_METRIC_KEYS) + "\n")
        cells = [str(episode)]
        for k in _RL_METRIC_KEYS:
            v = metrics.get(k, float("nan"))
            if v is None:
                cells.append("nan")
            else:
                cells.append(str(v))
        f.write(",".join(cells) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE molecular generation")
    parser.add_argument(
        "--config",
        default="train.yaml",
        help="YAML config (paths relative to config file directory)",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Override initial checkpoint (default: GENERATE.CKPT_PATH in config)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume a run dir or checkpoint .pt (restores episode/optimizer/baselines)",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_script_dir() / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = RLPipeline(
        str(cfg_path),
        ckpt_path=args.ckpt,
        resume_dir=args.resume,
    )
    pipeline.rl_train()


if __name__ == "__main__":
    main()
