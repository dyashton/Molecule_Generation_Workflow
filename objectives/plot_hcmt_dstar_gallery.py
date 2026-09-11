#!/usr/bin/env python3
"""Quick top-molecule gallery across HCMT arms.

Example:
  PYTHONPATH=. uv run python objectives/plot_hcmt_dstar_gallery.py \\
    --root outputs/druglike_dstar_arms_nn_c4 \\
    --arms no_tether dstar_100 dstar_130 dstar_200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectives.plot_hcmt_dstar_physchem import (
    DEFAULT_ARMS,
    DEFAULT_ROOT,
    _label,
    _late_arm_frame,
)
from objectives.plot_hcmt_dstar_tsne import _latest_run

SUB = (280, 240)
LABEL_H = 56  # title + meta need ~50px; 36 clipped the med-obj line
ROW_GAP = 12


def _top_mols(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    df = frame.copy()
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
    if "d_prime" in df.columns:
        df["d_prime"] = pd.to_numeric(df["d_prime"], errors="coerce")
    df = df.dropna(subset=["quality"]).sort_values("quality", ascending=False)
    out_rows = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can in seen:
            continue
        seen.add(can)
        out_rows.append(row)
        if len(out_rows) >= n:
            break
    return pd.DataFrame(out_rows)


def _legend(row: pd.Series) -> str:
    q = float(row["quality"])
    logp = float(pd.to_numeric(row.get("raw_ba"), errors="coerce"))
    d = float(pd.to_numeric(row.get("d_prime"), errors="coerce"))
    # Slot map: druglike raw_ba=logitQED; unbounded_proxy raw_ba=MolLogP.
    parts = [f"obj {q:.2f}", f"logP/QED {logp:.2f}"]
    if np.isfinite(d):
        parts.append(f"d′ {d:.2f}")
    return "  ·  ".join(parts)


def _font(size: int):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--n-snaps", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    out = args.out
    if out is None:
        out = root / "hcmt_dstar_gallery.png"
    elif not out.is_absolute():
        out = ROOT / out

    rows_img: list[Image.Image] = []
    title_font = _font(18)
    meta_font = _font(12)

    for arm in args.arms:
        run_dir = _latest_run(root / arm)
        frame = _late_arm_frame(run_dir, args.n_snaps)
        top = _top_mols(frame, args.top_n)
        print(
            f"{arm}: showing {len(top)}/{args.top_n} from last healthy snaps ({run_dir.name})",
            flush=True,
        )
        if top.empty:
            continue
        mols = [Chem.MolFromSmiles(str(s)) for s in top["smiles"]]
        legends = [_legend(row) for _, row in top.iterrows()]
        grid = Draw.MolsToGridImage(
            mols,
            molsPerRow=len(mols),
            subImgSize=SUB,
            legends=legends,
            returnPNG=False,
        )
        if not isinstance(grid, Image.Image):
            grid = Image.fromarray(np.asarray(grid))
        row_w = max(grid.width, SUB[0] * args.top_n)
        canvas = Image.new("RGB", (row_w, LABEL_H + grid.height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), _label(arm), fill=(30, 30, 30), font=title_font)
        n = len(top)
        med_q = float(top["quality"].median())
        med_d = (
            float(pd.to_numeric(top["d_prime"], errors="coerce").median())
            if "d_prime" in top.columns
            else float("nan")
        )
        meta = f"top {n} by objective  ·  med obj {med_q:.2f}"
        if np.isfinite(med_d):
            meta += f"  ·  med d′ {med_d:.2f}"
        draw.text((8, 30), meta, fill=(110, 110, 110), font=meta_font)
        canvas.paste(grid, (0, LABEL_H))
        rows_img.append(canvas)

    if not rows_img:
        raise SystemExit("No molecules to plot")

    width = max(im.width for im in rows_img)
    height = sum(im.height for im in rows_img) + ROW_GAP * (len(rows_img) - 1)
    final = Image.new("RGB", (width, height), "white")
    y = 0
    for im in rows_img:
        final.paste(im, (0, y))
        y += im.height + ROW_GAP

    out.parent.mkdir(parents=True, exist_ok=True)
    final.save(out, dpi=(160, 160))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
