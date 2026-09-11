# CO2 publication figures

Efficacy panels for the CO2-affine molecule generation workflow.

## Regenerate

```bash
cd Molecule_Generation_Workflow
uv run python publication/co2/make_figures.py \
  --checkpoint outputs/output_1/default/<run>/checkpoints/best_default_epN.pt \
  --train-graph outputs/output_1/default/<run>/graphs/final.png
```

Both `--checkpoint` and `--train-graph` are required.
Optional `--out-dir PATH` writes `figures/` and `data/` there instead of `publication/co2/`.

Generates **2,000** SMILES per model and scores every generated string before plotting.
Cached libraries and scores live in `data/` (auto-refreshed if size ≠ 2k); delete that folder to force a full resample.

### Checkpoint progression (opt-in)

With `--tsne-progression`, also sample each `periodic_*_ep*.pt` next to the checkpoint (**500** molecules each), then write:

```
<out-dir>/tsne/epXXXX.png + progression.gif     # shared t-SNE frames
<out-dir>/top_hits/epXXXX.png + progression.gif # top-12 galleries per episode
```

```bash
uv run python publication/co2/make_figures.py \
  --checkpoint outputs/output_1/<run>/checkpoints/best_....pt \
  --train-graph outputs/output_1/<run>/graphs/final.png \
  --out-dir outputs/output_1/<run>/analysis \
  --tsne-progression
```

Caches live under `<out-dir>/data/tsne_prog/`. Requires periodics on disk (`KEEP_LAST_N_PERIODIC: 0` keeps every `SAVE_EVERY` save).

## Figures

| File | Content |
|------|---------|
| `learning_curve.png` | Training mean raw objective vs episode |
| `baseline_reward.png` | Pretrained vs RL CO2 reward (all generated) |
| `baseline_ecdf.png` | ECDF of CO2 reward (all generated) |
| `property_space.png` | Property histograms (all generated; physchem where scored) |
| `top_hits_gallery.png` | Top-12 RL structures by reward |
| `chemotype_tsne.png` | Morgan FP t-SNE colored by reward (all generated) |
| `validity_uniqueness_gates.png` | Validity / uniqueness / gate rates |
| `physchem_overlay.png` | Physchem histograms: pretrained vs RL |
| `summary_stats.png` | Numeric summary table (all generated) |
| `tsne/epXXXX.png` + `progression.gif` | Opt-in chemotype progression over periodics |
| `top_hits/epXXXX.png` + `progression.gif` | Opt-in top-12 gallery progression over periodics |

**Note:** Physchem/PFC panels omit gate failures (those fields are undefined); reward always uses the full generated set.
