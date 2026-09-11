"""Hierarchical chemical-space tether: NN distances to a reference manifold.

Channels: Morgan Tanimoto, optional 2D embeddings, optional 3D embeddings.
Raw channel distances are divided by bank-median leave-one-out k-NN (``s_c``)
so combined ``d`` is in typical-NN units (on-manifold ≈ 1). ``λ`` remains a
vote after that conversion. Failed encodes are dropped and ``λ`` renormalized.
Distances are detached from θ; the policy term is REINFORCE on centered d
(``mean(log π · (d − d̄))``), not a batch-mean scalar (that has zero gradient).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

INVALID_DIST = 1.0
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
DEFAULT_2D_MODEL = "DeepChem/ChemBERTa-77M-MLM"
# |d − D*| band for tether_shell_hit (NN-units; typical d ≈ 1).
SHELL_HIT_BAND = 0.15


class VectorEncoder(Protocol):
    def encode(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (L2-normalized embeddings [B, D], valid mask [B])."""
        ...


def morgan_bits(smiles: str, *, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_N_BITS) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _nn_from_sim(sim: np.ndarray, k: int) -> np.ndarray:
    """Convert similarity matrix [B, N] → k-NN distance ``1 - mean(top-k sim)``."""
    if sim.size == 0:
        return np.full((sim.shape[0],), INVALID_DIST, dtype=np.float64)
    kth = min(max(int(k), 1), sim.shape[1])
    if kth <= 1:
        return (1.0 - sim.max(axis=1)).astype(np.float64)
    top = np.partition(sim, -kth, axis=1)[:, -kth:]
    return (1.0 - top.mean(axis=1)).astype(np.float64)


def tanimoto_nn_dist(query: np.ndarray, bank: np.ndarray, k: int = 1) -> np.ndarray:
    """query [B, D], bank [N, D] bit vectors → Tanimoto distance to k-NN."""
    if bank.size == 0 or query.size == 0:
        return np.full((query.shape[0],), INVALID_DIST, dtype=np.float64)
    q = query.astype(np.float32, copy=False)
    b = bank.astype(np.float32, copy=False)
    inter = q @ b.T
    union = q.sum(axis=1, keepdims=True) + b.sum(axis=1) - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return _nn_from_sim(sim, k)


def cosine_nn_dist(query: np.ndarray, bank: np.ndarray, k: int = 1) -> np.ndarray:
    """query/bank L2-normalized [B, D]/[N, D] → cosine distance to k-NN."""
    if bank.size == 0 or query.size == 0 or bank.shape[1] == 0:
        return np.full((query.shape[0],), INVALID_DIST, dtype=np.float64)
    sim = query.astype(np.float32, copy=False) @ bank.astype(np.float32, copy=False).T
    return _nn_from_sim(sim, k)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def _has_emb(emb: np.ndarray | None) -> bool:
    if emb is None:
        return False
    arr = np.asarray(emb)
    return arr.ndim == 2 and arr.size > 0 and arr.shape[1] > 0


def _loo_knn_from_sim(sim: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out k-NN distances from a square similarity matrix."""
    n = sim.shape[0]
    if n <= 1:
        return np.full((n,), np.nan, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64).copy()
    np.fill_diagonal(sim, -np.inf)
    kth = min(max(int(k), 1), n - 1)
    if kth <= 1:
        return (1.0 - sim.max(axis=1)).astype(np.float64)
    top = np.partition(sim, -kth, axis=1)[:, -kth:]
    return (1.0 - top.mean(axis=1)).astype(np.float64)


def _median_scale(dist: np.ndarray) -> float:
    """Median LOO distance; 1.0 if undefined (n<2 or collapsed bank)."""
    d = np.asarray(dist, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    med = float(np.median(d))
    # ponytail: duplicate/collapsed bank → skip unit conversion
    if not np.isfinite(med) or med <= 1e-12:
        return 1.0
    return med


def loo_channel_distances(
    morgan: np.ndarray,
    emb_2d: np.ndarray | None = None,
    emb_3d: np.ndarray | None = None,
    k: int = 1,
) -> dict[str, np.ndarray]:
    """Leave-one-out k-NN raw distances per channel (NaN if channel absent)."""
    m = np.asarray(morgan)
    n = int(m.shape[0]) if m.ndim == 2 else 0
    nan = np.full((max(n, 0),), np.nan, dtype=np.float64)
    out = {"morgan": nan.copy(), "2d": nan.copy(), "3d": nan.copy()}
    if m.ndim == 2 and m.shape[0] > 1 and m.shape[1] > 0:
        q = m.astype(np.float32, copy=False)
        inter = q @ q.T
        union = q.sum(axis=1, keepdims=True) + q.sum(axis=1) - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        out["morgan"] = _loo_knn_from_sim(sim, k)
    for key, emb in (("2d", emb_2d), ("3d", emb_3d)):
        if not _has_emb(emb):
            continue
        x = l2_normalize(np.asarray(emb, dtype=np.float32), axis=1)
        if x.shape[0] <= 1:
            continue
        out[key] = _loo_knn_from_sim(x @ x.T, k)
    return out


def compute_channel_scales(
    morgan: np.ndarray,
    emb_2d: np.ndarray | None = None,
    emb_3d: np.ndarray | None = None,
    k: int = 1,
) -> dict[str, float]:
    """Median leave-one-out k-NN distance per channel (bank-typical units)."""
    dists = loo_channel_distances(morgan, emb_2d, emb_3d, k)
    return {key: _median_scale(val) for key, val in dists.items()}


def _scales_from_npz(data) -> dict[str, float] | None:
    keys = ("scale_morgan", "scale_2d", "scale_3d")
    if not all(k in data.files for k in keys):
        return None
    out = {
        "morgan": float(np.asarray(data["scale_morgan"]).reshape(-1)[0]),
        "2d": float(np.asarray(data["scale_2d"]).reshape(-1)[0]),
        "3d": float(np.asarray(data["scale_3d"]).reshape(-1)[0]),
    }
    if not all(np.isfinite(v) and v > 0.0 for v in out.values()):
        return None
    return out


def tether_loss(
    sequence_log_probs: torch.Tensor,
    cost: torch.Tensor,
    *,
    baseline: float | None = None,
) -> torch.Tensor:
    """REINFORCE tether: ``mean(logπ · (cost − baseline))``. ``cost`` is detached.

    ``baseline``:
      - ``float``: use that value (e.g. EMA of batch-mean cost) — preferred so a
        whole-batch drift still yields nonzero advantages.
      - ``None``: fall back to this-batch mean (legacy). Needs ≥2 finite samples.
    """
    cost = cost.detach()
    finite = torch.isfinite(cost)
    n_fin = int(finite.sum())
    if n_fin == 0:
        return sequence_log_probs.new_zeros(())
    cost = cost[finite]
    lp = sequence_log_probs[finite]
    if baseline is None:
        if n_fin <= 1:
            return sequence_log_probs.new_zeros(())
        adv = cost - cost.mean()
    else:
        adv = cost - float(baseline)
    return (lp * adv).mean()


def explore_cost(d: torch.Tensor, d_star: float) -> torch.Tensor:
    """Novelty-shell cost ``(d − D*)²``. ``d`` is detached."""
    d = d.detach()
    return (d - float(d_star)) ** 2


def tether_cost(
    d: torch.Tensor,
    *,
    mode: str,
    d_star: float = 0.0,
) -> torch.Tensor:
    """Map manifold distance to the REINFORCE cost for Mode I / Mode II.

    - ``regularize``: cost = d (pull toward the manifold)
    - ``explore``: cost = (d − D*)² (pull toward novelty radius D*)
    """
    key = str(mode).strip().lower()
    if key == "regularize":
        return d.detach()
    if key == "explore":
        return explore_cost(d, d_star)
    raise ValueError(f"Unknown chem tether mode: {mode!r} (expected regularize|explore)")


class HashStubEncoder:
    """Deterministic hash vectors for tests (no pretrained weights)."""

    def __init__(self, dim: int = 16):
        self.dim = int(dim)

    def encode(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        dim = self.dim
        out = np.zeros((len(smiles), dim), dtype=np.float32)
        valid = np.zeros((len(smiles),), dtype=bool)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol, canonical=True)
            digest = hashlib.sha256(can.encode()).digest()
            raw = np.frombuffer(digest, dtype=np.uint8)
            vec = np.resize(raw.astype(np.float32), dim)
            out[i] = l2_normalize(vec, axis=0)
            valid[i] = True
        return out, valid


class ChemBERTaEncoder:
    """Frozen ChemBERTa last-hidden mean pool, L2-normalized."""

    def __init__(self, model_name: str = DEFAULT_2D_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ChemBERTa 2D encoder needs transformers. "
                "Install: uv sync --extra tether"
            ) from e
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)

    def encode(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        self._ensure()
        n = len(smiles)
        valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool)
        if n == 0:
            hidden = int(self._model.config.hidden_size)
            return np.zeros((0, hidden), dtype=np.float32), valid
        texts = [s if ok else "C" for s, ok in zip(smiles, valid)]
        tok = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            hidden = self._model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        emb = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
        emb[~valid] = 0.0
        return emb, valid


class UniMolEncoder:
    """Frozen Uni-Mol 3D representation via unimol_tools (SMILES in)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._repr = None
        self._dim: int | None = None

    def _ensure(self) -> None:
        if self._repr is not None:
            return
        try:
            from unimol_tools import UniMolRepr
        except ImportError as e:
            raise ImportError(
                "Uni-Mol 3D encoder needs unimol_tools. "
                "Install: uv sync --extra tether"
            ) from e
        use_cuda = str(self.device).startswith("cuda")
        try:
            self._repr = UniMolRepr(
                data_type="molecule", remove_hs=False, use_cuda=use_cuda
            )
        except TypeError:
            self._repr = UniMolRepr(data_type="molecule", remove_hs=False)

    def encode(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        self._ensure()
        n = len(smiles)
        valid = np.zeros((n,), dtype=bool)
        rows: list[np.ndarray | None] = [None] * n
        dim = self._dim
        for i, smi in enumerate(smiles):
            if Chem.MolFromSmiles(smi) is None:
                continue
            try:
                packed = self._repr.get_repr([smi])
                vec = _unimol_vec(packed)
            except Exception:
                continue
            vec = l2_normalize(np.asarray(vec, dtype=np.float32).reshape(-1), axis=0)
            rows[i] = vec
            valid[i] = True
            if dim is None:
                dim = int(vec.shape[0])
                self._dim = dim
        if dim is None:
            dim = 512
        out = np.zeros((n, dim), dtype=np.float32)
        for i, vec in enumerate(rows):
            if vec is not None and vec.shape[0] == dim:
                out[i] = vec
            elif vec is not None:
                valid[i] = False
        return out, valid


def _unimol_vec(packed) -> np.ndarray:
    if isinstance(packed, dict):
        for key in ("cls_repr", "repr", "cls"):
            if key in packed:
                arr = np.asarray(packed[key], dtype=np.float32)
                return arr[0] if arr.ndim > 1 else arr
        raise KeyError(f"UniMolRepr keys: {list(packed)}")
    arr = np.asarray(packed, dtype=np.float32)
    return arr[0] if arr.ndim > 1 else arr


class ChemicalTether:
    """1-NN (or k-NN) distances to a precomputed manifold, per channel."""

    def __init__(
        self,
        *,
        smiles: list[str],
        morgan: np.ndarray,
        emb_2d: np.ndarray | None = None,
        emb_3d: np.ndarray | None = None,
        lambda_morgan: float = 1.0,
        lambda_2d: float = 1.0,
        lambda_3d: float = 1.0,
        k: int = 1,
        encoder_2d: VectorEncoder | None = None,
        encoder_3d: VectorEncoder | None = None,
        model_2d: str = DEFAULT_2D_MODEL,
        device: str = "cpu",
        scales: dict[str, float] | None = None,
    ):
        self.smiles = list(smiles)
        self.morgan = np.asarray(morgan, dtype=np.uint8)
        self.emb_2d = (
            l2_normalize(np.asarray(emb_2d, dtype=np.float32), axis=1)
            if _has_emb(emb_2d)
            else None
        )
        self.emb_3d = (
            l2_normalize(np.asarray(emb_3d, dtype=np.float32), axis=1)
            if _has_emb(emb_3d)
            else None
        )
        self.lambda_morgan = float(lambda_morgan)
        self.lambda_2d = float(lambda_2d)
        self.lambda_3d = float(lambda_3d)
        self.k = max(int(k), 1)
        self.device_str = device
        self._encoder_2d = encoder_2d
        self._encoder_3d = encoder_3d
        self._model_2d = model_2d
        self._cache_2d: dict[str, np.ndarray] = {}
        self._cache_3d: dict[str, np.ndarray] = {}
        if scales is not None:
            self.scales = {
                "morgan": 1.0,
                "2d": 1.0,
                "3d": 1.0,
                **{str(name): float(val) for name, val in scales.items()},
            }
        else:
            self.scales = compute_channel_scales(
                self.morgan, self.emb_2d, self.emb_3d, self.k
            )
        self._validate_channels()

    def _validate_channels(self) -> None:
        if self.lambda_morgan > 0 and (self.morgan.ndim != 2 or self.morgan.shape[0] == 0):
            raise ValueError("CHEM_TETHER_LAMBDA_MORGAN > 0 but Morgan bank is empty")
        if self.lambda_2d > 0 and self.emb_2d is None:
            raise ValueError(
                "CHEM_TETHER_LAMBDA_2D > 0 but bank has no 2D embeddings. "
                "Rebuild: PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py"
            )
        if self.lambda_3d > 0 and self.emb_3d is None:
            raise ValueError(
                "CHEM_TETHER_LAMBDA_3D > 0 but bank has no 3D embeddings. "
                "Rebuild: PYTHONPATH=. uv run python objectives/build_chemical_tether_bank.py"
            )

    @classmethod
    def from_npz(
        cls,
        path: Path | str,
        **kwargs,
    ) -> "ChemicalTether":
        data = np.load(path, allow_pickle=True)
        smiles = [str(s) for s in data["smiles"].tolist()]
        morgan = data["morgan"]
        emb_2d = data["emb_2d"] if "emb_2d" in data.files else None
        emb_3d = data["emb_3d"] if "emb_3d" in data.files else None
        k = int(kwargs.get("k", 1))
        stored = _scales_from_npz(data)
        stored_k = (
            int(np.asarray(data["scale_k"]).reshape(-1)[0])
            if "scale_k" in data.files
            else None
        )
        if stored is not None and stored_k == k:
            kwargs.setdefault("scales", stored)
        return cls(smiles=smiles, morgan=morgan, emb_2d=emb_2d, emb_3d=emb_3d, **kwargs)

    def _get_encoder_2d(self) -> VectorEncoder:
        if self._encoder_2d is None:
            self._encoder_2d = ChemBERTaEncoder(self._model_2d, device=self.device_str)
        return self._encoder_2d

    def _get_encoder_3d(self) -> VectorEncoder:
        if self._encoder_3d is None:
            self._encoder_3d = UniMolEncoder(device=self.device_str)
        return self._encoder_3d

    def _morgan_dist(self, smiles: list[str]) -> np.ndarray:
        n = len(smiles)
        query = np.zeros((n, self.morgan.shape[1]), dtype=np.uint8)
        ok = np.zeros((n,), dtype=bool)
        for i, smi in enumerate(smiles):
            bits = morgan_bits(smi)
            if bits is None:
                continue
            query[i] = bits
            ok[i] = True
        d = tanimoto_nn_dist(query, self.morgan, self.k)
        d[~ok] = np.nan
        return d

    def _emb_dist(
        self,
        smiles: list[str],
        bank: np.ndarray,
        encoder: VectorEncoder,
        cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        n = len(smiles)
        dim = bank.shape[1]
        query = np.zeros((n, dim), dtype=np.float32)
        ok = np.zeros((n,), dtype=bool)
        missing_i: list[int] = []
        missing_s: list[str] = []
        for i, smi in enumerate(smiles):
            key = _canon_or_raw(smi)
            hit = cache.get(key)
            if hit is not None:
                if hit.shape[0] == dim:
                    query[i] = hit
                    ok[i] = True
                continue
            missing_i.append(i)
            missing_s.append(smi)
        if missing_s:
            emb, valid = encoder.encode(missing_s)
            if emb.shape[1] != dim:
                # Dim mismatch → drop channel for these molecules.
                valid = np.zeros((len(missing_s),), dtype=bool)
            for j, i in enumerate(missing_i):
                key = _canon_or_raw(missing_s[j])
                if not valid[j]:
                    continue
                vec = emb[j]
                cache[key] = vec
                query[i] = vec
                ok[i] = True
        d = cosine_nn_dist(query, bank, self.k)
        d[~ok] = np.nan
        return d

    def channel_distances(self, smiles: list[str]) -> dict[str, np.ndarray]:
        n = len(smiles)
        nan = np.full((n,), np.nan, dtype=np.float64)
        out = {"morgan": nan.copy(), "2d": nan.copy(), "3d": nan.copy()}
        if self.lambda_morgan > 0:
            out["morgan"] = self._morgan_dist(smiles)
        if self.lambda_2d > 0:
            out["2d"] = self._emb_dist(
                smiles, self.emb_2d, self._get_encoder_2d(), self._cache_2d
            )
        if self.lambda_3d > 0:
            out["3d"] = self._emb_dist(
                smiles, self.emb_3d, self._get_encoder_3d(), self._cache_3d
            )
        return out

    def combined_distance(self, channels: dict[str, np.ndarray], n: int) -> np.ndarray:
        """Weighted mean of calibrated active channels (typical bank NN ≈ 1).

        ``d'_c = d_c / s_c``. Failed (NaN) channels are dropped and ``λ``
        renormalized per molecule. All-fail → NaN.
        """
        specs = (
            ("morgan", self.lambda_morgan),
            ("2d", self.lambda_2d),
            ("3d", self.lambda_3d),
        )
        num = np.zeros((n,), dtype=np.float64)
        den = np.zeros((n,), dtype=np.float64)
        for name, lam in specs:
            if lam <= 0.0:
                continue
            raw = np.asarray(channels[name], dtype=np.float64)
            scale = float(self.scales.get(name, 1.0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            primed = raw / scale
            ok = np.isfinite(primed)
            num = num + np.where(ok, lam * primed, 0.0)
            den = den + np.where(ok, lam, 0.0)
        return np.divide(num, den, out=np.full((n,), np.nan, dtype=np.float64), where=den > 0.0)

    def calibrate_channels(self, channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Raw channel distances → NN-units ``d'_c = d_c / s_c``."""
        out: dict[str, np.ndarray] = {}
        for name in ("morgan", "2d", "3d"):
            raw = np.asarray(channels[name], dtype=np.float64)
            scale = float(self.scales.get(name, 1.0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            out[name] = raw / scale
        return out

    def distances(
        self, smiles: list[str], *, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        channels = self.channel_distances(smiles)
        combined = self.combined_distance(channels, len(smiles))
        t = torch.as_tensor(combined, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t, channels


def _canon_or_raw(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def save_bank(
    path: Path | str,
    smiles: list[str],
    morgan: np.ndarray,
    emb_2d: np.ndarray | None = None,
    emb_3d: np.ndarray | None = None,
    k: int = 1,
) -> dict[str, float]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(smiles)
    e2 = (
        np.asarray(emb_2d, dtype=np.float32)
        if _has_emb(emb_2d)
        else np.zeros((n, 0), dtype=np.float32)
    )
    e3 = (
        np.asarray(emb_3d, dtype=np.float32)
        if _has_emb(emb_3d)
        else np.zeros((n, 0), dtype=np.float32)
    )
    k = max(int(k), 1)
    scales = compute_channel_scales(
        morgan,
        e2 if e2.shape[1] > 0 else None,
        e3 if e3.shape[1] > 0 else None,
        k,
    )
    np.savez(
        path,
        smiles=np.asarray(smiles, dtype=object),
        morgan=np.asarray(morgan, dtype=np.uint8),
        emb_2d=e2,
        emb_3d=e3,
        scale_morgan=np.float64(scales["morgan"]),
        scale_2d=np.float64(scales["2d"]),
        scale_3d=np.float64(scales["3d"]),
        scale_k=np.int32(k),
    )
    return scales


def union_manifold_smiles(
    prior_smiles: list[str],
    axel_smiles: list[str],
) -> list[str]:
    """Canonical unique parseable SMILES, prior bank then AxelRolov."""
    out: list[str] = []
    seen: set[str] = set()
    for smi in list(prior_smiles) + list(axel_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    return out
