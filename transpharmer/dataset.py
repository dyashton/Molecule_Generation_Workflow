import json
import re
from pathlib import Path

import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transpharmer.utils.pharm.misc import get_pharma_fp, get_pharma_count_fp


pad_token = '<'
nan_token = ''
_DEFAULT_SMILES_TOKENS = [
    '#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    pad_token, '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]',
    '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]',
    '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]',
    '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]',
    '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]',
    '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]',
    '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o',
    'p', 's',
]
tokens = list(_DEFAULT_SMILES_TOKENS)
vocab_size = len(tokens)
stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for i, ch in enumerate(tokens)}

_SMILES_PATTERN = (
    "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|"
    "\\*|\\$|\\%[0-9]{2}|[0-9])"
)
_SELFIES_PATTERN = r"(\[[^\]]+\]|<)"
pattern = _SMILES_PATTERN
regex = re.compile(pattern)
_REPRESENTATION = "smiles"


def _resolve_vocab_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    cwd_hit = (Path.cwd() / path).resolve()
    if cwd_hit.is_file():
        return cwd_hit
    root_hit = (Path(__file__).resolve().parents[1] / path).resolve()
    if root_hit.is_file():
        return root_hit
    raise FileNotFoundError(f"VOCAB_PATH not found: {raw}")


def configure_from_config(config) -> None:
    """Load MODEL.VOCAB_PATH / REPRESENTATION into module-level stoi/itos (in-place)."""
    global vocab_size, pattern, regex, _REPRESENTATION, pad_token

    model = getattr(config, "MODEL", config)
    rep = str(getattr(model, "REPRESENTATION", "smiles") or "smiles").strip().lower()
    vocab_path = getattr(model, "VOCAB_PATH", None)

    if vocab_path:
        data = json.loads(_resolve_vocab_path(vocab_path).read_text())
        new_tokens = list(data["tokens"])
        pad_token = str(data.get("pad_token", "<"))
        rep = str(data.get("representation", rep) or rep).strip().lower()
    else:
        new_tokens = list(_DEFAULT_SMILES_TOKENS)
        pad_token = "<"

    tokens[:] = new_tokens
    stoi.clear()
    stoi.update({ch: i for i, ch in enumerate(tokens)})
    itos.clear()
    itos.update({i: ch for i, ch in enumerate(tokens)})
    vocab_size = len(tokens)
    _REPRESENTATION = rep
    pattern = _SELFIES_PATTERN if _REPRESENTATION == "selfies" else _SMILES_PATTERN
    regex = re.compile(pattern)

    expected = getattr(model, "VOCAB_SIZE", None)
    if expected is not None and int(expected) != vocab_size:
        raise ValueError(
            f"MODEL.VOCAB_SIZE={expected} does not match loaded vocab ({vocab_size})"
        )


def tokenize(smiles, max_len=0):
    ids = [stoi[s] for s in regex.findall(smiles)]
    if max_len > 0:
        ids += [stoi[pad_token]] * (max_len - len(ids))
        ids = ids[:max_len]
    return ids


def untokenize(token_ids):
    text = "".join([itos[int(t)] for t in token_ids])
    return text.replace(pad_token, nan_token)


def tokenize_from_smiles(smiles: str, max_len: int = 0):
    """SMILES → token ids (encodes to SELFIES when representation=selfies)."""
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    if _REPRESENTATION == "selfies":
        import selfies as sf

        try:
            text = sf.encoder(text)
        except Exception:
            return None
    try:
        return tokenize(text, max_len=max_len)
    except KeyError:
        return None


def tokens_to_smiles(token_ids) -> str:
    """Token ids → SMILES (decodes SELFIES when representation=selfies)."""
    raw = untokenize(token_ids)
    if not raw:
        return ""
    if _REPRESENTATION == "selfies":
        import selfies as sf

        try:
            return str(sf.decoder(raw))
        except Exception:
            return ""
    return raw


class SmileDataset(Dataset):
    def __init__(self, args, dataframe, block_size, num_props, aug_prob=0.5, isomericSmiles=True, canonical=False):
        self.args = args
        self.num_props = num_props
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = dataframe
        self.aug_prob = aug_prob
        self.isomericSmiles = isomericSmiles
        self.canonical = canonical
        self.buffer = None
    
    def __len__(self):
        return len(self.data)

    def get_prop(self, mol):
        try:
            if mol is not None:
                if self.num_props == 80:  # 72-bit fp & feature counts (8)
                    prop = get_pharma_count_fp(mol)
                else:
                    prop = get_pharma_fp(mol, n_dim=self.num_props)
                return prop
            # ct = get_pharma_count(mol)
            # fp = get_pharma_fp(mol)
            # fp = [t for t in fp]
            # prop = ct + fp
            # return prop
        except:
            return

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def randomize(self, smi):
        try:
            if np.random.uniform() < self.aug_prob:
                smi = self.randomize_smiles(smi)
            dix = tokenize(smi, self.max_len)
            return dix
        except:
            return

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        smi = data.smiles.strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            dix = self.randomize(smi)
            if dix is not None:
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
                if self.num_props:
                    prop = self.get_prop(mol)
                    if prop is not None:
                        p = torch.tensor(prop, dtype = torch.float)
                        if self.buffer is None:
                            self.buffer = (x, y, p)
                        return x, y, p
                else:
                    if self.buffer is None:
                        self.buffer = (x, y)
                    return x, y
        return self.buffer
    

def build_dataloader(dataset, is_train, batch_size, num_workers):
    dataloader = DataLoader(dataset, 
                            shuffle=is_train,
                            pin_memory=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    return dataloader
