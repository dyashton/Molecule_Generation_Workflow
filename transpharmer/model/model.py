import math
import torch
from torch import nn
from torch.nn import functional as F
from .embed import Embedding


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.N_EMBD % config.N_HEADS == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.query = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.value = nn.Linear(config.N_EMBD, config.N_EMBD)
        # regularization
        self.attn_drop = nn.Dropout(config.ATTN_PDROP)
        self.resid_drop = nn.Dropout(config.RESID_PDROP)
        # output projection
        self.proj = nn.Linear(config.N_EMBD, config.N_EMBD)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.MAX_LEN+config.NUM_PROPS, config.MAX_LEN+config.NUM_PROPS))
                                     .view(1, 1, config.MAX_LEN+config.NUM_PROPS, config.MAX_LEN+config.NUM_PROPS))
        self.n_head = config.N_HEADS

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.N_EMBD)
        self.ln2 = nn.LayerNorm(config.N_EMBD)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.N_EMBD, 4 * config.N_EMBD),
            nn.GELU(),
            nn.Linear(4 * config.N_EMBD, config.N_EMBD),
            nn.Dropout(config.RESID_PDROP))

    def forward(self, x):
        y = self.ln1(x)
        y = self.attn(y)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.config = config.MODEL
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.N_LAYERS)])
        self.embed = Embedding(config=self.config, device=self.device)
        self.ln_f = nn.LayerNorm(self.config.N_EMBD)
        self.head = nn.Linear(self.config.N_EMBD, self.config.VOCAB_SIZE, bias=False)
        self.block_size = self.config.MAX_LEN
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None, prop=None, return_hidden=False):
        x = self.embed(token=idx, prop=prop)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.config.NUM_PROPS:
            # logits = logits[:, self.config.NUM_PROPS:, :]
            pass
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, x
        return logits, loss


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def top_p_logits(logits, p):
    """Nucleus (top-p) filtering on softmax probs (Hugging Face style)."""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    sorted_remove = cumprobs > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_remove)
    out = logits.clone()
    out[indices_to_remove] = float('-inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample_=False, top_k=None, top_p=None, prop=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()   
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond, prop = prop)   # for liggpt
        # pluck the logits at the final step and scale by temperature
        # Guard against numerical issues during sampling:
        # - temperature near/at 0 can explode logits
        # - NaN/Inf logits (e.g., from unstable training) will break softmax/multinomial
        temperature = float(temperature)
        if temperature <= 0:
            temperature = 1e-6
        logits = logits[:, -1, :] / temperature
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        if top_p is not None:
            logits = top_p_logits(logits, top_p)
        # If filtering removed all mass for a row (all -inf), unmask to avoid NaNs.
        all_inf = torch.isneginf(logits).all(dim=-1, keepdim=True)
        if all_inf.any():
            logits = torch.where(all_inf, torch.zeros_like(logits), logits)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        denom = probs.sum(dim=-1, keepdim=True)
        zero = denom <= 0
        if zero.any():
            # Fallback: if a row is degenerate, pick argmax of (sanitized) logits.
            probs = torch.where(zero, torch.zeros_like(probs), probs)
            denom = torch.where(zero, torch.ones_like(denom), denom)
            probs = probs / denom
            ix = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat((x, ix), dim=1)
            continue
        probs = probs / denom
        # sample from the distribution or take the most likely
        if sample_:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    return x