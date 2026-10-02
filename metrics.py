import math
import torch
import torch.nn.functional as F


def contains(seq, kw):
    """True if kw appears as a contiguous run inside seq (both lists of token ids)."""
    m = len(kw)
    return any(seq[i:i + m] == kw for i in range(len(seq) - m + 1))


@torch.no_grad()
def perplexity(lm, prefix, ids):
    """PPL of generated tokens given the prefix, under plain GPT-2."""
    full = torch.cat([prefix, ids], dim=1)
    lg, _ = lm.discrete_forward(full)
    P = prefix.shape[1]
    nll = F.cross_entropy(lg[0, P - 1:-1], ids[0])   # logits at i predict token i+1
    return math.exp(nll.item())