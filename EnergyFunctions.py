import torch
import torch.nn.functional as F


def fluency_energy(span_probs, pred_logits):
    """Soft NLL of the span under the LM.
    span_probs:  (B, k, V)  soft tokens p_i
    pred_logits: (B, k, V)  LM's prediction for each span position
    Returns scalar: mean_i  -sum_v p_i(v) * log q_i(v)   (cross-entropy of p against q)
    """
    log_q = F.log_softmax(pred_logits, dim=-1)       # (B, k, V)
    return -(span_probs * log_q).sum(-1).mean()      # sum over V, mean over B,k


def keyword_energy(span_probs, kw_ids, temperature=1.0, eps=1e-10):
    """Push the keyword (sequence of token ids) to appear *somewhere* in the span.
    For each possible start offset s, cost = -sum_j log p_{s+j}(kw_j).
    A soft-min over offsets keeps it differentiable and lets the optimizer pick where.
    """
    B, k, V = span_probs.shape
    m = len(kw_ids)
    assert m <= k, "span shorter than keyword"
    kw = torch.as_tensor(kw_ids, device=span_probs.device)
    logp = torch.log(span_probs[:, :, kw] + eps)     # (B, k, m): logp[b,i,j] = log p_i(kw_j)
    costs = torch.stack(
        [-sum(logp[:, s + j, j] for j in range(m)) for s in range(k - m + 1)],
        dim=1,
    )                                                # (B, k-m+1)
    return (-temperature * torch.logsumexp(-costs / temperature, dim=1)).clamp(min=0).mean()


def attribute_energy(class_logits, target_class):
    """Generic attribute energy: CE of a classifier's output vs the desired class.
    class_logits: (B, C). Plug in a sentiment classifier later (optional)."""
    target = torch.full((class_logits.shape[0],), target_class, device=class_logits.device)
    return F.cross_entropy(class_logits, target)

def suffix_energy(logits_all, k, suffix_ids):
    """NLL of the fixed suffix tokens given the soft span before them.
    logits_all: (B, k+m, V) predictions aligned so row i predicts token i of [span; suffix]."""
    m = len(suffix_ids)
    tgt = torch.as_tensor(suffix_ids, device=logits_all.device).unsqueeze(0).expand(logits_all.shape[0], m)
    return F.cross_entropy(logits_all[:, k:].reshape(-1, logits_all.shape[-1]), tgt.reshape(-1))