import copy
import torch
import torch.nn.functional as F
from EnergyFunctions import fluency_energy, keyword_energy, suffix_energy


class ColdSpanEngine:
    def __init__(self, lm, iters=300, step_size=1.0, noise_std=0.05,
                 lambda_lm=1.0, lambda_hard=1.0, temperature=1.0, topk=50, y_cands=3):
        self.lm = lm
        self.iters, self.step_size, self.noise_std = iters, step_size, noise_std
        self.lambda_lm, self.lambda_hard = lambda_lm, lambda_hard
        self.temperature, self.topk, self.y_cands = temperature, topk, y_cands

    @torch.no_grad()
    def _init_logits(self, past, last_logits, k):
        """Start from the LM's own greedy rollout: row i = LM's logits for span token i.
        Returns (B, k, V). Works on a copy so the real prefix cache is untouched."""
        cache, cur, rows = copy.deepcopy(past), last_logits, []
        for _ in range(k):
            rows.append(cur)                                   # (B, V)
            tok = cur.argmax(-1, keepdim=True)                 # (B, 1)
            out, cache = self.lm.discrete_forward(tok, past_key_values=cache)
            cur = out[:, -1]
        return torch.stack(rows, dim=1)                        # (B, k, V)

    @torch.no_grad()
    def _discretize(self, y, past, last_logits):
        """Top-k guided: at each position, LM proposes top-K tokens given the tokens
        chosen so far; we pick the one the optimized logits y like best.
        Feeding chosen tokens also builds the committed KV-cache (the re-sync)."""
        cache, cur, ids = copy.deepcopy(past), last_logits, []
        for i in range(y.shape[1]):
            cand = cur.topk(self.topk, dim=-1).indices                     # (B, K) LM's candidates
            if self.y_cands > 0:                                           # + optimizer's favorites
                cand = torch.cat([cand, y[:, i].topk(self.y_cands, dim=-1).indices], dim=1)
            scores = y[:, i].gather(1, cand)                               # (B, K+y_cands)
            tok = cand.gather(1, scores.argmax(-1, keepdim=True))      # (B, 1)
            ids.append(tok)
            out, cache = self.lm.discrete_forward(tok, past_key_values=cache)
            cur = out[:, -1]                                           # (B, V) next-token logits
        return torch.cat(ids, dim=1), cache, cur                       # (B,k), cache, (B,V)

    def optimize(self, past, last_logits, kw_ids, k, suffix_ids=None, lambda_right=1.0):
        last_logits = last_logits.detach()                       # DETACH: prefix is a constant
        y = self._init_logits(past, last_logits, k)              # (B, k, V), no graph so far
        y.requires_grad_(True)                                   # the ONLY tensor we optimize
        kws = [] if not kw_ids else (kw_ids if isinstance(kw_ids[0], list) else [kw_ids])
        B = y.shape[0]
        zero = torch.zeros((), device=y.device)
        if suffix_ids:
            sfx = torch.as_tensor(suffix_ids, device=y.device)                    # (m,)
            sfx_emb = self.lm.embed_matrix[sfx].unsqueeze(0).expand(B, -1, -1)   # (B, m, d), fixed
        history = []

        for it in range(self.iters):
            cache = copy.deepcopy(past)                          # fresh prefix copy, no grad history
            probs = F.softmax(y / self.temperature, dim=-1)      # (B, k, V)
            embeds = self.lm.probs_to_embeds(probs)              # (B, k, d)
            if suffix_ids:
                embeds = torch.cat([embeds, sfx_emb], dim=1)     # (B, k+m, d): suffix sits after the span
            logits, _ = self.lm.soft_forward(embeds, past_key_values=cache)       # (B, k(+m), V)
            pred = torch.cat([last_logits.unsqueeze(1), logits[:, :-1]], dim=1)   # row i predicts token i

            e_lm = fluency_energy(probs, pred[:, :k])
            e_kw = sum((keyword_energy(probs, kw) for kw in kws), zero)
            e_right = zero
            if suffix_ids:                                       # how likely is the fixed suffix after the span?
                e_right = F.cross_entropy(pred[:, k:].reshape(-1, pred.shape[-1]), sfx.repeat(B))
            energy = self.lambda_lm * e_lm + self.lambda_hard * e_kw + lambda_right * e_right

            (grad,) = torch.autograd.grad(energy, y)             # graph freed here
            with torch.no_grad():                                # DETACH: update outside the graph
                sigma = self.noise_std * (1 - it / self.iters)
                y -= 0.5 * self.step_size * grad
                y += (self.step_size ** 0.5) * sigma * torch.randn_like(y)
            history.append((it, e_lm.item(), e_kw.item(), e_right.item()))

        y = y.detach()
        span_ids, new_past, next_logits = self._discretize(y, past, last_logits)
        return span_ids, new_past, next_logits, history, y