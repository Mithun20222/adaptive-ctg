import copy
import torch
import torch.nn.functional as F
from EnergyFunctions import fluency_energy, keyword_energy


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

    def optimize(self, past, last_logits, kw_ids, k):
        """past: frozen prefix cache. last_logits: (B, V) logits at the prefix's last position.
        Returns span_ids (B,k), new_past (prefix+span), next_logits (B,V), history."""
        last_logits = last_logits.detach()                 # DETACH: prefix is a constant
        y = self._init_logits(past, last_logits, k)        # (B, k, V), no graph so far
        y.requires_grad_(True)                             # the ONLY tensor we optimize
        history = []

        for it in range(self.iters):
            cache = copy.deepcopy(past)                    # fresh prefix copy, no grad history
            probs = F.softmax(y / self.temperature, dim=-1)           # (B, k, V)
            embeds = self.lm.probs_to_embeds(probs)                   # (B, k, d)
            logits, _ = self.lm.soft_forward(embeds, past_key_values=cache)   # (B, k, V)
            # shift by one: span token 0 is predicted by the prefix's last logits
            pred = torch.cat([last_logits.unsqueeze(1), logits[:, :-1]], dim=1)  # (B, k, V)
            
            e_lm = fluency_energy(probs, pred)
            kws = kw_ids if isinstance(kw_ids[0], list) else [kw_ids]   # one keyword or a list
            e_kw = sum(keyword_energy(probs, kw) for kw in kws)
            energy = self.lambda_lm * e_lm + self.lambda_hard * e_kw

            # autograd.grad (not backward): returns dE/dy only, accumulates nothing,
            # and frees the graph here. `cache` goes out of scope next iteration.
            (grad,) = torch.autograd.grad(energy, y)

            with torch.no_grad():                          # DETACH: update outside the graph
                eta = self.step_size
                sigma = self.noise_std * (1 - it / self.iters)   # annealed noise
                y -= 0.5 * eta * grad                            # drift down the energy
                y += (eta ** 0.5) * sigma * torch.randn_like(y)  # Langevin noise
            history.append((it, e_lm.item(), e_kw.item()))

        y = y.detach()                                     # leave the graph for good
        span_ids, new_past, next_logits = self._discretize(y, past, last_logits)
        return span_ids, new_past, next_logits, history, y