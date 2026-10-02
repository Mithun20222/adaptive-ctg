import copy
import torch
from metrics import contains


class HybridEngine:
    """BOLT by default; the router hands short spans to COLD when a constraint is at risk."""

    def __init__(self, lm, bolt, cold, router):
        self.lm, self.bolt, self.cold, self.router = lm, bolt, cold, router

    def generate(self, past, last_logits, kws, L):
        # 1) tune BOLT's bias ONCE (needs grad, so outside no_grad)
        bias, _ = self.bolt.tune(past, last_logits, kws, L)

        cache, cur = copy.deepcopy(past), last_logits.detach()   # cur: (B, V) next-token logits
        ids, escalations, t = [], [], 0

        while t < L:
            action = self.router.check(t, ids)
            if action is not None:
                kw, k = action
                # 2) COLD span: prefix cache is frozen inside optimize(); returns committed cache
                span, cache, cur, hist, _ = self.cold.optimize(cache, cur, [kw], k)
                ids += span[0].tolist()
                ok = contains(span[0].tolist(), kw)                       # did the keyword land in the span?
                escalations.append((t, k, ok, round(hist[-1][2], 3)))     # (start, span len, placed?, final E_kw)
                t += k
            else:
                # 3) one BOLT step with the tuned bias (no graph needed)
                with torch.no_grad():
                    z = cur + (1 - t / L) * bias[:, t]            # (B, V), your spec's decay
                    tok = z.argmax(-1, keepdim=True)              # (B, 1)
                    out, cache = self.lm.discrete_forward(tok, past_key_values=cache)
                    cur = out[:, -1]
                ids.append(tok.item())
                t += 1

        return torch.tensor([ids], device=last_logits.device), escalations