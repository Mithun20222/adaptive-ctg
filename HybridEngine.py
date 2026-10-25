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
                rem, k = action
                span, cache, cur, hist, _ = self.cold.optimize(cache, cur, rem, k)
                ids += span[0].tolist()
                placed = sum(contains(span[0].tolist(), kw) for kw in rem)
                escalations.append((t, k, f"{placed}/{len(rem)}", round(hist[-1][2], 3)))
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

    def generate_infill(self, past, last_logits, suffix_ids, bridge_len=6, lambda_right=1.0):
        cache, cur, ids, t = copy.deepcopy(past), last_logits.detach(), [], 0
        while t < self.router.L:
            k = self.router.check_suffix(t, bridge_len)
            if k is not None:
                span, cache, cur, hist, _ = self.cold.optimize(cache, cur, [], k, suffix_ids=suffix_ids, lambda_right=lambda_right)
                ids += span[0].tolist()
                return torch.tensor([ids], device=last_logits.device), hist
            with torch.no_grad():
                tok = cur.argmax(-1, keepdim=True)
                out, cache = self.lm.discrete_forward(tok, past_key_values=cache)
                cur = out[:, -1]
            ids.append(tok.item()); t += 1