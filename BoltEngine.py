import copy
import torch
import torch.nn.functional as F
from EnergyFunctions import fluency_energy, keyword_energy


class BoltEngine:
    def __init__(self, lm, iters=5, lr=3.0, tau=1.0, lambda_lm=1.0, lambda_hard=1.0):
        self.lm, self.iters, self.lr, self.tau = lm, iters, lr, tau
        self.lambda_lm, self.lambda_hard = lambda_lm, lambda_hard

    def _rollout(self, past, last_logits, bias, t0, steps, L):
        """Left-to-right decoding with biased logits + straight-through estimator."""
        cache = copy.deepcopy(past)              # fresh copy: no graph history from earlier iterations
        cur = last_logits                        # (B, V) LM logits for the next token
        ids, Ys, Ss, LMs = [], [], [], []
        for s in range(steps):
            t = t0 + s
            w = 1 - t / L                                        # linear decay of bias strength
            z = cur + w * bias[:, t]                             # (B, V) biased logits
            soft = F.softmax(z / self.tau, dim=-1)               # backward path
            hard = F.one_hot(z.argmax(-1), z.shape[-1]).float()  # forward path (real token)
            y = soft + (hard - soft).detach()                    # STE: forward=hard, grad=soft
            ids.append(z.argmax(-1)); Ys.append(y); Ss.append(soft); LMs.append(cur)
            emb = self.lm.probs_to_embeds(y).unsqueeze(1)        # (B, 1, d), differentiable
            out, cache = self.lm.soft_forward(emb, past_key_values=cache)
            cur = out[:, -1]                                     # (B, V) logits for token t+1
        stack = lambda xs: torch.stack(xs, dim=1)                # (B, steps, ...)
        return torch.stack(ids, dim=1), stack(Ys), stack(Ss), stack(LMs), cache, cur

    def generate(self, past, last_logits, kw_ids, L, t0=0, steps=None):
        steps = steps or (L - t0)
        last_logits = last_logits.detach()                       # prefix is a constant
        B, V = last_logits.shape
        bias = torch.zeros(B, L, V, device=last_logits.device, requires_grad=True)  # the only leaf
        history = []

        for it in range(self.iters):
            _, Y, S, LM, _, _ = self._rollout(past, last_logits, bias, t0, steps, L)
            e_lm = fluency_energy(Y, LM)         # Y: STE tokens (B,steps,V); LM[s] already predicts token s
            kws = kw_ids if isinstance(kw_ids[0], list) else [kw_ids]   # one keyword or a list
            e_kw = sum(keyword_energy(S, kw) for kw in kws)
            loss = self.lambda_lm * e_lm + self.lambda_hard * e_kw
            bias.grad = None
            loss.backward()                      # graph freed here; only `bias` gets a gradient
            with torch.no_grad():
                g = bias.grad / (bias.grad.abs().max() + 1e-12)  # normalize: largest entry moves by lr
                bias -= self.lr * g
            history.append((it, e_lm.item(), e_kw.item()))

        with torch.no_grad():                    # final decode with the tuned bias, no graph
            ids, _, _, _, cache, nxt = self._rollout(past, last_logits, bias, t0, steps, L)
        return ids, cache, nxt, history