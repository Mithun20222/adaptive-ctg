import time, torch
from transformers import GPT2TokenizerFast
from dual_path_lm import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from AdaptiveRouter import AdaptiveRouter
from HybridEngine import HybridEngine
from metrics import contains, perplexity

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last = logits[:, -1]

kws = [tok(w).input_ids for w in [" submarine", " treasure", " octopus", " lighthouse"]]
L = 10

def report(name, ids, dt):
    got = sum(contains(ids[0].tolist(), kw) for kw in kws)
    print(f"{name:<22} {got}/4 | ppl {perplexity(lm, prefix, ids):8.1f} | {L/dt:6.2f} tok/s | {tok.decode(ids[0])!r}")

bolt = BoltEngine(lm, iters=5, lr=30)
cold = ColdSpanEngine(lm, iters=150)

t0 = time.time(); ids, *_ = bolt.generate(past, last, kws, L); report("BOLT", ids, time.time() - t0)
t0 = time.time(); ids, *_ = cold.optimize(past, last, kws, k=L); report("COLD (full)", ids, time.time() - t0)

for tau in [5.0, 3.0, 2.0]:
    hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter(kws, L, tau_horizon=tau))
    t0 = time.time(); ids, esc = hyb.generate(past, last, L=L, kws=kws)
    report(f"HYBRID tau={tau}", ids, time.time() - t0)
    print(f"   escalations (t, k): {esc}")