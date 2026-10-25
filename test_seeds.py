import torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from AdaptiveRouter import AdaptiveRouter
from HybridEngine import HybridEngine
from metrics import contains, perplexity

tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last = logits[:, -1]
bolt, cold, L = BoltEngine(lm, iters=5, lr=30), ColdSpanEngine(lm, iters=150), 25

sets = {"rare1": [" bioluminescent", " hydrothermal", " leviathan", " periscope"],
        "rare2": [" phosphorescent", " submersible", " anglerfish", " trench"]}

for name, words in sets.items():
    kws = [tok(w).input_ids for w in words]
    sat = lambda ids: sum(contains(ids[0].tolist(), kw) for kw in kws)
    ids, *_ = bolt.generate(past, last, kws, L)             # deterministic: run once
    print(f"\n{name} BOLT: {sat(ids)}/4, ppl {perplexity(lm, prefix, ids):.1f}")
    for single in [False, True]:
        res = []
        for seed in [0, 1, 2]:
            torch.manual_seed(seed)                          # same start for every run
            hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter(kws, L, tau_horizon=2.0, single=single))
            ids, esc = hyb.generate(past, last, kws, L)
            res.append((sat(ids), perplexity(lm, prefix, ids)))
            print(f"  single={single} seed={seed}: {res[-1][0]}/4, ppl {res[-1][1]:.1f}, esc {esc}")
        print(f"  -> single={single} mean: {sum(r[0] for r in res)/3:.2f}/4, ppl {sum(r[1] for r in res)/3:.1f}")