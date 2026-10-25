import time, torch
from collections import defaultdict
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from AdaptiveRouter import AdaptiveRouter
from HybridEngine import HybridEngine
from metrics import contains, perplexity

RUN_COLD = False
torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
cold = ColdSpanEngine(lm, iters=150)

PROMPTS = ["In the deep dark ocean,", "The old castle stood on a hill,"]
POOL = [" bioluminescent", " hydrothermal", " leviathan", " periscope",
        " phosphorescent", " submersible", " anglerfish", " trench"]
KW_ALL = [tok(w).input_ids for w in POOL]
print("token lengths:", [len(k) for k in KW_ALL])

agg = defaultdict(list)       # (n_kw, L, lr, method) -> [(met_fraction, ppl, esc)]

for L in [25, 30]:
    for n in [4, 6, 8]:
        kws = KW_ALL[:n]
        cap = sum(len(k) for k in kws)
        if cap > L:                                   # cannot all fit: skip, not a fair test
            print(f"skip n={n} L={L}: keywords need {cap} tokens")
            continue
        for lr in [30, 60]:
            bolt = BoltEngine(lm, iters=5, lr=lr)
            for prompt in PROMPTS:
                prefix = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
                with torch.no_grad():
                    logits, past = lm.discrete_forward(prefix)
                last = logits[:, -1]
                hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter(kws, L, tau_horizon=2.0, single=True))
                runs = {"BOLT": lambda: (bolt.generate(past, last, kws, L)[0], 0),
                        "Hybrid": lambda: (lambda r: (r[0], len(r[1])))(hyb.generate(past, last, kws, L))}
                if RUN_COLD:
                    runs["COLD"] = lambda: (cold.optimize(past, last, kws, k=L)[0], 0)
                for name, fn in runs.items():
                    t0 = time.time(); ids, esc = fn(); dt = time.time() - t0
                    met = sum(contains(ids[0].tolist(), kw) for kw in kws)
                    ppl = perplexity(lm, prefix, ids)
                    agg[(n, L, lr, name)].append((met / n, ppl, esc))
                    print(f"n={n} L={L} lr={lr:<3} {prompt[:12]:<12} {name:<7} {met}/{n} | ppl {ppl:7.1f} | esc {esc} | {dt:5.1f}s")

print("\n=== SUMMARY (mean over prompts) ===")
for key in sorted(agg):
    v = agg[key]; m = len(v)
    print(f"n={key[0]} L={key[1]} lr={key[2]:<3} {key[3]:<7} met {100*sum(x[0] for x in v)/m:5.1f}% | "
          f"ppl {sum(x[1] for x in v)/m:7.1f} | escalations {sum(x[2] for x in v)}")