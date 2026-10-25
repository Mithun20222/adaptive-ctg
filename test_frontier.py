import torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from AdaptiveRouter import AdaptiveRouter
from HybridEngine import HybridEngine
from metrics import contains, perplexity

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
L = 25
cold = ColdSpanEngine(lm, iters=150)
PROMPTS = ["In the deep dark ocean,", "The old castle stood on a hill,", "Last night at the market,",
           "The scientists gathered in the laboratory,", "Early in the morning, the village",
           "After years of searching, they finally", "The storm rolled in quickly,",
           "At the edge of the forest,", "The old sailor looked at the horizon,", "During the long winter,"]
POOL = [" bioluminescent", " hydrothermal", " leviathan", " periscope",
        " phosphorescent", " submersible", " anglerfish", " trench"]
kws = [tok(w).input_ids for w in POOL]                       # n = 8
res = {}                                                     # method -> [(met, ppl)]

for prompt in PROMPTS:
    prefix = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    with torch.no_grad():
        logits, past = lm.discrete_forward(prefix)
    last = logits[:, -1]
    for lr in [35, 45, 55]:
        bolt = BoltEngine(lm, iters=5, lr=lr)
        ids, *_ = bolt.generate(past, last, kws, L)
        res.setdefault(f"BOLT lr={lr}", []).append((sum(contains(ids[0].tolist(), k) for k in kws), perplexity(lm, prefix, ids)))
    for lr in [30, 45]:
        bolt = BoltEngine(lm, iters=5, lr=lr)
        ids, _ = HybridEngine(lm, bolt, cold, AdaptiveRouter(kws, L, tau_horizon=2.0, single=True)).generate(past, last, kws, L)
        res.setdefault(f"Hybrid (BOLT lr={lr})", []).append((sum(contains(ids[0].tolist(), k) for k in kws), perplexity(lm, prefix, ids)))

print("\n=== FRONTIER (n=8, L=25, 10 prompts) ===")
for name, v in res.items():
    ppls = sorted(p for _, p in v)
    print(f"{name:<20} mean keywords {sum(m for m, _ in v)/len(v):.2f} | mean ppl {sum(p for _, p in v)/len(v):7.1f} | median ppl {ppls[len(ppls)//2]:7.1f}")