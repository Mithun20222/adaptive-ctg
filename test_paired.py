import time, json, torch
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
bolt, cold = BoltEngine(lm, iters=5, lr=30), ColdSpanEngine(lm, iters=150)

PROMPTS = ["In the deep dark ocean,", "The old castle stood on a hill,", "Last night at the market,",
           "The scientists gathered in the laboratory,", "Early in the morning, the village", 
           "After years of searching, they finally", "The storm rolled in quickly,",
           "At the edge of the forest,", "The old sailor looked at the horizon,", "During the long winter,"]
POOL = [" bioluminescent", " hydrothermal", " leviathan", " periscope",
        " phosphorescent", " submersible", " anglerfish", " trench"]
SETS = {"n6-fwd": POOL[:6], "n6-rev": POOL[::-1][:6], "n8": POOL}
rows = []

for prompt in PROMPTS:
    prefix = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    with torch.no_grad():
        logits, past = lm.discrete_forward(prefix)
    last = logits[:, -1]
    for sname, words in SETS.items():
        kws = [tok(w).input_ids for w in words]
        if sum(len(k) for k in kws) > L:
            continue                                         # cannot fit: skip
        ids_b, *_ = bolt.generate(past, last, kws, L)
        hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter(kws, L, tau_horizon=2.0, single=True))
        ids_h, esc = hyb.generate(past, last, kws, L)
        r = dict(prompt=prompt, set=sname, n=len(kws), esc=len(esc),
                 bolt_met=sum(contains(ids_b[0].tolist(), k) for k in kws),
                 hyb_met=sum(contains(ids_h[0].tolist(), k) for k in kws),
                 bolt_ppl=perplexity(lm, prefix, ids_b), hyb_ppl=perplexity(lm, prefix, ids_h))
        rows.append(r)
        print(f"{prompt[:26]:<26} {sname:<7} BOLT {r['bolt_met']}/{r['n']} ppl {r['bolt_ppl']:6.1f} | "
              f"Hybrid {r['hyb_met']}/{r['n']} ppl {r['hyb_ppl']:6.1f} | esc {r['esc']}")

json.dump(rows, open("paired_results.json", "w"), indent=1)

print("\n=== PAIRED SUMMARY ===")
for name in ["n6-fwd", "n6-rev", "n8", "ALL"]:
    rs = [r for r in rows if name == "ALL" or r["set"] == name]
    if not rs: continue
    d = [r["hyb_met"] - r["bolt_met"] for r in rs]
    esc = [r for r in rs if r["esc"] > 0]
    print(f"{name:<7} cases {len(rs):2d} | hybrid wins {sum(x > 0 for x in d)}, ties {sum(x == 0 for x in d)}, "
          f"losses {sum(x < 0 for x in d)} | mean keywords BOLT {sum(r['bolt_met'] for r in rs)/len(rs):.2f} "
          f"Hybrid {sum(r['hyb_met'] for r in rs)/len(rs):.2f} | mean ppl BOLT {sum(r['bolt_ppl'] for r in rs)/len(rs):.1f} "
          f"Hybrid {sum(r['hyb_ppl'] for r in rs)/len(rs):.1f} | escalated in {len(esc)}")