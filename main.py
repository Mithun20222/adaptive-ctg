import time, json, torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from AdaptiveRouter import AdaptiveRouter
from HybridEngine import HybridEngine
from metrics import contains, perplexity

L = 25
PROMPTS = ["In the deep dark ocean,", "The old castle stood on a hill,", "Last night at the market,"]
SETS = {
    "common": [" submarine", " treasure", " octopus", " lighthouse"],
    "rare1": [" bioluminescent", " hydrothermal", " leviathan", " periscope"],
    "rare2": [" phosphorescent", " submersible", " anglerfish", " trench"],
}
# control: common keywords on the first prompt only; rare sets on every prompt
CASES = [(PROMPTS[0], "common")] + [(p, s) for p in PROMPTS for s in ("rare1", "rare2")]

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
bolt = BoltEngine(lm, iters=5, lr=30)
cold = ColdSpanEngine(lm, iters=150)
rows = []

for prompt, sname in CASES:
    prefix = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    with torch.no_grad():
        logits, past = lm.discrete_forward(prefix)
    last = logits[:, -1]
    kws = [tok(w).input_ids for w in SETS[sname]]

    router = AdaptiveRouter(kws, L, tau_horizon=2.0, single=True)
    hybrid = HybridEngine(lm, bolt, cold, router)
    runs = {
        "BOLT": lambda: bolt.generate(past, last, kws, L)[0],
        "COLD": lambda: cold.optimize(past, last, kws, k=L)[0],
        "Hybrid": lambda: hybrid.generate(past, last, kws, L)[0],
    }
    for name, fn in runs.items():
        t0 = time.time(); ids = fn(); dt = time.time() - t0
        met = sum(contains(ids[0].tolist(), kw) for kw in kws)
        r = dict(prompt=prompt, set=sname, method=name, met=met,
                 ppl=perplexity(lm, prefix, ids), tps=L / dt, text=tok.decode(ids[0]))
        rows.append(r)
        print(f"{prompt[:22]:<22} {sname:<7} {name:<7} {met}/4 | ppl {r['ppl']:7.1f} | {r['tps']:5.2f} tok/s")

json.dump(rows, open("results.json", "w"), indent=1)

print("\n=== SUMMARY (mean over cases) ===")
for group, names in [("common", ["common"]), ("rare", ["rare1", "rare2"])]:
    for m in ["BOLT", "COLD", "Hybrid"]:
        rs = [r for r in rows if r["method"] == m and r["set"] in names]
        n = len(rs)
        print(f"{group:<7} {m:<7} n={n} | met {sum(r['met'] for r in rs)/n:.2f}/4 | "
              f"all-4 {sum(r['met'] == 4 for r in rs)}/{n} | ppl {sum(r['ppl'] for r in rs)/n:6.1f} | "
              f"{sum(r['tps'] for r in rs)/n:5.2f} tok/s")