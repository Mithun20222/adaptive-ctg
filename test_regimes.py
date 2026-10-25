import time, torch
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
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last = logits[:, -1]

common = [" submarine", " treasure", " octopus", " lighthouse"]
rare = [" bioluminescent", " hydrothermal", " leviathan", " periscope"]
bolt = BoltEngine(lm, iters=5, lr=30)
cold = ColdSpanEngine(lm, iters=150)


def run(name, kws, L, taus):
    print(f"\n=== {name}, L={L}, token lengths {[len(k) for k in kws]} ===")

    def report(tag, ids, dt, extra=""):
        got = sum(contains(ids[0].tolist(), kw) for kw in kws)
        print(f"{tag:<16} {got}/4 | ppl {perplexity(lm, prefix, ids):7.1f} | {dt:5.1f}s {extra}")

    t0 = time.time(); ids, *_ = bolt.generate(past, last, kws, L); report("BOLT", ids, time.time() - t0)
    t0 = time.time(); ids, *_ = cold.optimize(past, last, kws, k=L); report("COLD (full)", ids, time.time() - t0)
    for tau in taus:
        hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter(kws, L, tau_horizon=tau))
        t0 = time.time(); ids, esc = hyb.generate(past, last, kws, L)
        report(f"HYBRID tau={tau}", ids, time.time() - t0, f"| esc {esc}")


kw_rare = [tok(w).input_ids for w in rare]
rare2 = [" phosphorescent", " submersible", " anglerfish", " trench"]
kw_rare2 = [tok(w).input_ids for w in rare2]

run("rare set 1", kw_rare, 25, [5.0, 3.0, 2.0])
run("rare set 2", kw_rare2, 25, [5.0, 3.0, 2.0])