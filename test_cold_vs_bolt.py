import time, torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from metrics import contains, perplexity

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last = logits[:, -1]

words = [" submarine", " treasure", " octopus", " lighthouse"]
kws = [tok(w).input_ids for w in words]
print("keyword token lengths:", [len(k) for k in kws])


def report(name, ids, dt, L):
    seq = ids[0].tolist()
    got = sum(contains(seq, kw) for kw in kws)
    print(f"{name:<22} satisfied {got}/4 | ppl {perplexity(lm, prefix, ids):8.1f} | "
          f"{L / dt:6.2f} tok/s | {dt:6.1f}s | {tok.decode(ids[0])[:70]!r}")


for L in [10]:                      # run L=10 first; add 25 once this works (takes several minutes)
    print(f"\n=== L={L} ===")

    t0 = time.time()
    ids, *_ = BoltEngine(lm, iters=5, lr=30).generate(past, last, kws, L)
    report("BOLT (5 it, lr=30)", ids, time.time() - t0, L)

    for iters in [150, 300]:
        torch.manual_seed(0)
        t0 = time.time()
        ids, *_ = ColdSpanEngine(lm, iters=iters).optimize(past, last, kws, k=L)
        report(f"COLD ({iters} it)", ids, time.time() - t0, L)