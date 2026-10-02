import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from ColdSpanEngine import ColdSpanEngine
from EnergyFunctions import keyword_energy
from metrics import contains

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last = logits[:, -1]

words = [" submarine", " treasure", " octopus", " lighthouse"]
kws = [tok(w).input_ids for w in words]
L = 10

engine = ColdSpanEngine(lm, iters=150)
ids, _, _, hist, y = engine.optimize(past, last, kws, k=L)

# 1) per-keyword energy on the OPTIMIZED soft span (low = optimizer is happy with it)
probs = F.softmax(y / engine.temperature, dim=-1)
print("per-keyword energy (soft):")
for w, kw in zip(words, kws):
    print(f"  {w:<12} {keyword_energy(probs, kw).item():.3f}")

# 2) what the optimizer wants at each slot, with its confidence
print("\nper-slot optimizer argmax:")
for i in range(L):
    p = probs[0, i]
    top = p.topk(2)
    print(f"  slot {i}: {tok.decode(top.indices[0])!r} {top.values[0]:.2f} | "
          f"{tok.decode(top.indices[1])!r} {top.values[1]:.2f}")

# 3) discretization ablation on the SAME y
print("\ndiscretization ablation:")
for topk, y_cands in [(50, 3), (50, 10), (500, 3), (2000, 10)]:
    engine.topk, engine.y_cands = topk, y_cands
    d, _, _ = engine._discretize(y, past, last)
    got = sum(contains(d[0].tolist(), kw) for kw in kws)
    print(f"  topk={topk:<5} y_cands={y_cands:<3} satisfied {got}/4  {tok.decode(d[0])!r}")
