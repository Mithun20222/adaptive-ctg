import torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)

words = [" submarine", " treasure", " octopus", " lighthouse"]
kws = [tok(w).input_ids for w in words]

for L in [25, 10]:
    for iters, lr in [(5, 3), (5, 10), (5, 30), (20, 3), (20, 10)]:
        ids, _, _, hist = BoltEngine(lm, iters=iters, lr=lr).generate(past, logits[:, -1], kws, L)
        got = sum(all(t in ids[0].tolist() for t in kw) for kw in kws)
        print(f"L={L:2d} iters={iters:2d} lr={lr:<3} satisfied {got}/4  E_kw end {hist[-1][2]:.2f}  {tok.decode(ids[0])[:70]!r}")