import time, torch
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
for L in [25, 15, 10]:
    for m in [1, 2, 3, 4]:
        kws = [tok(w).input_ids for w in words[:m]]
        ids, _, _, _ = BoltEngine(lm, iters=5).generate(past, logits[:, -1], kws, L)
        got = [all(t in ids[0].tolist() for t in kw) for kw in kws]
        print(f"L={L:2d} keywords={m}  satisfied {sum(got)}/{m}  {tok.decode(ids[0])!r}")