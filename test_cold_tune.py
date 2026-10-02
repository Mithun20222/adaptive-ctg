import time, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from ColdSpanEngine import ColdSpanEngine
from EnergyFunctions import keyword_energy
from metrics import contains, perplexity

tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last = logits[:, -1]

words = [" submarine", " treasure", " octopus", " lighthouse"]
kws = [tok(w).input_ids for w in words]
L = 10

for lam in [1, 3, 10]:
    torch.manual_seed(0)
    eng = ColdSpanEngine(lm, iters=150, lambda_hard=lam)
    t0 = time.time()
    ids, _, _, hist, y = eng.optimize(past, last, kws, k=L)
    dt = time.time() - t0
    probs = F.softmax(y / eng.temperature, dim=-1)
    e = [round(keyword_energy(probs, kw).item(), 2) for kw in kws]
    got = sum(contains(ids[0].tolist(), kw) for kw in kws)
    print(f"lambda_hard={lam:<3} satisfied {got}/4 | ppl {perplexity(lm, prefix, ids):7.1f} | "
          f"{dt:4.0f}s | kw energies {e}\n   {tok.decode(ids[0])!r}")