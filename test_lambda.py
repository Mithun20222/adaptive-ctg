import time, torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from ColdSpanEngine import ColdSpanEngine
from metrics import suffix_nll

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
sfx = tok(" and the crew finally returned home.").input_ids
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last, L = logits[:, -1], 20

for lam, step in [(1, 1.0), (5, 1.0), (20, 1.0), (20, 5.0), (50, 5.0)]:
    cold = ColdSpanEngine(lm, iters=150, step_size=step)
    t0 = time.time()
    ids, _, _, hist, y = cold.optimize(past, last, [], L, suffix_ids=sfx, lambda_right=lam)
    print(f"lambda_right={lam:<3} step={step:<4} e_right {hist[0][3]:.2f}->{hist[-1][3]:.2f} | "
          f"e_lm {hist[0][1]:.2f}->{hist[-1][1]:.2f} | join NLL {suffix_nll(lm, prefix, ids, sfx):.2f} | {time.time()-t0:.0f}s")
    print(f"   {tok.decode(ids[0])!r}")