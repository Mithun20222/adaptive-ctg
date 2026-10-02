import time, torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
kw, L = tok(" submarine").input_ids, 25

with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
n = past.get_seq_length()

for iters in [0, 5]:                              # 0 = no tuning = plain greedy decoding
    engine = BoltEngine(lm, iters=iters)
    t0 = time.time()
    ids, new_past, nxt, hist = engine.generate(past, logits[:, -1], kw, L)
    dt = time.time() - t0
    for it, e_lm, e_kw in hist:
        print(f"  iter {it}  E_lm {e_lm:.3f}  E_kw {e_kw:.3f}")
    print(f"iters={iters}: {tok.decode(ids[0])!r}")
    print(f"  {L / dt:.2f} tokens/sec | keyword present: {all(t in ids[0].tolist() for t in kw)}\n")

    assert ids.shape == (1, L)
    assert past.get_seq_length() == n             # prefix cache untouched
    assert new_past.get_seq_length() == n + L
print("structure tests passed")