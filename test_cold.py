import time, torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from ColdSpanEngine import ColdSpanEngine

torch.manual_seed(0)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
kw = tok(" submarine").input_ids
k = len(kw) + 3

with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)
last_logits = logits[:, -1]
n = past.get_seq_length()

engine = ColdSpanEngine(lm)                      # default: topk=50, y_cands=3
t0 = time.time()
span, new_past, nxt, hist, y = engine.optimize(past, last_logits, kw, k)
dt = time.time() - t0

for it, e_lm, e_kw in hist[::50]:
    print(f"iter {it:3d}  E_lm {e_lm:.3f}  E_kw {e_kw:.3f}")
print(f"\nspan: {tok.decode(span[0])!r} | {k / dt:.2f} tokens/sec")
print("keyword present:", all(t in span[0].tolist() for t in kw))

# What did the optimizer want at each position (before discretization)?
print("\noptimizer's argmax per position:")
for i in range(k):
    print(f"  pos {i}: {tok.decode(y[0, i].argmax())!r}")

# Ablation: re-discretize the SAME optimized y with different settings (cheap)
print("\ndiscretization ablation:")
for topk, y_cands in [(50, 0), (50, 3), (500, 0)]:
    engine.topk, engine.y_cands = topk, y_cands
    ids, _, _ = engine._discretize(y, past, last_logits)
    ok = all(t in ids[0].tolist() for t in kw)
    print(f"  topk={topk:<4} y_cands={y_cands}  keyword={ok}  {tok.decode(ids[0])!r}")

assert past.get_seq_length() == n            # prefix cache untouched
assert new_past.get_seq_length() == n + k    # committed cache = prefix + span
assert hist[-1][2] < hist[0][2]              # keyword energy went down
print("\nstructure tests passed")