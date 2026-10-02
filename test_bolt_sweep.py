import math
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from metrics import contains, perplexity

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
prefix = tok("In the deep dark ocean,", return_tensors="pt").input_ids.to(lm.device)
with torch.no_grad():
    logits, past = lm.discrete_forward(prefix)

words = [" submarine", " treasure", " octopus", " lighthouse"]
kws = [tok(w).input_ids for w in words]


def contains(seq, kw):
    """True if kw appears as a contiguous run inside seq (both lists of token ids)."""
    m = len(kw)
    return any(seq[i:i + m] == kw for i in range(len(seq) - m + 1))


# unconstrained reference: greedy decoding, no tuning
for L in [25, 10]:
    ids, _, _, _ = BoltEngine(lm, iters=0).generate(past, logits[:, -1], kws, L)
    print(f"L={L:2d} greedy (no tuning)  satisfied {sum(contains(ids[0].tolist(), kw) for kw in kws)}/4  ppl {perplexity(lm, prefix, ids):7.1f}")

for L in [25, 10]:
    for iters, lr in [(5, 10), (5, 30), (5, 100)]:
        ids, _, _, hist = BoltEngine(lm, iters=iters, lr=lr).generate(past, logits[:, -1], kws, L)
        got = sum(contains(ids[0].tolist(), kw) for kw in kws)
        print(f"L={L:2d} iters={iters} lr={lr:<4} satisfied {got}/4  ppl {perplexity(lm, prefix, ids):7.1f}  {tok.decode(ids[0])[:60]!r}")