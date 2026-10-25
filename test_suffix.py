import time, torch
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from BoltEngine import BoltEngine
from ColdSpanEngine import ColdSpanEngine
from AdaptiveRouter import AdaptiveRouter
from HybridEngine import HybridEngine
from metrics import perplexity, suffix_nll

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
L, BRIDGE = 20, 6
cold = ColdSpanEngine(lm, iters=150)
bolt = BoltEngine(lm, iters=0)                      # no keywords: BOLT = greedy decoding

CASES = [("In the deep dark ocean,", " and the crew finally returned home."),
         ("The old castle stood on a hill,", " and nobody ever spoke of it again."),
         ("Last night at the market,", " so we decided to leave early.")]

for prompt, suffix in CASES:
    prefix = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    sfx = tok(suffix).input_ids
    with torch.no_grad():
        logits, past = lm.discrete_forward(prefix)
    last = logits[:, -1]
    hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter([], L))
    print(f"\n{prompt} ... {suffix}")

    def show(name, ids, dt, hist=None):
        sfx_t = torch.as_tensor(sfx, device=ids.device).unsqueeze(0)
        ppl = perplexity(lm, prefix, torch.cat([ids, sfx_t], dim=1))
        extra = f" | e_right {hist[0][3]:.2f}->{hist[-1][3]:.2f}" if hist else ""
        print(f"  {name:<7} join NLL {suffix_nll(lm, prefix, ids, sfx):5.2f} | ppl(all) {ppl:6.1f} | {dt:5.1f}s{extra}\n"
              f"          {tok.decode(ids[0])!r}")

    t0 = time.time(); ids, *_ = bolt.generate(past, last, [], L); show("BOLT", ids, time.time() - t0)
    t0 = time.time(); ids, _, _, hist, _ = cold.optimize(past, last, [], L, suffix_ids=sfx); show("COLD", ids, time.time() - t0, hist)
    t0 = time.time(); ids, hist = hyb.generate_infill(past, last, sfx, BRIDGE); show("Hybrid", ids, time.time() - t0, hist)
    t0 = time.time()
    out = lm.model.generate(prefix, max_new_tokens=L, do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    ids = out[:, prefix.shape[1]:]
    show("Greedy+rp", ids, time.time() - t0)