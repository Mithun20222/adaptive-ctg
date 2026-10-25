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
bolt = BoltEngine(lm, iters=0)
CASES = [("In the deep dark ocean,", " and the crew finally returned home."),
         ("The old castle stood on a hill,", " and nobody ever spoke of it again."),
         ("Last night at the market,", " so we decided to leave early.")]

for prompt, suffix in CASES:
    prefix = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    sfx = tok(suffix).input_ids
    sfx_t = torch.as_tensor(sfx, device=lm.device).unsqueeze(0)
    with torch.no_grad():
        logits, past = lm.discrete_forward(prefix)
    last = logits[:, -1]
    print(f"\n{prompt} ... {suffix}")

    def show(name, ids):
        print(f"  {name:<18} join NLL {suffix_nll(lm, prefix, ids, sfx):5.2f} | ppl(middle) {perplexity(lm, prefix, ids):6.1f} "
              f"| ppl(all) {perplexity(lm, prefix, torch.cat([ids, sfx_t], 1)):6.1f}\n      {tok.decode(ids[0])!r}")

    show("BOLT greedy", bolt.generate(past, last, [], L)[0])
    out = lm.model.generate(prefix, max_new_tokens=L, do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    show("Greedy+rep.penalty", out[:, prefix.shape[1]:])
    for lam in [5, 20, 50]:
        cold = ColdSpanEngine(lm, iters=150, step_size=5.0)
        hyb = HybridEngine(lm, bolt, cold, AdaptiveRouter([], L))
        ids, _ = hyb.generate_infill(past, last, sfx, BRIDGE, lambda_right=lam)
        show(f"Hybrid lam={lam}", ids)