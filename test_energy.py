import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM
from EnergyFunctions import fluency_energy, keyword_energy

tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
V = lm.embed_matrix.shape[0]

ids = tok("In the deep dark ocean, a submarine surfaced slowly", return_tensors="pt").input_ids.to(lm.device)
P = 6                                    # split: prefix = ids[:, :P], span = ids[:, P:]
prefix, span = ids[:, :P], ids[:, P:]

# Test 1: fluency energy on a one-hot span == ordinary NLL
with torch.no_grad():
    pre_logits, past = lm.discrete_forward(prefix)
    onehot = F.one_hot(span, V).float()
    soft_logits, _ = lm.soft_forward(lm.probs_to_embeds(onehot), past_key_values=past)
    pred = torch.cat([pre_logits[:, -1:], soft_logits[:, :-1]], dim=1)   # shift by one
    e = fluency_energy(onehot, pred)

    full, _ = lm.discrete_forward(ids)
    ref = F.cross_entropy(full[0, P - 1:-1], span[0])
print("energy:", e.item(), "reference:", ref.item())
assert torch.allclose(e, ref, atol=1e-3)

# Test 2: keyword energy is low when keyword is present, high when not
kw = tok(" submarine").input_ids
k = 8
has = F.one_hot(torch.tensor([[5] * 2 + kw + [5] * (k - 2 - len(kw))]), V).float()
uniform = torch.full((1, k, V), 1.0 / V)
e_has, e_uni = keyword_energy(has, kw), keyword_energy(uniform, kw)
print("with keyword:", e_has.item(), "uniform:", e_uni.item())
assert e_has < e_uni

# Test 3: gradients flow to span_probs
p = F.softmax(torch.randn(1, k, V), dim=-1).requires_grad_(True)
keyword_energy(p, kw).backward()
assert p.grad is not None and p.grad.abs().sum() > 0
print("all tests passed")