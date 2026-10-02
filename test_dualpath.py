import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from DualPathLM import DualPathLM

torch.manual_seed(0)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = DualPathLM()
ids = tok("In the deep dark ocean, a submarine", return_tensors="pt").input_ids.to(lm.device)
k = 3
V = lm.embed_matrix.shape[0]

# Test 1: soft path with one-hot input == discrete path
with torch.no_grad():
    full_logits, _ = lm.discrete_forward(ids)                    # whole sequence
    _, past = lm.discrete_forward(ids[:, :-k])                   # prefix cache only
    onehot = F.one_hot(ids[:, -k:], num_classes=V).float()       # (1, k, V)
    soft_logits, _ = lm.soft_forward(lm.probs_to_embeds(onehot), past_key_values=past)

diff = (full_logits[:, -k:] - soft_logits).abs().max().item()
print("max diff:", diff)
assert torch.allclose(full_logits[:, -k:], soft_logits, atol=1e-4)

# Test 2: gradients reach the soft input, not the weights
with torch.no_grad():
    _, past = lm.discrete_forward(ids[:, :-k])
probs = F.softmax(torch.randn(1, k, V, device=lm.device), dim=-1).requires_grad_(True)
logits, _ = lm.soft_forward(lm.probs_to_embeds(probs), past_key_values=past)
logits.sum().backward()
assert probs.grad is not None
assert all(p.grad is None for p in lm.model.parameters())
with torch.no_grad():
    y = engine._init_logits(past, logits[:, -1], k)   # just to inspect; replace with the optimized y if you return it

print("all tests passed")