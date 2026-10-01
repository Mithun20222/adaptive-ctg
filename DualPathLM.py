import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class DualPathLM(nn.Module):
    """Frozen GPT-2 with two entry points: token ids or soft embeddings."""

    def __init__(self, name="gpt2", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(name).to(self.device)
        self.model.eval()                      # no dropout
        for p in self.model.parameters():      # freeze weights
            p.requires_grad_(False)

    @property
    def embed_matrix(self):
        # (V, d) = (50257, 768) for gpt2
        return self.model.get_input_embeddings().weight

    def probs_to_embeds(self, probs):
        # probs: (B, k, V) @ (V, d) -> (B, k, d)
        return probs @ self.embed_matrix

    def discrete_forward(self, input_ids, past_key_values=None, attention_mask=None):
        # input_ids: (B, n) -> logits: (B, n, V), plus updated KV-cache
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.logits, out.past_key_values

    def soft_forward(self, soft_embeddings, past_key_values=None, attention_mask=None):
        # soft_embeddings: (B, k, d) -> logits: (B, k, V), plus updated KV-cache
        # attention_mask (if given) must have length past_len + k
        out = self.model(
            inputs_embeds=soft_embeddings,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.logits, out.past_key_values