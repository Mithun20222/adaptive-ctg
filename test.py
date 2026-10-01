from transformers import GPT2LMHeadModel
m = GPT2LMHeadModel.from_pretrained("gpt2")
print(sum(p.numel() for p in m.parameters()))