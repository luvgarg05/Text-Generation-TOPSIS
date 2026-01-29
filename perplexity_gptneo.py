import torch
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast

model_name = "EleutherAI/gpt-neo-125M"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)
model.eval()

text = "The future of artificial intelligence is"

encodings = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)

print("Perplexity:", round(perplexity.item(), 3))
