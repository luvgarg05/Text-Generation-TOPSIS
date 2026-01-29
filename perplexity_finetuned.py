import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model_path = "./finetuned-gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

text = "The future of artificial intelligence is"

encodings = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)

print("Perplexity:", round(perplexity.item(), 3))
