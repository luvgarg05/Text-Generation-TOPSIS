import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Model name
model_name = "distilgpt2"

# Load tokenizer & model from local cache only (no internet calls)
tokenizer = GPT2TokenizerFast.from_pretrained(
    model_name,
    local_files_only=True
)

model = GPT2LMHeadModel.from_pretrained(
    model_name,
    local_files_only=True
)

model.eval()

# Text prompt
text = "The future of artificial intelligence is"

# Tokenize
encodings = tokenizer(text, return_tensors="pt")

# Compute perplexity
with torch.no_grad():
    outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)

print("Perplexity:", round(perplexity.item(), 3))
