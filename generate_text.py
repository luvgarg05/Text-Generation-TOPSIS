import time
from transformers import pipeline

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"

start = time.time()
output = generator(prompt, max_length=50, num_return_sequences=1)
end = time.time()

generated_text = output[0]["generated_text"]

print("Generated Text:\n", generated_text)
print("\nTime Taken:", round(end - start, 3), "seconds")

