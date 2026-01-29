import time
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

prompt = "The future of artificial intelligence is"

start = time.time()
output = generator(prompt, max_length=50)
end = time.time()

print("Generated Text:\n")
print(output[0]["generated_text"])
print("\nTime Taken:", round(end - start, 3), "seconds")
