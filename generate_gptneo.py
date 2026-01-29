import time
from transformers import pipeline

# Load GPT-Neo 125M
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M"
)

prompt = "The future of artificial intelligence is"

start = time.time()
output = generator(prompt, max_length=50, num_return_sequences=1)
end = time.time()

print("Generated Text:\n")
print(output[0]["generated_text"])
print("\nTime Taken:", round(end - start, 3), "seconds")
