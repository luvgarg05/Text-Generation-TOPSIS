from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

model_name = "gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

dataset = load_dataset(
    "text",
    data_files={"train": "train.txt"}
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./finetuned-gpt2",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./finetuned-gpt2")
tokenizer.save_pretrained("./finetuned-gpt2")
