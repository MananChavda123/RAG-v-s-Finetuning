from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

from datasets import load_dataset

dataset = load_dataset("json", data_files="../data/train1.jsonl", split="train")


# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize)

# Data collator for language modeling
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
args = TrainingArguments(
    output_dir="../models/distilgpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("../models/distilgpt2")
tokenizer.save_pretrained("../models/distilgpt2")
