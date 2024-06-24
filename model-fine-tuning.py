import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

tokenizer = T5Tokenizer.from_pretrained("t5-small")
def tokenize_function(examples):
    return tokenizer(examples["cleaned_article"], examples["cleaned_summary"], truncation=True, padding=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./t5-small-trained",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_first_step=True,
)

model = T5ForConditionalGeneration.from_pretrained("t5-small")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)
    
trainer.train()
model.save_pretrained("./t5-small-trained")
