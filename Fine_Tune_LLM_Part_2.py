# -*- coding: utf-8 -*-
pip install datasets pandas torch transformers[torch] python-dotenv peft

import os
from dotenv import load_dotenv
load_dotenv()  # Load the .env file
hf_token = os.getenv("HF_TOKEN")

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login(token=hf_token)
model_name = "distilgpt2"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_name)
model

text = "ஒரு நாள் "
inputs = tokenizer(text, return_tensors="pt")
# Generate story
output = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

from datasets import load_dataset
# Load English-to-Colloquial Tamil dataset
raw_data = load_dataset("tniranjan/aitamilnadu_tamil_stories_no_instruct", split="train[:1000]")
data = raw_data.train_test_split(train_size=0.95)
data

tokenizer.pad_token = tokenizer.eos_token
def preprocess_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=200)
tokenized_dataset = data.map(
preprocess_batch,
batched=True,
batch_size =4,
remove_columns=data["train"].column_names,)
# Print dataset details
print(tokenized_dataset)

#Data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
data_collator

from peft import get_peft_model, LoraConfig, TaskType
# Define LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices (adjust for speed/memory tradeoff)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for stability
    bias="none",  # No extra bias parameters
    task_type=TaskType.CAUSAL_LM  # Since we're fine-tuning GPT-style models
)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)
# Print trainable parameters
model.print_trainable_parameters()

model.train() #This ensures layers like Dropout & BatchNorm are active during training.
from torch.optim import AdamW  # Correct import
# Define new optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
output_dir="./output",
evaluation_strategy="epoch",
#save_strategy="epoch",
save_steps=500,
learning_rate=1e-5,
weight_decay=0.01,
num_train_epochs=3,
per_device_train_batch_size=2,    # Batch size (adjust for GPU memory)
per_device_eval_batch_size=2,
logging_steps=50,
logging_dir="./logs",
resume_from_checkpoint=True
)
trainer = Trainer(
    model = model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    optimizers=(optimizer, None),
    data_collator=data_collator
)
trainer.train()

model = AutoModelForCausalLM.from_pretrained ("/content/drive/My Drive/fine_tuned_distilgpt2_Tamil")
model

text = "ஒரு நாள்"
inputs = tokenizer(text, return_tensors="pt")
# Generate story
output = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelForCausalLM
trainer.save_model("/content/drive/My Drive/fine_tuned_distilgpt2_Tamil")
tokenizer.save_pretrained("/content/drive/My Drive/fine_tuned_distilgpt2_Tamil")

model = AutoModelForCausalLM.from_pretrained ("/content/drive/My Drive/fine_tuned_distilgpt2_200")
model

text = "ஒரு நாள் "
inputs = tokenizer(text, return_tensors="pt")
# Generate story
output = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

model.train() #This ensures layers like Dropout & BatchNorm are active during training.
from torch.optim import AdamW  # Correct import
# Define new optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
output_dir="./output",
evaluation_strategy="epoch",
#save_strategy="epoch",
save_steps=500,
learning_rate=1e-5,
weight_decay=0.01,
num_train_epochs=3,
per_device_train_batch_size=2,    # Batch size (adjust for GPU memory)
per_device_eval_batch_size=2,
logging_steps=50,
logging_dir="./logs",
resume_from_checkpoint=True
)
trainer = Trainer(
    model = model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    optimizers=(optimizer, None),
    data_collator=data_collator
)
trainer.train()
