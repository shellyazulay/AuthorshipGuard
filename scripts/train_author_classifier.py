import json
import os
import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, pipeline)

# === Load and preprocess the dataset ===
dataset_path = "exp_main/data/author_style_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["text"] for entry in data]
labels = [entry["author"] for entry in data]

# Create label mapping
unique_labels = list(sorted(set(labels)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
numeric_labels = [label2id[label] for label in labels]

# Convert to HuggingFace Dataset
raw_dataset = Dataset.from_dict({
    "text": texts,
    "label": numeric_labels
})
raw_dataset = raw_dataset.train_test_split(test_size=0.2)

# === Tokenization ===
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./bert_output",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# === Train the model ===
trainer.train()

# === Evaluate the model ===
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids
print(classification_report(true_labels, preds, target_names=unique_labels))
