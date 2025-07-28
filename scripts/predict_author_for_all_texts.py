import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

# This script loads texts and runs author prediction model on them

# Load model and tokenizer
MODEL_NAME = "./exp_main/models/author_classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
trainer = Trainer(model=model)

# Path to the merged dataset (texts only, no labels)
MERGED_DATA_PATH = "exp_main/data/xsum_squad_merged_for_author_prediction.json"

with open(MERGED_DATA_PATH, "r") as f:
    examples = json.load(f)


texts = [ex["text"] for ex in examples if "text" in ex]

if len(texts) == 0:
    print("No texts found in the data file.")
    exit()

print("Loaded {len(texts)} texts for prediction.")

# Tokenize the texts
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Convert list of texts to HuggingFace Dataset
raw_dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = raw_dataset.map(tokenize_fn)

# Run prediction
predictions = trainer.predict(tokenized_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

# Get label names
label_names = model.config.id2label
print("Label names dict:", label_names)
print("Type of keys in label_names:", [type(k) for k in label_names.keys()])


# Count predictions per author
from collections import Counter
counts = Counter(predicted_labels)
print("Predicted label ids:", list(counts.keys()))

# Show predictions with percentages in a table-like format
total = sum(counts.values())
sorted_items = sorted(counts.items(), key=lambda x: -x[1])  # sort by count desc

print("Prediction summary by author:")
print(f"{'Author':<25}{'Count':<10}{'Percentage':<10}")
print("-" * 45)

for label_id, count in sorted_items:
    author = label_names.get(label_id, f"Unknown {label_id}")
    percentage = (count / total) * 100
    print(f"{author:<25}{count:<10}{percentage:>6.1f}%")

