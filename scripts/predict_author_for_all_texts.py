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
MERGED_DATA_PATH = "exp_main/data/authors_texts_datasets.json"

with open(MERGED_DATA_PATH, "r") as f:
    examples = json.load(f)


texts = [ex["text"] for ex in examples if "text" in ex]

if len(texts) == 0:
    print("No texts found in the data file.")
    exit()

# Extract true author labels from the dataset for accuracy calculation
true_authors = [ex["author"] for ex in examples if "author" in ex]

# Sanity check to make sure number of authors equals number of texts
if len(true_authors) != len(texts):
    print("Warning: Number of authors does not match number of texts!")
    exit()


print(f"Loaded {len(texts)} texts for prediction.")

# Tokenize the texts
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Convert list of texts to HuggingFace Dataset
raw_dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = raw_dataset.map(tokenize_fn)

# Run prediction
predictions = trainer.predict(tokenized_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

# Get label names from the model config
label_names = model.config.id2label

# Create a reverse mapping from author names to label IDs
label_name_to_id = {v: k for k, v in label_names.items()}

# Convert true author names to corresponding label IDs, mark unknown authors as -1
true_labels = []
for author in true_authors:
    if author in label_name_to_id:
        true_labels.append(label_name_to_id[author])
    else:
        print(f"Warning: author '{author}' not found in label mapping!")
        true_labels.append(-1)

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



