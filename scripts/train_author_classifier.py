import json
import joblib
import os
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Path to the dataset file
dataset_path = "exp_main/data/author_style_dataset.json"
output_model_path = "exp_main/models/author_classifier.joblib"

# Load the dataset from JSON file
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract texts and their corresponding author labels
texts = [entry["text"] for entry in data]
labels = [entry["author"] for entry in data]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Build a pipeline with TF-IDF vectorizer and logistic regression classifier
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
    LogisticRegression(max_iter=1000)
)

# Train the model using the training data
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model on the test set and display results
y_pred = model.predict(X_test)
print("\nTest set results:")
print(classification_report(y_test, y_pred))

# Ensure the output directory for models exists
os.makedirs("exp_main/models", exist_ok=True)

# Save the trained model to disk
joblib.dump(model, output_model_path)
print(f"\nModel saved to: {output_model_path}")
