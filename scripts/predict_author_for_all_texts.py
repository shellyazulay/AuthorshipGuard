import json
import joblib
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = "exp_main/models/author_classifier.joblib"
XSUM_DATA_PATH = "exp_main/data/xsum_gpt-neo-2.7B.raw_datas.json"
SQUAD_DATA_PATH = "exp_main/data/squad_gpt-neo-2.7B.raw_datas.json" # Assuming similar structure
# No need for AUTHOR_STYLE_DATASET_PATH if we're not explicitly comparing text content.

# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Please ensure the model file exists.")
    exit()

# --- Initialize metrics accumulators ---
# For precision: predicted_author -> {'correct': count, 'incorrect': count}
combined_author_prediction_counts = defaultdict(lambda: defaultdict(int))
# For recall: true_author -> total_occurrences_of_this_author
combined_author_true_occurrences = defaultdict(int)

total_overall_samples = 0
total_overall_correct_predictions = 0

# --- Helper function to load and process data ---
def process_file(file_path, model, prediction_counts, true_occurrences):
    global total_overall_samples, total_overall_correct_predictions

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Skipping.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        return

    if not raw_data:
        print(f"Warning: {file_path} is empty.")
        return

    for entry in raw_data:
        # We process both 'sampled' and 'original' texts
        texts_to_process = []
        true_author = entry.get("original_author") # Crucial for evaluation

        if true_author is None:
            # If original_author is missing, we can't evaluate metrics for this entry.
            # You might want to handle this differently, e.g., skip the entry or raise an error.
            continue

        if "sampled" in entry and entry["sampled"]:
            texts_to_process.append(entry["sampled"])
        if "original" in entry and entry["original"]:
            texts_to_process.append(entry["original"])

        for text in texts_to_process:
            if not text.strip(): # Skip empty strings
                continue

            total_overall_samples += 1
            predicted_author = model.predict([text])[0]

            # Update true occurrences for recall calculation
            true_occurrences[true_author] += 1

            # Update prediction counts for precision calculation
            if predicted_author == true_author:
                prediction_counts[predicted_author]["correct"] += 1
                total_overall_correct_predictions += 1
            else:
                prediction_counts[predicted_author]["incorrect"] += 1


# --- Process both XSUM and SQUAD data ---
print(f"Processing XSUM data from {XSUM_DATA_PATH}...")
process_file(XSUM_DATA_PATH, model, combined_author_prediction_counts, combined_author_true_occurrences)

print(f"Processing SQUAD data from {SQUAD_DATA_PATH}...")
process_file(SQUAD_DATA_PATH, model, combined_author_prediction_counts, combined_author_true_occurrences)

# --- Generate Classification Report Table ---
print("\n" + "="*50)
print("              Author Classification Report              ")
print("="*50)

# Collect all unique authors involved in predictions or ground truth
all_authors = sorted(list(set(combined_author_prediction_counts.keys()).union(set(combined_author_true_occurrences.keys()))))

header = "{:<20} {:>10} {:>10} {:>10} {:>10}".format(" ", "precision", "recall", "f1-score", "support")
print(header)
print("-" * len(header))

macro_precision_sum = 0
macro_recall_sum = 0
macro_f1_sum = 0
valid_authors_for_macro = 0 # Count authors with actual data for macro avg

weighted_precision_sum = 0
sweighted_recall_sum = 0
weighted_f1_sum = 0
total_support = sum(combined_author_true_occurrences.values())

for author in all_authors:
    true_positives = combined_author_prediction_counts[author]["correct"]
    false_positives = combined_author_prediction_counts[author]["incorrect"]
    # False negatives are true instances of this author that were NOT predicted as this author.
    # This is the 'support' minus true positives.
    false_negatives = combined_author_true_occurrences[author] - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    support = combined_author_true_occurrences[author]

    print("{:<20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format(author, precision, recall, f1_score, support))

    if support > 0: # Only include authors with actual occurrences for macro avg
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1_score
        valid_authors_for_macro += 1

    weighted_precision_sum += precision * support
    weighted_recall_sum += recall * support
    weighted_f1_sum += f1_score * support

print("-" * len(header))

overall_accuracy = total_overall_correct_predictions / total_overall_samples if total_overall_samples > 0 else 0.0

macro_avg_precision = macro_precision_sum / valid_authors_for_macro if valid_authors_for_macro > 0 else 0.0
macro_avg_recall = macro_recall_sum / valid_authors_for_macro if valid_authors_for_macro > 0 else 0.0
macro_avg_f1 = macro_f1_sum / valid_authors_for_macro if valid_authors_for_macro > 0 else 0.0

weighted_avg_precision = weighted_precision_sum / total_support if total_support > 0 else 0.0
weighted_avg_recall = weighted_recall_sum / total_support if total_support > 0 else 0.0
weighted_avg_f1 = weighted_f1_sum / total_support if total_support > 0 else 0.0


print("{:<20} {:>10} {:>10} {:>10.2f} {:>10}".format("accuracy", "", "", overall_accuracy, total_overall_samples))
print("{:<20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format("macro avg", macro_avg_precision, macro_avg_recall, macro_avg_f1, total_overall_samples))
print("{:<20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format("weighted avg", weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, total_overall_samples))
print("="*50)