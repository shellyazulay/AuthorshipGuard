import json

xsum_path = "exp_main/data/xsum_gpt-neo-2.7B.raw_data.json"
squad_path = "exp_main/data/squad_gpt-neo-2.7B.raw_data.json"
output_path = "exp_main/data/xsum_squad_merged_for_author_prediction.json"

# Load the raw data files for XSUM and SQuAD
with open(xsum_path, "r", encoding="utf-8") as f:
    xsum_data = json.load(f)

with open(squad_path, "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# Extracts texts categories in each dataset
def extract_texts(data):
    texts = []
    for category in ["original", "sampled"]:
        texts.extend(data.get(category, []))
    return texts

# Get list of all texts from each dataset
xsum_texts = extract_texts(xsum_data)
squad_texts = extract_texts(squad_data)

# Merge everything into a single list for prediction
combined_texts = xsum_texts + squad_texts

# Create a list of dictionaries, each containing a single "text" field, and save it to a JSON file
# This merges the XSUM and SQuAD texts into a single output file for later prediction
combined_results = [{"text": t} for t in combined_texts]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_results, f, indent=2, ensure_ascii=False)

print(f"✅ Combined {len(xsum_texts)} XSUM and {len(squad_texts)} SQuAD texts into {output_path}")
