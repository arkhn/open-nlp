import json
from pathlib import Path


def filter_entries_by_keyword_count(input_file, output_file, min_keywords=50, max_keywords=100):
    # Process each entry in the JSONL file
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            filtered_data = {}

            for key, entries in data.items():
                filtered_entries = [
                    entry
                    for entry in entries
                    if min_keywords <= len(entry["keywords"].split(",")) <= max_keywords
                ]
                if filtered_entries:
                    filtered_data[key] = filtered_entries

            if filtered_data:
                json.dump(filtered_data, outfile)
                outfile.write("\n")


# Define input and output file paths
Path("post-processed/").mkdir(parents=True, exist_ok=True)
input_file_path = "keywords_extraction/mimic-style-transfer.jsonl"
output_file_path = "post-processed/filtered_entries.jsonl"

# Call the function with file paths
filter_entries_by_keyword_count(input_file_path, output_file_path)
