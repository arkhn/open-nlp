import json
import os
from pathlib import Path
from openai import OpenAI
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def replace_tokens_with_gpt4(text):
    """Replace anonymized tokens with fake but realistic data using GPT-4."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that replaces anonymized medical tokens with realistic but fake data. Maintain medical context and consistency."},
                {"role": "user", "content": f"Replace anonymized tokens in this text with realistic but fake medical data. Preserve the medical context: {text}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in GPT-4 processing: {e}")
        return text  # Return original text if API call fails


def filter_entries_by_keyword_count(input_file, output_file, min_keywords=50, max_keywords=100):
    # Process each entry in the JSONL file
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            filtered_data = {}

            for key, entries in data.items():
                filtered_entries = []
                for entry in entries:
                    if min_keywords <= len(entry["keywords"].split(",")) <= max_keywords:
                        # Replace anonymized tokens in the text
                        entry["text"] = replace_tokens_with_gpt4(entry["text"])
                        # Add rate limiting to avoid API throttling
                        time.sleep(1)
                        filtered_entries.append(entry)
                
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
