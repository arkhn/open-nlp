import asyncio
import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm.asyncio import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def replace_tokens_with_gpt4(text):
    """Replace anonymized tokens with fake but realistic data using GPT-4."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a text processor that replaces anonymized tokens with "
                    "realistic data. Only output the processed text with no additional "
                    "commentary or explanations.",
                },
                {
                    "role": "user",
                    "content": f"Replace all anonymized tokens in this text with realistic data. "
                    f"Return only the processed text: {text}",
                },
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in GPT-4 processing: {e}")
        return text  # Return original text if API call fails


async def filter_entries_by_keyword_count(
    input_file, output_file, min_keywords=50, max_keywords=100
):
    # Process each entry in the JSONL file
    json_list = []
    tasks = []
    with open(input_file, "r") as infile:
        lines = infile.readlines()
        for line in lines:

            async def process_line(line):
                data = json.loads(line)
                filtered_data = {}

                for key, entries in data.items():
                    filtered_entries = []
                    for entry in entries:
                        if min_keywords <= len(entry["keywords"].split(",")) <= max_keywords:
                            # Replace anonymized tokens in the text
                            entry["text"] = replace_tokens_with_gpt4(entry["text"])
                            # Add rate limiting to avoid API throttling
                            filtered_entries.append(entry)

                    if filtered_entries:
                        filtered_data[key] = filtered_entries

                if filtered_data:
                    json_list.append(filtered_data)

            tasks.append(asyncio.create_task(process_line(line)))

        await tqdm.gather(*tasks)
        with open(output_file, "w") as outfile:
            for json_obj in json_list:
                json.dump(json_obj, outfile)
                outfile.write("\n")


# Define input and output file paths
Path("post-processed/").mkdir(parents=True, exist_ok=True)
input_file_path = "keywords_extraction/mimic-style-transfer.jsonl"
output_file_path = "post-processed/filtered_entries.jsonl"

# Call the function with file paths
asyncio.run(filter_entries_by_keyword_count(input_file_path, output_file_path))
