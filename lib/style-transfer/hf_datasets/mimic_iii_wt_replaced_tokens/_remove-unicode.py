import json

input_file = "post-processed/filtered_entries.jsonl"
output_file = "post-processed/filtered_entries-utf8.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(
    output_file, "w", encoding="utf-8"
) as f_out:
    for line in f_in:
        # Parse each line as JSON

        entry = json.loads(line.encode("utf-8"))

        # Write the JSON object back as a string without ASCII escaping
        f_out.write(" ".join(json.dumps(entry, ensure_ascii=False).encode("utf-8").decode()))
        f_out.write("\n")  # Newline to keep the JSONL format
