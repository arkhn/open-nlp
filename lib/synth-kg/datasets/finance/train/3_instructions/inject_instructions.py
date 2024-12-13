import pandas as pd

# Read the input parquet file and split into seed/gen datasets
df = pd.read_parquet("datasets/finance/train/2_keyword/keyword_data.parquet")
seed_df = df[:500].copy()
gen_df = df[500:].copy()

# Load prompt templates
prompt = open("datasets/finance/train/3_instructions/prompt.txt").read()
# Process seed dataset
seed_df["new_instruction"] = (
    seed_df["instruction_keywords"].str.join(", ").apply(lambda x: prompt.replace("{keywords}", x))
)
seed_df["new_instruction"] = seed_df.apply(
    lambda row: row["new_instruction"].replace("{sentiment}", row["response"]), axis=1
)

seed_df["output"] = seed_df["response"]
seed_df["response"] = seed_df["instruction"]
seed_df["instruction"] = seed_df["new_instruction"]

# Select and rename columns for seed dataset
output_columns = [
    "instruction",
    "response",
    "output",
]
seed_df = seed_df[output_columns]

# Save seed dataset
seed_output_path = "datasets/finance/train/4_generate_private_seed/private_seed_data.parquet"
seed_df.to_parquet(seed_output_path, index=False)

gen_df["new_instruction"] = (
    gen_df["instruction_keywords"].str.join(", ").apply(lambda x: prompt.replace("{keywords}", x))
)
gen_df["new_instruction"] = gen_df.apply(
    lambda row: row["new_instruction"].replace("{sentiment}", row["response"]), axis=1
)
gen_df["output"] = gen_df["response"]
gen_df["response"] = gen_df["instruction"]
gen_df["instruction"] = gen_df["new_instruction"]

# Save gen dataset
gen_output_path = "datasets/finance/train/3_instructions/gen_data.parquet"
gen_df.to_parquet(gen_output_path, index=False)
